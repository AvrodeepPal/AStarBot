"""Pipeline orchestrator.

`RAGEngine.achat` runs eight explicit steps and never raises to the caller:

    1. input guardrails      sanitise / bound the request
    2. regex injection screen
    3. follow-up rewrite     "how did he do it" -> "<previous question> how did he do it"
                             for RETRIEVAL only; the model answers the question as typed
    4. prompt-guard LLM  ┐   run CONCURRENTLY: neither reads the other's output,
    5. retrieval         ┘   so results are identical to running them in turn,
                             and the guard's round trip leaves the critical path
    6. prompt + LLM chain    primary -> fallback -> static fallback string;
                             the last few turns ride in the user prompt as data
    7. output guardrails     strip leaks, cap length
    8. summarisation         only when the client window is full

`astream_chat` shares steps 1-5 and streams step 6 through an incremental
step 7; summarisation is left to POST /summarize (see `asummarize`).
`chat` is a synchronous wrapper for the CLI.

Everything returned is a plain dict {"answer", "updated_summary"} plus an
optional "debug" block for the CLI/Streamlit interfaces.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from rag.config import settings
from rag.followup import standalone_query
from rag.guardrails import (
    GuardrailError,
    StreamSanitizer,
    contains_injection,
    sanitize_answer,
    validate_chat_input,
    validate_history,
)
from rag.knowledge import context_block
from rag.llm import LLMChain
from rag.memory import asummarize_conversation
from rag.prompt import (
    PROMPT_VERSION,
    REFUSAL_JAILBREAK,
    REFUSAL_OFFTOPIC,
    REFUSAL_UNSAFE,
    STATIC_FALLBACK,
    build_messages,
)
from rag.retriever import PineconeRetriever, RetrievalError

log = logging.getLogger("astarbot.engine")


@dataclass
class _Plan:
    """Outcome of steps 1-5: either an early answer or the messages for the LLM."""

    summary: str | None
    messages: list[tuple[str, str]] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    early_answer: str | None = None
    outcome: str = ""


class RAGEngine:
    def __init__(self) -> None:
        self.retriever = PineconeRetriever()
        self.llm = LLMChain()
        self._runner: asyncio.Runner | None = None
        log.info(
            "engine_ready",
            extra={
                "prompt_version": PROMPT_VERSION,
                "embedding_model": settings.embedding_model,
                "embedding_backend": settings.resolved_embedding_backend,
                "top_k": settings.top_k,
                "primary_model": settings.primary_llm_model,
            },
        )

    # ---- entry points ----------------------------------------------------

    def chat(
        self,
        question: str,
        recent_messages: list[dict] | None = None,
        summary: str | None = None,
        include_debug: bool = False,
        request_id: str | None = None,
    ) -> dict:
        """Synchronous wrapper for callers without an event loop (CLI, tests).

        One persistent loop is reused across calls: the Groq and HF clients
        pool connections per loop, and a fresh `asyncio.run` per turn would
        strand them. Never call this from inside a running loop; await
        `achat` there instead.
        """
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(self.achat(question, recent_messages, summary, include_debug, request_id))

    async def achat(
        self,
        question: str,
        recent_messages: list[dict] | None = None,
        summary: str | None = None,
        include_debug: bool = False,
        request_id: str | None = None,
    ) -> dict:
        request_id = request_id or uuid4().hex[:8]
        t0 = time.perf_counter()
        try:
            return await self._achat(question, recent_messages or [], summary, request_id, t0, include_debug)
        except Exception:  # noqa: BLE001 - last line of defence, always logged
            log.exception("chat_unhandled", extra={"request_id": request_id})
            return {"answer": STATIC_FALLBACK, "updated_summary": summary}

    async def astream_chat(
        self,
        question: str,
        recent_messages: list[dict] | None = None,
        summary: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield the answer as sanitised text chunks. Never raises.

        Refusals and fallbacks arrive as a single chunk. The summary is not
        touched here; clients call `asummarize` (POST /summarize) once their
        window is full, after the answer has already been shown.
        """
        request_id = request_id or uuid4().hex[:8]
        t0 = time.perf_counter()
        debug: dict = {}
        try:
            plan = await self._prepare(question, recent_messages or [], summary, request_id, debug)
        except Exception:  # noqa: BLE001
            log.exception("chat_unhandled", extra={"request_id": request_id, "stream": True})
            yield STATIC_FALLBACK
            return
        if plan.early_answer is not None:
            self._log_done(request_id, t0, debug, plan.outcome, None, stream=True)
            yield plan.early_answer
            return

        sanitizer = StreamSanitizer(request_id)
        tier, ttft_ms = -1, None
        try:
            async for chunk_tier, token in self.llm.astream_chat(plan.messages, request_id):
                tier = chunk_tier
                out = sanitizer.feed(token)
                if out:
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - t0) * 1000)
                    yield out
            tail = sanitizer.flush()
            if tail:
                yield tail
        except Exception:  # noqa: BLE001
            log.exception("chat_unhandled", extra={"request_id": request_id, "stream": True})

        if sanitizer.text:
            outcome = "answered"
        else:
            # tier is still -1 only if no model produced a single token.
            outcome = "llm_failed" if tier == -1 else "empty_answer"
            yield STATIC_FALLBACK
        self._log_done(request_id, t0, debug, outcome, tier, stream=True, ttft_ms=ttft_ms)

    async def asummarize(
        self,
        recent_messages: list[dict] | None,
        summary: str | None,
        request_id: str | None = None,
    ) -> str | None:
        """Fold the client's window into a fresh summary (streaming clients).

        Same trigger rule as the non-streaming path: below
        `summary_trigger_after` turns, or with summaries disabled, the summary
        comes back unchanged and no model is called. Never raises.
        """
        request_id = request_id or uuid4().hex[:8]
        try:
            msgs, summ = validate_history(recent_messages, summary)
            if not settings.enable_summary or len(msgs) < settings.summary_trigger_after:
                return summ
            return await asummarize_conversation(summ, msgs, self.llm, request_id)
        except Exception:  # noqa: BLE001
            log.exception("summarize_unhandled", extra={"request_id": request_id})
            return summary

    async def warmup(self) -> None:
        """Best-effort: wake the embedder, open the Pinecone connection and
        seed the retrieval cache with the most common opener. Never raises."""
        try:
            await self.retriever.aretrieve("Who is Avrodeep?", "warmup")
            log.info("warmup_ok")
        except Exception as exc:  # noqa: BLE001
            log.warning("warmup_fail", extra={"err": str(exc)[:200]})

    # ---- pipeline --------------------------------------------------------

    async def _prepare(
        self,
        question: str,
        recent_messages: list[dict],
        summary: str | None,
        request_id: str,
        debug: dict,
    ) -> _Plan:
        """Steps 1-5, shared by the blocking and the streaming paths."""
        # 1. input guardrails
        try:
            q, msgs, summ = validate_chat_input(question, recent_messages, summary)
        except GuardrailError as exc:
            log.info("input_rejected", extra={"request_id": request_id, "reason": str(exc)})
            return _Plan(summary, early_answer=STATIC_FALLBACK, outcome="input_rejected")
        log.debug("question", extra={"request_id": request_id, "q": q})

        # 2. regex injection screen
        if contains_injection(q):
            log.info("injection_blocked", extra={"request_id": request_id, "layer": "regex"})
            return _Plan(summ, early_answer=REFUSAL_JAILBREAK, outcome="injection_regex")

        # 3. follow-up rewrite (retrieval only)
        retrieval_q = standalone_query(q, msgs)
        if retrieval_q != q:
            log.info("followup_rewritten", extra={"request_id": request_id})
            debug["retrieval_query"] = retrieval_q

        # 4 + 5. prompt-guard LLM concurrently with retrieval. The guard is
        # checked first: an unsafe question is refused as unsafe even if
        # retrieval happened to fail. In that rare case the retrieval work
        # is simply discarded.
        retrieval = self.retriever.aretrieve(retrieval_q, request_id)
        if settings.enable_prompt_guard_llm:
            safe, contexts = await asyncio.gather(
                self.llm.acheck_safe(q, request_id), retrieval, return_exceptions=True
            )
            if safe is False:
                log.info("injection_blocked", extra={"request_id": request_id, "layer": "prompt_guard"})
                return _Plan(summ, early_answer=REFUSAL_UNSAFE, outcome="injection_guard")
            if isinstance(contexts, BaseException):
                if not isinstance(contexts, RetrievalError):
                    raise contexts
                return _Plan(summ, early_answer=STATIC_FALLBACK, outcome="retrieval_failed")
        else:
            try:
                contexts = await retrieval
            except RetrievalError:
                return _Plan(summ, early_answer=STATIC_FALLBACK, outcome="retrieval_failed")

        debug["sources"] = [
            {
                "id": c["id"],
                "source": c.get("source", ""),
                "title": c.get("title", ""),
                "score": round(c["score"], 4),
                "priority": c.get("priority"),
                "final_score": round(c.get("final_score", c["score"]), 4),
            }
            for c in contexts
        ]
        debug["top_score"] = contexts[0]["score"] if contexts else None
        if not contexts:
            return _Plan(summ, early_answer=REFUSAL_OFFTOPIC, outcome="no_context")

        # Prompt assembly. Each block carries title + text + links so the
        # model can name the topic and reuse the URLs; tags/ids stay out of
        # the prompt. The last few turns go in as a labelled data block,
        # never as real chat turns.
        messages = build_messages(
            [context_block(c) for c in contexts],
            summ,
            q,
            recent_messages=msgs,
            recent_turns=settings.recent_turns_in_prompt,
        )
        return _Plan(summ, messages=messages, history=msgs)

    async def _achat(
        self,
        question: str,
        recent_messages: list[dict],
        summary: str | None,
        request_id: str,
        t0: float,
        include_debug: bool,
    ) -> dict:
        debug: dict = {"request_id": request_id, "prompt_version": PROMPT_VERSION}

        def done(answer: str, updated_summary: str | None, outcome: str, tier: int | None = None):
            latency_ms = self._log_done(request_id, t0, debug, outcome, tier)
            result = {"answer": answer, "updated_summary": updated_summary}
            if include_debug:
                debug.update({"outcome": outcome, "llm_tier": tier, "latency_ms": latency_ms})
                result["debug"] = debug
            return result

        plan = await self._prepare(question, recent_messages, summary, request_id, debug)
        if plan.early_answer is not None:
            return done(plan.early_answer, plan.summary, plan.outcome)

        # 6. LLM chain
        raw_answer, tier = await self.llm.ainvoke_chat(plan.messages, request_id)
        if not raw_answer:
            return done(STATIC_FALLBACK, plan.summary, "llm_failed", tier)

        # 7. output guardrails
        answer = sanitize_answer(raw_answer, request_id)
        if not answer:
            return done(STATIC_FALLBACK, plan.summary, "empty_answer", tier)

        # 8. conditional summarisation
        new_summary = plan.summary
        if settings.enable_summary and len(plan.history) >= settings.summary_trigger_after:
            new_summary = await asummarize_conversation(plan.summary, plan.history, self.llm, request_id)
            debug["summarized"] = new_summary != plan.summary

        return done(answer, new_summary, "answered", tier)

    @staticmethod
    def _log_done(
        request_id: str,
        t0: float,
        debug: dict,
        outcome: str,
        tier: int | None,
        stream: bool = False,
        ttft_ms: int | None = None,
    ) -> int:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        extra = {
            "request_id": request_id,
            "prompt_version": PROMPT_VERSION,
            "embedding_model": settings.embedding_model,
            "embedding_backend": settings.resolved_embedding_backend,
            "top_k": settings.top_k,
            "retrieval_top_score": debug.get("top_score"),
            "llm_tier_used": tier,
            "outcome": outcome,
            "latency_ms": latency_ms,
            "stream": stream,
        }
        if stream:
            extra["ttft_ms"] = ttft_ms
        log.info("chat_done", extra=extra)
        return latency_ms
