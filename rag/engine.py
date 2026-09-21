"""Pipeline orchestrator.

`RAGEngine.chat` runs eight explicit steps and never raises to the caller:

    1. input guardrails      sanitise / bound the request
    2. regex injection screen
    3. prompt-guard LLM      fast safe/unsafe classification (optional)
    4. follow-up rewrite     "how did he do it" -> "<previous question> how did he do it"
                             for RETRIEVAL only; the model answers the question as typed
    5. retrieval             BGE query embedding -> Pinecone top-k (+ MMR)
    6. prompt + LLM chain    primary -> fallback -> static fallback string;
                             the last few turns ride in the user prompt as data
    7. output guardrails     strip leaks, cap length
    8. summarisation         only when the client window is full

Everything returned is a plain dict {"answer", "updated_summary"} plus an
optional "debug" block for the CLI/Streamlit interfaces.
"""

import logging
import time
from uuid import uuid4

from rag.config import settings
from rag.followup import standalone_query
from rag.guardrails import GuardrailError, contains_injection, sanitize_answer, validate_chat_input
from rag.knowledge import context_block
from rag.llm import LLMChain
from rag.memory import summarize_conversation
from rag.prompt import (
    PROMPT_VERSION,
    REFUSAL_JAILBREAK,
    REFUSAL_OFFTOPIC,
    REFUSAL_UNSAFE,
    STATIC_FALLBACK,
    build_messages,
)
from rag.retriever import PineconeRetriever

log = logging.getLogger("astarbot.engine")


class RAGEngine:
    def __init__(self) -> None:
        self.retriever = PineconeRetriever()
        self.llm = LLMChain()
        log.info(
            "engine_ready",
            extra={
                "prompt_version": PROMPT_VERSION,
                "embedding_model": settings.embedding_model,
                "top_k": settings.top_k,
                "primary_model": settings.primary_llm_model,
            },
        )

    def chat(
        self,
        question: str,
        recent_messages: list[dict] | None = None,
        summary: str | None = None,
        include_debug: bool = False,
    ) -> dict:
        request_id = uuid4().hex[:8]
        t0 = time.perf_counter()
        try:
            return self._chat(question, recent_messages or [], summary, request_id, t0, include_debug)
        except Exception:  # noqa: BLE001 - last line of defence, always logged
            log.exception("chat_unhandled", extra={"request_id": request_id})
            return {"answer": STATIC_FALLBACK, "updated_summary": summary}

    def _chat(
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
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "chat_done",
                extra={
                    "request_id": request_id,
                    "prompt_version": PROMPT_VERSION,
                    "embedding_model": settings.embedding_model,
                    "top_k": settings.top_k,
                    "retrieval_top_score": debug.get("top_score"),
                    "llm_tier_used": tier,
                    "outcome": outcome,
                    "latency_ms": latency_ms,
                },
            )
            result = {"answer": answer, "updated_summary": updated_summary}
            if include_debug:
                debug.update({"outcome": outcome, "llm_tier": tier, "latency_ms": latency_ms})
                result["debug"] = debug
            return result

        # 1. input guardrails
        try:
            q, msgs, summ = validate_chat_input(question, recent_messages, summary)
        except GuardrailError as exc:
            log.info("input_rejected", extra={"request_id": request_id, "reason": str(exc)})
            return done(STATIC_FALLBACK, summary, "input_rejected")
        log.debug("question", extra={"request_id": request_id, "q": q})

        # 2. regex injection screen
        if contains_injection(q):
            log.info("injection_blocked", extra={"request_id": request_id, "layer": "regex"})
            return done(REFUSAL_JAILBREAK, summ, "injection_regex")

        # 3. prompt-guard LLM
        if settings.enable_prompt_guard_llm and not self.llm.check_safe(q, request_id):
            log.info("injection_blocked", extra={"request_id": request_id, "layer": "prompt_guard"})
            return done(REFUSAL_UNSAFE, summ, "injection_guard")

        # 4. follow-up rewrite (retrieval only)
        retrieval_q = standalone_query(q, msgs)
        if retrieval_q != q:
            log.info("followup_rewritten", extra={"request_id": request_id})
            debug["retrieval_query"] = retrieval_q

        # 5. retrieval
        contexts = self.retriever.retrieve(retrieval_q, request_id)
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
            return done(REFUSAL_OFFTOPIC, summ, "no_context")

        # 6. prompt assembly + LLM chain
        # Each block carries title + text + links so the model can name the
        # topic and reuse the URLs; tags/ids stay out of the prompt. The last
        # few turns go in as a labelled data block, never as real chat turns.
        messages = build_messages(
            [context_block(c) for c in contexts],
            summ,
            q,
            recent_messages=msgs,
            recent_turns=settings.recent_turns_in_prompt,
        )
        raw_answer, tier = self.llm.invoke_chat(messages, request_id)
        if not raw_answer:
            return done(STATIC_FALLBACK, summ, "llm_failed", tier)

        # 7. output guardrails
        answer = sanitize_answer(raw_answer)
        if not answer:
            return done(STATIC_FALLBACK, summ, "empty_answer", tier)

        # 8. conditional summarisation
        new_summary = summ
        if settings.enable_summary and len(msgs) >= settings.summary_trigger_after:
            new_summary = summarize_conversation(summ, msgs, self.llm, request_id)
            debug["summarized"] = new_summary != summ

        return done(answer, new_summary, "answered", tier)
