"""Multi-tier Groq LLM chain.

Four models, each with one job:

    guard       meta-llama/llama-prompt-guard-2-86m  binary safe/unsafe screen
    primary     openai/gpt-oss-20b                   main grounded answer
    fallback    openai/gpt-oss-120b                  used when primary fails
    summarizer  openai/gpt-oss-20b                   3-5 line memory summary

Fallback is explicit try/except in `ainvoke_chat` / `astream_chat`; nothing
retries silently. Every call is logged with `request_id` and the tier that
served it (0 = primary, 1 = fallback, -1 = all failed).

The request path is async (`ainvoke` / `astream` on ChatGroq) so the guard
call can run concurrently with retrieval. `invoke_summarize` stays sync for
`rag.memory.summarize_conversation`'s synchronous callers.
"""

import logging
import re
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from rag.config import settings

log = logging.getLogger("astarbot.llm")

Messages = list[tuple[str, str]]  # [(role, content), ...]

_UNSAFE_WORDS = re.compile(r"\b(unsafe|jailbreak|malicious|injection)\b", re.IGNORECASE)
_FLOAT = re.compile(r"^\s*([01](?:\.\d+)?|\.\d+)\s*$")


def _make_llm(model: str, temperature: float, max_tokens: int, reasoning: bool) -> ChatGroq:
    """Build a ChatGroq client, passing reasoning_effort only where supported."""
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": settings.groq_api_key,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 0,  # fallback tier is our retry policy
    }
    if reasoning and settings.reasoning_effort and "reasoning_effort" in ChatGroq.model_fields:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    return ChatGroq(**kwargs)


def _to_lc_messages(messages: Messages):
    out = []
    for role, content in messages:
        out.append(SystemMessage(content=content) if role == "system" else HumanMessage(content=content))
    return out


def _raw_text(response) -> str:
    """A LangChain message/chunk's content as a string, whitespace intact."""
    content = getattr(response, "content", response)
    if isinstance(content, list):  # some providers return content parts
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def _text(response) -> str:
    """Normalise a LangChain response's content to a plain, stripped string.

    Never use this on stream chunks: the leading space of a token is content.
    """
    return _raw_text(response).strip()


class LLMChain:
    def __init__(self) -> None:
        self.guard = _make_llm(settings.guard_model, 0.0, 16, reasoning=False)
        self.primary = _make_llm(
            settings.primary_llm_model, settings.temperature, settings.max_answer_tokens, reasoning=True
        )
        self.fallback = _make_llm(
            settings.fallback_llm_model, settings.temperature, settings.max_answer_tokens, reasoning=True
        )
        self.summarizer = _make_llm(
            settings.summarizer_llm_model, settings.summarizer_temperature, 300, reasoning=True
        )
        self.tiers = (
            (0, settings.primary_llm_model, self.primary),
            (1, settings.fallback_llm_model, self.fallback),
        )

    # ---- guard -----------------------------------------------------------

    @staticmethod
    def parse_guard_output(text: str) -> bool:
        """Interpret prompt-guard output. Returns True when the input is SAFE.

        Prompt Guard 2 on Groq returns a jailbreak probability as text
        (e.g. "0.0012"); some deployments return a label instead. Both are
        handled; anything unparseable is treated as safe (fail-open).
        """
        text = (text or "").strip()
        m = _FLOAT.match(text)
        if m:
            return float(m.group(1)) < settings.guard_threshold
        if _UNSAFE_WORDS.search(text) and not re.search(r"\bsafe\b", text, re.IGNORECASE):
            return False
        return True

    async def acheck_safe(self, text: str, request_id: str = "-") -> bool:
        """Screen raw user input with the prompt-guard model.

        Fail-open: if the guard cannot be reached we still serve the request,
        because the regex screen already ran and the prompt itself carries
        injection defenses.
        """
        try:
            resp = await self.guard.ainvoke([HumanMessage(content=text)])
            raw = _text(resp)
            safe = self.parse_guard_output(raw)
            log.info("guard_ok", extra={"request_id": request_id, "safe": safe, "raw": raw[:32]})
            return safe
        except Exception as exc:  # noqa: BLE001 - deliberately broad, logged
            # ERROR, not WARNING: the request is about to be served with one
            # guardrail layer missing, and that must be visible in the logs at
            # the level anyone actually alerts on.
            log.error(
                "guard_fail_open",
                extra={
                    "request_id": request_id,
                    "err": str(exc)[:200],
                    "fail_open": True,
                    "guard_model": settings.guard_model,
                },
            )
            return True

    # ---- chat ------------------------------------------------------------

    async def ainvoke_chat(self, messages: Messages, request_id: str = "-") -> tuple[str | None, int]:
        """Try primary, then fallback. Returns (answer or None, tier_used)."""
        lc_messages = _to_lc_messages(messages)
        for tier, model, llm in self.tiers:
            try:
                answer = _text(await llm.ainvoke(lc_messages))
                if not answer:
                    # Typically the reasoning phase consumed max_tokens.
                    raise ValueError("empty completion")
                log.info("llm_ok", extra={"request_id": request_id, "tier": tier, "model": model})
                return answer, tier
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "llm_fail",
                    extra={"request_id": request_id, "tier": tier, "model": model, "err": str(exc)[:200]},
                )
        log.error("llm_all_failed", extra={"request_id": request_id})
        return None, -1

    async def astream_chat(
        self, messages: Messages, request_id: str = "-"
    ) -> AsyncIterator[tuple[int, str]]:
        """Yield (tier, text) chunks as the model generates them.

        A tier that fails or streams nothing BEFORE its first token hands over
        to the next tier. A tier that breaks AFTER tokens were sent stops the
        stream: the visitor already has half an answer, and splicing a second
        model's answer onto it would be worse than a short one. Yields nothing
        at all when every tier fails; the caller substitutes STATIC_FALLBACK.
        Output guardrails are the caller's job (guardrails.StreamSanitizer).
        """
        lc_messages = _to_lc_messages(messages)
        for tier, model, llm in self.tiers:
            ctx = {"request_id": request_id, "tier": tier, "model": model}
            yielded = False
            try:
                async for chunk in llm.astream(lc_messages):
                    text = _raw_text(chunk)
                    if text:
                        yielded = True
                        yield tier, text
                if yielded:
                    log.info("llm_stream_ok", extra=ctx)
                    return
                log.warning("llm_stream_empty", extra=ctx)
            except Exception as exc:  # noqa: BLE001
                if yielded:
                    log.error("llm_stream_broken", extra={**ctx, "err": str(exc)[:200]})
                    return
                log.warning("llm_stream_fail", extra={**ctx, "err": str(exc)[:200]})
        log.error("llm_stream_all_failed", extra={"request_id": request_id})

    # ---- summarise -------------------------------------------------------

    def invoke_summarize(self, messages: Messages, request_id: str = "-") -> str | None:
        """Single attempt on the summarizer tier; None on failure."""
        try:
            text = _text(self.summarizer.invoke(_to_lc_messages(messages)))
            log.info("summary_ok", extra={"request_id": request_id, "model": settings.summarizer_llm_model})
            return text or None
        except Exception as exc:  # noqa: BLE001
            log.warning("summary_fail", extra={"request_id": request_id, "err": str(exc)[:200]})
            return None

    async def ainvoke_summarize(self, messages: Messages, request_id: str = "-") -> str | None:
        """Async twin of `invoke_summarize`, for the request path."""
        try:
            text = _text(await self.summarizer.ainvoke(_to_lc_messages(messages)))
            log.info("summary_ok", extra={"request_id": request_id, "model": settings.summarizer_llm_model})
            return text or None
        except Exception as exc:  # noqa: BLE001
            log.warning("summary_fail", extra={"request_id": request_id, "err": str(exc)[:200]})
            return None
