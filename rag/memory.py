"""Stateless conversation summarisation.

The server keeps no session state. Once the client's message window reaches
`summary_trigger_after` turns, the engine folds the previous summary and the
recent turns into a fresh 3-5 line summary and returns it to the client, who
then trims its window and sends the summary back on the next request.
"""

import logging

from rag.config import settings
from rag.guardrails import CONTROL_CHARS
from rag.llm import LLMChain
from rag.prompt import SUMMARY_PROMPT

log = logging.getLogger("astarbot.memory")


def _format_transcript(previous_summary: str | None, recent_messages: list[dict]) -> str:
    parts: list[str] = []
    if previous_summary:
        parts.append(f"Previous summary:\n{previous_summary}")
    lines = [f"{m['role'].upper()}: {m['content']}" for m in recent_messages]
    parts.append("Recent messages:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _clip_lines(text: str, max_lines: int) -> str:
    lines = [ln.strip() for ln in CONTROL_CHARS.sub("", text).splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])


def summarize_conversation(
    previous_summary: str | None,
    recent_messages: list[dict],
    llm: LLMChain,
    request_id: str = "-",
) -> str | None:
    """Return an updated summary, or the previous one if nothing could be produced."""
    if not recent_messages:
        return previous_summary

    messages = [
        ("system", SUMMARY_PROMPT),
        ("user", _format_transcript(previous_summary, recent_messages)),
    ]
    raw = llm.invoke_summarize(messages, request_id)
    if not raw:
        return previous_summary

    summary = _clip_lines(raw, settings.summary_max_lines)
    if len(summary) > settings.max_summary_chars:
        summary = summary[: settings.max_summary_chars].rsplit(" ", 1)[0] + "…"
    return summary or previous_summary
