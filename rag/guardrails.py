"""Input and output guardrails.

Three cheap, deterministic defences that run without any model call:

    validate_chat_input  strip control characters, enforce length limits,
                         whitelist roles, bound the message window
    contains_injection   regex screen for common prompt-override phrasing
    sanitize_answer      remove leaked prompt markers / reasoning tags and
                         cap answer length

The prompt-guard LLM (rag.llm) and the INJECTION_DEFENSE prompt block are the
second and third layers; this module is the first and always runs.
"""

import re

from rag.config import settings
from rag.prompt import LEAK_MARKERS

ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant"})

# Null bytes, C0 controls (except \t \n \r), DEL. Tabs/newlines are kept
# because multi-line questions are legitimate.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ANSI_ESCAPES = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|all|your)\s*(previous|prior|above|instructions?|rules?)?",
    r"\byou\s+are\s+now\b",
    r"\bfrom\s+now\s+on\s+you\b",
    r"\bact\s+as\s+(a|an|my|the)\b",
    r"\bpretend\s+(to\s+be|you\s+are)\b",
    r"\brole-?play\s+as\b",
    r"\bsystem\s*prompt\b",
    r"\b(reveal|show|print|repeat|output|leak)\s+(me\s+)?(your|the|its)\s+(instructions?|prompt|rules|system\s+message)",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
    r"\bDAN\s+mode\b",
    r"\bdo\s+anything\s+now\b",
)
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)

# gpt-oss style reasoning tags; langchain normally separates them but we
# never want them reaching a user if a provider passes them through.
_THINK_TAGS = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_LEAK_RE = re.compile("|".join(re.escape(m) for m in LEAK_MARKERS))


class GuardrailError(ValueError):
    """Raised when input cannot be made safe (e.g. empty after cleaning)."""


def sanitize_text(text: str | None, limit: int) -> str:
    """Strip control/ANSI sequences, trim, and truncate at a word boundary."""
    cleaned = ANSI_ESCAPES.sub("", text or "")
    cleaned = CONTROL_CHARS.sub("", cleaned).strip()
    if not cleaned:
        raise GuardrailError("Empty text after sanitization")
    if len(cleaned) > limit:
        head = cleaned[:limit]
        cut = head.rsplit(" ", 1)[0] if " " in head else head
        cleaned = cut.rstrip() + "…"
    return cleaned


def validate_chat_input(
    question: str,
    recent_messages: list[dict] | None,
    summary: str | None,
) -> tuple[str, list[dict], str | None]:
    """Return (clean_question, clean_messages, clean_summary).

    - question: required, <= max_question_chars
    - recent_messages: last max_messages_sent only; unknown roles and
      empty contents are dropped silently
    - summary: optional, <= max_summary_chars
    """
    clean_question = sanitize_text(question, settings.max_question_chars)

    messages = list(recent_messages or [])
    if len(messages) > settings.max_messages_sent:
        messages = messages[-settings.max_messages_sent:]

    clean_messages: list[dict] = []
    for m in messages:
        role = (m or {}).get("role")
        if role not in ALLOWED_ROLES:
            continue
        try:
            content = sanitize_text(m.get("content"), settings.max_message_chars)
        except GuardrailError:
            continue
        clean_messages.append({"role": role, "content": content})

    clean_summary: str | None = None
    if summary:
        try:
            clean_summary = sanitize_text(summary, settings.max_summary_chars)
        except GuardrailError:
            clean_summary = None

    return clean_question, clean_messages, clean_summary


def contains_injection(text: str) -> bool:
    """True if the text matches a known prompt-override pattern."""
    return bool(_INJECTION_RE.search(text or ""))


def sanitize_answer(answer: str) -> str:
    """Remove leaked internal markers and cap the length of a model answer."""
    cleaned = _THINK_TAGS.sub("", answer or "")
    cleaned = _LEAK_RE.sub("", cleaned)
    cleaned = CONTROL_CHARS.sub("", cleaned).strip()
    if len(cleaned) > settings.max_answer_chars:
        head = cleaned[: settings.max_answer_chars]
        # Prefer the last sentence boundary, then a word boundary.
        idx = max(head.rfind(". "), head.rfind("! "), head.rfind("? "), head.rfind("\n"))
        if idx > settings.max_answer_chars // 2:
            cleaned = head[: idx + 1].rstrip()
        else:
            cleaned = head.rsplit(" ", 1)[0].rstrip() + "…"
    return cleaned
