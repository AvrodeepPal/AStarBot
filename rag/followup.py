"""Standalone-query rewriting for follow-up questions.

Retrieval sees one string. "How did he do it, what data and algos?" contains
nothing that points at the credit-risk entry, so Pinecone returns whatever
loosely matches "data" and "algorithms" and the model has nothing to answer
from, however much conversation history it can see.

`standalone_query` is a deterministic, zero-cost heuristic: if the question
leans on a referent (it / that / this / them ...) or opens like a
continuation ("and ...", "what about ...", "more details"), the previous
USER turn is prepended so the retrieval query carries the topic. The rewrite
is used for retrieval ONLY; the model still answers the question as typed.

Deliberately NOT a length heuristic. "any hobbies" after "introduce him" is
short but self-contained, and gluing the intro question onto it would pull
the wrong entries.
"""

import re

# Pronouns and demonstratives that need an antecedent. "he" / "his" are
# excluded on purpose: nearly every question about Avrodeep contains them.
_REFERENT_RE = re.compile(
    r"\b(it|its|that|this|those|these|them|the same|that one|this one|the project|the paper)\b",
    re.IGNORECASE,
)

# Openers that only make sense as a continuation of the previous turn.
_CONTINUATION_RE = re.compile(
    r"^\s*(and|also|but|what about|how about|why|how so|how come|more|tell me more|"
    r"elaborate|go on|details?|in detail|more details?|expand|explain further|"
    r"anything else|what else)\b",
    re.IGNORECASE,
)


def is_follow_up(question: str) -> bool:
    """True when the question leans on the previous turn for its topic."""
    q = question or ""
    return bool(_CONTINUATION_RE.search(q) or _REFERENT_RE.search(q))


def last_user_turn(recent_messages: list[dict]) -> str | None:
    for m in reversed(recent_messages or []):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return None


def standalone_query(question: str, recent_messages: list[dict]) -> str:
    """Return the string to embed for retrieval.

    Unchanged when the question stands alone or there is no previous user
    turn; otherwise "<previous user turn> <question>" so the topic words
    reach the vector.
    """
    if not is_follow_up(question):
        return question
    previous = last_user_turn(recent_messages)
    if not previous or previous.strip().lower() == question.strip().lower():
        return question
    return f"{previous} {question}"
