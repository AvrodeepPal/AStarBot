"""Streamlit beta-tester UI.

    streamlit run interfaces/streamlit_app.py

This talks to the AStarBot API (deployed on Railway) over HTTP rather than
instantiating RAGEngine in-process: the embedding model + torch don't fit in
Streamlit Community Cloud's 1 GB RAM cap, and the whole point of the split is
to run the engine on Railway. Point ASTARBOT_API_URL at the deployed API via
Streamlit secrets; it falls back to a local FastAPI instance for dev.

On Streamlit Community Cloud, secrets are exposed via `st.secrets`; they are
copied into the environment below so `ASTARBOT_API_URL` is visible to
`os.environ.get`.

Latency handling (v2.4.0):
    - once per browser session a background thread pings /health, so a
      Railway container that went to sleep starts waking while the visitor
      is still reading the page, not after they press Enter
    - answers stream from POST /chat/stream and render token by token;
      a spinner covers only the wait for the first token
    - the summary is refreshed with POST /summarize after the answer is on
      screen, and only once the window is full
    - if the API reports streaming disabled, the app falls back to /chat
"""

import itertools
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import requests
import streamlit as st

# Streamlit puts the script's directory on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # st.secrets raises if no secrets file exists locally
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:  # noqa: BLE001
    pass

from interfaces.session import trim_window  # noqa: E402

st.set_page_config(page_title="AStarBot", page_icon="⭐", layout="centered")

API_BASE = os.environ.get("ASTARBOT_API_URL", "http://localhost:8000").rstrip("/")

# Used only if the API is unreachable when the page first loads; once /version
# answers, the real prompt-defined greeting takes over.
FALLBACK_INITIAL_MESSAGE = (
    "Hi there! I'm AStarBot, Avrodeep's AI assistant, here to chat in his stead "
    "while he's offline. Ask me about his education, projects, skills, or what "
    "he's into outside of code. How about we start with a quick intro?"
)


# (connect, read) timeouts. The read timeout is the longest gap allowed
# between bytes, which covers a Railway cold start before the first token.
TIMEOUT = (15, 120)
DEFAULT_SUMMARY_TRIGGER = 10  # mirrors SUMMARY_TRIGGER_AFTER if /version is unreachable


def warm_backend() -> None:
    """Fire-and-forget /health ping, once per browser session.

    Runs in a daemon thread so the page renders immediately; by the time the
    visitor has typed a question the container is (or is nearly) awake.
    Deliberately NOT @st.cache_resource: that caches process-wide, so after
    the first visitor it would never ping again, even with the backend asleep.
    """
    if st.session_state.get("backend_warmed"):
        return
    st.session_state.backend_warmed = True

    def ping() -> None:
        try:
            requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
        except requests.RequestException:
            pass

    threading.Thread(target=ping, daemon=True).start()


def call_chat(question: str, recent_messages: list[dict], summary: str | None) -> dict:
    r = requests.post(
        f"{API_BASE}/chat",
        json={"question": question, "recent_messages": recent_messages, "summary": summary},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()  # {"answer": ..., "updated_summary": ...}


def open_stream(question: str, recent_messages: list[dict], summary: str | None) -> Iterator[str]:
    """POST /chat/stream and return an iterator over decoded text chunks."""
    r = requests.post(
        f"{API_BASE}/chat/stream",
        json={"question": question, "recent_messages": recent_messages, "summary": summary},
        stream=True,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    r.encoding = "utf-8"  # incremental decode: a split multi-byte char is never mangled
    return (chunk for chunk in r.iter_content(chunk_size=None, decode_unicode=True) if chunk)


def call_summarize(recent_messages: list[dict], summary: str | None) -> str | None:
    r = requests.post(
        f"{API_BASE}/summarize",
        json={"recent_messages": recent_messages, "summary": summary},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["updated_summary"]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_version() -> dict:
    # Raises on failure: st.cache_data does not cache exceptions, so an API
    # that was asleep on first load is retried on the next rerun instead of
    # being remembered as "unreachable" for five minutes.
    r = requests.get(f"{API_BASE}/version", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_version() -> dict | None:
    try:
        return _fetch_version()
    except requests.RequestException:
        return None


warm_backend()
version_info = fetch_version()
initial_message = (version_info or {}).get("initial_message", FALLBACK_INITIAL_MESSAGE)
# A reachable API that doesn't advertise streaming is pre-v2.4 or has it
# switched off; an unreachable one is assumed current (a 404 falls back anyway).
use_streaming = version_info.get("streaming_enabled", False) if version_info else True
summary_trigger = (
    (version_info or {}).get("client_window", {}).get("summary_trigger_after", DEFAULT_SUMMARY_TRIGGER)
)


def reset_conversation() -> None:
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
    st.session_state.summary = None


# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("AStarBot v2 · beta")
    if version_info:
        st.markdown(
            f"- **Prompt:** `{version_info['prompt_version']}`\n"
            f"- **Primary LLM:** `{version_info['primary_model']}`\n"
            f"- **Fallback LLM:** `{version_info['fallback_model']}`\n"
            f"- **Embeddings:** `{version_info['embedding_model']}` ({version_info['embedding_dim']}-d"
            f"{', ' + version_info['embedding_backend'] if version_info.get('embedding_backend') else ''})\n"
            f"- **Retrieval:** top-k = {version_info['top_k']}\n"
            f"- **Streaming:** {'on' if use_streaming else 'off'}"
        )
    else:
        st.info(f"Waking the AStarBot API at `{API_BASE}` — the first reply may take a little longer.")
    if st.button("Clear conversation", use_container_width=True):
        reset_conversation()
        st.rerun()
    if st.session_state.get("summary"):
        with st.expander("Conversation summary"):
            st.write(st.session_state.summary)

# ---- main ----------------------------------------------------------------
st.title("AStarBot")
st.caption("Avrodeep Pal's personal AI assistant")
st.info(
    "**Beta Preview**: This AI is experimental and may produce inaccurate information. "
    "For any good or bad qualities found, please feel free to contact and positively "
    "share your reviews. Thank you!"
)

if "messages" not in st.session_state:
    reset_conversation()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



def answer_streaming(question: str) -> tuple[str | None, str | None]:
    """Render the answer as it streams. Returns (answer, updated_summary)."""
    previous = st.session_state.summary
    try:
        with st.spinner("Thinking…"):
            chunks = open_stream(question, st.session_state.messages, previous)
            first = next(chunks, "")  # the spinner covers only the wait for this
        answer = st.write_stream(itertools.chain([first], chunks))
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return answer_blocking(question)  # streaming switched off since /version was cached
        st.error(f"Couldn't reach AStarBot's API: {exc}")
        return None, previous
    except requests.RequestException as exc:
        st.error(f"Couldn't reach AStarBot's API: {exc}")
        return None, previous
    if not isinstance(answer, str) or not answer:
        return None, previous

    # Summaries don't ride in a text stream. Once the window is full, fold
    # it (including this exchange) into a fresh summary; the answer is
    # already on screen, so this wait is off the perceived critical path.
    window = [
        *st.session_state.messages,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    updated = previous
    if len(window) >= summary_trigger:
        try:
            updated = call_summarize(window[-20:], previous)  # 20 = API's MAX_MESSAGES_SENT
        except requests.RequestException:
            pass  # keep the old summary; the next full window retries
    return answer, updated


def answer_blocking(question: str) -> tuple[str | None, str | None]:
    """Pre-v2.4 path via POST /chat, used when streaming is disabled."""
    previous = st.session_state.summary
    try:
        with st.spinner("Thinking…"):
            result = call_chat(question, st.session_state.messages, previous)
    except requests.RequestException as exc:
        st.error(f"Couldn't reach AStarBot's API: {exc}")
        return None, previous
    st.markdown(result["answer"])
    return result["answer"], result["updated_summary"]


user_input = st.chat_input("Ask me about Avrodeep…")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer, updated_summary = (answer_streaming if use_streaming else answer_blocking)(user_input)

    if answer:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.messages = trim_window(
            st.session_state.messages, st.session_state.summary, updated_summary
        )
        st.session_state.summary = updated_summary
