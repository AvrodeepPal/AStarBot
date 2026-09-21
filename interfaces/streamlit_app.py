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
"""

import os
import sys
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


def call_chat(question: str, recent_messages: list[dict], summary: str | None) -> dict:
    r = requests.post(
        f"{API_BASE}/chat",
        json={"question": question, "recent_messages": recent_messages, "summary": summary},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()  # {"answer": ..., "updated_summary": ...}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_version() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/version", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


version_info = fetch_version()
initial_message = (version_info or {}).get("initial_message", FALLBACK_INITIAL_MESSAGE)


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
            f"- **Embeddings:** `{version_info['embedding_model']}` ({version_info['embedding_dim']}-d)\n"
            f"- **Retrieval:** top-k = {version_info['top_k']}"
        )
    else:
        st.warning(f"Can't reach the AStarBot API at `{API_BASE}`.")
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

user_input = st.chat_input("Ask me about Avrodeep…")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    result = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = call_chat(
                    question=user_input,
                    recent_messages=st.session_state.messages,
                    summary=st.session_state.summary,
                )
            except requests.RequestException as exc:
                st.error(f"Couldn't reach AStarBot's API: {exc}")
        if result:
            st.markdown(result["answer"])

    if result:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.session_state.messages = trim_window(
            st.session_state.messages, st.session_state.summary, result["updated_summary"]
        )
        st.session_state.summary = result["updated_summary"]
