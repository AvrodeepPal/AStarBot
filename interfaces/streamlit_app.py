"""Streamlit beta-tester UI.

    streamlit run interfaces/streamlit_app.py

On Streamlit Community Cloud, secrets are exposed via `st.secrets`; they are
copied into the environment below so `rag.config.Settings` picks them up.
"""

import os
import sys
from pathlib import Path

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
from rag.config import settings  # noqa: E402
from rag.engine import RAGEngine  # noqa: E402
from rag.log import setup_logging  # noqa: E402
from rag.prompt import INITIAL_MESSAGE, PROMPT_VERSION  # noqa: E402

st.set_page_config(page_title="AStarBot", page_icon="⭐", layout="centered")


@st.cache_resource(show_spinner="Loading AStarBot (embedding model + vector index)…")
def load_engine() -> RAGEngine:
    setup_logging(settings.log_level)
    return RAGEngine()


def reset_conversation() -> None:
    st.session_state.messages = [{"role": "assistant", "content": INITIAL_MESSAGE}]
    st.session_state.summary = None


# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("AStarBot v2 · beta")
    st.markdown(
        f"- **Prompt:** `{PROMPT_VERSION}`\n"
        f"- **Primary LLM:** `{settings.primary_llm_model}`\n"
        f"- **Fallback LLM:** `{settings.fallback_llm_model}`\n"
        f"- **Embeddings:** `{settings.embedding_model}` ({settings.embedding_dim}-d)\n"
        f"- **Retrieval:** top-k = {settings.top_k}\n"
        f"- **Guard model:** `{settings.guard_model}`"
        + ("" if settings.enable_prompt_guard_llm else " (off)")
    )
    show_debug = st.toggle("Show retrieval debug", value=settings.debug)
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

engine = load_engine()

if "messages" not in st.session_state:
    reset_conversation()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me about Avrodeep…")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = engine.chat(
                question=user_input,
                recent_messages=st.session_state.messages,
                summary=st.session_state.summary,
                include_debug=show_debug,
            )
        st.markdown(result["answer"])
        if show_debug and "debug" in result:
            st.json(result["debug"], expanded=False)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.messages = trim_window(
        st.session_state.messages, st.session_state.summary, result["updated_summary"]
    )
    st.session_state.summary = result["updated_summary"]
