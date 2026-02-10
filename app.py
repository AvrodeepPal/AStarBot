# app.py (Streamlit)

import streamlit as st
from rag.engine import RAGEngine
from rag.prompt import INITIAL_MESSAGE

# Page configuration
st.set_page_config(
    page_title="AStarBot",
    page_icon="⭐",
    layout="centered",
)

st.title("AStarBot")
st.caption("Avrodeep Pal's personal AI assistant")
st.info("**Beta Preview**: This AI is experimental and may produce inaccurate information. Please verify details independently.")

# Load RAG engine once per session
@st.cache_resource
def load_engine():
    return RAGEngine()

engine = load_engine()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": INITIAL_MESSAGE}
    ]

if "summary" not in st.session_state:
    st.session_state.summary = None

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me about Avrodeep…")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    recent_messages = [
        m for m in st.session_state.messages
        if m["role"] in {"user", "assistant"}
    ][-12:]

    result = engine.chat(
        question=user_input,
        recent_messages=recent_messages,
        summary=st.session_state.summary,
    )

    answer = result["answer"]
    st.session_state.summary = result["updated_summary"]

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
