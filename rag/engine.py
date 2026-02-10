# rag/engine.py

import os
from groq import RateLimitError
from langchain_groq import ChatGroq

from rag.retriever import PineconeRetriever
from rag.prompt import build_prompt, FALLBACK_MESSAGE
from rag.memory import summarize_conversation


class RAGEngine:
    def __init__(self):
        self.retriever = PineconeRetriever()

        self.primary_llm = ChatGroq(
            model=os.getenv("PRIMARY_LLM_MODEL"),
            temperature=float(os.getenv("TEMPERATURE", 0.4)),
        )

        self.fallback_llm = ChatGroq(
            model=os.getenv("FALLBACK_LLM_MODEL"),
            temperature=float(os.getenv("TEMPERATURE", 0.4)),
        )

        self.max_recent = int(os.getenv("MAX_RECENT_MESSAGES", 6))
        self.enable_summary = os.getenv("ENABLE_SUMMARY", "true").lower() == "true"

    def chat(
        self,
        question: str,
        recent_messages: list[dict] | None = None,
        summary: str | None = None,
    ) -> dict:
        recent_messages = recent_messages or []

        contexts = self.retriever.retrieve(question)

        if not contexts:
            return {
                "answer": FALLBACK_MESSAGE,
                "updated_summary": summary,
            }

        context_blocks = [c["text"] for c in contexts]

        prompt = build_prompt(
            context_blocks=context_blocks,
            conversation_summary=summary,
            user_question=question,
        )

        try:
            response = self.primary_llm.invoke(prompt)
        except RateLimitError:
            response = self.fallback_llm.invoke(prompt)

        answer = response.content.strip()

        updated_summary = summary
        if self.enable_summary and len(recent_messages) >= self.max_recent:
            updated_summary = summarize_conversation(
                previous_summary=summary,
                recent_messages=recent_messages,
            )

        return {
            "answer": answer,
            "updated_summary": updated_summary,
        }
