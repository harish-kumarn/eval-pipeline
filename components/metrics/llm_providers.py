"""
LLM and Embedding provider adapters for the evaluation pipeline.

Provides LangChain-compatible wrappers around the Google Gemini API
that are also safe to pass to RAGAS (which expects LangChain interfaces).
Token usage is tracked via an optional :class:`~metrics.token_tracker.TokenTracker`.
"""

import os
from typing import Any, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM

from components.metrics.token_tracker import TokenTracker
from config.logger import setup_logger

load_dotenv()
logger = setup_logger(name="eval_pipeline", tag="LLM_PROVIDERS")


# ──────────────────────────────────────────────────────────────────────────────
# Gemini LLM  (LangChain + RAGAS compatible)
# ──────────────────────────────────────────────────────────────────────────────

class GeminiLLM(LLM):
    """LangChain LLM wrapper for Google Gemini with optional token tracking."""

    model_name: str = "gemini-2.5-flash"
    tracker: Optional[Any] = None  # TokenTracker (Any keeps Pydantic happy)

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        logger.info("GeminiLLM._call — model=%s, prompt_len=%d chars", self.model_name, len(prompt))

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)

        prompt_tokens = completion_tokens = 0
        if self.tracker is not None:
            try:
                usage = response.usage_metadata
                prompt_tokens     = usage.prompt_token_count or 0
                completion_tokens = usage.candidates_token_count or 0
                self.tracker.record_llm(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception as te:
                logger.info("Token tracking failed: %s", te)

        logger.info(
            "GeminiLLM response — prompt_tokens=%d, completion_tokens=%d, response_len=%d chars",
            prompt_tokens, completion_tokens,
            len(response.text or "") if response else 0,
        )

        text = response.text if response and response.text else ""

        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token)[0]

        return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Gemini Embeddings  (LangChain + RAGAS compatible)
# ──────────────────────────────────────────────────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for Google Gemini with optional token tracking."""

    def __init__(self, model_name: str = "gemini-embedding-001", tracker: Optional[TokenTracker] = None):
        self.model_name = model_name
        self.tracker = tracker
        self._dim: Optional[int] = None
        logger.info("GeminiEmbeddings initialized (model=%s)", model_name)

    def _embed(self, text: str) -> List[float]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        estimated_tokens = len(text.split())
        logger.info("GeminiEmbeddings._embed — model=%s, ~%d tokens", self.model_name, estimated_tokens)

        genai.configure(api_key=api_key)
        response = genai.embed_content(
            model=f"models/{self.model_name}",
            content=text,
            task_type="retrieval_document",
        )
        vec = response["embedding"]

        if self._dim is None:
            self._dim = len(vec)
            logger.info("GeminiEmbeddings — embedding dimension: %d", self._dim)

        if self.tracker is not None:
            try:
                self.tracker.record_embedding(estimated_tokens)
            except Exception as te:
                logger.info("Embedding token tracking failed: %s", te)

        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            if not text or not text.strip():
                vectors.append([0.0] * (self._dim or 768))
            else:
                vectors.append(self._embed(text))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * (self._dim or 768)
        return self._embed(text)
