"""
vrag_client.py
--------------
HTTP client for the VisionRAG streaming API.

Responsibilities:
  - Build the request payload.
  - Stream the response and separate retrieved-doc metadata from answer text.
  - Return a structured dict for the caller to consume.
"""

from typing import Any, Dict, List, Optional

import httpx
import json

from config.config_loader import load_config
from config.logger import setup_logger
from .retriever_utils import parse_json

logger = setup_logger(name="eval_pipeline", tag="VRAG_CLIENT")
CONFIG = load_config("config/config.yml")

# Sentinel that the VisionRAG backend injects before the retrieved-doc JSON blob.
_DOCS_MARKER = "__RETRIEVED_DOCS__"


class VisionRAGClient:
    """Handles VisionRAG API interactions and response parsing."""

    def __init__(
        self,
        query_reformulation: bool = False,
        use_reranking: bool = False,
        use_hybrid: bool = False,
    ) -> None:

        ### Initialize instance variables   
        self.query_reformulation = query_reformulation
        self.use_reranking    = use_reranking
        self.use_hybrid       = use_hybrid
        self._url             = CONFIG["vrag_client"].get("VISIONRAG_URL", "http://localhost:8000")
        self.vector_top_k     = CONFIG["vrag_client"].get("vector_top_k", 5)
        self.bm25_top_k       = CONFIG["vrag_client"].get("bm25_top_k", 5)
        self.rerank_top_k     = CONFIG["vrag_client"].get("rerank_top_k", 5)
        self.final_top_k      = CONFIG["vrag_client"].get("final_top_k", 5)
        self.memory_enabled   = CONFIG["vrag_client"].get("memory_enabled", True)
        self.timeout          = CONFIG["vrag_client"].get("timeout", 30)
        

        logger.info(
            f"VisionRAGClient initialized — query_reformulation={self.query_reformulation}, "
            f"use_reranking={self.use_reranking}, use_hybrid={self.use_hybrid}, url={self._url}",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_payload(
        self,
        user_id: str,
        question: str,
        thread_id: str,
        vrag_file_name: Optional[str],
        is_last: bool,
    ) -> Dict[str, Any]:
        """Assemble the JSON payload sent to the VisionRAG API."""
        
        return {
            "user_id":            user_id,
            "messages":           [question],
            "thread_id":          thread_id,
            "vrag_file_name":     vrag_file_name,
            "is_last":            is_last,
            "query_reformulation": self.query_reformulation,
            "use_reranking":      self.use_reranking,
            "use_hybrid":         self.use_hybrid,
            "vector_top_k":      self.vector_top_k,
            "bm25_top_k":        self.bm25_top_k,
            "rerank_top_k":      self.rerank_top_k,
            "final_top_k":      self.final_top_k,
            "memory_enabled":     self.memory_enabled,
        }


    async def _parse_stream(self, response) -> tuple[List[str], Optional[List[Dict]], str]:
        """
        Consume the streaming response and separate retrieved docs from answer text.

        Returns:
            raw_chunks     : every raw text chunk received (for debugging).
            retrieved_docs : parsed doc list, or None if the marker was absent.
            answer         : concatenated answer text.
        """
        raw_chunks: List[str] = []
        retrieved_docs: Optional[List[Dict]] = None
        answer_parts: List[str] = []

        async for chunk in response.aiter_text():
            raw_chunks.append(chunk)

            if chunk.startswith(_DOCS_MARKER):
                payload = chunk[len(_DOCS_MARKER):]
                parsed = parse_json(payload)
                if parsed is not None:
                    retrieved_docs = parsed
                    logger.info("Parsed %d retrieved docs from stream marker", len(retrieved_docs))
                else:
                    logger.warning("Stream marker found but JSON parse failed — payload: %.120s", payload)
            else:
                answer_parts.append(chunk)

        return raw_chunks, retrieved_docs, "".join(answer_parts).strip()

    # ── Public API ────────────────────────────────────────────────────────────

    async def  query(
        self,
        user_id: str,
        question: str,
        thread_id: Optional[str] = None,
        vrag_file_name: Optional[str] = None,
        is_last: bool = False,
    ) -> Dict[str, Any]:
        """
        Query VisionRAG and return parsed results.

        Args:
            user_id:        User identifier used to route the request.
            question:       The natural-language query.
            thread_id:      Conversation thread identifier (auto-generated if omitted).
            vrag_file_name: Name of the current evaluation run (written to the backend log).
            is_last:        Signal to the backend that this is the final query in the batch.

        Returns:
            {
                "question":          str,
                "retrieved_docs":    List[Dict],
                "answer":            str,
            }
        """
        effective_thread_id = thread_id or f"{vrag_file_name}_retrieval_eval"
        url = self._url.format(user_id=user_id)
        payload = self._build_payload(user_id, question, effective_thread_id, vrag_file_name, is_last)
        full_response = ""

        async with httpx.AsyncClient(timeout=self.timeout) as client:

            async with client.stream("POST", url, json=payload) as response:

                logger.info("VisionRAG response status: HTTP %d",response.status_code)

                # Read streamed response properly
                async for chunk in response.aiter_text():

                    if chunk:
                        full_response += chunk
                        logger.info("Received chunk: %s", chunk)

                # Handle non-200 responses
                if response.status_code != 200:

                    logger.info(
                        "VisionRAG returned HTTP %d — body: %.500s",
                        response.status_code,
                        full_response,
                    )

                    raise RuntimeError(
                        f"VisionRAG error {response.status_code}: "
                        f"{full_response}"
                    )

        logger.info("VisionRAG execution completed successfully")
        logger.info("Full streamed response: %s", full_response)

        # Try parsing JSON response
        parsed_json = None

        try:
            parsed_json = json.loads(full_response)
        except Exception:
            logger.info("Response is not valid JSON")
        logger.info("=============================================================")

        return {
            "success": True,
            "status_code": response.status_code,
            "question": question,
            "response_text": full_response,
            "response_json": parsed_json,
        }



