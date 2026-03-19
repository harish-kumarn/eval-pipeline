from typing import List, Dict, Any, Optional
import httpx
from config.config_loader import load_config
from config.logger import setup_logger
from .retriever_utils import _extract_docs_from_stream, _parse_json

config = load_config("config/config.yml")
VISIONRAG_URL = config["VISIONRAG_URL"]

logger = setup_logger(name="eval_pipeline", tag="VRAG_CLIENT")
logger.info(VISIONRAG_URL)


class VisionRAGClient:
    """Handles VisionRAG API interactions and response parsing."""

    def __init__(self, timeout: int = 120, collection_name: str = "sprayers"):
        self.timeout = timeout
        self.collection_name = collection_name
        logger.info(
            "VisionRAGClient initialized (collection=%s, timeout=%ds)",
            collection_name, timeout,
        )

    async def query(
        self,
        user_id: str,
        question: str,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query VisionRAG and parse retrieved documents.

        Args:
            user_id: User identifier
            question: Query question
            thread_id: Optional thread identifier

        Returns:
            Dict containing question, retrieved_docs, answer, and raw_stream_preview
        """
        effective_thread_id = thread_id or f"{user_id}_retrieval_eval"
        payload = {
            "user_id": user_id,
            "message": question,
            "thread_id": effective_thread_id,
            "collection_names": [self.collection_name]
        }

        logger.info(
            "POST %s — user_id=%s, thread_id=%s, question=%.80s…",
            VISIONRAG_URL.format(user_id=user_id), user_id, effective_thread_id, question,
        )

        raw_chunks: List[str] = []
        retrieved_docs: Optional[List[Dict]] = None
        answer = ""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                VISIONRAG_URL.format(user_id=user_id),
                json=payload
            ) as response:
                logger.info("VisionRAG response status: HTTP %d", response.status_code)
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(
                        "VisionRAG returned HTTP %d — body: %s", response.status_code, body[:500]
                    )
                    raise RuntimeError(
                        f"VisionRAG error {response.status_code}: {body}"
                    )
                async for chunk in response.aiter_text():
                    raw_chunks.append(chunk)

                    if chunk.startswith("__RETRIEVED_DOCS__"):
                        raw = chunk[len("__RETRIEVED_DOCS__"):]
                        retrieved_docs = _parse_json(raw)
                        if retrieved_docs is not None:
                            logger.info(
                                "Parsed %d retrieved docs from __RETRIEVED_DOCS__ marker",
                                len(retrieved_docs),
                            )
                        else:
                            logger.info("__RETRIEVED_DOCS__ marker found but JSON parse failed")
                    else:
                        answer += chunk

        # Fallback parsing if marker-based extraction failed
        if not retrieved_docs:
            logger.info("Marker-based doc extraction failed — attempting fallback stream parse")
            retrieved_docs = _extract_docs_from_stream(raw_chunks)
            if retrieved_docs:
                logger.info("Fallback parse succeeded — %d docs extracted", len(retrieved_docs))
            else:
                logger.info("Fallback parse also failed — no retrieved docs found")

        if retrieved_docs is None:
            retrieved_docs = []

        logger.info(
            "Query complete — %d docs, answer length=%d chars, raw chunks=%d",
            len(retrieved_docs), len(answer.strip()), len(raw_chunks),
        )

        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "answer": answer.strip(),
            "raw_stream_preview": raw_chunks[:5],
        }

