"""VisionRAG retriever: queries the VisionRAG API and returns structured results."""

import logging

from core.data_models import Question, RetrievalResult
from .vrag_client import VisionRAGClient
from .chunk_extractor import ChunkMetadataExtractor

logger = logging.getLogger(__name__)


class VisionRAGRetriever:
    """Retriever implementation that queries the VisionRAG API."""

    def __init__(
        self,
        user_id: str = "1234",
        timeout: int = 120,
        chunk_format: str = "chunk{id}",
        collection_name: str = "sprayers",
    ):
        logger.debug(
            "Initializing VisionRAGRetriever (user_id=%s, timeout=%ds, "
            "chunk_format=%s, collection=%s)",
            user_id, timeout, chunk_format, collection_name,
        )
        self.user_id      = user_id
        self.chunk_format = chunk_format
        self.client       = VisionRAGClient(timeout=timeout, collection_name=collection_name)
        logger.debug("✓ VisionRAGRetriever initialized")

    async def retrieve(self, question: Question) -> RetrievalResult:
        """Query VisionRAG and return a structured RetrievalResult."""
        logger.debug("retrieve() called for question %s: %s…", question.qid, question.text[:60])
        try:
            logger.debug(
                "Querying VisionRAG API (user_id=%s, thread_id=%s_%s)",
                self.user_id, self.user_id, question.qid,
            )
            response = await self.client.query(
                user_id=self.user_id,
                question=question.text,
                thread_id=f"{self.user_id}_{question.qid}",
            )
            logger.debug("✓ VisionRAG API response received")

            retrieved_docs = response.get("retrieved_docs", [])
            logger.debug("Retrieved %d documents", len(retrieved_docs))
            logger.info("Retrieved docs: %s", retrieved_docs)

            retrieved_chunk_ids = ChunkMetadataExtractor.extract_chunk_ids(
                retrieved_docs,
                chunk_format=self.chunk_format,
            )

            answer_text = response.get("answer", "")

            result = RetrievalResult(
                qid=question.qid,
                question=question.text,
                ground_truth_chunk=question.chunk_id,
                retrieved_chunks=[c for c in retrieved_chunk_ids if c],
                retrieved_docs=retrieved_docs,
                generated_answer=answer_text,
            )
            logger.debug("✓ RetrievalResult created for %s", question.qid)
            return result

        except Exception as e:
            logger.error("✗ Error retrieving for question %s: %s", question.qid, e, exc_info=True)
            return RetrievalResult(
                qid=question.qid,
                question=question.text,
                ground_truth_chunk=question.chunk_id,
                error=str(e),
            )
