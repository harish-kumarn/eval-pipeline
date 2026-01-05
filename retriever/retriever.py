import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

from core.interfaces import IRetriever
from core.data_models import Question, RetrievalResult

# Both VisionRAGClient and ChunkMetadataExtractor are in the same directory as this file
from .vrag_client import VisionRAGClient
from .chunk_extractor import ChunkMetadataExtractor

logger.debug("Retriever imports completed")


class VisionRAGRetriever(IRetriever):
    """Retriever that hits VisionRAG API."""

    def __init__(
        self,
        user_id: str = "434",
        timeout: int = 120,
        chunk_format: str = "chunk{id}",
    ):
        logger.debug(f"Initializing VisionRAGRetriever (user_id={user_id}, timeout={timeout}s, chunk_format={chunk_format})")
        self.user_id = user_id
        self.chunk_format = chunk_format
        self.client = VisionRAGClient(timeout=timeout)
        logger.debug("✓ VisionRAGRetriever initialized")

    async def retrieve(self, question: Question) -> RetrievalResult:
        """Public interface to retrieve results for a question."""
        return await self.vrag_client(question)

    async def vrag_client(self, question: Question) -> RetrievalResult:
        """Query VisionRAG and return retrieval result."""
        logger.debug(f"retrieve() called for question {question.qid}: {question.text[:60]}...")
        try:
            logger.debug(f"  Querying VisionRAG API (user_id={self.user_id}, thread_id={self.user_id}_{question.qid})")
            response = await self.client.query(
                user_id=self.user_id,
                question=question.text,
                thread_id=f"{self.user_id}_{question.qid}",
            )
            logger.debug(f"  ✓ VisionRAG API response received")

            retrieved_docs = response.get("retrieved_docs", [])
            logger.debug(f"  Retrieved {len(retrieved_docs)} documents")
            
            retrieved_chunk_ids = ChunkMetadataExtractor.extract_chunk_ids(
                retrieved_docs,
                chunk_format=self.chunk_format,
            )
            logger.debug(f"  Extracted {len([c for c in retrieved_chunk_ids if c])} chunk IDs")

            answer_text = response.get("answer", "")
            logger.debug(f"  Generated answer length: {len(answer_text)} chars")

            result = RetrievalResult(
                qid=question.qid,
                question=question.text,
                ground_truth_chunk=question.chunk_id,
                retrieved_chunks=[c for c in retrieved_chunk_ids if c],
                retrieved_docs=retrieved_docs,
                generated_answer=answer_text,
            )
            logger.debug(f"✓ RetrievalResult created for {question.qid}")
            return result

        except Exception as e:
            logger.error(f"✗ Error retrieving for question {question.qid}: {e}", exc_info=True)
            return RetrievalResult(
                qid=question.qid,
                question=question.text,
                ground_truth_chunk=question.chunk_id,
                error=str(e),
            )
