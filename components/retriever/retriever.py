"""VisionRAG retriever: queries the VisionRAG API and returns structured results."""

import random
from utils.support_evals.data_models import Question, RetrievalResult
from .vrag_client import VisionRAGClient
from utils.data_extractor import ChunkMetadataExtractor
from typing import Dict, Optional
from config.config_loader import load_config
from config.logger import setup_logger

logger = setup_logger(name="eval_pipeline", tag="RETRIEVER")

_cfg = load_config("config/config.yml")


class VisionRAGRetriever:
    """Retriever implementation that queries the VisionRAG API."""

    def __init__(
        self,
        user_id: str = _cfg["user_id"],
        timeout: int = _cfg["timeout"],
        chunk_format: str = _cfg["chunk_format"],
        collection_name: str = _cfg["collection_name"],
        query_reformulation: bool = False,
        use_reranking: bool = False,
        use_hybrid: bool = False,
    ):
        logger.info(
            "Initializing VisionRAGRetriever (user_id=%s, timeout=%ds, "
            "chunk_format=%s, collection=%s)",
            user_id, timeout, chunk_format, collection_name,
        )
        self.user_id      = user_id
        self.chunk_format = chunk_format
        self.client       = VisionRAGClient(
            timeout=timeout,
            collection_name=collection_name,
            query_reformulation=query_reformulation,
            use_reranking=use_reranking,
            use_hybrid=use_hybrid,
        )
        # Generate a unique thread ID for this run instance
        self.run_thread_id = f"{random.randint(100000, 999999)}_{self.user_id}_eval"
        logger.info("✓ VisionRAGRetriever initialized with run_thread_id: %s", self.run_thread_id)

    async def retrieve(self, question: Question, eval_file_name: Optional[str] = None, is_last: bool = False) -> RetrievalResult:
        """Query VisionRAG and return a structured RetrievalResult."""
        logger.info("retrieve() called for question %s: %s…", question.qid, question.text[:60])
        try:
            logger.info(
                "Querying VisionRAG API (user_id=%s, thread_id=%s, eval_file_name=%s, is_last=%s)",
                self.user_id, self.run_thread_id, eval_file_name, is_last,
            )
            response = await self.client.query(
                user_id=self.user_id,
                question=question.text,
                thread_id=self.run_thread_id,
                eval_file_name=eval_file_name,
                is_last=is_last,
            )
            logger.info("✓ VisionRAG API response received")

            retrieved_docs = response.get("retrieved_docs", [])
            logger.info("Retrieved %d documents", len(retrieved_docs))
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
            logger.info("✓ RetrievalResult created for %s", question.qid)
            return result

        except Exception as e:
            logger.error("✗ Error retrieving for question %s: %s", question.qid, e, exc_info=True)
            return RetrievalResult(
                qid=question.qid,
                question=question.text,
                ground_truth_chunk=question.chunk_id,
                error=str(e),
            )




