"""
retriever.py
------------
VisionRAG retriever: queries the VisionRAG API and returns structured results.
"""

import random
import requests
from typing import Optional

from openai import base_url

from config.config_loader import load_config
from config.logger import setup_logger
from utils.data_extractor import ChunkMetadataExtractor
from utils.support_evals.data_models import Question, RetrievalResult
from .vrag_client import VisionRAGClient

logger = setup_logger(name="eval_pipeline", tag="RETRIEVER")
CONFIG = load_config("config/config.yml")


class VisionRAGRetriever:
    """Retriever that queries the VisionRAG API and returns structured results."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        query_reformulation: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        chunk_format: Optional[str] = None, 
    ) -> None:
        # Resolve all values from config when not explicitly provided.
        self.user_id      = user_id      or CONFIG["vrag_retriever"]["user_id"]
        self.collection_name  = collection_name  or CONFIG["vrag_retriever"]["collection_name"]
        self.chunk_format = chunk_format or CONFIG["vrag_retriever"]["chunk_format"]

        self.client = VisionRAGClient(
            query_reformulation = query_reformulation if query_reformulation is not None else CONFIG["vrag_retriever"]["query_reformulation"],
            use_reranking    = use_reranking    if use_reranking    is not None else CONFIG["vrag_retriever"]["use_reranking"],
            use_hybrid       = use_hybrid       if use_hybrid       is not None else CONFIG["vrag_retriever"]["use_hybrid"],
        )

        # Unique thread ID scoped to this retriever instance / evaluation run.
        self.run_thread_id = f"{random.randint(100_000, 999_999)}_{self.user_id}_eval"

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        question: Question,
        vrag_file_name: Optional[str] = None,
        is_last: bool = False,
    ) -> RetrievalResult:
        """Query VisionRAG for *question* and return a structured RetrievalResult."""
        logger.info("qid=%s: %.60s…", question.qid, question.text)

        try:

            response = await self.client.query(
                user_id       = self.user_id,
                question      = question.text,
                thread_id     = self.run_thread_id,
                vrag_file_name= vrag_file_name,
                is_last       = is_last,
            )
            return response 

        except Exception as e:
            logger.error("✗ Error retrieving qid=%s: %s", question.qid, e, exc_info=True)


    def reinitialize_chat(self):
        """
        Calls the reinit chat API for a given user and document.
        """
        base_url = "http://localhost:8000/api/v1"
        url = f"{base_url}/users/{self.user_id}/chat/reinit"

        # query params
        params = {
            "doc_id": self.collection_name
        }

        ### Before calling the reinit chat API tag user with doc in localhost 8000
        try:
            response = requests.post(url, params=params)

            # Raise exception for bad status codes
            response.raise_for_status()
            logger.info(f"Chat reinitialized: {response.json()}")
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            return {
                "status": "error",
                "message": f"HTTP error occurred: {http_err}",
                "response": response.text
            }

        except Exception as err:
            return {
                "status": "error",
                "message": f"Other error occurred: {err}"
            }


