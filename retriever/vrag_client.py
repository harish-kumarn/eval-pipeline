import json
import re
from typing import List, Dict, Any, Optional
import httpx


VISIONRAG_URL = "http://127.0.0.1:8000/api/v1/users/{user_id}/chat"


class VisionRAGClient:
    """Handles VisionRAG API interactions and response parsing."""

    """Handles VisionRAG API interactions and response parsing."""

    def __init__(self, timeout: int = 120, collection_name: str = "sprayers"):
        self.timeout = timeout
        self.collection_name = collection_name

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
        payload = {
            "user_id": user_id,
            "message": question,
            "thread_id": thread_id or f"{user_id}_retrieval_eval",
            "collection_names": [self.collection_name]
        }

        raw_chunks: List[str] = []
        retrieved_docs: Optional[List[Dict]] = None
        answer = ""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                VISIONRAG_URL.format(user_id=user_id),
                json=payload
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise RuntimeError(
                        f"VisionRAG error {response.status_code}: {body}"
                    )

                async for chunk in response.aiter_text():
                    raw_chunks.append(chunk)

                    # Check for retrieval output marker
                    if chunk.startswith("__RETRIEVED_DOCS__"):
                        raw = chunk[len("__RETRIEVED_DOCS__"):]
                        retrieved_docs = self._parse_json(raw)
                    else:
                        answer += chunk

        # Fallback parsing if marker-based extraction failed
        if not retrieved_docs:
            retrieved_docs = self._extract_docs_from_stream(raw_chunks)

        if retrieved_docs is None:
            retrieved_docs = []

        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "answer": answer.strip(),
            "raw_stream_preview": raw_chunks[:5]
        }

    @staticmethod
    def _parse_json(text: str) -> Optional[List[Dict]]:
        """Safely parse JSON text."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_docs_from_stream(raw_chunks: List[str]) -> Optional[List[Dict]]:
        """Extract JSON from combined stream as fallback."""
        combined = "".join(raw_chunks)

        # Try marker-based extraction
        if "__RETRIEVED_DOCS__" in combined:
            raw = combined.split("__RETRIEVED_DOCS__", 1)[1]
            parsed = VisionRAGClient._parse_json(raw)
            if parsed:
                return parsed

        # Try regex-based JSON array extraction
        match = re.search(r"(\[\s*\{.*?\}\s*\])", combined, re.DOTALL)
        if match:
            parsed = VisionRAGClient._parse_json(match.group(1))
            if parsed:
                return parsed

        return None