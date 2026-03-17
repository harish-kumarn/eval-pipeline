from typing import List, Dict, Any, Optional


class ChunkMetadataExtractor:
    """Extracts and standardizes chunk IDs from retrieved documents."""

    CHUNK_ID_FIELDS = ["chunk_id", "chunkId", "id", "source"]

    @classmethod
    def extract_chunk_ids(
        cls,
        retrieved_docs: List[Dict],
        chunk_format: str = "chunk{id}"
    ) -> List[Optional[str]]:
        """
        Extract chunk IDs from documents with fallback strategies.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            chunk_format: Format string for chunk ID (default: "chunk{id}")
                         Use "chunk{id}" for "chunk1", "{id}" for "1", etc.
        
        Returns:
            List of standardized chunk IDs
        """
        chunk_ids = []

        for doc in retrieved_docs:
            chunk_id = cls._extract_from_doc(doc)
            if chunk_id is not None:
                # Standardize format
                standardized_id = cls._standardize_chunk_id(chunk_id, chunk_format)
                chunk_ids.append(standardized_id)
            else:
                chunk_ids.append(None)

        return chunk_ids

    @classmethod
    def _extract_from_doc(cls, doc: Dict) -> Optional[Any]:
        """Extract raw chunk ID from a single document."""
        if not isinstance(doc, dict):
            return None

        # Try metadata fields first
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            for field in cls.CHUNK_ID_FIELDS:
                if field in metadata and metadata[field]:
                    return metadata[field]

        # Try top-level fields
        for field in cls.CHUNK_ID_FIELDS:
            if field in doc and doc[field]:
                return doc[field]

        return None

    @staticmethod
    def _standardize_chunk_id(raw_id: Any, format_string: str) -> str:
        """
        Standardize chunk ID to match expected format.
        
        Args:
            raw_id: Raw chunk ID (could be int, string, etc.)
            format_string: Format template (e.g., "chunk{id}", "{id}")
        
        Returns:
            Standardized chunk ID string
        """
        # Extract numeric part if it's a string like "chunk1"
        if isinstance(raw_id, str) and raw_id.startswith("chunk"):
            numeric_part = raw_id.replace("chunk", "")
        else:
            numeric_part = str(raw_id)

        # Format according to template
        return format_string.format(id=numeric_part)


        