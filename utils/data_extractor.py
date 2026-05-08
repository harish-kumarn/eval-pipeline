import logging
from typing import List, Dict, Any, Optional
from utils.support_evals.data_models import Question
from utils.support_evals.file_handler import FileHandler
from config.logger import setup_logger

logger = setup_logger(name="eval_pipeline", tag="DATA_EXTRACTOR")


class DataExtractor:
    """Extracts questions from target_chunks.json."""

    def __init__(self, target_chunks_file: str):
        self.target_chunks_file = target_chunks_file
        self.target_chunks: Dict[str, Any] = {}
        self.question_map: Dict[str, Question] = {}

    def load_chunks(self) -> Dict[str, Any]:
        """Load target chunks from file."""
        self.target_chunks = FileHandler.load_json(self.target_chunks_file)
        logger.info("Loaded %d chunks from %s", len(self.target_chunks), self.target_chunks_file)
        return self.target_chunks

    def extract_questions(self) -> List[Question]:
        """Extract all questions from target chunks."""
        if not self.target_chunks:
            self.load_chunks()

        questions: List[Question] = []
        type_counts: Dict[str, int] = {}

        for chunk_id, chunk_data in self.target_chunks.items():
            questions_section = chunk_data.get("questions", {})
            
            # Global counter for this chunk
            chunk_question_idx = 0

            # Extract questions from direct, indirect, image_based
            for qtype_str in ["direct", "indirect", "image_based"]:
                items = questions_section.get(qtype_str, [])

                for question_obj in items:
                    chunk_question_idx += 1
                    prefix = "q"
                    qid = f"{prefix}{chunk_question_idx}_{chunk_id}"

                    raw_question_text = question_obj.get("question", "").strip()
                    question = Question(
                        qid=qid,
                        text=raw_question_text,
                        chunk_id=chunk_id,
                        question_type=qtype_str,
                        metadata=question_obj,
                    )

                    questions.append(question)
                    self.question_map[qid] = question
                    type_counts[qtype_str] = type_counts.get(qtype_str, 0) + 1

        logger.info(
            "Extracted %d questions from %d chunks — breakdown: %s",
            len(questions), len(self.target_chunks), type_counts,
        )
        return questions



    # used in compute_metrics of orchestrator.py
    
    def get_question(self, qid: str) -> Optional[Question]:
        """Get question by ID."""
        return self.question_map.get(qid)

    def get_ground_truth(self, qid: str) -> Optional[str]:
        """Extract ground truth answer for a question."""
        question = self.get_question(qid)
        if not question:
            return None

        metadata = question.metadata
        for key in ("answer_span", "answer_rationale", "grounding", "ground_truth", "answer_extract", "answer_synthesis"):
            if key in metadata and metadata[key]:
                val = metadata[key]
                if isinstance(val, list):
                    return " ".join(str(v) for v in val)
                return str(val)
        return None

    def get_answer_extract(self, qid: str) -> Optional[str]:
        """Return the answer_extract field specifically (for answer_correctness ground truth).

        Falls back to get_ground_truth() if answer_extract is not present so the
        method always returns something when a ground truth exists.
        """
        question = self.get_question(qid)
        if not question:
            return None
        ae = question.metadata.get("answer_extract")
        if ae:
            if isinstance(ae, list):
                return " ".join(str(v) for v in ae)
            return str(ae)
        # Fall back to the generic ground truth lookup
        return self.get_ground_truth(qid)

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Get chunk text by chunk ID."""
        chunk = self.target_chunks.get(chunk_id)
        if chunk:
            return chunk.get("text", "")
        logger.info("Chunk %s not found in target_chunks", chunk_id)
        return None



### used in REtreiver for extracting chunkIDS
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
            logger.info("Skipping non-dict document in retrieved_docs: %s", type(doc))
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

        logger.info("No recognisable chunk-ID field in doc keys: %s", list(doc.keys())[:10])
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


        