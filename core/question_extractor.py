from typing import List, Dict, Any, Optional
from core.data_models import Question
from utils.file_handler import FileHandler


class QuestionExtractor:
    """Extracts questions from target_chunks.json."""

    def __init__(self, target_chunks_file: str):
        self.target_chunks_file = target_chunks_file
        self.target_chunks: Dict[str, Any] = {}
        self.question_map: Dict[str, Question] = {}

    def load_chunks(self) -> Dict[str, Any]:
        """Load target chunks from file."""
        self.target_chunks = FileHandler.load_json(self.target_chunks_file)
        return self.target_chunks

    def extract_questions(self) -> List[Question]:
        """Extract all questions from target chunks."""
        if not self.target_chunks:
            self.load_chunks()

        questions: List[Question] = []

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
                        text= raw_question_text,
                        chunk_id=chunk_id,
                        question_type=qtype_str,
                        metadata=question_obj,
                    )

                    questions.append(question)
                    self.question_map[qid] = question

        return questions


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

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Get chunk text by chunk ID."""
        chunk = self.target_chunks.get(chunk_id)
        if chunk:
            return chunk.get("text", "")
        return None
