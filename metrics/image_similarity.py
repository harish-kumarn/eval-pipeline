import numpy as np
from PIL import Image, ImageFile
import imagehash
import base64
from io import BytesIO
import os
from transformers import CLIPProcessor, CLIPModel
import torch

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageSimilarity:
    """
    Multi-stage image similarity:

    A) Grayscale similarity (cheap)
    B) Binary-mask similarity (shape focused)
    C) pHash similarity (robust to resizing/compression)

    All scores normalized to [0,1]
    """

    def __init__(
        self,
        phash_threshold: int = 32,
        resize_hw=(256, 256),
        binarize_threshold: int = 128,
    ):
        self.phash_threshold = phash_threshold
        self.resize_hw = resize_hw
        self.binarize_threshold = binarize_threshold

    # --------------------------
    # Utilities
    # --------------------------
    def _load_image(self, img_input) -> Image.Image:
        if isinstance(img_input, str):
            if img_input.startswith("data:image"):
                # Handle base64
                header, data = img_input.split(",", 1)
                # Fix padding if necessary
                missing_padding = len(data) % 4
                if missing_padding:
                    data += "=" * (4 - missing_padding)
                return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
            else:
                # Handle file path
                return Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

    def _to_gray_array(self, img: Image.Image) -> np.ndarray:
        gray = img.convert("L").resize(self.resize_hw, Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32) / 255.0  # normalize [0,1]
        return arr

    def _to_binary_array(self, gray_arr: np.ndarray) -> np.ndarray:
        # threshold -> {0,1}
        thr = self.binarize_threshold / 255.0
        return (gray_arr >= thr).astype(np.uint8)

    # --------------------------
    # A) Grayscale similarity
    # --------------------------
    def grayscale_similarity(self, img1_input, img2_input) -> float:
        """
        Cosine similarity of grayscale normalized pixels.
        Good for structural similarity (cheap).
        """
        img1 = self._load_image(img1_input)
        img2 = self._load_image(img2_input)

        g1 = self._to_gray_array(img1).reshape(-1)
        g2 = self._to_gray_array(img2).reshape(-1)

        # cosine similarity
        denom = (np.linalg.norm(g1) * np.linalg.norm(g2)) + 1e-12
        cos = float(np.dot(g1, g2) / denom)

        # cosine is in [0,1] for non-negative vectors
        return float(np.clip(cos, 0.0, 1.0))

    # --------------------------
    # B) Binary similarity
    # --------------------------
    def binary_iou_similarity(self, img1_input, img2_input) -> float:
        """
        IoU similarity after binarization.
        Great for UI/logos/docs.
        """
        img1 = self._load_image(img1_input)
        img2 = self._load_image(img2_input)

        g1 = self._to_gray_array(img1)
        g2 = self._to_gray_array(img2)

        b1 = self._to_binary_array(g1)
        b2 = self._to_binary_array(g2)

        inter = np.logical_and(b1, b2).sum()
        union = np.logical_or(b1, b2).sum()

        if union == 0:
            # both empty masks
            return 1.0

        iou = inter / union
        return float(np.clip(iou, 0.0, 1.0))

    # --------------------------
    # C) pHash similarity
    # --------------------------
    def phash_similarity(self, img1_input, img2_input) -> float:
        """
        Returns pHash similarity score in [0, 1].
        """
        img1 = self._load_image(img1_input)
        img2 = self._load_image(img2_input)

        h1 = imagehash.phash(img1)  # default: 64-bit
        h2 = imagehash.phash(img2)

        dist = h1 - h2  # Hamming distance
        score = 1.0 - min(dist, self.phash_threshold) / float(self.phash_threshold)
        return float(max(0.0, score))
  

    # --------------------------
    # Final Combined Score
    # --------------------------
    def similarity_score(
        self,
        img1_input,
        img2_input,
        w_gray: float = 0.15,
        w_bin: float = 0.20,
        w_phash: float = 0.25,
    ) -> dict:
        """
        Weighted score.
        All weights should sum to 1.0 ideally.
        """

        gray_score = self.grayscale_similarity(img1_input, img2_input)
        bin_score = self.binary_iou_similarity(img1_input, img2_input)
        phash_score = self.phash_similarity(img1_input, img2_input)

        # Normalize weights if they don't sum to 1
        weights = np.array([w_gray, w_bin, w_phash], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-12)

        parts = np.array(
            [
                gray_score,
                bin_score,
                phash_score,
            ],
            dtype=np.float32,
        )

        final = float(np.dot(weights, parts))

        return {
            "grayscale_score": round(float(gray_score), 4),
            "binary_iou_score": round(float(bin_score), 4),
            "phash_score": round(float(phash_score), 4),
            "final_score": round(final, 4),
        }
