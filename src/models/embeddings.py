"""Text and image embeddings in ALIGN's shared vector space."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AlignModel, AlignProcessor

BATCH_SIZE = 16
MAX_TEXT_TOKENS = 64


def _unit_length(features: torch.Tensor) -> list[list[float]]:
    return torch.nn.functional.normalize(features, dim=-1).tolist()


class MultiModalEmbedder:
    """Embeds text and images separately into the same space.

    Because both live in one space, a text query can be compared directly with
    text chunks and with images.
    """

    def __init__(self, api_key: str | None = None, align_model_name: str = "kakaobrain/align-base"):
        self.processor = AlignProcessor.from_pretrained(align_model_name, token=api_key)
        self.model = AlignModel.from_pretrained(align_model_name, token=api_key).eval()
        self.dimension: int = self.model.config.projection_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            inputs = self.processor.tokenizer(
                texts[start : start + BATCH_SIZE],
                padding="max_length",
                truncation=True,
                max_length=MAX_TEXT_TOKENS,
                return_tensors="pt",
            )
            with torch.no_grad():
                features = self.model.get_text_features(**inputs).pooler_output
            vectors += _unit_length(features)
        return vectors

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(images), BATCH_SIZE):
            inputs = self.processor.image_processor(
                images=images[start : start + BATCH_SIZE], return_tensors="pt"
            )
            with torch.no_grad():
                features = self.model.get_image_features(**inputs).pooler_output
            vectors += _unit_length(features)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
