from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import Settings
from models.embeddings import MultiModalEmbedder
from models.llm import LLM_CLIENTS
from utils.preprocessing import Document, Image_Text_Extractor, PageImage
from utils.vectordb import ChromaVectorDB

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Answer:
    text: str
    pages: list[int]
    images: list[PageImage]


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.embedder = MultiModalEmbedder(api_key=settings.hf_token)
        self.vectordb = ChromaVectorDB(settings.chroma_path)
        self.llm = LLM_CLIENTS[settings.llm_provider](settings.llm_api_key, settings.llm_model)
        logger.info("Answers come from %s (%s)", settings.llm_provider, settings.llm_model)

    def ingest(self, pdf: str | Path | bytes) -> Document:
        document = Image_Text_Extractor(pdf).extract_text_and_images()
        namespace = document.doc_id

        expected = len(document.chunks) + len(document.images)
        stored = self.vectordb.count(namespace)
        if stored == expected:
            logger.info("Document %s is already indexed (%d vectors)", namespace, stored)
            return document
        if stored:
            self.vectordb.clear(namespace)

        text_vectors = self.embedder.embed_texts([chunk.text for chunk in document.chunks])
        image_vectors = self.embedder.embed_images([item.image for item in document.images])

        vectors = [
            {
                "id": f"text-{i}",
                "values": values,
                "metadata": {"type": "text", "page": chunk.page, "text": chunk.text},
            }
            for i, (chunk, values) in enumerate(zip(document.chunks, text_vectors))
        ]
        vectors += [
            {
                "id": f"image-{i}",
                "values": values,
                "metadata": {"type": "image", "page": item.page, "image_index": i},
            }
            for i, (item, values) in enumerate(zip(document.images, image_vectors))
        ]
        self.vectordb.upsert(vectors, namespace)
        logger.info("Indexed document %s (%d vectors)", namespace, len(vectors))
        return document

    def answer(
        self, question: str, document: Document, top_k_text: int = 5, top_k_images: int = 3
    ) -> Answer:
        query = self.embedder.embed_query(question)

        text_hits = self.vectordb.search(query, document.doc_id, "text", top_k_text)
        image_hits = self.vectordb.search(query, document.doc_id, "image", top_k_images)

        pages = list(dict.fromkeys(hit["page"] for hit in text_hits))
        images = [document.images[hit["image_index"]] for hit in image_hits]
        context = "\n\n".join(f"[Page {page}]\n{document.pages[page - 1]}" for page in pages)

        return Answer(self.llm.generate(question, context, images), pages, images)
