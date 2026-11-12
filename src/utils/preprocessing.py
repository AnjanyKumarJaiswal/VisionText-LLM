"""Extract the text and embedded images of a PDF."""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

logger = logging.getLogger(__name__)

CHUNK_WORDS = 40
CHUNK_OVERLAP = 10
MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 1600


@dataclass(frozen=True)
class TextChunk:
    page: int
    text: str


@dataclass(frozen=True)
class PageImage:
    page: int
    image: Image.Image


@dataclass(frozen=True)
class Document:
    doc_id: str
    pages: list[str]
    chunks: list[TextChunk]
    images: list[PageImage]


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    starts = range(0, max(len(words) - overlap, 1), size - overlap)
    return [" ".join(words[start : start + size]) for start in starts]


class Image_Text_Extractor:
    def __init__(self, pdf: str | Path | bytes):
        """`pdf` is a path to a PDF file or the raw bytes of one."""
        self.pdf = pdf

    def extract_text_and_images(self) -> Document:
        data = self.pdf if isinstance(self.pdf, bytes) else Path(self.pdf).read_bytes()

        pages: list[str] = []
        chunks: list[TextChunk] = []
        images: list[PageImage] = []
        seen_xrefs: set[int] = set()

        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except pymupdf.FileDataError as exc:
            raise ValueError("The file could not be read as a PDF.") from exc

        with pdf:
            if pdf.needs_pass:
                raise ValueError("The PDF is password-protected. Remove the password and try again.")

            for number, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                pages.append(text)
                chunks.extend(TextChunk(number, chunk) for chunk in chunk_text(text))

                for xref, *_ in page.get_images(full=True):
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    image = self._load_image(pdf, xref, number)
                    if image is not None:
                        images.append(PageImage(number, image))

        if not chunks and not images:
            raise ValueError("The PDF has no extractable text or images.")

        return Document(hashlib.sha256(data).hexdigest()[:16], pages, chunks, images)

    @staticmethod
    def _load_image(pdf: pymupdf.Document, xref: int, page: int) -> Image.Image | None:
        try:
            raw = pdf.extract_image(xref)["image"]
            image = Image.open(io.BytesIO(raw))
            if min(image.size) < MIN_IMAGE_SIDE:
                return None
            image.draft(None, (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            image = image.convert("RGB")
        except Exception as exc:
            logger.warning("Skipping unreadable image on page %d: %s", page, exc)
            return None
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        return image
