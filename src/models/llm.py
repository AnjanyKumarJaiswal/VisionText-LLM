from __future__ import annotations

import base64
import io

from google import genai
from google.genai import errors, types
from openai import OpenAI
from PIL import Image

from utils.preprocessing import PageImage

SYSTEM_PROMPT = (
    "You answer questions about a PDF document. Use only the page text and the images "
    "you are given; look at the images directly when the question is about them. "
    'If the answer is not in the text or the images, say "I don\'t know."'
)
TIMEOUT_SECONDS = 60


def _document_text(context: str) -> str:
    return f"Text from the document:\n{context or '(no text was retrieved)'}"


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class GenerateGeminiResponse:

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(attempts=2),
            ),
        )
        self.model = model

    def generate(self, question: str, context: str, images: list[PageImage]) -> str:
        contents: list = [_document_text(context)]
        for item in images:
            contents += [
                f"Image from page {item.page}:",
                types.Part.from_bytes(data=_jpeg(item.image), mime_type="image/jpeg"),
            ]
        contents.append(f"Question: {question}")

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
        except errors.APIError as exc:
            raise RuntimeError(f"Gemini error {exc.code}: {exc.message}") from exc
        if not response.text:
            raise RuntimeError("Gemini returned no text for this question.")
        return response.text


def _data_url(image: Image.Image) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(image)).decode("ascii")


class GenerateOpenAIResponse:

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=1)
        self.model = model

    def generate(self, question: str, context: str, images: list[PageImage]) -> str:
        content: list[dict] = [{"type": "text", "text": _document_text(context)}]
        for item in images:
            content += [
                {"type": "text", "text": f"Image from page {item.page}:"},
                {"type": "image_url", "image_url": {"url": _data_url(item.image)}},
            ]
        content.append({"type": "text", "text": f"Question: {question}"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("OpenAI returned no text for this question.")
        return text


LLM_CLIENTS = {"Gemini": GenerateGeminiResponse, "OpenAI": GenerateOpenAIResponse}
