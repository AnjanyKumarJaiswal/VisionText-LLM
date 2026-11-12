"""Runtime settings, read from environment variables, a .env file or Streamlit secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = {
    "GEMINI_API_KEY": ("Gemini", "gemini-3.5-flash-lite"),
    "OPENAI_API_KEY": ("OpenAI", "gpt-6-luna"),
}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str
    llm_model: str
    chroma_path: Path
    hf_token: str | None
    pdf_path: Path

    @classmethod
    def load(cls, overrides: Mapping[str, str] | None = None) -> Settings:
        load_dotenv()
        values = {**os.environ, **(overrides or {})}

        key_name = next((name for name in PROVIDERS if values.get(name)), None)
        if key_name is None:
            raise RuntimeError(
                f"No API key found. Set {' or '.join(PROVIDERS)} in a .env file, "
                "as an environment variable or in Streamlit secrets."
            )
        provider, default_model = PROVIDERS[key_name]

        return cls(
            llm_provider=provider,
            llm_api_key=values[key_name],
            llm_model=values.get("LLM_MODEL") or default_model,
            chroma_path=Path(values.get("CHROMA_PATH") or PROJECT_ROOT / "chroma_db"),
            hf_token=values.get("HF_TOKEN") or values.get("HF_KEY") or None,
            pdf_path=Path(values.get("PDF_PATH") or PROJECT_ROOT / "test.pdf"),
        )
