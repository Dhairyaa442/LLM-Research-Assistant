"""
Central configuration for the LLM Research Assistant.
All settings are loaded from environment variables via .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

# ── RAG ──────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH: Path = Path(os.getenv("FAISS_INDEX_PATH", str(BASE_DIR / "data" / "faiss_index")))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
EMBEDDING_MODEL: str = "text-embedding-3-small"

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: Path = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
