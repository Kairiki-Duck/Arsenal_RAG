"""Shared configuration and project-relative path defaults."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENT_DIR = Path(os.getenv("ARSENAL_DOCUMENT_DIR", PROJECT_ROOT / "data" / "documents"))
RAW_DATA_DIR = Path(os.getenv("ARSENAL_RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw_data"))
INDEX_PATH = Path(os.getenv("ARSENAL_INDEX_PATH", PROJECT_ROOT / "vector_db" / "index.faiss"))
METADATA_PATH = Path(
	os.getenv("ARSENAL_METADATA_PATH", PROJECT_ROOT / "vector_db" / "metadata.pkl")
)
EMBEDDING_MODEL = os.getenv(
	"ARSENAL_EMBEDDING_MODEL",
	"Qwen/Qwen3-Embedding-4B",
)
GENERATOR_MODEL = os.getenv(
	"ARSENAL_GENERATOR_MODEL",
	"Qwen/Qwen2.5-14B-Instruct",
)
RERANKER_MODEL = os.getenv(
	"ARSENAL_RERANKER_MODEL",
	"Qwen/Qwen3-Reranker-4B",
)


def resolve_path(value, base_dir=PROJECT_ROOT):
	"""Resolve relative application paths against the project root."""
	path = Path(value)
	return path if path.is_absolute() else Path(base_dir) / path


__all__ = [
	"DOCUMENT_DIR",
	"EMBEDDING_MODEL",
	"GENERATOR_MODEL",
	"INDEX_PATH",
	"METADATA_PATH",
	"PROJECT_ROOT",
	"RERANKER_MODEL",
	"resolve_path",
]
