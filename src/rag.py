"""Application orchestration for the Arsenal retrieval-augmented generator."""

from dataclasses import dataclass
import logging
from os import PathLike
from pathlib import Path

from config import INDEX_PATH, METADATA_PATH, RERANKER_MODEL
from src.generator import Generator
from src.prompt import PromptBuilder
from src.retriever import Retriever
from src.reranker import Reranker


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    """Runtime configuration for the RAG pipeline."""

    index_path: str | PathLike[str] = INDEX_PATH
    metadata_path: str | PathLike[str] = METADATA_PATH
    reranker_model: str | None = RERANKER_MODEL
    default_top_k: int = 3

    def __post_init__(self):
        for name in ("index_path", "metadata_path"):
            value = getattr(self, name)
            if not isinstance(value, (str, PathLike)) or not str(value).strip():
                raise ValueError(f"{name} must be a non-empty path")
        if self.reranker_model is not None and (
            not isinstance(self.reranker_model, str) or not self.reranker_model.strip()
        ):
            raise ValueError("reranker_model must be a non-empty string or None")
        if (
            not isinstance(self.default_top_k, int)
            or isinstance(self.default_top_k, bool)
            or self.default_top_k <= 0
        ):
            raise ValueError("default_top_k must be a positive integer")

    @property
    def index_file(self):
        return Path(self.index_path)

    @property
    def metadata_file(self):
        return Path(self.metadata_path)


class RAG:
    """Coordinate retrieval, prompt construction, and answer generation."""

    def __init__(
        self,
        config=None,
        retriever=None,
        prompt_builder=None,
        generator=None,
    ):
        self.config = RAGConfig() if config is None else config
        if not isinstance(self.config, RAGConfig):
            raise TypeError("config must be an RAGConfig instance")

        self.retriever = (
            Retriever(
                index_path=self.config.index_file,
                metadata_path=self.config.metadata_file,
                reranker=(
                    Reranker(model_path=self.config.reranker_model)
                    if self.config.reranker_model is not None
                    else None
                ),
            )
            if retriever is None
            else retriever
        )
        self.prompt_builder = PromptBuilder() if prompt_builder is None else prompt_builder
        self.generator = Generator() if generator is None else generator

        LOGGER.info(
            "RAG initialized with index=%s, metadata=%s",
            self.config.index_file,
            self.config.metadata_file,
        )

    @staticmethod
    def _validate_query(query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return query.strip()

    @staticmethod
    def _validate_top_k(top_k):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k

    def ask(self, query, top_k=None, **generation_options):
        """Answer one query using retrieved context."""
        normalized_query = self._validate_query(query)
        resolved_top_k = self.config.default_top_k if top_k is None else top_k
        resolved_top_k = self._validate_top_k(resolved_top_k)

        retrieved_results = self.retriever.retrieve(
            normalized_query,
            top_k=resolved_top_k,
        )
        prompt = self.prompt_builder.build(normalized_query, retrieved_results)
        answer = self.generator.generate(prompt, **generation_options)

        if not isinstance(answer, str):
            raise TypeError("generator.generate must return a string")
        return answer.strip()


__all__ = ["RAG", "RAGConfig"]