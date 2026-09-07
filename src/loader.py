"""Document loading utilities for the indexing pipeline."""

from pathlib import Path
from os import PathLike


class DocumentLoadError(RuntimeError):
    """Raised when a document cannot be read or decoded."""


def load_markdown(file_path, encoding="utf-8"):
    """Load one Markdown file as text with a useful error boundary."""
    if not isinstance(file_path, (str, PathLike)):
        raise TypeError("file_path must be a path-like value")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding must be a non-empty string")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"document file does not exist: {path}")

    try:
        return path.read_text(encoding=encoding)
    except (OSError, UnicodeError) as exc:
        raise DocumentLoadError(f"failed to load document: {path}") from exc


def load_documents(directory, pattern="*.md", recursive=True, encoding="utf-8"):
    """Load Markdown documents in deterministic path order.

    Args:
        directory: Directory containing source documents.
        pattern: Glob pattern used to select files.
        recursive: Search nested directories when true. Defaults to true so
            documents in nested folders are included by the indexing pipeline.
        encoding: Text encoding used for every document.
    """
    if not isinstance(directory, (str, PathLike)):
        raise TypeError("directory must be a path-like value")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    if not isinstance(recursive, bool):
        raise TypeError("recursive must be a boolean")

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"document directory does not exist: {root}")

    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    documents = []
    for file_path in sorted(path for path in paths if path.is_file()):
        documents.append({
            "text": load_markdown(file_path, encoding=encoding),
            "source": str(file_path.relative_to(root)),
        })
    return documents


__all__ = ["DocumentLoadError", "load_documents", "load_markdown"]