"""Clean Wikipedia-style Markdown files for the local knowledge base."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REMOVED_SECTION_HEADING = re.compile(
	r"^\s*#{1,6}\s+(?:references|see\s+also)\s*$",
	re.IGNORECASE | re.MULTILINE,
)
INLINE_CITATION = re.compile(
	r"\[\s*(?:\d+(?:\s+\d+)*|citation\s+needed|vague|tone|when\?|"
	r"according\s+to\s+whom\?|contradictory|not\s+verified|"
	r"unreliable\s+source\?)\s*\]",
	re.IGNORECASE,
)
PRONUNCIATION_CLAUSE = re.compile(
	r"\(\s*[A-Za-z-]+\s+(?:pronunciation|pronounced):\s*"
	r"\[[^\]]*\]\s*(?:ⓘ\s*)?;?\s*",
	re.IGNORECASE,
)
PRONUNCIATION_ONLY = re.compile(
	r"\(\s*[A-Za-z-]+\s+(?:pronunciation|pronounced):\s*"
	r"\[[^\]]*\]\s*(?:ⓘ\s*)?\)",
	re.IGNORECASE,
)


def clean_markdown(text: str) -> str:
	"""Remove references, inline citations, and leading pronunciation notes."""
	removed_section = REMOVED_SECTION_HEADING.search(text)
	if removed_section:
		text = text[: removed_section.start()].rstrip()

	text = PRONUNCIATION_ONLY.sub("", text)
	text = PRONUNCIATION_CLAUSE.sub("(", text)
	text = INLINE_CITATION.sub("", text)

	text = re.sub(r"\(\s*\)", "", text)
	return text.strip() + "\n"


def clean_directory(source_dir: Path, target_dir: Path) -> int:
	"""Clean every Markdown file while preserving its relative directory."""
	files = sorted(source_dir.rglob("*.md"))
	for source_path in files:
		target_path = target_dir / source_path.relative_to(source_dir)
		target_path.parent.mkdir(parents=True, exist_ok=True)
		target_path.write_text(
			clean_markdown(source_path.read_text(encoding="utf-8")),
			encoding="utf-8",
		)
	return len(files)


def main() -> None:
	parser = argparse.ArgumentParser(description="Clean raw Markdown knowledge files.")
	parser.add_argument(
		"source",
		nargs="?",
		type=Path,
		default=Path(__file__).parent / "data" / "raw_data" / "players",
		help="directory containing raw Markdown files",
	)
	parser.add_argument(
		"target",
		nargs="?",
		type=Path,
		default=Path(__file__).parent / "data" / "documents" /"players",
		help="directory receiving cleaned Markdown files",
	)
	args = parser.parse_args()

	if not args.source.is_dir():
		parser.error(f"source directory does not exist: {args.source}")

	count = clean_directory(args.source, args.target)
	print(f"Cleaned {count} Markdown file(s): {args.source} -> {args.target}")


if __name__ == "__main__":
	main()
