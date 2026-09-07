"""Command-line entry point for the Arsenal RAG assistant."""

import argparse
import logging
import sys

from config import INDEX_PATH, METADATA_PATH


LOGGER = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description="阿森纳 RAG 知识助手")
    parser.add_argument(
        "--index-path",
        default=INDEX_PATH,
        help="FAISS index path",
    )
    parser.add_argument(
        "--metadata-path",
        default=METADATA_PATH,
        help="retrieval metadata path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="maximum number of generated tokens",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging level",
    )
    return parser


def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_chat(rag, top_k, max_new_tokens, input_fn=input, output_fn=print):
    """Run the interactive loop and return a process exit code."""
    output_fn("\n这里是阿森纳 RAG 知识助手!")
    output_fn("请输入您的问题，输入 exit 退出。")

    while True:
        try:
            query = input_fn("\nUser: ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\n已退出。")
            return 0

        if query.strip().lower() in {"exit", "quit"}:
            output_fn("已退出。")
            return 0
        if not query.strip():
            output_fn("问题不能为空，请重新输入。")
            continue

        try:
            answer = rag.ask(
                query,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
        except (ValueError, TypeError) as exc:
            output_fn(f"输入参数错误：{exc}")
            continue
        except Exception:
            LOGGER.exception("Failed to answer query")
            output_fn("回答失败，请检查模型和索引配置后重试。")
            continue

        output_fn(f"\nAssistant:\n{answer}")


def main(argv=None, rag=None, input_fn=input, output_fn=print):
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    if args.top_k <= 0 or args.max_new_tokens <= 0:
        LOGGER.error("--top-k and --max-new-tokens must be positive integers")
        return 2

    try:
        if rag is None:
            from src.rag import RAG, RAGConfig

        assistant = (
            RAG(
                config=RAGConfig(
                    index_path=args.index_path,
                    metadata_path=args.metadata_path,
                    default_top_k=args.top_k,
                )
            )
            if rag is None
            else rag
        )
    except Exception:
        LOGGER.exception("Failed to initialize RAG")
        return 1

    return run_chat(
        assistant,
        args.top_k,
        args.max_new_tokens,
        input_fn=input_fn,
        output_fn=output_fn,
    )


if __name__ == "__main__":
    sys.exit(main())