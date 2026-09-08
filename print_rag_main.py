"""Run the RAG assistant and print retrieval/reranking details."""

import argparse
import logging
import sys

from config import INDEX_PATH, METADATA_PATH, RERANKER_MODEL


LOGGER = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description="输出检索结果和 reranker 排序的阿森纳 RAG 助手")
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
        "--reranker-model",
        default=RERANKER_MODEL,
        help="reranker model; use an empty value to disable",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
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
    parser.add_argument(
        "query",
        nargs="?",
        help="问题；不传时进入交互模式",
    )
    return parser


def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def print_results(results, output_fn=print):
    """Print the retrieved chunks and their final reranker order."""
    output_fn("\n" + "=" * 72)
    output_fn("[检索结果 / Reranker 排序]")
    if not results:
        output_fn("没有检索到结果。")
        return

    for position, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        text = metadata.get("text", "")
        source = metadata.get("source", "unknown")
        retrieval_score = result.get("retrieval_score")
        final_score = result.get("score")
        rank = result.get("rank", position)
        output_fn(f"\n结果 {position} | 最终排名: {rank}")
        output_fn(f"来源: {source}")
        output_fn(f"Chunk ID: {metadata.get('chunk_id', 'unknown')}")
        output_fn(f"检索类型: {result.get('retrieval_type', 'unknown')}")
        if retrieval_score is not None:
            output_fn(f"向量召回分数: {retrieval_score:.6f}")
        if isinstance(final_score, (int, float)):
            output_fn(f"Reranker 分数: {final_score:.6f}")
        output_fn(f"内容:\n{text}")


def answer_once(query, rag, top_k, max_new_tokens, output_fn=print):
    """Retrieve, print ranking details, then generate the normal RAG answer."""
    normalized_query = rag._validate_query(query)
    resolved_top_k = rag.config.default_top_k if top_k is None else top_k
    resolved_top_k = rag._validate_top_k(resolved_top_k)

    retrieved_results = rag.retriever.retrieve(
        normalized_query,
        top_k=resolved_top_k,
    )
    print_results(retrieved_results, output_fn=output_fn)

    prompt = rag.prompt_builder.build(normalized_query, retrieved_results)
    answer = rag.generator.generate(prompt, max_new_tokens=max_new_tokens)
    if not isinstance(answer, str):
        raise TypeError("generator.generate must return a string")

    output_fn("\n[模型回答]")
    output_fn(answer.strip())
    output_fn("=" * 72)


def run_chat(rag, top_k, max_new_tokens, input_fn=input, output_fn=print):
    output_fn("\n这里是阿森纳 RAG 检索调试助手!")
    output_fn("请输入问题，输入 exit 退出。")

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
            answer_once(query, rag, top_k, max_new_tokens, output_fn=output_fn)
        except (ValueError, TypeError) as exc:
            output_fn(f"输入参数错误：{exc}")
        except Exception:
            LOGGER.exception("Failed to answer query")
            output_fn("回答失败，请检查模型和索引配置后重试。")


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    if args.top_k <= 0 or args.max_new_tokens <= 0:
        LOGGER.error("--top-k and --max-new-tokens must be positive integers")
        return 2

    try:
        from src.rag import RAG, RAGConfig

        rag = RAG(
            config=RAGConfig(
                index_path=args.index_path,
                metadata_path=args.metadata_path,
                reranker_model=args.reranker_model or None,
                default_top_k=args.top_k,
            )
        )
    except Exception:
        LOGGER.exception("Failed to initialize RAG")
        return 1

    if args.query:
        try:
            answer_once(args.query, rag, args.top_k, args.max_new_tokens)
        except Exception:
            LOGGER.exception("Failed to answer query")
            return 1
        return 0

    return run_chat(rag, args.top_k, args.max_new_tokens)


if __name__ == "__main__":
    sys.exit(main())
