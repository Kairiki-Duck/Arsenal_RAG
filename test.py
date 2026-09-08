"""Compare the same language model with and without RAG retrieval."""

import argparse

from config import INDEX_PATH, METADATA_PATH, RERANKER_MODEL


BASELINE_PROMPT = """你是一个专业的阿森纳足球俱乐部知识助手。
请直接回答用户问题。如果你不确定，请明确说明。

用户问题：
{query}

请回答："""


def build_parser():
    from config import GENERATOR_MODEL

    parser = argparse.ArgumentParser(description="对比 RAG 和 baseline 的回答")
    parser.add_argument(
        "query",
        nargs="?",
        help="要测试的问题；不传时进入交互模式",
    )
    parser.add_argument("--model-path", default=GENERATOR_MODEL, help="生成模型路径")
    parser.add_argument("--index-path", default=INDEX_PATH)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument(
        "--reranker-model",
        default=RERANKER_MODEL,
        help="RAG 使用的 reranker 模型；传入空值可禁用",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def compare_once(query, rag, baseline_generator, top_k, max_new_tokens):
    """Generate and print both answers for one query."""
    query = query.strip()
    if not query:
        raise ValueError("问题不能为空")

    rag_answer = rag.ask(
        query,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    baseline_answer = baseline_generator.generate(
        BASELINE_PROMPT.format(query=query),
        max_new_tokens=max_new_tokens,
    )

    print("\n" + "=" * 72)
    print(f"问题：{query}")
    print("\n[模型 1：RAG]")
    print(rag_answer)
    print("\n[模型 2：Baseline，无检索]")
    print(baseline_answer)
    print("=" * 72)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.top_k <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--top-k 和 --max-new-tokens 必须是正整数")

    from src.generator import Generator
    from src.rag import RAG, RAGConfig

    print(f"正在加载模型：{args.model_path}")
    print("模型 1 使用 RAG + reranker，模型 2 仅使用生成模型。")

    rag_generator = Generator(model_path=args.model_path)
    baseline_generator = Generator(model_path=args.model_path)
    rag = RAG(
        config=RAGConfig(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            reranker_model=args.reranker_model or None,
            default_top_k=args.top_k,
        ),
        generator=rag_generator,
    )

    if args.query:
        compare_once(
            args.query,
            rag,
            baseline_generator,
            args.top_k,
            args.max_new_tokens,
        )
        return

    print("请输入问题，输入 exit 退出。")
    while True:
        try:
            query = input("\n问题：")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if query.strip().lower() in {"exit", "quit"}:
            print("已退出。")
            return
        if not query.strip():
            print("问题不能为空，请重新输入。")
            continue
        try:
            compare_once(
                query,
                rag,
                baseline_generator,
                args.top_k,
                args.max_new_tokens,
            )
        except Exception as exc:
            print(f"回答失败：{exc}")


if __name__ == "__main__":
    main()