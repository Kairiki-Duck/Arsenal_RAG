from collections.abc import Mapping


QWEN_RERANKER_MARKER = "qwen3-reranker"
QWEN_INSTRUCTION = (
    "Given a search query, retrieve relevant passages that answer the query"
)
QWEN_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    """Rank retrieved chunks with a cross-encoder model."""

    def __init__(
        self,
        model_path="BAAI/bge-reranker-v2-m3",
        model=None,
        tokenizer=None,
        device=None,
        device_map="auto",
        max_length=8192,
        batch_size=4,
    ):
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.is_qwen = QWEN_RERANKER_MARKER in model_path.lower()
        if self.is_qwen:
            self._init_qwen(model, tokenizer, device, device_map)
        else:
            if model is None:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(model_path)
            self.model = model

    def _init_qwen(self, model, tokenizer, device, device_map):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.true_token_id is None or self.false_token_id is None:
            raise ValueError("Qwen reranker tokenizer must contain yes and no tokens")

        if model is None:
            load_kwargs = {"trust_remote_code": True}
            if device_map is not None:
                load_kwargs["device_map"] = device_map
            else:
                load_kwargs["torch_dtype"] = (
                    torch.bfloat16 if self.device.startswith("cuda") else torch.float32
                )
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
            if device_map is None:
                model.to(self.device)
        self.model = model.eval()

    @staticmethod
    def _qwen_pair(query, document):
        return (
            f"<Instruct>: {QWEN_INSTRUCTION}\n\n"
            f"<Query>: {query}\n\n<Document>: {document}"
        )

    def _qwen_inputs(self, pairs):
        prefix_tokens = self.tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
        available_length = self.max_length - len(prefix_tokens) - len(suffix_tokens)
        if available_length <= 0:
            raise ValueError("max_length is too small for the Qwen reranker template")

        tokenized = self.tokenizer(
            pairs,
            padding=False,
            truncation=True,
            max_length=available_length,
            return_attention_mask=False,
        )
        for index, input_ids in enumerate(tokenized["input_ids"]):
            tokenized["input_ids"][index] = prefix_tokens + input_ids + suffix_tokens
        inputs = self.tokenizer.pad(tokenized, padding=True, return_tensors="pt")
        model_device = getattr(self.model, "device", self.device)
        return {key: value.to(model_device) for key, value in inputs.items()}

    def _qwen_scores(self, pairs):
        import torch

        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                inputs = self._qwen_inputs(pairs[start:start + self.batch_size])
                logits = self.model(**inputs).logits[:, -1, :]
                binary_logits = torch.stack(
                    [logits[:, self.false_token_id], logits[:, self.true_token_id]],
                    dim=1,
                )
                scores.extend(torch.softmax(binary_logits, dim=1)[:, 1].float().tolist())
        return scores

    def rerank(self, query, results, top_k):
        candidates = [
            result for result in results
            if isinstance(result, Mapping)
            and isinstance(result.get("metadata"), Mapping)
            and isinstance(result["metadata"].get("text"), str)
        ]
        if not candidates:
            return []

        pairs = [(query, result["metadata"]["text"]) for result in candidates]
        if self.is_qwen:
            scores = self._qwen_scores(
                [self._qwen_pair(query, document) for query, document in pairs]
            )
        else:
            scores = self.model.predict(pairs)
        if len(scores) != len(candidates):
            raise ValueError("reranker returned an unexpected number of scores")

        ranked = []
        for result, score in zip(candidates, scores):
            ranked.append({
                **result,
                "retrieval_score": result.get("score"),
                "score": float(score),
                "retrieval_type": "reranked",
            })
        ranked.sort(key=lambda item: item["score"], reverse=True)
        for rank, result in enumerate(ranked[:top_k], start=1):
            result["rank"] = rank
        return ranked[:top_k]


__all__ = ["Reranker"]