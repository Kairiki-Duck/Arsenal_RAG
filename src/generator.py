import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GENERATOR_MODEL

MODEL_PATH = GENERATOR_MODEL


class Generator:
    """Production-oriented text generator for the RAG answer stage."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        device=None,
        dtype=None,
        device_map="auto",
        max_input_tokens=4096,
        tokenizer=None,
        model=None,
        trust_remote_code=True,
    ):
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        if not isinstance(max_input_tokens, int) or isinstance(max_input_tokens, bool):
            raise TypeError("max_input_tokens must be an integer")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if device is not None and not isinstance(device, str):
            raise TypeError("device must be a string or None")
        if model is not None and tokenizer is None:
            raise ValueError("tokenizer is required when model is injected")

        self.model_path = model_path
        self.max_input_tokens = max_input_tokens
        self.device = device or self._default_device()

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        resolved_dtype = dtype or self._default_dtype(self.device)
        if model is None:
            load_kwargs = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": resolved_dtype,
            }
            if device_map is not None:
                load_kwargs["device_map"] = device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )
            if device_map is None:
                self.model.to(self.device)
        else:
            self.model = model
            if device_map is None and device is not None:
                self.model.to(device)

        self.model.eval()
        self._configure_tokenizer()

    @staticmethod
    def _default_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _default_dtype(device):
        return torch.bfloat16 if device.startswith("cuda") else torch.float32

    def _configure_tokenizer(self):
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _input_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return torch.device(self.device)

    def _build_input(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (AttributeError, ValueError):
                text = prompt
        else:
            text = prompt

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return {
            key: value.to(self._input_device())
            for key, value in inputs.items()
        }

    def generate(
        self,
        prompt,
        max_new_tokens=4096,
        do_sample=False,
        temperature=None,
        top_p=None,
    ):
        """Generate one answer while preserving the existing string API."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a boolean")
        if temperature is not None and (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be positive")
        if top_p is not None and (
            not isinstance(top_p, (int, float))
            or isinstance(top_p, bool)
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError("top_p must be between 0 and 1")
        if not do_sample and (temperature is not None or top_p is not None):
            raise ValueError("temperature and top_p require do_sample=True")

        inputs = self._build_input(prompt)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if temperature is not None:
            generation_kwargs["temperature"] = temperature
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if self.tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_length:]
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return answer.strip()
