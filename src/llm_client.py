import json
import re
from typing import Any, Dict, Tuple

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


class LocalLLM:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Install it or run with --no_llm.")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def _format_qwen_chat(self, user_prompt: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are a strict JSON generator. Output exactly one valid JSON object and no other text.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate(self, prompt: str) -> str:
        formatted_prompt = self._format_qwen_chat(prompt)
        result = self.llm(
            formatted_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            repeat_penalty=1.12,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return result["choices"][0]["text"].strip()

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        data, _ = self.generate_json_with_raw(prompt)
        return data

    def generate_json_with_raw(self, prompt: str) -> Tuple[Dict[str, Any], str]:
        text = self.generate(prompt)
        try:
            return json.loads(text), text
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"Invalid JSON:\n{text}")
        json_text = match.group(0)
        try:
            return json.loads(json_text), text
        except Exception as e:
            raise ValueError(f"Invalid JSON:\n{text}") from e
