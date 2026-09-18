"""
Local Qwen-7B-Instruct model loading and inference
==================================================
The model is loaded as a singleton; its path is taken from the LOCAL_MODEL_PATH
environment variable. All LLM calls are routed through this module, replacing the
original OpenAI API / vLLM API calls.

Usage:
    from local_qwen import get_local_llm
    llm = get_local_llm()
    response = llm.generate(prompt, max_tokens=500)
    responses = llm.batch_generate(prompts, max_tokens=500)  # batched
"""

import os
import threading
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    """Wrapper around a local model, supporting single and batched inference."""

    def __init__(self, model_path: str):
        print(f"[LocalLLM] Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self._lock = threading.Lock()  # keeps batched inference thread-safe
        print("[LocalLLM] Model loaded")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Single-prompt inference."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.01,
            top_p=top_p,
            do_sample=(temperature > 0),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        """
        Batched inference: all prompts are encoded at once and generated together.
        Note: memory usage grows linearly with the number of prompts, so it is
        recommended to call this in chunks.
        """
        if not prompts:
            return []

        messages_list = [
            [{"role": "user", "content": p}] for p in prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]

        with self._lock:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                top_p=top_p,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        responses = []
        input_lens = inputs["input_ids"].shape[1]
        for i, out_ids in enumerate(outputs):
            generated = out_ids[input_lens:]
            # The real input length may differ because of padding; a more precise way:
            actual_len = (inputs["attention_mask"][i] == 1).sum().item()
            generated = out_ids[actual_len:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            responses.append(response.strip())

        return responses

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)


# ─── Singleton ────────────────────────────────────────────────────────────
_local_llm: Optional[LocalLLM] = None
_llm_lock = threading.Lock()


def get_local_llm(model_path: Optional[str] = None) -> LocalLLM:
    """
    Return the global LocalLLM singleton.

    Args:
        model_path: path to the model. If None, it is read from the
            LOCAL_MODEL_PATH environment variable.
    """
    global _local_llm

    if model_path is None:
        model_path = os.getenv("LOCAL_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "No model path specified. Set the LOCAL_MODEL_PATH environment "
                "variable or pass the model_path argument."
            )

    if _local_llm is None:
        with _llm_lock:
            if _local_llm is None:
                _local_llm = LocalLLM(model_path)

    return _local_llm


def reset_local_llm():
    """Reset the singleton (used for testing)."""
    global _local_llm
    with _llm_lock:
        _local_llm = None
