# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
import os.path as osp
import re
from datetime import datetime
from time import sleep


class OpenAILLM:
    """OpenAI API call (kept for the comparison experiments)."""

    def __init__(self, model_name="gpt-4o-mini-2024-07-18"):
        from openai import OpenAI

        self.model_name = model_name

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", ""),
        )

    def __call__(
        self,
        prompt,
        system_prompt=None,
        temperature=0,
        top_p=1.0,
        max_tokens=1024,
        seed=42,
        max_num_retries=2,
        return_full=False,
    ) -> str:
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        retry = 0
        while retry < max_num_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                content = completion.choices[0].message.content
                if not return_full:
                    return content

                ret_dict = {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "model_name": self.model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response": content,
                    "response_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                return ret_dict

            except Exception as e:
                retry += 1
                sleep(5)
                print(f"Error: {e}", flush=True)

        raise RuntimeError(
            "Calling OpenAI failed after retrying for " f"{retry} times."
        )


class LocalLLM:
    """
    Local-model wrapper (uses the singleton from src/local_qwen.py).
    Keeps the same interface signature as the original LocalLLM.
    """

    def __init__(self, model_name_or_path=None):
        # Store the path in the environment so that the local_qwen singleton can use it
        if model_name_or_path:
            os.environ["LOCAL_MODEL_PATH"] = model_name_or_path
        # Lazy loading: the model is only loaded on the first call
        self._llm = None

    def _ensure_loaded(self):
        if self._llm is None:
            from local_qwen import get_local_llm
            self._llm = get_local_llm()
        return self._llm

    def __call__(
        self,
        prompt: str,
        temperature: float = 0.9,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        seed: int = 42,
    ):
        llm = self._ensure_loaded()
        return llm.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
