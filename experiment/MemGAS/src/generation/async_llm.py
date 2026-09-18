"""
Asynchronous LLM calls (local-model version)
============================================
Replaces the original aiohttp + OpenAI API calls with direct inference through a
local Qwen-7B-Instruct model.

The original interface is preserved:
    run_async(prompts, model="qwen-7b") -> List[str]

The `model` argument no longer selects the model that is actually used; every call
goes to the local model given by the LOCAL_MODEL_PATH environment variable.
"""

import os
import sys
from typing import List

# Make `local_qwen` importable regardless of the current working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_qwen import get_local_llm


async def create_completion(session, prompt, model="gpt-4o-mini"):
    """
    Single-prompt inference. The original signature is kept for compatibility;
    `session` is accepted but never used. The local model has no real async API,
    so generation is performed synchronously.
    """
    llm = get_local_llm()
    response = llm.generate(prompt, max_tokens=4000, temperature=0.0)
    print(response[:200])
    print('-----------------------')
    return response


async def run_async(prompts: List[str], model="gpt-4o-mini") -> List[str]:
    """
    Batch inference using the local model's `batch_generate`.

    Args:
        prompts: list of input prompts
        model: kept for compatibility, no longer used (the local model is always used)

    Returns:
        responses: list of generated answers
    """
    if not prompts:
        return []

    llm = get_local_llm()

    # Large batches may OOM, so process them in chunks
    BATCH_SIZE = 8
    all_responses = []

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        batch_responses = llm.batch_generate(batch, max_tokens=4000, temperature=0.0)
        all_responses.extend(batch_responses)
        print(f"[batch] {i+len(batch)}/{len(prompts)} done")

    return all_responses


if __name__ == "__main__":
    import asyncio
    prompts = ['who are you?', 'what you like?']
    responses = asyncio.run(run_async(prompts))
    print('--------')
    print(responses)
