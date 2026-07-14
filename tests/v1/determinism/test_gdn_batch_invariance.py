# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in validation for GDN_ATTN batch invariance.

This test is intentionally not enabled by default. It is a contribution spike
for validating whether enabling GDN_ATTN at the BIC support gate is sufficient
for Qwen3.5/Qwen3.6-style GDN models.
"""

import contextlib
import os
import random
from dataclasses import dataclass

import pytest
from utils import skip_unsupported

from vllm import LLM, SamplingParams


@dataclass
class GDNMismatch:
    prompt_idx: int
    kind: str
    detail: str


def _enabled() -> bool:
    return os.getenv("VLLM_RUN_GDN_BIC_VALIDATION") == "1"


def _gdn_prefill_backends() -> list[str | None]:
    raw = os.getenv("VLLM_TEST_GDN_PREFILL_BACKENDS", "auto")
    backends: list[str | None] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        backends.append(None if value == "auto" else value)
    return backends or [None]


def _make_prompts(batch_size: int) -> list[str]:
    bases = [
        "Question: What is the capital of France? Answer:",
        "Explain how photosynthesis works in simple terms:",
        "Write a short story about a lighthouse:",
        "To implement a binary search tree in Python:",
    ]
    prompts: list[str] = []
    for idx in range(batch_size):
        padding = " This is padding context." * (idx * 8)
        prompts.append(bases[idx % len(bases)] + padding)
    return prompts


def _extract_tokens_and_logprobs(request_output) -> tuple[list[int], list[float]]:
    inner = request_output.outputs[0]
    token_ids = list(inner.token_ids)
    logprobs: list[float] = []
    for step, token_id in enumerate(token_ids):
        logprobs.append(float(inner.logprobs[step][token_id].logprob))
    return token_ids, logprobs


@skip_unsupported
@pytest.mark.skipif(
    not _enabled(),
    reason="Set VLLM_RUN_GDN_BIC_VALIDATION=1 to run this opt-in GDN BIC check.",
)
@pytest.mark.parametrize("gdn_prefill_backend", _gdn_prefill_backends())
def test_qwen35_gdn_logprobs_are_batch_invariant(
    gdn_prefill_backend: str | None,
) -> None:
    """Compare Qwen3.5 GDN outputs for bs=1 runs against one bs=N run.

    This is modeled after the existing BIC determinism tests, but uses a public
    GDN architecture and optionally forces the GDN prefill backend.
    """

    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    model = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3.5-0.8B")
    batch_size = int(os.getenv("VLLM_TEST_GDN_BATCH_SIZE", "32"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    max_tokens = int(os.getenv("VLLM_TEST_GDN_MAX_TOKENS", "8"))
    gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5"))

    assert batch_size >= 2

    prompts = _make_prompts(batch_size)
    sampling_params = SamplingParams(
        temperature=float(os.getenv("VLLM_TEST_GDN_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("VLLM_TEST_GDN_TOP_P", "1.0")),
        max_tokens=max_tokens,
        seed=int(os.getenv("VLLM_TEST_GDN_SAMPLING_SEED", "42")),
        logprobs=int(os.getenv("VLLM_TEST_GDN_LOGPROBS", "5")),
    )

    llm: LLM | None = None
    try:
        llm = LLM(
            model=model,
            max_num_seqs=batch_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="auto",
            enable_prefix_caching=False,
            gdn_prefill_backend=gdn_prefill_backend,
        )

        bs1_results: list[tuple[list[int], list[float]]] = []
        for prompt in prompts:
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            assert len(outputs) == 1
            bs1_results.append(_extract_tokens_and_logprobs(outputs[0]))

        batched_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        assert len(batched_outputs) == len(prompts)
        bsn_results = [
            _extract_tokens_and_logprobs(output) for output in batched_outputs
        ]

        mismatches: list[GDNMismatch] = []
        for idx, ((bs1_tokens, bs1_logprobs), (bsn_tokens, bsn_logprobs)) in enumerate(
            zip(bs1_results, bsn_results)
        ):
            if bs1_tokens != bsn_tokens:
                mismatches.append(
                    GDNMismatch(
                        prompt_idx=idx,
                        kind="tokens",
                        detail=f"bs1={bs1_tokens} bsn={bsn_tokens}",
                    )
                )
                continue

            if bs1_logprobs != bsn_logprobs:
                max_diff = max(
                    abs(a - b) for a, b in zip(bs1_logprobs, bsn_logprobs)
                )
                mismatches.append(
                    GDNMismatch(
                        prompt_idx=idx,
                        kind="logprobs",
                        detail=f"max_abs_diff={max_diff}",
                    )
                )

        if mismatches:
            examples = "\n".join(str(mismatch) for mismatch in mismatches[:8])
            pytest.fail(
                f"GDN BIC mismatches: {len(mismatches)}/{batch_size} "
                f"for gdn_prefill_backend={gdn_prefill_backend}\n{examples}"
            )

    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
