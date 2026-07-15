# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process vLLM batch-invariance probe for generated-token logprobs.

This is the next rung above the OpenAI-compatible server harness. It removes
HTTP handling and OpenAI chat response plumbing from the experiment while still
using vLLM's real engine, scheduler, model runner, sampler, and output
processor.

The probe compares a target prompt run alone against the same prompt run inside
an identical batch. It requests one generated token and raw pre-sampling
logprobs for that token.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Write a three-sentence note about why deterministic inference matters "
    "for evaluation pipelines."
)
DEFAULT_DISTRACTORS = [
    "Summarize the tradeoff between latency and throughput in one paragraph.",
    "List three ways a GPU inference server can become bottlenecked.",
    "Explain prefix caching to a new engineer in two short bullets.",
    "Write a concise status update for a model-serving experiment.",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _chat_prompt(model: str, prompt: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _parse_logprob_row(row: Any) -> list[dict[str, Any]]:
    parsed = []
    for token_id, item in row.items():
        parsed.append(
            {
                "token_id": int(token_id),
                "logprob": float(item.logprob),
                "rank": int(item.rank) if item.rank is not None else None,
                "decoded_token": item.decoded_token,
            }
        )
    return sorted(parsed, key=lambda item: (item["rank"] is None, item["rank"] or 0))


def _logprob_rows(raw: Any) -> list[list[dict[str, Any]]]:
    if not raw:
        return []
    return [_parse_logprob_row(row) for row in raw]


def _fingerprint(output: Any) -> dict[str, Any]:
    completion = output.outputs[0]
    return {
        "text": completion.text,
        "token_ids": [int(token_id) for token_id in completion.token_ids],
        "logprobs_by_step": _logprob_rows(completion.logprobs),
    }


def _compare_logprobs(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {item["token_id"]: item for item in left}
    right_by_id = {item["token_id"]: item for item in right}
    shared_ids = sorted(set(left_by_id) & set(right_by_id))
    diffs = []
    for token_id in shared_ids:
        left_item = left_by_id[token_id]
        right_item = right_by_id[token_id]
        if left_item["logprob"] != right_item["logprob"]:
            diffs.append(
                {
                    "token_id": token_id,
                    "decoded_token": left_item["decoded_token"],
                    "left_logprob": left_item["logprob"],
                    "right_logprob": right_item["logprob"],
                    "abs_diff": abs(left_item["logprob"] - right_item["logprob"]),
                }
            )
    return {
        "shared_token_count": len(shared_ids),
        "left_only_token_ids": sorted(set(left_by_id) - set(right_by_id)),
        "right_only_token_ids": sorted(set(right_by_id) - set(left_by_id)),
        "logprob_diffs": diffs,
        "max_abs_diff": max((item["abs_diff"] for item in diffs), default=0.0),
    }


def _compare_logprob_steps(
    left: list[list[dict[str, Any]]], right: list[list[dict[str, Any]]]
) -> dict[str, Any]:
    step_count = min(len(left), len(right))
    step_comparisons = [
        _compare_logprobs(left[index], right[index]) for index in range(step_count)
    ]
    first_diff_step = None
    max_abs_diff = 0.0
    for index, comparison in enumerate(step_comparisons):
        if (
            comparison["left_only_token_ids"]
            or comparison["right_only_token_ids"]
            or comparison["logprob_diffs"]
        ) and first_diff_step is None:
            first_diff_step = index
        max_abs_diff = max(max_abs_diff, comparison["max_abs_diff"])
    if len(left) != len(right) and first_diff_step is None:
        first_diff_step = step_count
    return {
        "left_steps": len(left),
        "right_steps": len(right),
        "first_diff_step": first_diff_step,
        "max_abs_diff": max_abs_diff,
        "steps": step_comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        help="Batch the target with different distractor prompts instead of exact copies.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Position of the target prompt inside the offline batch.",
    )
    parser.add_argument("--distractor-prompt", action="append", default=[])
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--gdn-prefill-backend", default="triton")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

    from vllm import LLM, SamplingParams

    prompt = _chat_prompt(args.model, args.prompt)
    raw_distractors = args.distractor_prompt or DEFAULT_DISTRACTORS
    distractors = [_chat_prompt(args.model, item) for item in raw_distractors]
    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        logprobs=args.logprobs,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        additional_config={"gdn_prefill_backend": args.gdn_prefill_backend},
    )

    # Warm one request so first-use JIT does not pollute the measurement run.
    llm.generate([prompt], params, use_tqdm=False)

    solo_outputs = llm.generate([prompt], params, use_tqdm=False)
    solo_repeat_outputs = llm.generate([prompt], params, use_tqdm=False)
    if args.heterogeneous:
        if args.target_index < 0 or args.target_index >= args.batch_size:
            raise ValueError("--target-index must be inside --batch-size")
        batch_prompts = []
        for index in range(args.batch_size - 1):
            batch_prompts.append(distractors[index % len(distractors)])
        batch_prompts.insert(args.target_index, prompt)
    else:
        batch_prompts = [prompt] * args.batch_size
    batched_outputs = llm.generate(batch_prompts, params, use_tqdm=False)

    solo = _fingerprint(solo_outputs[0])
    solo_repeat = _fingerprint(solo_repeat_outputs[0])
    batched = [_fingerprint(output) for output in batched_outputs]

    comparisons = {
        "solo_vs_solo_repeat": {
            "token_ids_equal": solo["token_ids"] == solo_repeat["token_ids"],
            "text_equal": solo["text"] == solo_repeat["text"],
            "logprobs": _compare_logprob_steps(
                solo["logprobs_by_step"], solo_repeat["logprobs_by_step"]
            ),
        },
        "solo_vs_batched": [],
    }
    for index, fp in enumerate(batched):
        is_target = (not args.heterogeneous) or index == args.target_index
        comparisons["solo_vs_batched"].append(
            {
                "batch_index": index,
                "is_target": is_target,
                "token_ids_equal": solo["token_ids"] == fp["token_ids"],
                "text_equal": solo["text"] == fp["text"],
                "logprobs": (
                    _compare_logprob_steps(
                        solo["logprobs_by_step"], fp["logprobs_by_step"]
                    )
                    if is_target
                    else None
                ),
            }
        )

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": vars(args),
        "rendered_prompt": prompt,
        "batch_prompts": batch_prompts,
        "solo": solo,
        "solo_repeat": solo_repeat,
        "batched": batched,
        "comparisons": comparisons,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )

    print(json.dumps(report["comparisons"], indent=2, sort_keys=True))
    if args.output is not None:
        print(args.output)


if __name__ == "__main__":
    main()
