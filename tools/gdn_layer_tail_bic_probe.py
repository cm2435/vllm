# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN layer-tail batch-invariance probe.

This investigation tool isolates the last part of
``QwenGatedDeltaNetAttention._output_projection``:

* RMSNormGated over the per-head recurrent output
* flatten heads into the model dimension
* output projection GEMM

It compares a target sequence run alone against the same target sequence when
embedded in a larger packed-token batch. Earlier probes cover the recurrent GDN
core; this probe checks whether the dense layer tail is shape-sensitive.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from vllm.model_executor.layers.layernorm import RMSNormGated


NUM_V_HEADS = 8
HEAD_V_DIM = 128
INPUT_DIM = NUM_V_HEADS * HEAD_V_DIM
OUTPUT_DIM = 2048


def _parse_lengths(raw: str) -> list[int]:
    lengths = [int(part) for part in raw.split(",") if part]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"invalid lengths: {raw!r}")
    return lengths


def _offset(lengths: list[int], index: int) -> int:
    return sum(lengths[:index])


def _make_inputs(
    lengths: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    total_tokens = sum(lengths)
    return {
        "core_attn_out": torch.randn(
            total_tokens, NUM_V_HEADS, HEAD_V_DIM, device=device, dtype=dtype
        )
        * 0.05,
        "z": torch.randn(total_tokens, NUM_V_HEADS, HEAD_V_DIM, device=device, dtype=dtype)
        * 0.05,
        "norm_weight": torch.randn(HEAD_V_DIM, device=device, dtype=torch.float32) * 0.05,
        "proj_weight": torch.randn(OUTPUT_DIM, INPUT_DIM, device=device, dtype=dtype)
        * 0.05,
        "proj_bias": torch.randn(OUTPUT_DIM, device=device, dtype=dtype) * 0.05,
    }


def _torch_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def _matmul_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight.t() + bias


def _run_tail(
    *,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    linear_impl: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    z_shape_og = z.shape
    norm_out = RMSNormGated.forward_static(
        core_attn_out.reshape(-1, core_attn_out.shape[-1]),
        z.reshape(-1, z.shape[-1]),
        norm_weight,
        1e-5,
        core_attn_out.dtype,
        group_size=None,
        norm_before_gate=True,
        activation="swish",
    )
    norm_out = norm_out.reshape(z_shape_og)
    flat = norm_out.flatten(-2)
    projected = linear_impl(flat, proj_weight, proj_bias)
    torch.cuda.synchronize()
    return {
        "norm_out": norm_out,
        "flat": flat,
        "projected": projected,
    }


def _compare(name: str, single: torch.Tensor, batched: torch.Tensor) -> tuple[bool, str]:
    equal = torch.equal(single, batched)
    diff = (single.float() - batched.float()).abs()
    return equal, f"{name}: equal={equal} max={diff.max().item()} mean={diff.mean().item()}"


def run_case(
    *,
    lengths: list[int],
    target_index: int,
    dtype: torch.dtype,
    linear_name: str,
    linear_impl: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> bool:
    torch.manual_seed(9012)
    device = torch.device("cuda")
    inputs = _make_inputs(lengths, dtype, device)
    target_start = _offset(lengths, target_index)
    target_len = lengths[target_index]
    target_slice = slice(target_start, target_start + target_len)

    single_inputs = {
        "core_attn_out": inputs["core_attn_out"][target_slice].clone(),
        "z": inputs["z"][target_slice].clone(),
        "norm_weight": inputs["norm_weight"],
        "proj_weight": inputs["proj_weight"],
        "proj_bias": inputs["proj_bias"],
        "linear_impl": linear_impl,
    }
    batched_inputs = dict(inputs)
    batched_inputs["linear_impl"] = linear_impl

    single = _run_tail(**single_inputs)
    batched = _run_tail(**batched_inputs)

    comparisons = [
        _compare("norm_out", single["norm_out"], batched["norm_out"][target_slice]),
        _compare("flat", single["flat"], batched["flat"][target_slice]),
        _compare("projected", single["projected"], batched["projected"][target_slice]),
    ]
    ok = True
    print(
        f"case linear={linear_name} dtype={dtype} lengths={lengths} "
        f"target_index={target_index}"
    )
    for equal, message in comparisons:
        ok = ok and equal
        print("  " + message)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", action="append", default=[])
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--linear", choices=["torch", "matmul"], default="torch")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    linear_impl = _torch_linear if args.linear == "torch" else _matmul_linear
    length_sets = (
        [_parse_lengths(raw) for raw in args.lengths]
        if args.lengths
        else [
            [73, 129, 17, 256],
            [64, 5, 512, 300],
            [1, 65, 130, 7, 409],
            [1024, 32, 1536],
        ]
    )

    all_ok = True
    for lengths in length_sets:
        target_indices = sorted({0, len(lengths) // 2, len(lengths) - 1})
        for target_index in target_indices:
            all_ok = (
                run_case(
                    lengths=lengths,
                    target_index=target_index,
                    dtype=dtype,
                    linear_name=args.linear,
                    linear_impl=linear_impl,
                )
                and all_ok
            )

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
