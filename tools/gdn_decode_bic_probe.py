# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-by-stage GDN decode batch-invariance probe.

This mirrors the standard non-spec decode path in
``QwenGatedDeltaNetAttention._forward_core``:

* causal_conv1d_update for the one-token q/k/v stream
* rearrange into q/k/v heads
* fused_sigmoid_gating_delta_rule_update for the recurrent decode update

It compares a target decode request run alone against the same target request
embedded in a larger decode batch.
"""

from __future__ import annotations

import argparse

import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update


NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM


def _cu_decode(num_reqs: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_reqs + 1, dtype=torch.int32, device=device)


def _make_inputs(
    num_reqs: int,
    state_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dtype = torch.bfloat16
    return {
        "mixed_qkv": torch.randn(num_reqs, CONV_DIM, device=device, dtype=dtype) * 0.05,
        "a": torch.randn(num_reqs, NUM_V_HEADS, device=device, dtype=dtype) * 0.05,
        "b": torch.randn(num_reqs, NUM_V_HEADS, device=device, dtype=dtype) * 0.05,
        "conv_weight": torch.randn(
            CONV_DIM, CONV_KERNEL, device=device, dtype=dtype
        )
        * 0.05,
        "conv_bias": torch.randn(CONV_DIM, device=device, dtype=dtype) * 0.05,
        "conv_state": torch.randn(
            num_reqs + 1,
            CONV_DIM,
            CONV_KERNEL - 1,
            device=device,
            dtype=dtype,
        )
        * 0.05,
        "ssm_state": torch.randn(
            num_reqs + 1,
            NUM_V_HEADS,
            HEAD_V_DIM,
            HEAD_K_DIM,
            device=device,
            dtype=state_dtype,
        )
        * 0.05,
        "A_log": torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32) * 0.05,
        "dt_bias": torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32) * 0.05,
    }


def _rearrange_mixed_qkv(mixed_qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
    query, key, value = torch.split(mixed_qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    query = query.view(1, mixed_qkv.shape[0], NUM_K_HEADS, HEAD_K_DIM)
    key = key.view(1, mixed_qkv.shape[0], NUM_K_HEADS, HEAD_K_DIM)
    value = value.view(1, mixed_qkv.shape[0], NUM_V_HEADS, HEAD_V_DIM)
    return query, key, value


def _run_decode(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = mixed_qkv.device
    num_reqs = mixed_qkv.shape[0]
    state_indices = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)

    conv_state_after = conv_state.clone()
    conv_out = causal_conv1d_update(
        mixed_qkv,
        conv_state_after,
        conv_weight,
        conv_bias,
        "silu",
        conv_state_indices=state_indices,
        validate_data=True,
    )
    q, k, v = _rearrange_mixed_qkv(conv_out)
    ssm_state_after = ssm_state.clone()
    core_out, last_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=ssm_state_after,
        inplace_final_state=True,
        cu_seqlens=_cu_decode(num_reqs, device),
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    return {
        "conv_out": conv_out,
        "conv_state_after": conv_state_after,
        "q": q.squeeze(0),
        "k": k.squeeze(0),
        "v": v.squeeze(0),
        "core_out": core_out.squeeze(0),
        "last_state": last_state,
        "ssm_state_after": ssm_state_after,
    }


def _run_packed_decode(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = mixed_qkv.device
    num_reqs = mixed_qkv.shape[0]
    state_indices = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)

    ssm_state_after = ssm_state.clone()
    out = torch.empty(
        num_reqs, 1, NUM_V_HEADS, HEAD_V_DIM, device=device, dtype=mixed_qkv.dtype
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=HEAD_K_DIM**-0.5,
        initial_state=ssm_state_after,
        out=out,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    return {
        "core_out": out.squeeze(1),
        "ssm_state_after": ssm_state_after,
    }


def _compare(name: str, single: torch.Tensor, batched: torch.Tensor) -> tuple[bool, str]:
    equal = torch.equal(single, batched)
    diff = (single.float() - batched.float()).abs()
    return equal, f"{name}: equal={equal} max={diff.max().item()} mean={diff.mean().item()}"


def run_case(num_reqs: int, target_index: int, state_dtype: torch.dtype) -> bool:
    torch.manual_seed(5678)
    device = torch.device("cuda")
    inputs = _make_inputs(num_reqs, state_dtype, device)
    target_cache_index = target_index + 1

    single_inputs = {
        "mixed_qkv": inputs["mixed_qkv"][target_index : target_index + 1].clone(),
        "a": inputs["a"][target_index : target_index + 1].clone(),
        "b": inputs["b"][target_index : target_index + 1].clone(),
        "conv_weight": inputs["conv_weight"],
        "conv_bias": inputs["conv_bias"],
        "conv_state": torch.cat(
            [
                inputs["conv_state"][:1].clone(),
                inputs["conv_state"][target_cache_index : target_cache_index + 1].clone(),
            ],
            dim=0,
        ),
        "ssm_state": torch.cat(
            [
                inputs["ssm_state"][:1].clone(),
                inputs["ssm_state"][target_cache_index : target_cache_index + 1].clone(),
            ],
            dim=0,
        ),
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
    }

    single = _run_decode(**single_inputs)
    batched = _run_decode(**inputs)

    comparisons = [
        _compare("conv_out", single["conv_out"], batched["conv_out"][target_index : target_index + 1]),
        _compare(
            "conv_state_after",
            single["conv_state_after"][1:2],
            batched["conv_state_after"][target_cache_index : target_cache_index + 1],
        ),
        _compare("q", single["q"], batched["q"][target_index : target_index + 1]),
        _compare("k", single["k"], batched["k"][target_index : target_index + 1]),
        _compare("v", single["v"], batched["v"][target_index : target_index + 1]),
        _compare(
            "core_out",
            single["core_out"],
            batched["core_out"][target_index : target_index + 1],
        ),
        _compare("last_state", single["last_state"][1:2], batched["last_state"][target_cache_index : target_cache_index + 1]),
        _compare(
            "ssm_state_after",
            single["ssm_state_after"][1:2],
            batched["ssm_state_after"][target_cache_index : target_cache_index + 1],
        ),
    ]
    ok = all(equal for equal, _ in comparisons)
    print(
        {
            "num_reqs": num_reqs,
            "target_index": target_index,
            "state_dtype": str(state_dtype),
            "ok": ok,
        }
    )
    for _, line in comparisons:
        print("  " + line)
    return ok


def run_packed_case(num_reqs: int, target_index: int, state_dtype: torch.dtype) -> bool:
    torch.manual_seed(91011)
    device = torch.device("cuda")
    inputs = _make_inputs(num_reqs, state_dtype, device)
    target_cache_index = target_index + 1

    single_inputs = {
        "mixed_qkv": inputs["mixed_qkv"][target_index : target_index + 1].clone(),
        "a": inputs["a"][target_index : target_index + 1].clone(),
        "b": inputs["b"][target_index : target_index + 1].clone(),
        "ssm_state": torch.cat(
            [
                inputs["ssm_state"][:1].clone(),
                inputs["ssm_state"][target_cache_index : target_cache_index + 1].clone(),
            ],
            dim=0,
        ),
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
    }
    batch_inputs = {
        "mixed_qkv": inputs["mixed_qkv"],
        "a": inputs["a"],
        "b": inputs["b"],
        "ssm_state": inputs["ssm_state"],
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
    }

    single = _run_packed_decode(**single_inputs)
    batched = _run_packed_decode(**batch_inputs)

    comparisons = [
        _compare(
            "packed_core_out",
            single["core_out"],
            batched["core_out"][target_index : target_index + 1],
        ),
        _compare(
            "packed_ssm_state_after",
            single["ssm_state_after"][1:2],
            batched["ssm_state_after"][target_cache_index : target_cache_index + 1],
        ),
    ]
    ok = all(equal for equal, _ in comparisons)
    print(
        {
            "mode": "packed",
            "num_reqs": num_reqs,
            "target_index": target_index,
            "state_dtype": str(state_dtype),
            "ok": ok,
        }
    )
    for _, line in comparisons:
        print("  " + line)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-reqs", type=int, default=8)
    parser.add_argument("--target-index", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["standard", "packed", "both"],
        default="both",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_indices = sorted(set([0, args.target_index, args.num_reqs - 1]))
    all_ok = True
    for target_index in target_indices:
        for state_dtype in (torch.bfloat16, torch.float32):
            if args.mode in ("standard", "both"):
                all_ok &= run_case(args.num_reqs, target_index, state_dtype)
            if args.mode in ("packed", "both"):
                all_ok &= run_packed_case(args.num_reqs, target_index, state_dtype)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
