# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-by-stage GDN prefill batch-invariance probe.

This is investigation tooling, not a default test. It mirrors the prefill-only
parts of ``QwenGatedDeltaNetAttention._forward_core`` and compares a target
sequence run alone against the same target sequence embedded in a larger varlen
batch.

The purpose is to find the first stage that breaks exact batch invariance:

* causal conv1d output
* conv-state writeback
* fused post-conv q/k/v/g/beta prep
* chunk_gated_delta_rule output/final state
* SSM-state writeback
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_post_conv_prep,
)
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata


NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM


def _cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return cu


def _conv_metadata(cu_cpu: torch.Tensor, device: torch.device) -> SimpleNamespace:
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        cu_cpu, device=device
    )
    return SimpleNamespace(
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


def _chunk_metadata(cu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        prepare_chunk_indices(cu.cpu(), FLA_CHUNK_SIZE).to(cu.device),
        prepare_chunk_offsets(cu.cpu(), FLA_CHUNK_SIZE).to(cu.device),
    )


def _make_inputs(
    lengths: list[int],
    state_dtype: torch.dtype,
    device: torch.device,
    has_initial_state: bool,
) -> dict[str, torch.Tensor]:
    total_tokens = sum(lengths)
    dtype = torch.bfloat16
    return {
        "mixed_qkv": torch.randn(total_tokens, CONV_DIM, device=device, dtype=dtype)
        * 0.05,
        "a": torch.randn(total_tokens, NUM_V_HEADS, device=device, dtype=dtype) * 0.05,
        "b": torch.randn(total_tokens, NUM_V_HEADS, device=device, dtype=dtype) * 0.05,
        "conv_weight": torch.randn(
            CONV_DIM, CONV_KERNEL, device=device, dtype=dtype
        )
        * 0.05,
        "conv_bias": torch.randn(CONV_DIM, device=device, dtype=dtype) * 0.05,
        "conv_state": torch.randn(
            len(lengths) + 1,
            CONV_DIM,
            CONV_KERNEL - 1,
            device=device,
            dtype=dtype,
        )
        * (0.05 if has_initial_state else 0.0),
        "ssm_state": torch.randn(
            len(lengths) + 1,
            NUM_V_HEADS,
            HEAD_V_DIM,
            HEAD_K_DIM,
            device=device,
            dtype=state_dtype,
        )
        * (0.05 if has_initial_state else 0.0),
        "A_log": torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32) * 0.05,
        "dt_bias": torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32) * 0.05,
    }


def _run_stages(
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
    lengths: list[int],
    has_initial_state: bool,
) -> dict[str, torch.Tensor]:
    device = mixed_qkv.device
    cu = _cu_seqlens(lengths, device)
    cu_cpu = cu.cpu()
    # State/cache index 0 is NULL_BLOCK_ID in vLLM metadata, so real cache
    # entries must use positive indices.
    state_indices = torch.arange(1, len(lengths) + 1, dtype=torch.int32, device=device)
    has_state = torch.full(
        (len(lengths),), has_initial_state, dtype=torch.bool, device=device
    )
    metadata = _conv_metadata(cu_cpu, device)

    conv_state_after = conv_state.clone()
    conv_out = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        conv_weight,
        conv_bias,
        conv_states=conv_state_after,
        has_initial_state=has_state,
        cache_indices=state_indices,
        query_start_loc=cu,
        metadata=metadata,
    ).transpose(0, 1)

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=NUM_K_HEADS,
        head_k_dim=HEAD_K_DIM,
        head_v_dim=HEAD_V_DIM,
        apply_l2norm=True,
        output_g_exp=False,
    )
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    g = g.unsqueeze(0)
    beta = beta.unsqueeze(0)

    chunk_indices, chunk_offsets = _chunk_metadata(cu)
    initial_state = ssm_state[state_indices].clone()
    if not has_initial_state:
        initial_state.zero_()

    chunk_out, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        use_qk_l2norm_in_kernel=False,
    )
    assert final_state is not None

    ssm_state_after = ssm_state.clone()
    ssm_state_after[state_indices] = final_state.to(ssm_state_after.dtype)
    torch.cuda.synchronize()

    return {
        "conv_out": conv_out,
        "conv_state_after": conv_state_after,
        "q": q.squeeze(0),
        "k": k.squeeze(0),
        "v": v.squeeze(0),
        "g": g.squeeze(0),
        "beta": beta.squeeze(0),
        "chunk_out": chunk_out.squeeze(0),
        "final_state": final_state,
        "ssm_state_after": ssm_state_after,
    }


def _compare(
    name: str,
    single: torch.Tensor,
    batched: torch.Tensor,
) -> tuple[bool, str]:
    equal = torch.equal(single, batched)
    diff = (single.float() - batched.float()).abs()
    return (
        equal,
        f"{name}: equal={equal} max={diff.max().item()} mean={diff.mean().item()}",
    )


def run_case(
    lengths: list[int],
    target_index: int,
    state_dtype: torch.dtype,
    has_initial_state: bool,
) -> bool:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    inputs = _make_inputs(lengths, state_dtype, device, has_initial_state)
    target_start = sum(lengths[:target_index])
    target_stop = target_start + lengths[target_index]
    target_slice = slice(target_start, target_stop)

    single_inputs = {
        "mixed_qkv": inputs["mixed_qkv"][target_slice].clone(),
        "a": inputs["a"][target_slice].clone(),
        "b": inputs["b"][target_slice].clone(),
        "conv_weight": inputs["conv_weight"],
        "conv_bias": inputs["conv_bias"],
        "conv_state": torch.cat(
            [
                inputs["conv_state"][:1].clone(),
                inputs["conv_state"][target_index + 1 : target_index + 2].clone(),
            ],
            dim=0,
        ),
        "ssm_state": torch.cat(
            [
                inputs["ssm_state"][:1].clone(),
                inputs["ssm_state"][target_index + 1 : target_index + 2].clone(),
            ],
            dim=0,
        ),
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "lengths": [lengths[target_index]],
        "has_initial_state": has_initial_state,
    }
    batch_inputs = {
        **inputs,
        "lengths": lengths,
        "has_initial_state": has_initial_state,
    }

    single = _run_stages(**single_inputs)
    batched = _run_stages(**batch_inputs)

    comparisons = [
        _compare("conv_out", single["conv_out"], batched["conv_out"][target_slice]),
        _compare(
            "conv_state_after",
            single["conv_state_after"][1:2],
            batched["conv_state_after"][target_index + 1 : target_index + 2],
        ),
        _compare("q", single["q"], batched["q"][target_slice]),
        _compare("k", single["k"], batched["k"][target_slice]),
        _compare("v", single["v"], batched["v"][target_slice]),
        _compare("g", single["g"], batched["g"][target_slice]),
        _compare("beta", single["beta"], batched["beta"][target_slice]),
        _compare(
            "chunk_out", single["chunk_out"], batched["chunk_out"][target_slice]
        ),
        _compare(
            "final_state",
            single["final_state"][0:1],
            batched["final_state"][target_index : target_index + 1],
        ),
        _compare(
            "ssm_state_after",
            single["ssm_state_after"][1:2],
            batched["ssm_state_after"][target_index + 1 : target_index + 2],
        ),
    ]

    ok = all(equal for equal, _ in comparisons)
    print(
        {
            "lengths": lengths,
            "target_index": target_index,
            "state_dtype": str(state_dtype),
            "has_initial_state": has_initial_state,
            "ok": ok,
        }
    )
    for _, line in comparisons:
        print("  " + line)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[73, 129, 17, 256],
        help="Varlen query lengths to embed the target inside.",
    )
    parser.add_argument("--target-index", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_ok = True
    target_indices = sorted(set([0, args.target_index, len(args.lengths) - 1]))
    for target_index in target_indices:
        for state_dtype in (torch.bfloat16, torch.float32):
            for has_initial_state in (False, True):
                all_ok &= run_case(
                    list(args.lengths), target_index, state_dtype, has_initial_state
                )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
