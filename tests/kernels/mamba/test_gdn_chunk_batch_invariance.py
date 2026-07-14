# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GDN chunk-core batch-invariance validation.

This isolates the varlen GDN chunk prefill kernel from the rest of the model.
It compares a target sequence run alone against the same target sequence placed
inside a larger unrelated varlen batch. A failure here means enabling
``GDNAttentionBackend.supports_batch_invariance`` is not sufficient: the GDN
prefill core itself is batch-shape dependent.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("GDN chunk batch-invariance validation requires CUDA.", allow_module_level=True)

from tests.v1.attention.utils import create_vllm_config  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fla.ops.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
)


def _enabled() -> bool:
    return os.getenv("VLLM_RUN_GDN_CORE_BIC_VALIDATION") == "1"


def _backend_list() -> list[str]:
    raw = os.getenv("VLLM_TEST_GDN_PREFILL_BACKENDS", "auto,triton")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _make_chunk_rule(backend: str) -> ChunkGatedDeltaRule:
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        hf_config_override={"linear_key_head_dim": 128},
    )
    cfg.additional_config = {"gdn_prefill_backend": backend}
    with set_current_vllm_config(cfg):
        return ChunkGatedDeltaRule()


def _make_cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return cu_seqlens


def _run_chunk_rule(
    rule: ChunkGatedDeltaRule,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_indices = prepare_chunk_indices(cu_seqlens.cpu(), FLA_CHUNK_SIZE).to(
        cu_seqlens.device
    )
    chunk_offsets = prepare_chunk_offsets(cu_seqlens.cpu(), FLA_CHUNK_SIZE).to(
        cu_seqlens.device
    )
    out, final_state = rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        use_qk_l2norm_in_kernel=False,
    )
    assert final_state is not None
    torch.cuda.synchronize()
    return out, final_state


@pytest.mark.skipif(
    not _enabled(),
    reason="Set VLLM_RUN_GDN_CORE_BIC_VALIDATION=1 to run this opt-in check.",
)
@pytest.mark.parametrize("backend", _backend_list())
@pytest.mark.parametrize("target_index", [0, 2])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_chunk_core_is_batch_invariant_for_target_sequence(
    backend: str,
    target_index: int,
    state_dtype: torch.dtype,
) -> None:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    lengths = [73, 129, 17, 256]
    target_len = lengths[target_index]
    total_tokens = sum(lengths)
    target_start = sum(lengths[:target_index])
    target_stop = target_start + target_len

    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 128

    q_batch = torch.randn(
        1, total_tokens, num_k_heads, head_k_dim, device=device, dtype=dtype
    )
    k_batch = torch.randn_like(q_batch)
    v_batch = torch.randn(
        1, total_tokens, num_v_heads, head_v_dim, device=device, dtype=dtype
    )
    q_batch = F.normalize(q_batch.float(), p=2, dim=-1).to(dtype)
    k_batch = F.normalize(k_batch.float(), p=2, dim=-1).to(dtype)
    g_batch = -torch.rand(
        1, total_tokens, num_v_heads, device=device, dtype=dtype
    ) * 0.02
    beta_batch = torch.rand(
        1, total_tokens, num_v_heads, device=device, dtype=dtype
    )
    initial_state_batch = (
        torch.randn(
            len(lengths),
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        * 0.05
    )

    q_single = q_batch[:, target_start:target_stop].clone()
    k_single = k_batch[:, target_start:target_stop].clone()
    v_single = v_batch[:, target_start:target_stop].clone()
    g_single = g_batch[:, target_start:target_stop].clone()
    beta_single = beta_batch[:, target_start:target_stop].clone()
    initial_state_single = initial_state_batch[target_index : target_index + 1].clone()

    rule = _make_chunk_rule(backend)
    single_cu = _make_cu_seqlens([target_len], device)
    batch_cu = _make_cu_seqlens(lengths, device)

    single_out, single_final_state = _run_chunk_rule(
        rule,
        q_single,
        k_single,
        v_single,
        g_single,
        beta_single,
        initial_state_single,
        single_cu,
    )
    batch_out, batch_final_state = _run_chunk_rule(
        rule,
        q_batch,
        k_batch,
        v_batch,
        g_batch,
        beta_batch,
        initial_state_batch,
        batch_cu,
    )

    target_out = batch_out[:, target_start:target_stop]
    target_final_state = batch_final_state[target_index : target_index + 1]

    out_equal = torch.equal(single_out, target_out)
    state_equal = torch.equal(single_final_state, target_final_state)
    if not (out_equal and state_equal):
        out_diff = (single_out.float() - target_out.float()).abs()
        state_diff = (single_final_state.float() - target_final_state.float()).abs()
        pytest.fail(
            "GDN chunk core is not batch invariant for target sequence "
            f"(backend={backend}, target_index={target_index}, "
            f"state_dtype={state_dtype}, "
            f"out_equal={out_equal}, state_equal={state_equal}, "
            f"out_max_diff={out_diff.max().item()}, "
            f"out_mean_diff={out_diff.mean().item()}, "
            f"state_max_diff={state_diff.max().item()}, "
            f"state_mean_diff={state_diff.mean().item()})"
        )
