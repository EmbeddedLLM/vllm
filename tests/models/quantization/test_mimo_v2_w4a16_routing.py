# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing support for the AITER Triton MXFP4 W4A16 MoE backend.

MiMo-V2.6 uses an ungrouped sigmoid router with a per-expert correction bias
(``scoring_func=sigmoid``, ``topk_method=noaux_tc``, ``n_group == 1``), which
``get_routing_method_type`` classifies as ``RoutingMethodType.DeepSeekV3``.
``AiterW4A16ExpertsMonolithic`` used to reject that router ("kernel does not
support routing method ..."), and on gfx942 it is the only MXFP4 kernel whose
device gate accepts the card, so the checkpoint could not be served natively at
all.

This module covers both halves of the fix: the backend advertises support, and
the degenerate grouping it uses actually reproduces the model's router.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a16_moe import (
    AiterW4A16ExpertsMonolithic,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

MIMO_EXPERTS = 256
MIMO_TOP_K = 8
MIMO_PRO_EXPERTS = 384  # MiMo-V2.6-Pro is above the grouped top-k's 256 cap


def mimo_routing_method() -> RoutingMethodType:
    """MiMo-V2.6: sigmoid + correction bias, a single expert group."""
    return get_routing_method_type(
        scoring_func="sigmoid",
        top_k=MIMO_TOP_K,
        renormalize=True,
        num_expert_group=1,
        has_e_score_bias=True,
        routed_scaling_factor=None,
    )


def mimo_moe_config(num_experts: int = MIMO_EXPERTS) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=MIMO_TOP_K,
        hidden_dim=4096,
        intermediate_size=2048,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        routing_method=mimo_routing_method(),
        router_logits_dtype=torch.float32,
    )


def test_ungrouped_sigmoid_router_is_deepseekv3() -> None:
    """Guard the premise of the fix: MiMo's router is DeepSeekV3-classified."""
    assert mimo_routing_method() == RoutingMethodType.DeepSeekV3


def test_aiter_w4a16_supports_mimo_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the reported failure.

    The device gate is stubbed out so the routing check is what decides; before
    the fix ``is_supported_config`` returned ``(False, "kernel does not support
    routing method ...")`` here, which is what aborted model construction.
    """
    monkeypatch.setattr(
        AiterW4A16ExpertsMonolithic,
        "_supports_current_device",
        staticmethod(lambda: True),
    )

    supported, reason = AiterW4A16ExpertsMonolithic.is_supported_config(
        AiterW4A16ExpertsMonolithic,
        mimo_moe_config(),
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert supported, reason


def test_expert_count_above_grouped_topk_limit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MiMo-V2.6-Pro (384 experts) must be turned away by the routing gate.

    The ungrouped sigmoid path goes through aiter's grouped top-k, which caps out
    at 256 experts. Declining it here keeps the failure at construction time
    instead of an assert on the first forward.
    """
    monkeypatch.setattr(
        AiterW4A16ExpertsMonolithic,
        "_supports_current_device",
        staticmethod(lambda: True),
    )

    supported, reason = AiterW4A16ExpertsMonolithic.is_supported_config(
        AiterW4A16ExpertsMonolithic,
        mimo_moe_config(num_experts=MIMO_PRO_EXPERTS),
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported, reason
    assert reason is not None and "256 experts" in reason


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="only runs on ROCm with a supported AITER install",
)
def test_degenerate_grouped_routing_matches_mimo_reference() -> None:
    """The routing the fix requests must equal MiMo's own router math.

    MiMo's router is ungrouped, and aiter's flat top-k only transforms scores
    with softmax/sqrtsoftplus, so the wrapper asks the *grouped* kernel for every
    group (``topk_group == num_expert_group``). That makes the group stage a
    no-op, i.e. a global top-k over ``sigmoid(logits) + bias``.
    """
    aiter_routing = pytest.importorskip(
        "aiter.ops.triton.moe.moe_routing.routing"
    ).routing

    torch.manual_seed(0)
    top_k, num_experts, num_tokens = MIMO_TOP_K, MIMO_EXPERTS, 16
    logits = torch.randn(num_tokens, num_experts, device="cuda") * 2.0
    bias = torch.randn(num_experts, device="cuda") * 0.5

    # Reference: select on the biased score, return the unbiased score, renorm.
    scores = torch.sigmoid(logits.float())
    ref_ids = (scores + bias).topk(top_k, dim=-1).indices
    ref_weights = scores.gather(1, ref_ids)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)

    def routed(score_mode: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        routing_data, topk_indx, _ = aiter_routing(
            logits,
            top_k,
            score_mode=score_mode,
            bias=bias if score_mode is not None else None,
            renorm=True,
            routed_scaling_factor=1.0,
            use_grouped_topk=score_mode is not None,
            num_expert_group=2,
            topk_group=2,
        )
        # `topk_indx[i]` is the token-slot index (token * top_k + slot) of the
        # i-th entry of the expert-sorted array whose weights are gate_scal.
        hist = routing_data.expt_hist.to(torch.long).cpu()
        expert_of_slot = torch.repeat_interleave(torch.arange(hist.numel()), hist)
        slot = topk_indx.reshape(-1).to(torch.long).cpu()
        weights = routing_data.gate_scal.reshape(-1).to(torch.float32).cpu()
        ids = torch.full((num_tokens, top_k), -1, dtype=torch.long)
        out = torch.zeros(num_tokens, top_k)
        for i in range(slot.numel()):
            token, position = int(slot[i]) // top_k, int(slot[i]) % top_k
            ids[token, position] = int(expert_of_slot[i])
            out[token, position] = float(weights[i])
        return ids, out

    got_ids, got_weights = routed("sigmoid")
    assert torch.equal(
        torch.sort(got_ids, dim=-1).values, torch.sort(ref_ids.cpu(), dim=-1).values
    )
    assert torch.allclose(got_weights, ref_weights.cpu(), atol=1e-3)

    # Control: falling back to the softmax path (score_mode=None) is NOT
    # equivalent, which is why the allow-list alone cannot be relaxed.
    _, softmax_weights = routed(None)
    assert not torch.allclose(softmax_weights, ref_weights.cpu(), atol=1e-3)
