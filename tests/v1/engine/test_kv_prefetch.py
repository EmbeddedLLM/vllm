# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("prompt", "expected_lookahead"),
    [
        (list(range(32)), 31),
        (list(range(33)), 32),
    ],
)
def test_gpu_prefetch_keeps_a_synthetic_lookahead(
    prompt: list[int], expected_lookahead: int
) -> None:
    engine_core = object.__new__(EngineCore)
    engine_core._prefetch_hash_block_size = 16
    engine_core.request_block_hasher = None
    engine_core.scheduler = Mock()
    engine_core.scheduler.has_hbm_prefetch.return_value = False

    result = engine_core.start_kv_prefetch(
        prefetch_id="test-prefetch",
        prompts=[prompt],
        target_tier="gpu",
    )

    request = engine_core.scheduler.add_hbm_prefetch.call_args.args[0]
    assert request.prompt_token_ids == prompt[:32] + [expected_lookahead]
    assert request.num_tokens == 33
    assert result["total_blocks"] == 2
