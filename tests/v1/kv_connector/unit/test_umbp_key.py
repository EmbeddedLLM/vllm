# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.key import UMBPKeySpace


def identity():
    return UMBPKeySpace("deployment-a", "model", "commit", "canonical-layout", "sha256")


@pytest.mark.parametrize(
    "field", ["deployment", "model", "revision", "layout", "hash_algorithm"]
)
def test_incompatible_kv_cannot_reuse_a_key(field):
    first = identity()
    second = replace(first, **{field: "different"})
    assert first.block_key(b"hash", group=0, shard=0) != second.block_key(
        b"hash", group=0, shard=0
    )


def test_pd_and_offload_share_keys_only_for_the_same_group_shard_and_hash():
    producer, consumer = identity(), identity()
    key = producer.block_key(b"prefix", group=0, shard=0)
    assert key == consumer.block_key(b"prefix", group=0, shard=0)
    assert (
        len(
            {
                key,
                consumer.block_key(b"salted", group=0, shard=0),
                consumer.block_key(b"prefix", group=1, shard=0),
                consumer.block_key(b"prefix", group=0, shard=1),
            }
        )
        == 4
    )


def test_field_boundaries_do_not_alias():
    first = replace(identity(), deployment="a:b", model="c")
    second = replace(identity(), deployment="a", model="b:c")
    assert first.prefix != second.prefix


@pytest.mark.parametrize(
    "field", ["deployment", "model", "revision", "layout", "hash_algorithm"]
)
def test_missing_identity_is_rejected(field):
    with pytest.raises(ValueError, match="nonempty"):
        replace(identity(), **{field: " "})


@pytest.mark.parametrize(
    "block_hash,group,shard",
    [(b"", 0, 0), ("text", 0, 0), (b"hash", -1, 0), (b"hash", 0, True)],
)
def test_invalid_hash_or_shard_is_not_serialized(block_hash, group, shard):
    with pytest.raises(ValueError):
        identity().block_key(block_hash, group=group, shard=shard)
