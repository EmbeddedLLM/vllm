import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return None


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _calls_in_function(method: ast.FunctionDef) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            calls.append((node.lineno, func.id))
        elif isinstance(func, ast.Attribute):
            calls.append((node.lineno, func.attr))
    return sorted(calls)


def test_vllm_runtime_does_not_import_atom_package():
    offenders: list[str] = []
    for path in (REPO_ROOT / "vllm").rglob("*.py"):
        modules = _imported_modules(path)
        if any(module == "atom" or module.startswith("atom.") for module in modules):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_installed_aiter_does_not_export_atom_sparse_contract():
    aiter = pytest.importorskip("aiter")
    exported = set(dir(aiter))
    atom_sparse_contract = {
        "fused_compress_attn",
        "sparse_attn_v4_paged_decode",
        "sparse_attn_v4_paged_prefill",
        "swa_write",
        "update_compressor_states",
    }

    assert atom_sparse_contract.isdisjoint(exported)


def test_atom_native_abi_probe_reports_installed_aiter_gap():
    from vllm.models.deepseek_v4.amd.atom_native_abi import (
        probe_atom_native_abi,
        require_atom_native_abi,
    )

    status = probe_atom_native_abi()
    audit = (
        REPO_ROOT / "docs" / "deepseek_v4_atom_op_surface_audit.md"
    ).read_text()

    assert status.aiter_available
    assert not status.has_full_native_packed_main_path
    assert not status.packed_fp8_ds_mla_compressor
    assert not status.packed_fp8_ds_mla_attention
    assert not status.mhc_fused_post_pre
    assert not status.maybe_dual_stream_forward
    assert {
        "packed_fp8_ds_mla_compressor",
        "packed_fp8_ds_mla_attention",
        "mhc_fused_post_pre",
        "maybe_dual_stream_forward",
    } <= set(status.missing)
    assert "aiter.ops.flydsl.kernels.fused_compress_attn" in status.checked_modules
    assert "aiter.ops.pa_sparse_prefill_opus" in status.checked_modules
    assert "aiter.ops.triton.attention.pa_mqa_logits" in status.checked_modules
    assert "probe_atom_native_abi()" in audit
    assert "packed_fp8_ds_mla_compressor=False" in audit
    assert "packed_fp8_ds_mla_attention=False" in audit
    assert "packed_fp8_ds_mla_compressor" in status.missing_summary()
    assert "aiter.ops.pa_sparse_prefill_opus" in status.checked_modules_summary()
    with pytest.raises(RuntimeError, match="packed DeepSeek-V4 ATOM"):
        require_atom_native_abi()


def test_require_native_atom_abi_guard_is_registered_and_fail_fast(monkeypatch):
    envs_source = (REPO_ROOT / "vllm" / "envs.py").read_text()
    model_state_source = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "model_state.py"
    ).read_text()

    assert "VLLM_ROCM_DSV4_REQUIRE_NATIVE_ATOM_ABI" in envs_source
    assert "VLLM_ROCM_DSV4_REQUIRE_NATIVE_ATOM_ABI" in model_state_source
    assert "_check_required_native_atom_abi()" in model_state_source
    assert "require_atom_native_abi" in model_state_source

    from vllm.models.deepseek_v4.amd import model_state

    monkeypatch.setattr(model_state, "_REQUIRE_NATIVE_ATOM_ABI", True)
    with pytest.raises(RuntimeError, match="packed DeepSeek-V4 ATOM"):
        model_state._check_required_native_atom_abi()

    monkeypatch.setattr(model_state, "_REQUIRE_NATIVE_ATOM_ABI", False)
    model_state._check_required_native_atom_abi()


def test_installed_aiter_mhc_contract_exposes_pre_post_not_fused_post_pre():
    aiter = pytest.importorskip("aiter")

    assert hasattr(aiter, "mhc_pre")
    assert hasattr(aiter, "mhc_post")
    assert not hasattr(aiter, "mhc_fused_post_pre")

    mhc = pytest.importorskip("aiter.ops.mhc")
    assert hasattr(mhc, "mhc_pre")
    assert hasattr(mhc, "mhc_post")
    assert not hasattr(mhc, "mhc_fused_post_pre")


def test_rocm_mhc_fused_post_pre_is_not_aiter_backed_until_aiter_exports_it():
    mhc_path = REPO_ROOT / "vllm" / "model_executor" / "layers" / "mhc.py"
    tree = ast.parse(mhc_path.read_text(), filename=str(mhc_path))
    fused_op = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MHCFusedPostPreOp"
    )
    forward_hip = next(
        node
        for node in fused_op.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_hip"
    )
    source = ast.unparse(forward_hip)

    assert "mhc_fused_post_pre_tilelang" in source
    assert "mhc_fused_post_pre_aiter" not in source
    assert "rocm_aiter_ops" not in source


def test_atom_split_kv_decode_always_passes_ordered_metadata():
    rocm_path = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "rocm.py"
    )
    tree = ast.parse(rocm_path.read_text(), filename=str(rocm_path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sparse_attn_v4_paged_decode_split_kv"
    ]

    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}

    csa_positions = ast.unparse(keywords["csa_positions"])
    csa_window_size = ast.unparse(keywords["csa_window_size"])

    assert csa_positions == "positions"
    assert csa_window_size == "self.window_size"


def test_atom_attention_packed_deployment_routes_through_split_kv_branches():
    rocm_path = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "rocm.py"
    )
    tree = ast.parse(rocm_path.read_text(), filename=str(rocm_path))

    split_decode_gate = ast.unparse(
        _function_def(tree, "_should_use_atom_split_kv_decode")
    )
    assert "if unified_kv is None:\n        return True" in split_decode_gate

    decode_atom = _function_def(tree, "_maybe_forward_decode_atom")
    run_paged_decode = [
        node
        for node in decode_atom.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_paged_decode"
    ]
    assert len(run_paged_decode) == 1
    run_decode_source = ast.unparse(run_paged_decode[0])
    split_decode = run_decode_source.find(
        "if use_split_kv_decode and unified_kv_scales is None:"
    )
    homogeneous_decode_guard = run_decode_source.find("if unified_kv is None:")
    homogeneous_decode = run_decode_source.find("sparse_attn_v4_paged_decode(")
    assert -1 not in (split_decode, homogeneous_decode_guard, homogeneous_decode)
    assert split_decode < homogeneous_decode_guard < homogeneous_decode
    assert "compressed_kv_layout=split_kv_layout" in run_decode_source

    prefill_atom = _function_def(tree, "_maybe_forward_prefill_atom")
    prefill_source = ast.unparse(prefill_atom)
    split_prefill = prefill_source.find("if use_split_kv_prefill and unified_kv is None:")
    homogeneous_prefill = prefill_source.find("sparse_attn_v4_paged_prefill(")
    assert -1 not in (split_prefill, homogeneous_prefill)
    assert split_prefill < homogeneous_prefill
    assert "compressed_kv_layout=split_kv_layout" in prefill_source


def test_deepseek_v4_atom_model_state_does_not_use_workspace_manager():
    model_state_path = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "model_state.py"
    )
    tree = ast.parse(model_state_path.read_text(), filename=str(model_state_path))
    modules = _imported_modules(model_state_path)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attr_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "vllm.v1.worker.workspace" not in modules
    assert "current_workspace_manager" not in calls
    assert "get_simultaneous" not in attr_calls


def test_dsv4_rocm_kv_workspace_plan_documents_current_ownership_split():
    doc = (
        REPO_ROOT / "docs" / "deepseek_v4_rocm_kv_workspace_plan.md"
    ).read_text()
    workspace_source = (REPO_ROOT / "vllm" / "v1" / "worker" / "workspace.py").read_text()
    kv_interface_source = (
        REPO_ROOT / "vllm" / "v1" / "kv_cache_interface.py"
    ).read_text()
    kv_utils_source = (
        REPO_ROOT / "vllm" / "v1" / "core" / "kv_cache_utils.py"
    ).read_text()

    required_doc_phrases = [
        "vLLM's scheduler owns request-to-block allocation",
        "fixed_prefix_size_bytes",
        "requires_strided_kv_cache_view",
        "inner_block_stride_bytes",
        "DeepseekV4AtomMLAAttentionSpec.atom_swa_prefix_bytes",
        "compressed tail pages shaped as `uint8 [num_blocks, k_per_block, 584]`",
        "layout string `fp8_ds_mla`",
        "no sidecar scale tensor",
        "State that belongs in ModelState",
        "is a scratch allocator",
        "only until the next `get_simultaneous(...)` call",
        "State that does not belong in WorkspaceManager",
        "No GPU worker changes are required for persistent request state",
        "CUDA should stay untouched",
        "Choice A: Keep vLLM Split Packed Layout",
        "Choice B: Expose A Homogeneous Native View",
        "full native ATOM kernel benefit",
        "not yet available",
    ]
    missing = [phrase for phrase in required_doc_phrases if phrase not in doc]
    assert missing == []

    assert "class WorkspaceManager" in workspace_source
    assert "def get_simultaneous" in workspace_source
    assert "dbo_current_ubatch_id()" in workspace_source
    assert "fixed_prefix_size_bytes" in kv_interface_source
    assert "inner_block_stride_bytes" in kv_interface_source
    assert "DeepseekV4AtomMLAAttentionSpec" in kv_interface_source
    assert "_get_kv_cache_config_deepseek_v4" in kv_utils_source
    assert "fixed_prefix_size=spec.atom_swa_prefix_bytes" in kv_utils_source


def test_generic_worker_reshape_uses_kv_cache_spec_contract_not_dsv4_type():
    worker_paths = [
        REPO_ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py",
        REPO_ROOT / "vllm" / "v1" / "worker" / "gpu" / "attn_utils.py",
        REPO_ROOT / "vllm" / "v1" / "worker" / "utils.py",
    ]

    for path in worker_paths:
        source = path.read_text()
        assert "DeepseekV4AtomMLAAttentionSpec" not in source
        assert "fixed_prefix_size_bytes" in source
        if path.name != "utils.py":
            assert "requires_strided_kv_cache_view" in source
            assert "inner_block_stride_bytes" in source


def test_generic_worker_code_does_not_import_deepseek_v4_model_modules():
    worker_paths = [
        REPO_ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py",
        REPO_ROOT / "vllm" / "v1" / "worker" / "gpu" / "attn_utils.py",
        REPO_ROOT / "vllm" / "v1" / "worker" / "utils.py",
        REPO_ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py",
    ]
    offenders: list[str] = []

    for path in worker_paths:
        modules = _imported_modules(path)
        for module in sorted(modules):
            if module == "vllm.models.deepseek_v4" or module.startswith(
                "vllm.models.deepseek_v4."
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{module}")

    assert offenders == []


def test_nvidia_deepseek_v4_path_does_not_import_rocm_atom_modules():
    nvidia_root = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "nvidia"
    forbidden_modules = {
        "vllm.models.deepseek_v4.amd.model_state",
        "vllm.models.deepseek_v4.amd.v4_kernels",
    }
    forbidden_names = {
        "DeepseekV4AtomMLAAttentionSpec",
        "DeepseekV4RocmAtomModelState",
    }
    offenders: list[str] = []

    for path in nvidia_root.rglob("*.py"):
        source = path.read_text()
        modules = _imported_modules(path)
        for module in sorted(modules):
            if module in forbidden_modules or any(
                module.startswith(f"{forbidden}.") for forbidden in forbidden_modules
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{module}")
        for name in sorted(forbidden_names):
            if name in source:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{name}")

    assert offenders == []


def test_atom_kv_spec_uses_generic_full_attention_manager():
    manager_path = REPO_ROOT / "vllm" / "v1" / "core" / "single_type_kv_cache_manager.py"
    source = manager_path.read_text()

    assert "DeepseekV4AtomMLAAttentionSpec" in source
    assert "DeepseekV4AtomMLAAttentionSpec,\n        FullAttentionManager" in source
    assert (
        "DeepseekV4AtomMLAAttentionSpec,\n"
        "        FullAttentionManager,\n"
        "        uniform_type_base_spec=FullAttentionSpec"
    ) in source


def test_core_prefix_sizing_uses_kv_cache_spec_contract_not_dsv4_type():
    core_path = REPO_ROOT / "vllm" / "v1" / "core" / "kv_cache_utils.py"
    tree = ast.parse(core_path.read_text(), filename=str(core_path))

    representative = _function_def(tree, "_representative_scheduler_spec")
    scalable_blocks = _function_def(tree, "_scalable_blocks_per_request")
    uniform_specs = _function_def(tree, "is_kv_cache_spec_uniform")

    for method in (representative, scalable_blocks, uniform_specs):
        source = ast.unparse(method)
        assert "DeepseekV4AtomMLAAttentionSpec" not in source
        assert "fixed_prefix_size_bytes" in source


def test_packed_atom_kv_spec_is_split_prefix_plus_compressed_tail_contract():
    kv_interface = (REPO_ROOT / "vllm" / "v1" / "kv_cache_interface.py").read_text()
    attention = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "attention.py"
    ).read_text()
    model_state = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "model_state.py"
    ).read_text()

    assert "class DeepseekV4AtomMLAAttentionSpec" in kv_interface
    assert "return self.atom_swa_prefix_bytes" in kv_interface
    assert "448 FP8 NoPE bytes" in kv_interface
    assert "64 BF16 RoPE values" in kv_interface
    assert "8 embedded UE8M0 scale bytes per compressed token" in kv_interface

    assert 'self.atom_vllm_compressed_layout = "fp8_ds_mla"' in attention
    assert 'spec_dtype = torch.uint8' in attention
    assert 'spec_cache_dtype = "fp8_ds_mla"' in attention
    assert '"atom_compressed_layout": self.atom_vllm_compressed_layout' in attention

    assert 'compressed_kv_layout[layer_id] = "fp8_ds_mla"' in model_state
    assert 'setattr(attn, "atom_split_kv_layout", "fp8_ds_mla")' in model_state
    assert 'if hasattr(attn, "atom_unified_kv"):' in model_state
    assert "delattr(attn, \"atom_unified_kv\")" in model_state


def test_atom_main_compressor_preserves_read_before_update_ordering():
    compressor_path = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "compressor.py"
    tree = ast.parse(compressor_path.read_text(), filename=str(compressor_path))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_maybe_atom_main_compressor_forward"
    ]
    assert len(methods) == 1

    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(methods[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"fused_compress_attn", "update_compressor_states"}
    ]
    ordered_call_names = [name for _, name in sorted(calls)]

    assert ordered_call_names == ["fused_compress_attn", "update_compressor_states"]


def test_rocm_attention_runtime_uses_atom_attention_ops():
    rocm_path = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "rocm.py"
    tree = ast.parse(rocm_path.read_text(), filename=str(rocm_path))
    imported = _imported_modules(rocm_path)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "vllm.models.deepseek_v4.amd.v4_kernels" in imported
    assert (
        "vllm.models.deepseek_v4.amd.v4_kernels.qk_norm_rope_maybe_quant"
        in imported
    )
    assert {
        "qk_norm_rope_maybe_quant",
        "swa_write",
        "csa_translate_pack",
        "sparse_attn_v4_paged_decode",
        "sparse_attn_v4_paged_decode_split_kv",
        "sparse_attn_v4_paged_prefill",
        "sparse_attn_v4_paged_prefill_split_kv",
    } <= calls


def test_rocm_atom_decode_sequence_matches_atom_modeling_order():
    rocm_path = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "rocm.py"
    tree = ast.parse(rocm_path.read_text(), filename=str(rocm_path))

    qk_method = _function_def(tree, "_fused_qnorm_rope_kv_insert")
    qk_calls = _calls_in_function(qk_method)
    assert "qk_norm_rope_maybe_quant" in [name for _, name in qk_calls]

    forward_mqa = _function_def(tree, "forward_mqa")
    forward_calls = _calls_in_function(forward_mqa)
    pure_decode_swa_write = [
        lineno for lineno, name in forward_calls if name == "_maybe_atom_swa_write"
    ][0]
    decode_dispatch = [
        lineno for lineno, name in forward_calls if name == "_forward_decode"
    ][0]
    assert pure_decode_swa_write < decode_dispatch

    decode_atom = _function_def(tree, "_maybe_forward_decode_atom")
    decode_calls = _calls_in_function(decode_atom)
    call_lines: dict[str, list[int]] = {}
    for lineno, name in decode_calls:
        call_lines.setdefault(name, []).append(lineno)

    assert {"csa_translate_pack", "run_paged_decode"} <= set(call_lines)
    assert min(call_lines["csa_translate_pack"]) < min(call_lines["run_paged_decode"])

    nested = [
        node
        for node in decode_atom.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_paged_decode"
    ]
    assert len(nested) == 1
    nested_calls = {name for _, name in _calls_in_function(nested[0])}
    assert {
        "sparse_attn_v4_paged_decode",
        "sparse_attn_v4_paged_decode_split_kv",
    } <= nested_calls


def test_rocm_atom_prefill_sequence_matches_atom_modeling_order():
    rocm_path = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "rocm.py"
    tree = ast.parse(rocm_path.read_text(), filename=str(rocm_path))

    prefill_atom = _function_def(tree, "_maybe_forward_prefill_atom")
    call_lines: dict[str, list[int]] = {}
    for lineno, name in _calls_in_function(prefill_atom):
        call_lines.setdefault(name, []).append(lineno)

    assert {
        "write_v4_paged_prefill_indices",
        "csa_translate_pack",
        "sparse_attn_v4_paged_prefill",
        "sparse_attn_v4_paged_prefill_split_kv",
        "swa_write",
    } <= set(call_lines)

    index_write = min(call_lines["write_v4_paged_prefill_indices"])
    csa_pack = min(call_lines["csa_translate_pack"])
    first_sparse_prefill = min(
        min(call_lines["sparse_attn_v4_paged_prefill"]),
        min(call_lines["sparse_attn_v4_paged_prefill_split_kv"]),
    )
    post_attn_swa_write = min(call_lines["swa_write"])

    assert index_write < csa_pack < first_sparse_prefill < post_attn_swa_write


def test_rocm_compressor_runtime_uses_atom_compressor_ops():
    compressor_path = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "compressor.py"
    tree = ast.parse(compressor_path.read_text(), filename=str(compressor_path))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"fused_compress_attn", "update_compressor_states"} <= calls


def test_rocm_indexer_has_opt_in_atom_explicit_scale_sequence():
    attention_path = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "attention.py"
    envs_path = REPO_ROOT / "vllm" / "envs.py"
    tree = ast.parse(attention_path.read_text(), filename=str(attention_path))
    module_source = attention_path.read_text()
    envs_source = envs_path.read_text()
    sequence = _function_def(tree, "_maybe_atom_indexer_sequence")
    sequence_source = ast.unparse(sequence)
    indexer_cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4Indexer"
    )
    forward = next(
        node
        for node in indexer_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    forward_source = ast.unparse(forward)

    assert "VLLM_ROCM_DSV4_ATOM_INDEXER_SEQUENCE" in module_source
    assert "VLLM_ROCM_DSV4_ATOM_INDEXER_SEQUENCE" in envs_source
    assert "_ATOM_INDEXER_SEQUENCE_ENABLED" in sequence_source
    assert "rotary_embedding" in sequence_source
    assert "per_token_group_quant_fp8" in sequence_source
    assert "scale_indexer_weights" in sequence_source
    assert "_maybe_atom_indexer_sequence" in forward_source
    assert "fused_indexer_q_rope_quant" in forward_source


def test_rocm_indexer_has_opt_in_atom_score_topk_dispatch():
    attention_path = REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "attention.py"
    envs_path = REPO_ROOT / "vllm" / "envs.py"
    tree = ast.parse(attention_path.read_text(), filename=str(attention_path))
    module_source = attention_path.read_text()
    envs_source = envs_path.read_text()
    fallback = _function_def(tree, "_atom_indexer_score_topk")
    fallback_source = ast.unparse(fallback)
    indexer_method = _function_def(tree, "indexer_score_topk")
    indexer_method_source = ast.unparse(indexer_method)
    indexer_cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4Indexer"
    )
    forward = next(
        node
        for node in indexer_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    forward_source = ast.unparse(forward)

    assert "VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH" in module_source
    assert "VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH" in envs_source
    assert '_atom_torch_op_exists("aiter", "indexer_score_topk")' in module_source
    assert "target_lib=_ATOM_AITER_FALLBACK_LIB" in module_source
    assert "_ATOM_INDEXER_DISPATCH_REGISTRY[self.prefix] = self" in module_source
    assert "indexer.indexer_score_topk(q_fp8, weights, topk)" in fallback_source
    assert "_maybe_atom_decode_indexer_fastpath" in indexer_method_source
    assert "self.indexer_op(q_fp8, q_fp8, None, weights)" in indexer_method_source
    assert "torch.ops.aiter.indexer_score_topk" in forward_source


def test_full_atom_indexer_prefill_decode_dispatch_is_not_default_integrated():
    atom_model = REPO_ROOT.parent / "ATOM" / "atom" / "models" / "deepseek_v4.py"
    if not atom_model.exists():
        pytest.skip(f"ATOM model file is unavailable: {atom_model}")
    atom_source = atom_model.read_text()
    attention_source = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "attention.py"
    ).read_text()
    rocm_sparse_ops = (
        REPO_ROOT
        / "vllm"
        / "v1"
        / "attention"
        / "ops"
        / "rocm_aiter_mla_sparse.py"
    ).read_text()

    assert "def _score_topk_prefill(" in atom_source
    assert "cp_gather_indexer_k_quant_cache(" in atom_source
    assert "fp8_mqa_logits(" in atom_source
    assert "top_k_per_row_prefill(" in atom_source
    assert "def _score_topk_decode(" in atom_source
    assert "deepgemm_fp8_paged_mqa_logits(" in atom_source
    assert "top_k_per_row_decode(" in atom_source

    assert "cp_gather_indexer_k_quant_cache_triton(" in rocm_sparse_ops
    assert "rocm_fp8_mqa_logits(" in rocm_sparse_ops
    assert "rocm_fp8_paged_mqa_logits(" in rocm_sparse_ops
    assert "_top_k_per_row_prefill(" in rocm_sparse_ops
    assert "_top_k_per_row_decode(" in rocm_sparse_ops

    assert "_maybe_atom_decode_indexer_fastpath" in attention_source
    assert "rocm_fp8_paged_mqa_logits" in attention_source
    assert "_top_k_per_row_decode" in attention_source
    assert "cp_gather_indexer_k_quant_cache" not in attention_source
    assert "rocm_fp8_mqa_logits" not in attention_source
    assert "return self.indexer_op(hidden_states, q_quant, k, weights)" in (
        attention_source
    )


def test_atom_dual_stream_moe_and_aux_compressor_overlap_are_not_integrated():
    atom_model = REPO_ROOT.parent / "ATOM" / "atom" / "models" / "deepseek_v4.py"
    if not atom_model.exists():
        pytest.skip(f"ATOM model file is unavailable: {atom_model}")
    atom_source = atom_model.read_text()
    vllm_model_sources = "\n".join(
        path.read_text()
        for path in (REPO_ROOT / "vllm" / "models" / "deepseek_v4").rglob("*.py")
        if path.name != "atom_native_abi.py"
    )
    probe_source = (
        REPO_ROOT
        / "vllm"
        / "models"
        / "deepseek_v4"
        / "amd"
        / "atom_native_abi.py"
    ).read_text()
    amd_model = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "model.py"
    ).read_text()

    assert "torch.ops.aiter.maybe_dual_stream_forward" in atom_source
    assert "dual_stream_moe_forward" in atom_source
    assert "self.alt_stream: Optional[torch.cuda.Stream]" in atom_source
    assert "self.compress_stream: Optional[torch.cuda.Stream]" in atom_source
    assert "with torch.cuda.stream(self.alt_stream):" in atom_source
    assert "with torch.cuda.stream(self.compress_stream):" in atom_source

    assert "maybe_dual_stream_forward" not in vllm_model_sources
    assert "maybe_dual_stream_forward" in probe_source
    assert "torch.ops.aiter.maybe_dual_stream_forward" not in probe_source
    assert "aux_stream_list = (" in amd_model
    assert "None\n            if current_platform.is_rocm()" in amd_model


def test_packed_fp8_ds_mla_layout_is_shared_by_writer_and_readers():
    fused_compress = (
        REPO_ROOT
        / "vllm"
        / "models"
        / "deepseek_v4"
        / "amd"
        / "v4_kernels"
        / "fused_compress.py"
    ).read_text()
    paged_decode = (
        REPO_ROOT
        / "vllm"
        / "models"
        / "deepseek_v4"
        / "amd"
        / "v4_kernels"
        / "paged_decode.py"
    ).read_text()
    paged_prefill = (
        REPO_ROOT
        / "vllm"
        / "models"
        / "deepseek_v4"
        / "amd"
        / "v4_kernels"
        / "paged_prefill.py"
    ).read_text()

    assert "TOKEN_DATA_SIZE: tl.constexpr = 576" in fused_compress
    assert "TOKEN_SCALE_DIM: tl.constexpr = 8" in fused_compress
    assert "+ slot_in_block * TOKEN_DATA_SIZE" in fused_compress
    assert "+ k_per_block * TOKEN_DATA_SIZE" in fused_compress
    assert "+ slot_in_block * TOKEN_SCALE_DIM" in fused_compress
    assert "not in\n            # kv_cache[block, slot, 576:584]" in fused_compress

    for reader in (paged_decode, paged_prefill):
        assert "token_data_base = packed_base + tail_pos[:, None] * 576" in reader
        assert "packed_base + PACKED_BLOCK_SIZE * 576 + tail_pos[:, None] * 8" in reader
        assert "compressed_kv_ptr + token_data_base + 448" in reader

    assert "_PACKED_TOKEN_DATA_SIZE = 576" in paged_decode
    assert "_PACKED_SCALE_DIM = 8" in paged_decode
    assert "data_start = slot * _PACKED_TOKEN_DATA_SIZE" in paged_decode
    assert (
        "scale_start = block_size * _PACKED_TOKEN_DATA_SIZE + "
        "slot * _PACKED_SCALE_DIM"
    ) in paged_decode


def test_launch_defaults_select_rocm_atom_benchmark_path():
    launch_path = REPO_ROOT.parent / "launchdeepseekgraph.sh"
    launch = launch_path.read_text()

    assert "MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}" in launch
    assert "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}" in launch
    assert "ENFORCE_EAGER=${ENFORCE_EAGER:-0}" in launch
    assert "BLOCK_SIZE=${BLOCK_SIZE:-128}" in launch
    assert "VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}" in launch
    assert "VLLM_ROCM_DSV4_USE_AITER_MHC=${VLLM_ROCM_DSV4_USE_AITER_MHC:-0}" in launch
    assert (
        "VLLM_ROCM_DSV4_USE_AITER_HC_HEAD="
        "${VLLM_ROCM_DSV4_USE_AITER_HC_HEAD:-0}"
    ) in launch
    assert "VLLM_ROCM_DSV4_ATOM_STATE=${VLLM_ROCM_DSV4_ATOM_STATE:-1}" in launch
    assert (
        "VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM="
        "${VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM:-1}"
    ) in launch
    assert "VLLM_ROCM_DSV4_ATOM_MIXED_KV=${VLLM_ROCM_DSV4_ATOM_MIXED_KV:-0}" in launch
    assert "VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=${VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN:-1}" in launch
    assert (
        "VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR="
        "${VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR:-1}"
    ) in launch
    assert "VLLM_ROCM_DSV4_ATOM_ATTENTION=${VLLM_ROCM_DSV4_ATOM_ATTENTION:-1}" in launch
    assert "ATOM_USE_FUSED_Q_NORM_QUANT=${ATOM_USE_FUSED_Q_NORM_QUANT:-1}" in launch
    assert "--kv-cache-dtype fp8" in launch


def test_rocm_atom_mixed_decode_prefill_keeps_generic_sparse_metadata():
    model_state_path = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "model_state.py"
    )
    tree = ast.parse(model_state_path.read_text(), filename=str(model_state_path))
    source = model_state_path.read_text()

    assert "VLLM_ROCM_DSV4_ATOM_SKIP_MIXED_DECODE_METADATA" in source
    assert (
        'os.environ.get("VLLM_ROCM_DSV4_ATOM_SKIP_MIXED_DECODE_METADATA", "0") == "1"'
        in source
    )
    assert "_ATOM_PREFILL_ALLOW_MIXED" in source
    assert "_ATOM_SKIP_PAGED_PREFILL" in source

    skip_guard = ast.unparse(_function_def(tree, "_can_skip_generic_atom_decode_metadata"))
    assert "return True" in skip_guard
    assert "return False" in skip_guard
    assert "_ATOM_SKIP_MIXED_GENERIC_DECODE_METADATA" in skip_guard
    assert "_ATOM_PREFILL_ALLOW_MIXED" in skip_guard
    assert "_ATOM_SKIP_PAGED_PREFILL" in skip_guard
    assert "_atom_mixed_batch_is_decode_then_prefill(input_batch)" in skip_guard

    ordering_guard = ast.unparse(
        _function_def(tree, "_atom_mixed_batch_is_decode_then_prefill")
    )
    assert "np.all(decode_mask[:num_decodes])" in ordering_guard
    assert "not np.any(decode_mask[num_decodes:])" in ordering_guard

    prepare_attn = ast.unparse(_function_def(tree, "prepare_attn"))
    assert "_attach_minimal_atom_decode_metadata" in prepare_attn
    assert "skip_generic_atom_decode_metadata" in prepare_attn


def test_rocm_atom_mixed_decode_prefill_preserves_indexer_metadata_alias():
    model_state_path = (
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "model_state.py"
    )
    sparse_ops_path = (
        REPO_ROOT / "vllm" / "v1" / "attention" / "ops" / "rocm_aiter_mla_sparse.py"
    )
    tree = ast.parse(model_state_path.read_text(), filename=str(model_state_path))
    source = model_state_path.read_text()
    sparse_source = sparse_ops_path.read_text()

    indexer_guard = ast.unparse(_function_def(tree, "_can_skip_generic_indexer_metadata"))
    assert "_ATOM_SKIP_MIXED_GENERIC_DECODE_METADATA" not in indexer_guard
    assert "_atom_mixed_batch_is_decode_then_prefill" not in indexer_guard
    assert "return pure_decode_one_token" in indexer_guard

    attach_source = ast.unparse(_function_def(tree, "_attach_minimal_atom_decode_metadata"))
    assert "DeepseekV32IndexerMetadata" in attach_source
    assert "_ATOM_INDEXER_METADATA_ALIAS_SUFFIX" in attach_source
    assert "existing_metadata = attn_metadata.get(layer_name)" in attach_source
    assert "layer_name + _ATOM_INDEXER_METADATA_ALIAS_SUFFIX" in attach_source
    assert '".__rocm_atom_indexer_metadata"' in source

    assert '".__rocm_atom_indexer_metadata"' in sparse_source
    assert "indexer_metadata_alias = attn_metadata.get" in sparse_source
    assert "isinstance(indexer_metadata_alias, DeepseekV32IndexerMetadata)" in sparse_source


def test_deepseek_v4_atom_env_lookups_are_import_time_cached():
    hot_files = [
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "attention.py",
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "compressor.py",
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "sparse_mla.py",
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "rocm.py",
        REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "model_state.py",
        (
            REPO_ROOT
            / "vllm"
            / "models"
            / "deepseek_v4"
            / "amd"
            / "v4_kernels"
            / "paged_decode.py"
        ),
        (
            REPO_ROOT
            / "vllm"
            / "models"
            / "deepseek_v4"
            / "amd"
            / "v4_kernels"
            / "paged_prefill.py"
        ),
        (
            REPO_ROOT
            / "vllm"
            / "models"
            / "deepseek_v4"
            / "amd"
            / "v4_kernels"
            / "fused_compress.py"
        ),
        (
            REPO_ROOT
            / "vllm"
            / "models"
            / "deepseek_v4"
            / "amd"
            / "v4_kernels"
            / "state_writes.py"
        ),
    ]
    allowed_env_helpers = {"_env_int", "_env_float"}
    offenders: list[str] = []

    for path in hot_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            env_get = (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
            )
            helper_call = isinstance(func, ast.Name) and func.id in allowed_env_helpers
            if not env_get and not helper_call:
                continue

            enclosing = _enclosing_function(node, parents)
            if enclosing is None:
                continue
            if env_get and enclosing.name in allowed_env_helpers:
                continue
            offenders.append(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{enclosing.name}"
            )

    assert offenders == []


def test_rocm_atom_separate_inverse_rope_path_is_opt_in():
    rocm = (REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "amd" / "rocm.py").read_text()
    init = (
        REPO_ROOT
        / "vllm"
        / "models"
        / "deepseek_v4"
        / "amd"
        / "v4_kernels"
        / "__init__.py"
    ).read_text()
    inverse_rope = (
        REPO_ROOT
        / "vllm"
        / "models"
        / "deepseek_v4"
        / "amd"
        / "v4_kernels"
        / "inverse_rope.py"
    ).read_text()
    envs = (REPO_ROOT / "vllm" / "envs.py").read_text()

    assert "VLLM_ROCM_DSV4_ATOM_SEPARATE_INVERSE_ROPE" in envs
    assert "_ATOM_SEPARATE_INVERSE_ROPE" in rocm
    assert "if _ATOM_SEPARATE_INVERSE_ROPE:" in rocm
    assert "inverse_rope_inplace(" in rocm
    assert "rocm_inv_rope_einsum(" in rocm
    assert rocm.index("if _ATOM_SEPARATE_INVERSE_ROPE:") < rocm.index(
        "rocm_inv_rope_einsum("
    )
    assert "inverse_rope_inplace" in init
    assert "def inverse_rope_inplace" in inverse_rope
    assert "_inverse_rope_gptj_inplace_kernel" in inverse_rope
