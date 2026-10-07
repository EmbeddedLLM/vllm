# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC model loader: maps post-quantized weights from a local weight cache
daemon via CUDA IPC instead of loading from disk."""

import dataclasses
import socket
import time
import types
from collections import deque
from collections.abc import Callable
from copy import copy
from typing import TypeVar

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    get_draft_load_config,
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    CacheConfigMismatchError,
    UnsupportedQuantForIPCError,
    WeightCacheKey,
    WeightCacheState,
    WeightCacheUnavailableError,
    check_ipc_platform_support,
    check_ipc_quant_support,
    get_current_device_uuid,
    get_socket_path,
    recv_msg,
    send_msg,
    verify_socket_owner,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    is_draft_model_cacheable,
)
from vllm.model_executor.utils import weights_already_processed
from vllm.tracing import instrument
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)

_CONNECT_TIMEOUT_S = 5.0
_STATE_TIMEOUT_S = 300.0
_STARTUP_RETRY_INTERVAL_S = 0.5

# Fail-open bound for the stale-reference rebind walk in _apply_entries: the
# model object graph is expected to stay far below this, so tripping it means
# the holder graph is bigger than anticipated and the walk logs a warning
# instead of running unbounded.
_REBIND_NODE_BOUND = 200_000
# Registration containers owned by nn.Module; already replaced correctly by
# _register, so the rebind walk must never rewrite them.
_MODULE_REGISTRATION_ATTRS = frozenset({"_parameters", "_buffers", "_modules"})

_T = TypeVar("_T")


class IpcModelLoader(BaseModelLoader):
    """Loads a model by mapping the weight cache daemon's tensors via CUDA IPC.

    The model is initialized on the meta device and every parameter/buffer is
    replaced by the daemon's post-quantized tensor, so
    process_weights_after_loading is skipped entirely. In "zero_copy" mode the
    engine shares the daemon's GPU memory; in "copy" mode the tensors are
    cloned into engine-owned memory and the daemon is asked to release its
    cache afterwards.

    Extra config keys (via --model-loader-extra-config):

    - socket_path: explicit daemon socket path. Defaults to a per-GPU path
      derived from the physical GPU uuid and the cache role (target/draft).
    - socket_dir: directory containing the daemon sockets.
    - mode: "zero_copy" (default) or "copy".
    - fallback: fall back to disk loading when the daemon is unavailable or
      the fingerprints mismatch (default: True).
    - connect_timeout_s: socket connect timeout (default: 5.0).
    - state_timeout_s: timeout for the weight-transfer request (default: 300.0).

    Note: in zero-copy mode the weights live in the daemon's CUDA IPC
    allocations, so sleep mode (CuMemAllocator weight offloading) must not be
    used with this loader.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = copy(load_config.model_loader_extra_config or {})
        self.socket_path: str | None = extra_config.pop("socket_path", None)
        self.socket_dir: str | None = extra_config.pop("socket_dir", None)
        # Internal: set by the engine when routing a speculative draft to the
        # daemon's draft group.
        self.is_draft = bool(extra_config.pop("is_draft", False))
        if self.is_draft and self.socket_path is not None:
            raise ValueError(
                "socket_path cannot be combined with the draft weight cache role; "
                "use socket_dir so the target and draft sockets are derived "
                "independently"
            )
        self.mode: str = extra_config.pop("mode", "zero_copy")
        self.fallback: bool = extra_config.pop("fallback", True)
        self.connect_timeout_s: float = float(
            extra_config.pop("connect_timeout_s", _CONNECT_TIMEOUT_S)
        )
        self.state_timeout_s: float = float(
            extra_config.pop("state_timeout_s", _STATE_TIMEOUT_S)
        )
        if self.mode not in ("zero_copy", "copy"):
            raise ValueError(
                f"Invalid weight cache mode {self.mode!r}, "
                "expected 'zero_copy' or 'copy'"
            )
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: {sorted(extra_config)}"
            )

    def get_external_weight_memory(self, vllm_config: VllmConfig) -> int:
        # Copy mode clones the weights into this process; nothing external.
        if self.mode != "zero_copy":
            return 0
        total = self._daemon_memory()
        if is_draft_model_cacheable(vllm_config.speculative_config):
            # The draft group is queried with the same config the engine
            # would load the draft with.
            draft_config = get_draft_load_config(vllm_config)
            if draft_config.load_format == "ipc_cache":
                draft_loader = IpcModelLoader(draft_config)
                # Only a zero-copy draft-group daemon holds external weights;
                # an explicit draft_load_config without is_draft would resolve
                # back to the target socket and double-count it.
                if draft_loader.is_draft and draft_loader.mode == "zero_copy":
                    total += draft_loader._daemon_memory()
        return total

    def _daemon_memory(self) -> int:
        if not self.fallback:
            # The loader waits for the daemon at load time, so the weights
            # will be zero-copy mapped for sure; mirror that wait here since
            # returning 0 would over-grant the memory budget.
            return self._with_startup_wait(self._query_daemon_memory)
        # An unreachable daemon means a disk load, i.e. nothing external.
        try:
            return self._query_daemon_memory()
        except (WeightCacheUnavailableError, ConnectionError, OSError) as e:
            logger.warning(
                "Cannot query weight cache daemon memory (%s); "
                "assuming the weights are not externally held",
                e,
            )
            return 0

    def _query_daemon_memory(self) -> int:
        with self._connect(self.connect_timeout_s) as conn:
            send_msg(conn, {"cmd": "get_memory"})
            response = recv_msg(conn)
        if response.get("status") != "ok":
            raise WeightCacheUnavailableError(
                "Weight cache daemon rejected the memory query: "
                f"{response.get('message')}"
            )
        return int(response.get("memory_bytes", 0))

    def download_model(self, model_config: ModelConfig) -> None:
        DefaultModelLoader(self._fallback_load_config()).download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Best-effort in-place reload for an already-initialized model.

        Copies daemon tensors into matching parameters/buffers. The model is
        expected to already be in the post-quantized layout (e.g. previously
        loaded through this loader).
        """
        device_index = torch.accelerator.current_device_index()
        entries = self._fetch_entries(model_config).entries
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        for name, entry in entries.items():
            target = params.get(name, buffers.get(name))
            source = entry.rebuild(device_index)
            if target is None or target.shape != source.shape:
                logger.warning("Skipping mismatched cached tensor %s", name)
                continue
            target.data.copy_(source)

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        # An unsupported platform is a permanent misconfiguration rather than
        # a transient daemon outage, so it is raised even when fallback is on.
        check_ipc_platform_support()
        state_fetched = False
        try:
            # Cross-check the routing flag against the identity of the model
            # being loaded: a draft load that lost its flag (or a target load
            # that got one) would hit the wrong daemon group and
            # fingerprint-mismatch.
            spec = vllm_config.speculative_config
            inferred = spec is not None and model_config is spec.draft_model_config
            if inferred != self.is_draft:
                raise CacheConfigMismatchError(
                    f"Weight cache role mismatch: loading "
                    f"{'draft' if inferred else 'target'} model but the loader "
                    f"was configured for the "
                    f"{'draft' if self.is_draft else 'target'} group"
                )
            state = self._fetch_entries(model_config)
            state_fetched = True
            return self._build_model(vllm_config, model_config, prefix, state)
        except (WeightCacheUnavailableError, CacheConfigMismatchError) as e:
            if not self.fallback:
                raise
            logger.warning(
                "Weight cache unusable (%s); falling back to disk loading", e
            )
        except UnsupportedQuantForIPCError:
            # Unsupported quantization is a permanent misconfiguration rather
            # than a transient daemon outage, so it is raised even when
            # fallback is on.
            raise
        except Exception:
            if not self.fallback:
                raise
            logger.exception(
                "Weight cache IPC loading failed; falling back to disk loading"
            )
            # _build_model failed after fetching state without reaching its
            # copy-mode release, so the daemon still holds the full cache;
            # release it so the disk fallback does not OOM against it.
            if state_fetched and self.mode == "copy":
                self._send_release()
            torch.accelerator.empty_cache()
        return self._fallback_load(vllm_config, model_config, prefix)

    def _build_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str,
        state: WeightCacheState,
    ) -> nn.Module:
        device_config = vllm_config.device_config
        load_device = (
            device_config.device
            if self.load_config.device is None
            else self.load_config.device
        )
        target_device = torch.device(load_device)
        device_index = (
            target_device.index
            if target_device.index is not None
            else torch.accelerator.current_device_index()
        )
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            check_ipc_quant_support(model)
            self._apply_entries(model, state, device_index)
            # Flags that load_weights would have set (e.g. EAGLE ownership of
            # embed_tokens / lm_head); the daemon ran it, this process did not.
            for name, value in state.attrs.items():
                setattr(model, name, value)
            # The daemon exports tensors that already went through
            # process_weights_after_loading; re-run it in pre-processed mode
            # so quant methods only rebuild Python-side state (e.g. the MoE
            # kernel). Leftovers are materialized afterwards so that
            # placeholders the daemon-side post-processing consumed are
            # dropped rather than filled with uninitialized memory.
            with weights_already_processed():
                process_weights_after_loading(model, model_config, target_device)
            _materialize_remaining_meta_tensors(
                model, torch.device(target_device.type, device_index)
            )
        if self.mode == "copy":
            self._send_release()
        logger.info(
            "Mapped %d tensors from the weight cache daemon (%s mode)",
            len(state.entries),
            self.mode,
        )
        return model.eval()

    def _apply_entries(
        self,
        model: nn.Module,
        state: WeightCacheState,
        device_index: int,
    ) -> None:
        # remove_duplicate=False keeps tied module aliases reachable by name:
        # a tied lm_head *is* the embedding module, so the deduplicated view
        # would not contain "lm_head" at all.
        modules = dict(model.named_modules(remove_duplicate=False))
        registered: dict[str, torch.Tensor] = {}
        # id(old registration object) -> object registered in its place.
        # Non-module holders that captured a tensor by value at model
        # construction (e.g. GroupedTopKRouter.e_score_correction_bias) keep
        # the popped object; the rebind sweep below fixes them up afterwards.
        stale_map: dict[int, torch.Tensor] = {}
        # Pinned old objects so their ids cannot be recycled before the sweep.
        retired: list[torch.Tensor] = []

        def _register(name: str, tensor: torch.Tensor, is_param: bool) -> None:
            module_name, _, leaf = name.rpartition(".")
            module = modules.get(module_name)
            if module is None:
                raise RuntimeError(f"Cached tensor {name} has no matching module")
            # Capture the old object so by-value holders of it can be
            # rebound once registration is done.
            old = module._parameters.get(leaf)
            if old is None:
                old = module._buffers.get(leaf)
            # Replace via registration rather than param.data assignment,
            # which fails for meta tensors. Entries may also introduce
            # post-quantization tensors absent from the meta model.
            module._parameters.pop(leaf, None)
            module._buffers.pop(leaf, None)
            if is_param:
                obj: torch.Tensor = (
                    tensor
                    if isinstance(tensor, nn.Parameter)
                    else nn.Parameter(tensor, requires_grad=False)
                )
                module.register_parameter(leaf, obj)
            else:
                obj = tensor
                module.register_buffer(leaf, obj)
            registered[name] = obj
            if isinstance(old, torch.Tensor) and old is not obj:
                old_id = id(old)
                prior = stale_map.get(old_id)
                if prior is None:
                    stale_map[old_id] = obj
                    retired.append(old)
                elif prior is not obj:
                    logger.warning(
                        "Cached tensor %s remaps stale object %#x that was "
                        "already rebound to a different object",
                        name,
                        old_id,
                    )

        for name, entry in state.entries.items():
            tensor = entry.rebuild(device_index)
            if self.mode == "copy":
                tensor = tensor.clone()
            _register(name, tensor, entry.kind == "param")

        # Re-establish tied-weight aliases by registering the *same* object the
        # canonical name resolved to, so parameter identity (and the tie) is
        # preserved instead of allocating uninitialized memory.
        for alias_name, canonical_name in state.aliases.items():
            obj = registered.get(canonical_name)
            if obj is None:
                logger.warning(
                    "Cached alias %s references missing canonical tensor %s",
                    alias_name,
                    canonical_name,
                )
                continue
            _register(alias_name, obj, isinstance(obj, nn.Parameter))

        # One rebind sweep for by-value holders the registration swap cannot
        # reach (plain attributes of non-module objects, e.g. the MoE router's
        # e_score_correction_bias which stays meta and crashes profiling).
        if stale_map:
            holders = _rebind_stale_references(modules, stale_map)
            logger.info(
                "ipc_cache: rebound %d stale plain-attribute tensor references",
                len(holders),
            )

    def _fetch_entries(self, model_config: ModelConfig) -> WeightCacheState:
        dp_group = get_dp_group()
        pp_group = get_pp_group()
        cache_config = WeightCacheKey.from_model_config(
            model_config,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
            pp_size=pp_group.world_size,
            pp_rank=pp_group.rank_in_group,
            dp_size=dp_group.world_size,
            dp_rank=dp_group.rank_in_group,
            is_draft=self.is_draft,
        )
        if not self.fallback:
            return self._request_state_with_startup_wait(cache_config)
        return self._request_state(cache_config)

    def _request_state_with_startup_wait(
        self, cache_config: WeightCacheKey
    ) -> WeightCacheState:
        return self._with_startup_wait(lambda: self._request_state(cache_config))

    def _with_startup_wait(self, op: Callable[[], _T]) -> _T:
        """Retry op until the daemon answers or the state timeout elapses;
        the daemon may still be loading the model when the engine starts."""
        deadline = time.monotonic() + self.state_timeout_s
        while True:
            try:
                return op()
            except (WeightCacheUnavailableError, ConnectionError, OSError) as e:
                if time.monotonic() >= deadline:
                    raise WeightCacheUnavailableError(
                        "Weight cache daemon did not become ready within "
                        f"{self.state_timeout_s:.1f}s: {e}"
                    ) from e
                logger.info_once(
                    "Waiting up to %.1fs for the weight cache daemon to start",
                    self.state_timeout_s,
                )
                time.sleep(
                    max(
                        0.0,
                        min(_STARTUP_RETRY_INTERVAL_S, deadline - time.monotonic()),
                    )
                )

    def _request_state(self, cache_config: WeightCacheKey) -> WeightCacheState:
        with self._connect(self.state_timeout_s) as conn:
            send_msg(conn, {"cmd": "get_state", "cache_config": cache_config})
            response = recv_msg(conn)
        status = response.get("status")
        if status == "mismatch":
            raise CacheConfigMismatchError(
                f"WeightCacheKey mismatch on fields: {response.get('fields')}"
            )
        if status != "ok":
            raise WeightCacheUnavailableError(
                f"Weight cache daemon error: {response.get('message')}"
            )
        self._check_gpu_uuid(response.get("gpu_uuid"))
        return WeightCacheState(
            entries=response["entries"],
            aliases=response.get("aliases", {}),
            attrs=response.get("attrs", {}),
        )

    def _connect(self, timeout: float) -> socket.socket:
        socket_path = self._resolve_socket_path()
        # The auto-derived per-user directory is locked to 0700 and checked
        # strictly. When the operator explicitly configures a path they own the
        # trust decision, so only ownership/symlink safety is enforced.
        strict_perms = self.socket_path is None and self.socket_dir is None
        try:
            verify_socket_owner(socket_path, strict_perms=strict_perms)
        except OSError as e:
            raise WeightCacheUnavailableError(
                f"Weight cache socket {socket_path} is unavailable: {e}"
            ) from e
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(socket_path)
        except OSError as e:
            sock.close()
            raise WeightCacheUnavailableError(
                f"Cannot connect to weight cache daemon at {socket_path}: {e}"
            ) from e
        return sock

    def _resolve_socket_path(self) -> str:
        if self.socket_path is not None:
            return self.socket_path
        return get_socket_path(
            get_current_device_uuid(),
            self.socket_dir,
            is_draft=self.is_draft,
        )

    def _check_gpu_uuid(self, daemon_uuid: str | None) -> None:
        if daemon_uuid is None:
            return
        local_uuid = get_current_device_uuid()
        if daemon_uuid != local_uuid:
            raise CacheConfigMismatchError(
                f"Daemon GPU {daemon_uuid} != engine GPU {local_uuid}; "
                "check the socket path / GPU mapping"
            )

    def _send_release(self) -> None:
        try:
            with self._connect(self.connect_timeout_s) as conn:
                send_msg(conn, {"cmd": "release"})
                recv_msg(conn)
        except (WeightCacheUnavailableError, ConnectionError, OSError):
            logger.warning("Failed to ask the weight cache daemon to release")

    def _fallback_load_config(self) -> LoadConfig:
        # DefaultModelLoader must not see load_format="ipc_cache" or the ipc
        # extra config keys.
        return dataclasses.replace(
            self.load_config,
            load_format="auto",
            model_loader_extra_config={},
        )

    def _fallback_load(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str
    ) -> nn.Module:
        loader = DefaultModelLoader(self._fallback_load_config())
        return loader.load_model(
            vllm_config=vllm_config, model_config=model_config, prefix=prefix
        )


def _may_scan(value: object) -> bool:
    """Whether the stale-reference walk may descend into value.__dict__.

    Allows plain instances with a real instance __dict__; skips primitives,
    functions/types/modules, and torch library objects (their state is not
    model-tensor holders and scanning it is noise at best).
    """
    if isinstance(
        value,
        (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            type,
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.CodeType,
            types.ModuleType,
        ),
    ):
        return False
    cls_module = type(value).__module__ or ""
    if cls_module == "torch" or cls_module.startswith("torch."):
        return False
    return isinstance(getattr(value, "__dict__", None), dict)


def _rebind_stale_references(
    modules: dict[str, nn.Module],
    stale_map: dict[int, torch.Tensor],
) -> list[str]:
    """Rebind by-value tensor references that registration could not reach.

    ``IpcModelLoader._apply_entries`` replaces registered parameters/buffers
    with NEW objects, so any holder that captured the old tensor by value into
    a plain attribute (e.g. ``GroupedTopKRouter.e_score_correction_bias``, a
    non-module object built at model construction) keeps the stale object.

    BFS from every module in ``modules``: scan each owner's plain attributes
    (never ``_parameters``/``_buffers``/``_modules`` — registration already
    fixed those), rebind ``stale_map`` hits in place, descend into non-module,
    non-tensor instances, and rebind stale tensors at 1 level inside
    list/dict/tuple attribute values. Returns the rebound holder paths; the
    walk is bounded and fails open (warns and stops early) if the bound trips;
    with nothing stale it is a pure no-op.
    """
    if not stale_map:
        return []

    def _rebound(value: object) -> object | None:
        if isinstance(value, torch.Tensor):
            new = stale_map.get(id(value))
            if new is not None and new is not value:
                return new
        return None

    visited: set[int] = set()
    queue: deque[tuple[object, str]] = deque()
    for module_path, module in modules.items():
        if id(module) not in visited:
            visited.add(id(module))
            queue.append((module, module_path))

    holders: list[str] = []
    nodes = 0
    while queue:
        nodes += 1
        if nodes > _REBIND_NODE_BOUND:
            logger.warning(
                "ipc_cache: stale-reference rebind walk hit the %d-node "
                "bound with %d rebind(s) done; continuing (fail-open)",
                _REBIND_NODE_BOUND,
                len(holders),
            )
            break
        owner, owner_path = queue.popleft()
        for key, value in list(owner.__dict__.items()):
            if key in _MODULE_REGISTRATION_ATTRS:
                continue
            holder_path = f"{owner_path}.{key}"
            new = _rebound(value)
            if new is not None:
                # Raw __dict__ write: plain attribute assignment on purpose
                # (setattr would re-register Parameter values in clone form).
                owner.__dict__[key] = new
                holders.append(holder_path)
            elif isinstance(value, dict):
                for dkey, dvalue in list(value.items()):
                    dnew = _rebound(dvalue)
                    if dnew is not None:
                        value[dkey] = dnew
                        holders.append(f"{holder_path}[{dkey}]")
            elif isinstance(value, list):
                for i, elem in enumerate(value):
                    enew = _rebound(elem)
                    if enew is not None:
                        value[i] = enew
                        holders.append(f"{holder_path}[{i}]")
            elif isinstance(value, tuple):
                # Tuples are immutable: rebuild only if something is stale.
                new_elems = [_rebound(elem) for elem in value]
                stale_idx = [i for i, e in enumerate(new_elems) if e is not None]
                for i in stale_idx:
                    holders.append(f"{holder_path}[{i}]")
                if stale_idx:
                    stale_set = set(stale_idx)
                    owner.__dict__[key] = tuple(
                        new_elems[i] if i in stale_set else elem
                        for i, elem in enumerate(value)
                    )
            elif isinstance(value, nn.Module):
                # Modules are BFS seeds already; do not descend here.
                continue
            elif id(value) not in visited and _may_scan(value):
                # Non-module, non-tensor instance (this is what reaches the
                # router): scan its plain attributes on a later turn.
                visited.add(id(value))
                queue.append((value, holder_path))

    if holders:
        logger.debug(
            "ipc_cache: rebound stale plain-attribute holders: %s",
            ", ".join(holders),
        )
    return holders


def _materialize_remaining_meta_tensors(model: nn.Module, device: torch.device) -> None:
    """Allocate any tensors the daemon did not provide.

    These are typically parameters removed on the daemon side by
    process_weights_after_loading; they are not expected to be read at
    runtime, so they are left uninitialized.
    """
    for module_name, module in model.named_modules():
        for leaf, param in list(module._parameters.items()):
            if param is not None and param.device.type == "meta":
                logger.warning(
                    "Materializing empty parameter %s.%s missing from the weight cache",
                    module_name,
                    leaf,
                )
                module._parameters[leaf] = nn.Parameter(
                    torch.empty_like(param, device=device), requires_grad=False
                )
        for leaf, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[leaf] = torch.empty_like(buffer, device=device)
