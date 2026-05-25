#  Copyright (c) 2026, TensorCast Team.
"""Framework-neutral recipe build identity and cache helpers."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorcast.serving.binding_plan import ServingBindingPlan

_LOGGER = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def stable_recipe_build_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def compute_trace_cache_key(
    identity: ServingBindingPlan,
    *,
    metadata_fingerprint: str,
) -> str:
    return stable_recipe_build_hash(
        identity.trace_payload(metadata_fingerprint=metadata_fingerprint)
    )


def compute_recipe_cache_key(
    identity: ServingBindingPlan,
    *,
    metadata_fingerprint: str,
) -> str:
    return stable_recipe_build_hash(
        identity.recipe_payload(metadata_fingerprint=metadata_fingerprint)
    )


def trace_cache_path(*, cache_dir: str, cache_key: str, tp_rank: int) -> str:
    return os.path.join(cache_dir, f"tensorcast_trace_{cache_key}_tp{tp_rank}.json")


def recipe_cache_path(*, cache_dir: str, cache_key: str, tp_rank: int) -> str:
    return os.path.join(cache_dir, f"tensorcast_recipe_{cache_key}_tp{tp_rank}.json")


@dataclass(frozen=True)
class RecipeBuildCacheConfig:
    cache_dirs: tuple[str, ...] = ()
    trace_write_dirs: tuple[str, ...] = ()
    recipe_cache_dirs: tuple[str, ...] = ()
    recipe_cache_write_dirs: tuple[str, ...] = ()
    debug_output_dir: Path | None = None
    allow_cache: bool = True
    allow_recipe_cache: bool = True
    allow_trace: bool = True
    trace_tp_slices: bool = True
    debug_dump_trace: bool = False
    trace_cache_schema_version: int = 1
    synchronous_cache_write: bool = True
    synchronous_recipe_cache_write: bool = False


DEFAULT_RECIPE_BUILD_MEMORY_CACHE_ENTRIES = 128


class RecipeBuildMemoryCache(MutableMapping[str, Any]):
    """Thread-safe LRU memory cache for trace plans and compiled recipes."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_RECIPE_BUILD_MEMORY_CACHE_ENTRIES,
    ) -> None:
        self._max_entries = max(0, int(max_entries))
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, Any] = OrderedDict()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def size(self) -> int:
        return len(self)

    @property
    def enabled(self) -> bool:
        return self._max_entries > 0

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            value = self._entries[key]
            self._entries.move_to_end(key)
            return value

    def __setitem__(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            self._evict_lru_locked()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._entries[key]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(tuple(self._entries.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _evict_lru_locked(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


TRACE_PLAN_MEMORY_CACHE: RecipeBuildMemoryCache = RecipeBuildMemoryCache()
COMPILED_RECIPE_MEMORY_CACHE: RecipeBuildMemoryCache = RecipeBuildMemoryCache()


@dataclass(frozen=True)
class RecipeCacheLookupResult:
    value: Any | None
    cache_key: str
    cache_path: str | None
    cache_lookup_sec: float
    memory_hit: bool = False
    disk_hit: bool = False


@dataclass(frozen=True)
class RecipeCacheWriteResult:
    cache_path: str
    cache_write_sec: float


@dataclass(frozen=True)
class RecipeBuildRunResult:
    recipe: Any
    diagnostics: dict[str, Any]


def _metadata_fingerprint(source_catalog: Any) -> str:
    return str(getattr(source_catalog, "metadata_fingerprint", ""))


def _cache_config_attr(cache_config: Any, name: str, default: Any) -> Any:
    return getattr(cache_config, name, default)


class RecipeBuildSession:
    """Small core-owned shell for stable recipe build cache identity."""

    def __init__(self, identity: ServingBindingPlan) -> None:
        self.identity = identity

    def trace_cache_key(self, *, metadata_fingerprint: str) -> str:
        return compute_trace_cache_key(
            self.identity,
            metadata_fingerprint=metadata_fingerprint,
        )

    def recipe_cache_key(self, *, metadata_fingerprint: str) -> str:
        return compute_recipe_cache_key(
            self.identity,
            metadata_fingerprint=metadata_fingerprint,
        )

    def trace_cache_path(
        self,
        *,
        metadata_fingerprint: str,
        cache_dir: str,
    ) -> str:
        return trace_cache_path(
            cache_dir=cache_dir,
            cache_key=self.trace_cache_key(metadata_fingerprint=metadata_fingerprint),
            tp_rank=self.identity.tp_rank,
        )

    def recipe_cache_path(
        self,
        *,
        metadata_fingerprint: str,
        cache_dir: str,
    ) -> str:
        return recipe_cache_path(
            cache_dir=cache_dir,
            cache_key=self.recipe_cache_key(metadata_fingerprint=metadata_fingerprint),
            tp_rank=self.identity.tp_rank,
        )

    def compile_identity(self, *, serving_facts: Any) -> Any:
        return ServingBindingPlan(
            model_id=self.identity.model_id,
            model_revision=self.identity.model_revision,
            dtype=self.identity.dtype,
            model_hash=self.identity.model_hash,
            runtime_version=self.identity.runtime_version,
            framework_name=getattr(
                serving_facts, "framework_name", self.identity.framework_name
            ),
            adapter_version=getattr(
                serving_facts, "adapter_version", self.identity.adapter_version
            ),
            serving_abi_version=getattr(
                serving_facts,
                "serving_abi_version",
                self.identity.serving_abi_version,
            ),
            framework_version=getattr(
                serving_facts, "framework_version", self.identity.framework_version
            ),
            trace_cache_schema_version=self.identity.trace_cache_schema_version,
            tp_rank=self.identity.tp_rank,
            tp_world_size=self.identity.tp_world_size,
            topology_ref=self.identity.topology_ref,
            member_ref=self.identity.member_ref,
            placement=self.identity.placement,
        )

    def compile_recipe(self, *, inputs: Any) -> Any:
        return self.compile_recipe_from_inputs(
            identity=self.compile_identity(serving_facts=inputs.serving_facts),
            inputs=inputs,
        )

    def build_recipe(
        self,
        *,
        model_config: Any,
        framework_config: Any | None = None,
        source_catalog: Any,
        framework_adapter: Any,
        build_meta_model: Callable[[], Any],
        cache_config: Any,
        is_reserved_serving_tensor_name: Callable[[str], bool],
        semantic_validation_spec: object | None = None,
        trace_capture_fn: Callable[[Any, list[str], dict[str, Any]], Any] | None = None,
        trace_plan_memory_cache: MutableMapping[str, Any] | None = None,
        compiled_recipe_memory_cache: MutableMapping[str, Any] | None = None,
        placement: Any | None = None,
        debug_extra: dict[str, Any] | None = None,
        profile_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> RecipeBuildRunResult:
        total_start = time.perf_counter()
        allow_cache = bool(_cache_config_attr(cache_config, "allow_cache", True))
        allow_recipe_cache = bool(
            _cache_config_attr(cache_config, "allow_recipe_cache", True)
        )
        allow_trace = bool(_cache_config_attr(cache_config, "allow_trace", True))
        debug_dump_trace = bool(
            _cache_config_attr(cache_config, "debug_dump_trace", False)
        )
        trace_tp_slices = bool(
            _cache_config_attr(cache_config, "trace_tp_slices", True)
        )
        cache_dirs = tuple(_cache_config_attr(cache_config, "cache_dirs", ()))
        recipe_cache_dirs = tuple(
            _cache_config_attr(cache_config, "recipe_cache_dirs", ()) or cache_dirs
        )
        trace_write_dirs = tuple(
            _cache_config_attr(cache_config, "trace_write_dirs", ()) or cache_dirs
        )
        recipe_cache_write_dirs = tuple(
            _cache_config_attr(cache_config, "recipe_cache_write_dirs", ())
            or recipe_cache_dirs
        )
        debug_output_dir = _cache_config_attr(cache_config, "debug_output_dir", None)
        trace_cache_schema_version = int(
            _cache_config_attr(
                cache_config,
                "trace_cache_schema_version",
                self.identity.trace_cache_schema_version,
            )
        )
        synchronous_cache_write = bool(
            _cache_config_attr(cache_config, "synchronous_cache_write", True)
        )
        synchronous_recipe_cache_write = bool(
            _cache_config_attr(cache_config, "synchronous_recipe_cache_write", False)
        )

        cache_hit_path = None
        cache_lookup_sec = 0.0
        recipe_cache_hit_path = None
        recipe_cache_lookup_sec = 0.0
        recipe_cache_write_sec = 0.0
        recipe_cache_write_deferred = False
        trace_build_sec = 0.0
        cache_write_sec = 0.0
        cache_write_deferred = False

        if trace_plan_memory_cache is None:
            trace_plan_memory_cache = TRACE_PLAN_MEMORY_CACHE
        if compiled_recipe_memory_cache is None:
            compiled_recipe_memory_cache = COMPILED_RECIPE_MEMORY_CACHE

        if allow_cache and allow_recipe_cache:
            cached_recipe, recipe_cache_hit_path, recipe_cache_lookup_sec = (
                self._lookup_compiled_recipe_cache_value(
                    source_catalog=source_catalog,
                    cache_dirs=recipe_cache_dirs,
                    memory_cache=compiled_recipe_memory_cache,
                    placement=placement,
                )
            )
            self._emit_profile(
                profile_sink,
                "recipe.lookup_recipe_cache",
                {
                    "cache_lookup_sec": recipe_cache_lookup_sec,
                    "cache_hit": cached_recipe is not None,
                    "cache_hit_path": recipe_cache_hit_path,
                },
            )
            if cached_recipe is not None:
                recipe = self.rebind_cached_recipe_template(
                    cached_recipe,
                    source_catalog=source_catalog,
                )
                self._dump_trace_plan_debug(
                    recipe.trace_plan,
                    model_config=model_config,
                    cache_path=recipe_cache_hit_path,
                    cache_hit=True,
                    output_dir=debug_output_dir,
                    trace_cache_schema_version=trace_cache_schema_version,
                    extra=debug_extra,
                )
                diagnostics = self._recipe_build_diagnostics(
                    recipe,
                    total_start=total_start,
                    cache_hit=True,
                    cache_hit_path=None,
                    cache_lookup_sec=0.0,
                    recipe_cache_hit=True,
                    recipe_cache_hit_path=recipe_cache_hit_path,
                    recipe_cache_lookup_sec=recipe_cache_lookup_sec,
                    trace_build_sec=0.0,
                    cache_write_sec=0.0,
                    cache_write_deferred=False,
                    recipe_cache_write_sec=0.0,
                    recipe_cache_write_deferred=False,
                )
                self._emit_profile(profile_sink, "recipe.summary", diagnostics)
                return RecipeBuildRunResult(recipe=recipe, diagnostics=diagnostics)

        meta_model = build_meta_model()
        self._emit_profile(
            profile_sink,
            "recipe.build_meta_model",
            {
                "model_type": getattr(model_config, "model_type", None),
                "model_name": getattr(model_config, "model", None),
                "meta_model_class": type(meta_model).__name__,
            },
        )
        serving_facts = self.collect_serving_facts(
            meta_model,
            model_config,
            framework_adapter,
        )
        tensor_schema = self.collect_tensor_schema(
            meta_model,
            runtime_only_tensor_names=serving_facts.runtime_only_tensor_names,
            is_reserved_serving_tensor_name=is_reserved_serving_tensor_name,
        )
        resolved_semantic_validation_spec = self.resolve_semantic_validation_spec(
            meta_model,
            model_config,
            framework_adapter,
            semantic_validation_spec,
        )
        self._emit_profile(
            profile_sink,
            "recipe.collect_model_metadata",
            {
                "support_level": getattr(
                    serving_facts.support_level, "value", serving_facts.support_level
                ),
                "process_after_load_class": getattr(
                    serving_facts.process_after_load_class,
                    "value",
                    serving_facts.process_after_load_class,
                ),
                "post_bind_finalize_class": getattr(
                    serving_facts.post_bind_finalize_class,
                    "value",
                    serving_facts.post_bind_finalize_class,
                ),
                "runtime_only_tensor_count": len(
                    serving_facts.runtime_only_tensor_names
                ),
                "tensor_schema_count": len(tensor_schema),
            },
        )

        trace_plan = None
        if allow_cache:
            trace_plan, cache_hit_path, cache_lookup_sec = (
                self._lookup_trace_plan_cache_value(
                    source_catalog=source_catalog,
                    cache_dirs=cache_dirs,
                    memory_cache=trace_plan_memory_cache,
                )
            )
            self._emit_profile(
                profile_sink,
                "recipe.lookup_trace_cache",
                {
                    "cache_lookup_sec": cache_lookup_sec,
                    "cache_hit": trace_plan is not None,
                    "cache_hit_path": cache_hit_path,
                },
            )
            if trace_plan is not None:
                self._dump_trace_plan_debug(
                    trace_plan,
                    model_config=model_config,
                    cache_path=cache_hit_path,
                    cache_hit=True,
                    output_dir=debug_output_dir,
                    trace_cache_schema_version=trace_cache_schema_version,
                    extra=debug_extra,
                )

        if trace_plan is None:
            if not trace_tp_slices:
                raise RuntimeError(
                    "tensorcast trace_tp_slices is disabled but no cache found"
                )
            if not allow_trace:
                raise RuntimeError(
                    "Tensorcast trace plan cache miss during reload; expected "
                    "startup load_model() to have already produced the trace plan"
                )
            if trace_capture_fn is None:
                trace_capture_fn = getattr(framework_adapter, "trace_model_load", None)
            if not callable(trace_capture_fn):
                raise RuntimeError(
                    "TensorCast recipe build requires a framework trace callback"
                )
            stage_start = time.perf_counter()
            trace_kwargs: dict[str, Any] = {}
            try:
                parameters = inspect.signature(trace_capture_fn).parameters
                if "debug_dump_trace" in parameters or any(
                    param.kind is inspect.Parameter.VAR_KEYWORD
                    for param in parameters.values()
                ):
                    trace_kwargs["debug_dump_trace"] = debug_dump_trace
            except (TypeError, ValueError):
                pass
            trace_plan = trace_capture_fn(
                meta_model,
                list(source_catalog.ordered_names),
                source_catalog.meta_by_name,
                **trace_kwargs,
            )
            trace_build_sec = time.perf_counter() - stage_start
            self._emit_profile(
                profile_sink,
                "recipe.trace_model_load",
                {
                    "ordered_source_tensor_count": len(source_catalog.ordered_names),
                    **self.trace_plan_summary_fields(trace_plan),
                },
            )
            self.remember_trace_plan_cache(
                source_catalog=source_catalog,
                trace_plan=trace_plan,
                memory_cache=trace_plan_memory_cache,
            )
            trace_cache_paths = self.trace_cache_write_paths(
                source_catalog=source_catalog,
                cache_dirs=trace_write_dirs,
            )
            self._dump_trace_plan_debug(
                trace_plan,
                model_config=model_config,
                cache_path=trace_cache_paths[0] if trace_cache_paths else None,
                cache_hit=False,
                output_dir=debug_output_dir,
                trace_cache_schema_version=trace_cache_schema_version,
                extra=debug_extra,
            )
            for cache_path in trace_cache_paths:
                if synchronous_cache_write:
                    write_result = self.store_trace_plan_cache(
                        cache_path=cache_path,
                        trace_plan=trace_plan,
                    )
                    cache_write_sec += write_result.cache_write_sec
                    self._emit_profile(
                        profile_sink,
                        "recipe.write_trace_cache",
                        {
                            "cache_path": cache_path,
                            "deferred": False,
                            "cache_write_sec": write_result.cache_write_sec,
                        },
                    )
                else:
                    self.defer_trace_plan_cache_write(
                        cache_path=cache_path,
                        trace_plan=trace_plan,
                    )
                    cache_write_deferred = True
                    self._emit_profile(
                        profile_sink,
                        "recipe.defer_trace_cache_write",
                        {
                            "cache_path": cache_path,
                            "deferred": True,
                        },
                    )

        recipe = self.compile_recipe(
            inputs=self._recipe_compile_inputs(
                source_catalog=source_catalog,
                trace_plan=trace_plan,
                serving_facts=serving_facts,
                tensor_schema=tensor_schema,
                semantic_validation_spec=resolved_semantic_validation_spec,
            )
        )
        self._cleanup_after_recipe_build(
            framework_adapter,
            meta_model=meta_model,
            model_config=model_config,
            framework_config=framework_config,
        )
        if allow_cache and allow_recipe_cache:
            self.remember_compiled_recipe_cache(
                source_catalog=source_catalog,
                recipe=recipe,
                memory_cache=compiled_recipe_memory_cache,
            )
            for cache_path in self.recipe_cache_write_paths(
                source_catalog=source_catalog,
                cache_dirs=recipe_cache_write_dirs,
            ):
                if synchronous_recipe_cache_write:
                    write_result = self.store_compiled_recipe_cache(
                        cache_path=cache_path,
                        recipe=recipe,
                    )
                    recipe_cache_write_sec += write_result.cache_write_sec
                    self._emit_profile(
                        profile_sink,
                        "recipe.write_recipe_cache",
                        {
                            "cache_path": cache_path,
                            "deferred": False,
                            "cache_write_sec": write_result.cache_write_sec,
                        },
                    )
                else:
                    self.defer_compiled_recipe_cache_write(
                        cache_path=cache_path,
                        recipe=recipe,
                    )
                    recipe_cache_write_deferred = True
                    self._emit_profile(
                        profile_sink,
                        "recipe.defer_recipe_cache_write",
                        {
                            "cache_path": cache_path,
                            "deferred": True,
                        },
                    )

        diagnostics = self._recipe_build_diagnostics(
            recipe,
            total_start=total_start,
            cache_hit=cache_hit_path is not None or trace_build_sec == 0.0,
            cache_hit_path=cache_hit_path,
            cache_lookup_sec=cache_lookup_sec,
            recipe_cache_hit=False,
            recipe_cache_hit_path=recipe_cache_hit_path,
            recipe_cache_lookup_sec=recipe_cache_lookup_sec,
            trace_build_sec=trace_build_sec,
            cache_write_sec=cache_write_sec,
            cache_write_deferred=cache_write_deferred,
            recipe_cache_write_sec=recipe_cache_write_sec,
            recipe_cache_write_deferred=recipe_cache_write_deferred,
        )
        self._emit_profile(profile_sink, "recipe.summary", diagnostics)
        return RecipeBuildRunResult(recipe=recipe, diagnostics=diagnostics)

    def _lookup_trace_plan_cache_value(
        self,
        *,
        source_catalog: Any,
        cache_dirs: Sequence[str],
        memory_cache: MutableMapping[str, Any] | None,
    ) -> tuple[Any | None, str | None, float]:
        result = self.lookup_trace_plan_cache(
            source_catalog=source_catalog,
            cache_dirs=cache_dirs,
            memory_cache=memory_cache,
        )
        return result.value, result.cache_path, result.cache_lookup_sec

    def _lookup_compiled_recipe_cache_value(
        self,
        *,
        source_catalog: Any,
        cache_dirs: Sequence[str],
        memory_cache: MutableMapping[str, Any] | None,
        placement: Any | None,
    ) -> tuple[Any | None, str | None, float]:
        result = self.lookup_compiled_recipe_cache(
            source_catalog=source_catalog,
            cache_dirs=cache_dirs,
            memory_cache=memory_cache,
            placement=placement,
        )
        return result.value, result.cache_path, result.cache_lookup_sec

    @staticmethod
    def _cleanup_after_recipe_build(
        framework_adapter: Any,
        *,
        meta_model: Any,
        model_config: Any,
        framework_config: Any | None,
    ) -> None:
        cleanup = getattr(framework_adapter, "cleanup_after_recipe_build", None)
        if callable(cleanup):
            cleanup(meta_model, model_config, framework_config=framework_config)

    @staticmethod
    def _recipe_compile_inputs(
        *,
        source_catalog: Any,
        trace_plan: Any,
        serving_facts: Any,
        tensor_schema: Any,
        semantic_validation_spec: Any,
    ) -> Any:
        from tensorcast.serving.builder import compiler as tc_compiler

        return tc_compiler.RecipeCompileInputs(
            source_catalog=source_catalog,
            trace_plan=trace_plan,
            serving_facts=serving_facts,
            tensor_schema=tensor_schema,
            semantic_validation_spec=semantic_validation_spec,
        )

    def _dump_trace_plan_debug(
        self,
        trace_plan: Any,
        *,
        model_config: Any,
        cache_path: str | None,
        cache_hit: bool,
        output_dir: Any | None,
        trace_cache_schema_version: int,
        extra: dict[str, Any] | None,
    ) -> None:
        if output_dir is None:
            return
        compute_hash = getattr(model_config, "compute_hash", None)
        payload: dict[str, Any] = {
            "pid": os.getpid(),
            "model_hash": compute_hash()
            if callable(compute_hash)
            else self.identity.model_hash,
            "version": self.identity.runtime_version,
            "tp_rank": self.identity.tp_rank,
            "tp_world_size": self.identity.tp_world_size,
        }
        payload.update(extra or {})
        self.dump_trace_plan_debug(
            trace_plan,
            output_dir=output_dir,
            filename=(
                f"tensorcast_trace_plan_tp{self.identity.tp_rank}_pid{os.getpid()}.json"
            ),
            cache_path=cache_path,
            cache_hit=cache_hit,
            trace_cache_schema_version=trace_cache_schema_version,
            extra=payload,
        )

    def _recipe_build_diagnostics(
        self,
        recipe: Any,
        *,
        total_start: float,
        cache_hit: bool,
        cache_hit_path: str | None,
        cache_lookup_sec: float,
        recipe_cache_hit: bool,
        recipe_cache_hit_path: str | None,
        recipe_cache_lookup_sec: float,
        trace_build_sec: float,
        cache_write_sec: float,
        cache_write_deferred: bool,
        recipe_cache_write_sec: float,
        recipe_cache_write_deferred: bool,
    ) -> dict[str, Any]:
        return {
            "compile_key": recipe.compile_key,
            "cache_hit": cache_hit,
            "cache_hit_path": cache_hit_path,
            "cache_lookup_sec": cache_lookup_sec,
            "recipe_cache_hit": recipe_cache_hit,
            "recipe_cache_hit_path": recipe_cache_hit_path,
            "recipe_cache_lookup_sec": recipe_cache_lookup_sec,
            "trace_build_sec": trace_build_sec,
            "cache_write_sec": cache_write_sec,
            "cache_write_deferred": cache_write_deferred,
            "recipe_cache_write_sec": recipe_cache_write_sec,
            "recipe_cache_write_deferred": recipe_cache_write_deferred,
            "total_sec": time.perf_counter() - total_start,
            **self.recipe_summary_fields(recipe),
        }

    @staticmethod
    def _emit_profile(
        profile_sink: Callable[[dict[str, Any]], None] | None,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        if callable(profile_sink):
            profile_sink({"stage": stage, **payload})

    def rebind_cached_recipe_template(
        self,
        cached_recipe: Any,
        *,
        source_catalog: Any,
    ) -> Any:
        from dataclasses import replace

        from tensorcast.serving.builder import compiler as tc_compiler
        from tensorcast.serving.source_catalog import (
            resolve_source_artifact_ref,
        )

        source_artifact_ref = resolve_source_artifact_ref(
            source_catalog.source_artifact_ref
        )
        source_metadata_fingerprint = str(source_catalog.metadata_fingerprint)
        identity = self.compile_identity(serving_facts=cached_recipe.serving_facts)
        realization_plan_proto = bytes(cached_recipe.realization_plan_proto or b"")
        binding_plan = identity.with_compiled_artifacts(
            source_artifact_ref=source_artifact_ref,
            source_metadata_fingerprint=source_metadata_fingerprint,
            serving_facts=cached_recipe.serving_facts,
            trace_plan=cached_recipe.trace_plan,
            tensor_schema=tuple(cached_recipe.tensor_schema),
            source_hull=tuple(cached_recipe.source_hull),
            source_schema_hash=str(
                getattr(source_catalog, "canonical_index_hash", "")
                or source_metadata_fingerprint
            ),
            tensor_schema_hash=tc_compiler.target_tensor_schema_hash(
                cached_recipe.tensor_schema
            ),
            realization_plan=tuple(cached_recipe.realization_plan),
            realization_fallback_plan=tuple(cached_recipe.realization_fallback_plan),
            realization_plan_proto=realization_plan_proto,
            realization_plan_digest=tc_compiler.realization_plan_digest(
                realization_plan_proto
            ),
            realization_plan_count=tc_compiler.compiled_recipe_realization_plan_count(
                cached_recipe
            ),
            semantic_validation_spec=cached_recipe.semantic_validation_spec,
        )
        compile_key = tc_compiler.compute_recipe_compile_key(
            identity=binding_plan,
            source_artifact_ref=source_artifact_ref,
            source_metadata_fingerprint=source_metadata_fingerprint,
            serving_facts=cached_recipe.serving_facts,
            tensor_schema=cached_recipe.tensor_schema,
            semantic_validation_spec=cached_recipe.semantic_validation_spec,
        )
        return replace(
            cached_recipe,
            compile_key=compile_key,
            source_artifact_ref=source_artifact_ref,
            source_metadata_fingerprint=source_metadata_fingerprint,
            topology_ref=identity.topology_ref,
            member_ref=identity.member_ref,
            binding_plan=binding_plan,
        )

    def cached_recipe_matches_context(
        self,
        recipe: Any,
        *,
        source_catalog: Any,
        placement: Any | None = None,
    ) -> bool:
        if str(recipe.source_metadata_fingerprint) != str(
            source_catalog.metadata_fingerprint
        ):
            return False
        if placement is not None:
            serving_placement = getattr(placement, "serving_placement", placement)
            placement_topology = getattr(serving_placement, "topology", None)
            placement_member = getattr(serving_placement, "member", None)
            recipe_topology = getattr(recipe, "topology_ref", None)
            recipe_member = getattr(recipe, "member_ref", None)
            if recipe_topology is not None and recipe_topology != placement_topology:
                return False
            if recipe_member is not None and recipe_member != placement_member:
                return False
        return True

    def lookup_trace_plan_cache(
        self,
        *,
        source_catalog: Any,
        cache_dirs: Sequence[str],
        memory_cache: MutableMapping[str, Any] | None = None,
    ) -> RecipeCacheLookupResult:
        stage_start = time.perf_counter()
        metadata_fingerprint = _metadata_fingerprint(source_catalog)
        cache_key = self.trace_cache_key(metadata_fingerprint=metadata_fingerprint)
        cached = None if memory_cache is None else memory_cache.get(cache_key)
        cache_lookup_sec = time.perf_counter() - stage_start
        if cached is not None:
            return RecipeCacheLookupResult(
                value=cached,
                cache_key=cache_key,
                cache_path=None,
                cache_lookup_sec=cache_lookup_sec,
                memory_hit=True,
            )
        for cache_dir in cache_dirs:
            cache_path = self.trace_cache_path(
                metadata_fingerprint=metadata_fingerprint,
                cache_dir=cache_dir,
            )
            stage_start = time.perf_counter()
            cached = self.load_trace_plan_cache(cache_path)
            cache_lookup_sec += time.perf_counter() - stage_start
            if cached is not None:
                if memory_cache is not None:
                    memory_cache[cache_key] = cached
                return RecipeCacheLookupResult(
                    value=cached,
                    cache_key=cache_key,
                    cache_path=cache_path,
                    cache_lookup_sec=cache_lookup_sec,
                    disk_hit=True,
                )
        return RecipeCacheLookupResult(
            value=None,
            cache_key=cache_key,
            cache_path=None,
            cache_lookup_sec=cache_lookup_sec,
        )

    def lookup_compiled_recipe_cache(
        self,
        *,
        source_catalog: Any,
        cache_dirs: Sequence[str],
        memory_cache: MutableMapping[str, Any] | None = None,
        placement: Any | None = None,
    ) -> RecipeCacheLookupResult:
        stage_start = time.perf_counter()
        metadata_fingerprint = _metadata_fingerprint(source_catalog)
        cache_key = self.recipe_cache_key(metadata_fingerprint=metadata_fingerprint)
        cached = None if memory_cache is None else memory_cache.get(cache_key)
        cache_lookup_sec = time.perf_counter() - stage_start
        if cached is not None and self.cached_recipe_matches_context(
            cached,
            source_catalog=source_catalog,
            placement=placement,
        ):
            return RecipeCacheLookupResult(
                value=cached,
                cache_key=cache_key,
                cache_path=None,
                cache_lookup_sec=cache_lookup_sec,
                memory_hit=True,
            )
        for cache_dir in cache_dirs:
            cache_path = self.recipe_cache_path(
                metadata_fingerprint=metadata_fingerprint,
                cache_dir=cache_dir,
            )
            stage_start = time.perf_counter()
            cached = self.load_compiled_recipe_cache(cache_path)
            cache_lookup_sec += time.perf_counter() - stage_start
            if cached is not None and self.cached_recipe_matches_context(
                cached,
                source_catalog=source_catalog,
                placement=placement,
            ):
                if memory_cache is not None:
                    memory_cache[cache_key] = cached
                return RecipeCacheLookupResult(
                    value=cached,
                    cache_key=cache_key,
                    cache_path=cache_path,
                    cache_lookup_sec=cache_lookup_sec,
                    disk_hit=True,
                )
        return RecipeCacheLookupResult(
            value=None,
            cache_key=cache_key,
            cache_path=None,
            cache_lookup_sec=cache_lookup_sec,
        )

    def remember_trace_plan_cache(
        self,
        *,
        source_catalog: Any,
        trace_plan: Any,
        memory_cache: MutableMapping[str, Any] | None = None,
    ) -> str:
        cache_key = self.trace_cache_key(
            metadata_fingerprint=_metadata_fingerprint(source_catalog)
        )
        if memory_cache is not None:
            memory_cache[cache_key] = trace_plan
        return cache_key

    def remember_compiled_recipe_cache(
        self,
        *,
        source_catalog: Any,
        recipe: Any,
        memory_cache: MutableMapping[str, Any] | None = None,
    ) -> str:
        cache_key = self.recipe_cache_key(
            metadata_fingerprint=_metadata_fingerprint(source_catalog)
        )
        if memory_cache is not None:
            memory_cache[cache_key] = recipe
        return cache_key

    def trace_cache_write_paths(
        self,
        *,
        source_catalog: Any,
        cache_dirs: Sequence[str],
    ) -> tuple[str, ...]:
        metadata_fingerprint = _metadata_fingerprint(source_catalog)
        return tuple(
            self.trace_cache_path(
                metadata_fingerprint=metadata_fingerprint,
                cache_dir=cache_dir,
            )
            for cache_dir in cache_dirs
        )

    def recipe_cache_write_paths(
        self,
        *,
        source_catalog: Any,
        cache_dirs: Sequence[str],
    ) -> tuple[str, ...]:
        metadata_fingerprint = _metadata_fingerprint(source_catalog)
        return tuple(
            self.recipe_cache_path(
                metadata_fingerprint=metadata_fingerprint,
                cache_dir=cache_dir,
            )
            for cache_dir in cache_dirs
        )

    def store_trace_plan_cache(
        self,
        *,
        cache_path: str,
        trace_plan: Any,
    ) -> RecipeCacheWriteResult:
        stage_start = time.perf_counter()
        self.write_trace_plan_cache(cache_path, trace_plan)
        return RecipeCacheWriteResult(
            cache_path=cache_path,
            cache_write_sec=time.perf_counter() - stage_start,
        )

    def store_compiled_recipe_cache(
        self,
        *,
        cache_path: str,
        recipe: Any,
    ) -> RecipeCacheWriteResult:
        stage_start = time.perf_counter()
        self.write_compiled_recipe_cache(cache_path, recipe)
        return RecipeCacheWriteResult(
            cache_path=cache_path,
            cache_write_sec=time.perf_counter() - stage_start,
        )

    def defer_trace_plan_cache_write(
        self,
        *,
        cache_path: str,
        trace_plan: Any,
    ) -> None:
        def _worker() -> None:
            try:
                self.store_trace_plan_cache(
                    cache_path=cache_path,
                    trace_plan=trace_plan,
                )
            except Exception:
                _LOGGER.exception("Failed to write deferred trace plan cache")

        threading.Thread(
            target=_worker,
            name="tensorcast-trace-cache-write",
            daemon=True,
        ).start()

    def defer_compiled_recipe_cache_write(
        self,
        *,
        cache_path: str,
        recipe: Any,
    ) -> None:
        def _worker() -> None:
            try:
                self.store_compiled_recipe_cache(
                    cache_path=cache_path,
                    recipe=recipe,
                )
            except Exception:
                _LOGGER.exception("Failed to write deferred compiled recipe cache")

        threading.Thread(
            target=_worker,
            name="tensorcast-recipe-cache-write",
            daemon=True,
        ).start()

    @staticmethod
    def collect_serving_facts(
        model: Any,
        model_config: Any,
        framework_adapter: Any,
    ) -> Any:
        from tensorcast.serving.builder import compiler as tc_compiler

        return tc_compiler.TensorcastServingFacts(
            framework_name=framework_adapter.framework_name(),
            framework_version=framework_adapter.framework_version(),
            adapter_version=framework_adapter.adapter_version(),
            serving_abi_version=framework_adapter.serving_abi_version(model_config),
            support_level=framework_adapter.support_level(model, model_config),
            runtime_only_tensor_names=framework_adapter.runtime_only_tensor_names(
                model
            ),
            process_after_load_class=framework_adapter.process_after_load_class(
                model, model_config
            ),
            post_bind_finalize_class=framework_adapter.post_bind_finalize_class(
                model, model_config
            ),
        )

    @staticmethod
    def collect_tensor_schema(
        model: Any,
        *,
        runtime_only_tensor_names: tuple[str, ...],
        is_reserved_serving_tensor_name: Any,
    ) -> tuple[Any, ...]:
        from tensorcast.serving.builder import compiler as tc_compiler

        excluded = set(runtime_only_tensor_names)
        entries: list[Any] = []
        # Serving tensor schema should use canonical tensor names. Alias paths
        # for the same Parameter/Buffer do not represent additional tensors.
        for name, param in model.named_parameters(remove_duplicate=True):
            if name in excluded:
                continue
            if is_reserved_serving_tensor_name(name):
                raise RuntimeError(
                    f"Model tensor name '{name}' collides with Tensorcast reserved names"
                )
            tensor = param.data
            entries.append(
                tc_compiler.TensorSchemaEntry(
                    name=name,
                    dtype=str(tensor.dtype),
                    shape=tuple(int(dim) for dim in tensor.shape),
                    stride=tuple(int(dim) for dim in tensor.stride()),
                )
            )
        for name, buf in model.named_buffers(remove_duplicate=True):
            if name in excluded:
                continue
            if is_reserved_serving_tensor_name(name):
                raise RuntimeError(
                    f"Model tensor name '{name}' collides with Tensorcast reserved names"
                )
            entries.append(
                tc_compiler.TensorSchemaEntry(
                    name=name,
                    dtype=str(buf.dtype),
                    shape=tuple(int(dim) for dim in buf.shape),
                    stride=tuple(int(dim) for dim in buf.stride()),
                )
            )
        if not entries:
            raise RuntimeError(
                "Tensorcast runtime binding requires at least one model tensor"
            )
        return tuple(sorted(entries, key=lambda item: item.name))

    @staticmethod
    def resolve_semantic_validation_spec(
        model: Any,
        model_config: Any,
        framework_adapter: Any,
        explicit_spec: object | None,
    ) -> Any:
        from tensorcast.serving.builder import compiler as tc_compiler

        if explicit_spec is not None:
            if isinstance(explicit_spec, tc_compiler.TensorcastSemanticValidationSpec):
                return explicit_spec
            return tc_compiler.TensorcastSemanticValidationSpec(
                kind="explicit",
                payload=_jsonable(explicit_spec),
            )
        probes = framework_adapter.semantic_probes(model, model_config)
        if probes is None:
            return tc_compiler.TensorcastSemanticValidationSpec.empty()
        return tc_compiler.TensorcastSemanticValidationSpec(
            kind="framework_semantic_probes",
            payload=_jsonable(probes),
        )

    @staticmethod
    def trace_plan_summary_fields(trace_plan: Any) -> dict[str, int]:
        return {
            "copy_plan_count": len(trace_plan.copy_plan),
            "expected_src_count": len(trace_plan.expected_src_names),
            "expected_dst_count": len(trace_plan.expected_dst_names),
            "tensorcast_slice_count": len(trace_plan.tensorcast_slices),
        }

    @staticmethod
    def recipe_summary_fields(recipe: Any) -> dict[str, int]:
        from tensorcast.serving.builder import compiler as tc_compiler

        return {
            "tensor_schema_count": len(recipe.tensor_schema),
            **RecipeBuildSession.trace_plan_summary_fields(recipe.trace_plan),
            "realization_plan_count": tc_compiler.compiled_recipe_realization_plan_count(
                recipe
            ),
            "realization_fallback_count": len(recipe.realization_fallback_plan),
        }

    @staticmethod
    def load_trace_plan_cache(cache_path: str | None) -> Any:
        from tensorcast.serving.builder import trace_cache as tc_trace_cache

        return tc_trace_cache.load_trace_plan_cache(cache_path)

    @staticmethod
    def write_trace_plan_cache(cache_path: str, trace_plan: Any) -> None:
        from tensorcast.serving.builder import trace_cache as tc_trace_cache

        tc_trace_cache.write_trace_plan_cache(cache_path, trace_plan)

    @staticmethod
    def dump_trace_plan_debug(*args: Any, **kwargs: Any) -> Any:
        from tensorcast.serving.builder import trace_cache as tc_trace_cache

        return tc_trace_cache.dump_trace_plan_debug(*args, **kwargs)

    @staticmethod
    def load_compiled_recipe_cache(cache_path: str | None) -> Any:
        from tensorcast.serving.builder import recipe_cache as tc_recipe_cache

        return tc_recipe_cache.load_compiled_recipe_cache(cache_path)

    @staticmethod
    def write_compiled_recipe_cache(cache_path: str, recipe: Any) -> None:
        from tensorcast.serving.builder import recipe_cache as tc_recipe_cache

        tc_recipe_cache.write_compiled_recipe_cache(cache_path, recipe)

    @staticmethod
    def compute_recipe_compile_key(*args: Any, **kwargs: Any) -> str:
        from tensorcast.serving.builder import compiler as tc_compiler

        return tc_compiler.compute_recipe_compile_key(*args, **kwargs)

    @staticmethod
    def compile_recipe_from_inputs(*args: Any, **kwargs: Any) -> Any:
        from tensorcast.serving.builder import compiler as tc_compiler

        return tc_compiler.compile_serving_recipe(*args, **kwargs)


__all__ = [
    "ServingBindingPlan",
    "RecipeBuildMemoryCache",
    "RecipeBuildCacheConfig",
    "RecipeBuildRunResult",
    "RecipeCacheLookupResult",
    "RecipeCacheWriteResult",
    "RecipeBuildSession",
    "COMPILED_RECIPE_MEMORY_CACHE",
    "DEFAULT_RECIPE_BUILD_MEMORY_CACHE_ENTRIES",
    "TRACE_PLAN_MEMORY_CACHE",
    "compute_recipe_cache_key",
    "compute_trace_cache_key",
    "recipe_cache_path",
    "stable_recipe_build_hash",
    "trace_cache_path",
]
