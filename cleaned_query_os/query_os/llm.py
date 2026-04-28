from __future__ import annotations

import random
import re
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class VLLMEndpointConfig:
    name: str
    base_url: str
    api_key: str = "EMPTY"
    model: Optional[str] = None
    weight: float = 1.0
    max_inflight: int = 0


class _EndpointRuntime:
    def __init__(self, config: VLLMEndpointConfig, timeout: Optional[float] = None) -> None:
        self.config = config
        self.client = create_vllm_client(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=timeout,
        )
        self.inflight = 0
        self.failures = 0
        self.cooldown_until = 0.0

    def available(self, now: float) -> bool:
        if self.cooldown_until and now < self.cooldown_until:
            return False
        if self.config.max_inflight > 0 and self.inflight >= self.config.max_inflight:
            return False
        return True


_VLLM_RUNTIME_POOL: Dict[str, _EndpointRuntime] = {}
_VLLM_RUNTIME_POOL_LOCK = threading.RLock()


class OpenAIChatBackend:
    """Single-endpoint backend for the hosted OpenAI API."""

    provider = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        tracer: Optional[Any] = None,
    ) -> None:
        self.client = create_openai_client(api_key=api_key, base_url=base_url, timeout=timeout)
        self.tracer = tracer

    def chat_completion(self, role: str = "generic", **kwargs: Any) -> Any:
        start = time.monotonic()
        try:
            response = _create_chat_completion_compatible(self.client, **kwargs)
            self._emit_route(role, kwargs.get("model"), time.monotonic() - start, "ok")
            return response
        except Exception as exc:
            self._emit_route(role, kwargs.get("model"), time.monotonic() - start, "error", safe_llm_error(exc))
            raise

    def _emit_route(
        self,
        role: str,
        model: Optional[str],
        latency: float,
        status: str,
        error: str = "",
    ) -> None:
        if not self.tracer:
            return
        self.tracer.emit(
            "llm_call",
            str(role),
            "OpenAI LLM call completed.",
            status=status,
            payload={
                "provider": self.provider,
                "role": role,
                "endpoint": "openai",
                "model": model,
                "latency_ms": int(latency * 1000),
                "error": error,
            },
        )


class VLLMRouterBackend:
    """Client-side router for multiple OpenAI-compatible vLLM endpoints."""

    provider = "vllm"

    def __init__(
        self,
        endpoints: Sequence[VLLMEndpointConfig],
        *,
        strategy: str = "least_inflight",
        role_pools: Optional[Dict[str, Dict[str, Any]]] = None,
        max_retries: int = 2,
        cooldown_seconds: float = 30.0,
        request_timeout_seconds: Optional[float] = None,
        tracer: Optional[Any] = None,
    ) -> None:
        if not endpoints:
            raise ValueError("vLLM provider requires at least one endpoint")
        self.strategy = _normalize_strategy(strategy)
        self.max_retries = max(0, int(max_retries or 0))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds or 0.0))
        self.tracer = tracer
        self._lock = _VLLM_RUNTIME_POOL_LOCK
        self._rr_counter = 0
        self._role_rr_counters: Dict[str, int] = {}
        self._endpoints: Dict[str, _EndpointRuntime] = {}
        for endpoint in endpoints:
            if endpoint.name in self._endpoints:
                raise ValueError(f"duplicate vLLM endpoint name: {endpoint.name}")
            self._endpoints[endpoint.name] = _get_shared_vllm_runtime(endpoint, timeout=request_timeout_seconds)
        self.role_pools = _normalize_role_pools(role_pools or {}, set(self._endpoints))

    def chat_completion(self, role: str = "generic", **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        attempts = self.max_retries + 1
        for retry_idx in range(attempts):
            endpoint = self._reserve_endpoint(role)
            request_kwargs = dict(kwargs)
            if endpoint.config.model:
                request_kwargs["model"] = endpoint.config.model
            start = time.monotonic()
            try:
                response = _create_chat_completion_compatible(endpoint.client, **request_kwargs)
                latency = time.monotonic() - start
                with self._lock:
                    endpoint.failures = 0
                    endpoint.cooldown_until = 0.0
                self._emit_route(role, endpoint, request_kwargs.get("model"), retry_idx, latency, "ok")
                return response
            except Exception as exc:
                last_exc = exc
                latency = time.monotonic() - start
                self._mark_endpoint_failure(endpoint)
                self._emit_route(
                    role,
                    endpoint,
                    request_kwargs.get("model"),
                    retry_idx,
                    latency,
                    "error",
                    safe_llm_error(exc),
                )
            finally:
                with self._lock:
                    endpoint.inflight = max(0, endpoint.inflight - 1)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("vLLM router exhausted retries without making a request")

    def _reserve_endpoint(self, role: str) -> _EndpointRuntime:
        with self._lock:
            endpoint = self._pick_endpoint_locked(role)
            endpoint.inflight += 1
            return endpoint

    def _pick_endpoint_locked(self, role: str) -> _EndpointRuntime:
        pool = self.role_pools.get(role) or {}
        allowed_names = pool.get("endpoints") or list(self._endpoints)
        strategy = _normalize_strategy(str(pool.get("strategy") or self.strategy))
        candidates = [self._endpoints[name] for name in allowed_names if name in self._endpoints]
        if not candidates:
            candidates = list(self._endpoints.values())
        now = time.monotonic()
        available = [endpoint for endpoint in candidates if endpoint.available(now)]
        if not available:
            # Prefer making progress over blocking indefinitely. If all endpoints
            # are busy/cooling down, pick from the role pool anyway.
            available = candidates
        if strategy == "random":
            weights = [max(0.01, endpoint.config.weight) for endpoint in available]
            return random.choices(available, weights=weights, k=1)[0]
        if strategy == "round_robin":
            key = role if role in self.role_pools else "__global__"
            idx = self._role_rr_counters.get(key, self._rr_counter)
            endpoint = available[idx % len(available)]
            self._role_rr_counters[key] = idx + 1
            self._rr_counter += 1
            return endpoint
        return min(
            available,
            key=lambda endpoint: (
                endpoint.inflight / max(0.01, endpoint.config.weight),
                endpoint.failures,
                endpoint.config.name,
            ),
        )

    def _mark_endpoint_failure(self, endpoint: _EndpointRuntime) -> None:
        with self._lock:
            endpoint.failures += 1
            if self.cooldown_seconds > 0:
                endpoint.cooldown_until = time.monotonic() + self.cooldown_seconds

    def _emit_route(
        self,
        role: str,
        endpoint: _EndpointRuntime,
        model: Optional[str],
        retry_idx: int,
        latency: float,
        status: str,
        error: str = "",
    ) -> None:
        if not self.tracer:
            return
        self.tracer.emit(
            "llm_call",
            str(role),
            "vLLM routed LLM call completed.",
            status=status,
            payload={
                "provider": self.provider,
                "role": role,
                "endpoint": endpoint.config.name,
                "base_url": endpoint.config.base_url,
                "model": model,
                "retry": retry_idx,
                "latency_ms": int(latency * 1000),
                "inflight_after": endpoint.inflight,
                "error": error,
            },
        )


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The OpenAI Python SDK is required. Install it with `pip install openai` "
            "or `pip install -r requirements.txt` from cleaned_query_os."
        ) from exc

    client_args: Dict[str, Any] = {}
    if api_key:
        client_args["api_key"] = api_key
    if base_url:
        client_args["base_url"] = base_url
    if timeout:
        client_args["timeout"] = timeout
    return OpenAI(**client_args)


def create_vllm_client(api_key: str = "EMPTY", base_url: Optional[str] = None, timeout: Optional[float] = None) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The OpenAI Python SDK is required for the vLLM OpenAI-compatible client. "
            "Install it with `pip install openai` or `pip install -r requirements.txt` "
            "from cleaned_query_os."
        ) from exc

    client_args: Dict[str, Any] = {"api_key": api_key or "EMPTY"}
    if base_url:
        client_args["base_url"] = base_url
    if timeout:
        client_args["timeout"] = timeout
    return OpenAI(**client_args)


def create_llm_backend(
    *,
    provider: str = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    router_config: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    tracer: Optional[Any] = None,
) -> Any:
    provider_name = (provider or "openai").strip().lower()
    normalized_timeout = float(timeout) if timeout else None
    if provider_name == "vllm":
        cfg = router_config or {}
        endpoints = _load_vllm_endpoints(cfg, api_key=api_key, base_url=base_url, model=model)
        return VLLMRouterBackend(
            endpoints,
            strategy=str(cfg.get("strategy") or "least_inflight"),
            role_pools=cfg.get("role_pools") if isinstance(cfg.get("role_pools"), dict) else {},
            max_retries=int(cfg.get("max_retries", 2)),
            cooldown_seconds=float(cfg.get("cooldown_seconds", 30)),
            request_timeout_seconds=float(cfg.get("request_timeout_seconds", normalized_timeout or 0)) or None,
            tracer=tracer,
        )
    if provider_name != "openai":
        raise ValueError(f"unknown llm provider: {provider}")
    return OpenAIChatBackend(api_key=api_key, base_url=base_url, timeout=normalized_timeout, tracer=tracer)


def create_chat_completion(client: Any, role: str = "generic", **kwargs: Any) -> Any:
    if hasattr(client, "chat_completion"):
        return client.chat_completion(role=role, **kwargs)
    return _create_chat_completion_compatible(client, **kwargs)


def _create_chat_completion_compatible(client: Any, **kwargs: Any) -> Any:
    """Call Chat Completions with token-limit compatibility across models.

    Some newer chat models reject `max_tokens` and require
    `max_completion_tokens`. Keep the rest of the code using the older, clearer
    internal name and retry with the newer API parameter only when required.
    """
    request_kwargs = dict(kwargs)
    removed_optional_params = set()
    converted_max_tokens = False

    while True:
        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if (
                _unsupported_parameter(exc, "max_tokens")
                and "max_tokens" in request_kwargs
                and not converted_max_tokens
            ):
                request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens")
                converted_max_tokens = True
                continue

            # Some OpenAI-compatible servers, including some vLLM deployments,
            # reject newer optional OpenAI parameters while still supporting
            # tools. Remove only optional control parameters; never drop `tools`.
            removable = None
            for param in ("parallel_tool_calls", "tool_choice"):
                if (
                    param in request_kwargs
                    and param not in removed_optional_params
                    and _unsupported_parameter(exc, param)
                ):
                    removable = param
                    break
            if removable:
                request_kwargs.pop(removable, None)
                removed_optional_params.add(removable)
                continue
            raise


def _load_vllm_endpoints(
    cfg: Dict[str, Any],
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
) -> List[VLLMEndpointConfig]:
    raw_endpoints = cfg.get("endpoints")
    endpoints: List[VLLMEndpointConfig] = []
    if isinstance(raw_endpoints, list) and raw_endpoints:
        for idx, raw in enumerate(raw_endpoints, start=1):
            if not isinstance(raw, dict):
                raise ValueError("each llm_router endpoint must be a mapping/object")
            endpoint_base_url = str(raw.get("base_url") or "").strip()
            if not endpoint_base_url:
                raise ValueError("each vLLM endpoint requires base_url")
            endpoints.append(
                VLLMEndpointConfig(
                    name=str(raw.get("name") or f"vllm-{idx}"),
                    base_url=endpoint_base_url,
                    api_key=str(raw.get("api_key") or api_key or "EMPTY"),
                    model=str(raw.get("model") or model or ""),
                    weight=float(raw.get("weight", 1.0) or 1.0),
                    max_inflight=int(raw.get("max_inflight", 0) or 0),
                )
            )
    elif base_url:
        endpoints.append(
            VLLMEndpointConfig(
                name="vllm-default",
                base_url=base_url,
                api_key=api_key or "EMPTY",
                model=model,
                weight=1.0,
                max_inflight=int(cfg.get("max_inflight", 0) or 0),
            )
        )
    else:
        raise ValueError("vLLM provider requires either llm_router.endpoints or top-level base_url")
    return endpoints


def _get_shared_vllm_runtime(endpoint: VLLMEndpointConfig, timeout: Optional[float]) -> _EndpointRuntime:
    key = "|".join(
        [
            endpoint.name,
            endpoint.base_url,
            endpoint.api_key,
            endpoint.model or "",
            str(endpoint.weight),
            str(endpoint.max_inflight),
            str(timeout or ""),
        ]
    )
    with _VLLM_RUNTIME_POOL_LOCK:
        runtime = _VLLM_RUNTIME_POOL.get(key)
        if runtime is None:
            runtime = _EndpointRuntime(endpoint, timeout=timeout)
            _VLLM_RUNTIME_POOL[key] = runtime
        return runtime


def _normalize_role_pools(raw_role_pools: Dict[str, Any], endpoint_names: set[str]) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for role, raw_pool in raw_role_pools.items():
        if not isinstance(raw_pool, dict):
            continue
        names = raw_pool.get("endpoints") or []
        if isinstance(names, str):
            names = [names]
        valid_names = [str(name) for name in names if str(name) in endpoint_names]
        normalized[str(role)] = {
            "strategy": _normalize_strategy(str(raw_pool.get("strategy") or "")),
            "endpoints": valid_names,
        }
    return normalized


def _normalize_strategy(strategy: str) -> str:
    strategy = (strategy or "").strip().lower()
    if strategy in {"round_robin", "random", "least_inflight"}:
        return strategy
    return "least_inflight"


def safe_llm_error(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", text)
    return text


def is_auth_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return (
        "authentication" in name
        or "invalid_api_key" in text
        or "incorrect api key" in text
        or "401" in text
    )


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "insufficient_quota" in text
        or "exceeded your current quota" in text
        or "check your plan and billing" in text
    )


def is_fatal_llm_error(exc: Exception) -> bool:
    return is_auth_error(exc) or is_quota_error(exc)


def _unsupported_parameter(exc: Exception, parameter: str) -> bool:
    text = str(exc).lower()
    param = parameter.lower()
    return (
        ("unsupported parameter" in text and param in text)
        or ("unexpected keyword" in text and param in text)
        or ("extra inputs are not permitted" in text and param in text)
        or ("extra_forbidden" in text and param in text)
    )
