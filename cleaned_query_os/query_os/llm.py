from __future__ import annotations

import re
import os
from typing import Any, Dict, Optional


def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Any:
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
    elif base_url and not os.environ.get("OPENAI_API_KEY"):
        # OpenAI-compatible local servers such as vLLM often ignore the API key,
        # but the OpenAI SDK still requires a non-empty value.
        client_args["api_key"] = "EMPTY"
    if base_url:
        client_args["base_url"] = base_url
    return OpenAI(**client_args)


def create_chat_completion(client: Any, **kwargs: Any) -> Any:
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
