from __future__ import annotations

from typing import Any, Dict

from mistake_prompts import build_system_prompt, build_user_prompt
from query_os.llm import create_chat_completion
from text_utils import parse_json_object, sanitize_output_obj


def extract_and_route_record(
    *,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    record: Dict[str, Any],
    taxonomy: Dict[str, Any],
    active_preview_limit: int,
    proposed_preview_limit: int,
    record_max_chars: int,
) -> Dict[str, Any]:
    response = create_chat_completion(
        client,
        role="general_mistake_extractor",
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(
                    record=record,
                    taxonomy=taxonomy,
                    active_preview_limit=active_preview_limit,
                    proposed_preview_limit=proposed_preview_limit,
                    record_max_chars=record_max_chars,
                ),
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return sanitize_output_obj(parse_json_object(content))

