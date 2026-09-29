"""Turn dictated lines into one ready-to-paste prompt.

The verbatim lines stay on the card. This call only produces the clean text
the copy button uses. Clips never come through here.
"""

from __future__ import annotations

from typing import Any

from .calibration_policy import OWNER_APPROVED_MODEL
from .judges import APPROVED_GEMINI_BASE_URL, JudgeError, _load_json_response, pinned_gemini_client


CLEAN_PROMPT_INSTRUCTIONS = """Turn these dictated meeting lines into one clean prompt that can be pasted into an AI tool.

Keep the intent only. Remove filler, hesitation, greetings, names of bystanders, and asides.
The lines may be consecutive pieces of one instruction. Include every action that is actually present.
Do not add requirements that were not asked for.
Write the prompt in the language the speaker used: Swiss German or Standard German becomes Standard German (Hochdeutsch, Swiss spelling with "ss" instead of "ß"); English stays English. Only switch language when the speaker clearly asks for it.
Return JSON {"prompt": "<the prompt>"} and nothing else.
"""


def clean_prompt_text(
    verbatim: str,
    *,
    api_key: str | None = None,
    client: Any | None = None,
    types_module: Any | None = None,
    model: str = OWNER_APPROVED_MODEL,
    timeout_seconds: float = 30,
) -> str:
    """Return a clean prompt. Raises ``JudgeError`` when the model answer is unusable."""
    text = verbatim.strip()
    if not text:
        raise JudgeError("clean prompt needs verbatim lines")
    if client is None:
        client, types_module = pinned_gemini_client(api_key)
    schema = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    }
    prompt = CLEAN_PROMPT_INSTRUCTIONS + "\nDictated lines:\n" + text
    last: Exception | None = None
    for _attempt in range(2):
        try:
            return _clean_once(client, types_module, model, prompt, schema, timeout_seconds)
        except JudgeError as exc:
            last = exc
    assert last is not None
    raise last


def _clean_once(client, types_module, model, prompt, schema, timeout_seconds: float) -> str:
    if types_module is None:
        config: Any = {
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": schema,
            "http_options": {
                "base_url": APPROVED_GEMINI_BASE_URL,
                "api_version": "v1beta",
                "timeout": int(timeout_seconds * 1000),
            },
        }
    else:
        config = types_module.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1024,
            thinking_config=types_module.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=schema,
            http_options=types_module.HttpOptions(
                base_url=APPROVED_GEMINI_BASE_URL,
                api_version="v1beta",
                timeout=int(timeout_seconds * 1000),
            ),
        )
    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
    except Exception as exc:
        raise JudgeError(f"clean prompt request failed: {type(exc).__name__}") from exc
    payload = _load_json_response(response)
    cleaned = payload.get("prompt")
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise JudgeError("clean prompt response was empty")
    return cleaned.strip()
