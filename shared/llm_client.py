"""
Single shared wrapper around the Anthropic API.

Every LLM-powered module in this project (credit_risk/llm_features.py,
fraud/explain.py, agent/claude_agent.py) imports from here instead of calling
the API directly. That keeps model choice, retries, and JSON-parsing logic
in one place.

Requires the ANTHROPIC_API_KEY environment variable to be set. If it isn't,
`is_available()` returns False and callers should fall back gracefully
(see credit_risk/llm_features.py for the expected pattern) rather than
crashing — this lets the rest of the pipeline run and be reviewed even
without an API key configured.
"""

from __future__ import annotations

import json
import os
from typing import Any

MODEL = "claude-sonnet-4-6"


def is_available() -> bool:
    """Whether an API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    import anthropic  # imported lazily so the package is only required if used

    return anthropic.Anthropic()


def complete(prompt: str, system: str | None = None, max_tokens: int = 1000) -> str:
    """Send a single prompt, return the text of the response."""
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return "".join(block.text for block in response.content if block.type == "text")


def complete_json(prompt: str, system: str | None = None, max_tokens: int = 1000) -> dict:
    """
    Send a prompt that asks for a JSON object back, and parse it.

    The caller's prompt/system should explicitly instruct the model to
    respond with ONLY a JSON object and nothing else (no markdown fences,
    no preamble) — this function strips code fences defensively but a
    clear prompt is the main safeguard.
    """
    raw = complete(prompt, system=system, max_tokens=max_tokens)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {raw!r}") from exc
