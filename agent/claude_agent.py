"""
The agentic EDA layer: Claude is given a dataset and a general instruction,
and decides for itself which profiling tools to call and in what order,
via the Anthropic tool-use API. This is the "AI" component of the EDA
module — profiler.py does the actual computation, this file does the
planning.

Requires ANTHROPIC_API_KEY. If it isn't set, run() falls back to calling
every profiler tool once in a fixed order, so the module is still runnable
without a live key (see the note printed in that case).

Usage:
    python agent/claude_agent.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

from agent.profiler import PROFILER_TOOLS
from shared import llm_client

TOOL_DEFINITIONS = [
    {
        "name": "missingness_report",
        "description": "Returns missing-value counts and percentages per column.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "numeric_summary",
        "description": "Returns descriptive statistics (mean, std, quartiles) for numeric columns.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "categorical_summary",
        "description": "Returns the most frequent values for each categorical column.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "correlation_matrix",
        "description": "Returns numeric column pairs with correlation above a threshold.",
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "description": "Minimum absolute correlation to include, e.g. 0.5"}
            },
        },
    },
]

SYSTEM_PROMPT = """You are a data science agent. You have tools available \
to profile a dataset. Decide which tools are useful to understand this \
dataset's quality and structure, call them, then write a short (4-6 \
sentence) plain-English summary of what you found: data quality issues, \
notable distributions, and any strong correlations worth flagging. Don't \
call every tool if it isn't useful — use judgment."""


def _fallback_run(df: pd.DataFrame) -> str:
    """Calls every tool once in a fixed order — used only without an API key."""
    results = {
        "missingness_report": PROFILER_TOOLS["missingness_report"](df),
        "numeric_summary": PROFILER_TOOLS["numeric_summary"](df),
        "categorical_summary": PROFILER_TOOLS["categorical_summary"](df),
        "correlation_matrix": PROFILER_TOOLS["correlation_matrix"](df, threshold=0.5),
    }
    return (
        "[No ANTHROPIC_API_KEY set — ran all profiling tools directly instead of "
        "letting the agent plan which to use.]\n\n" + json.dumps(results, indent=2, default=str)
    )


def run(df: pd.DataFrame, max_turns: int = 6) -> str:
    if not llm_client.is_available():
        return _fallback_run(df)

    client = llm_client._get_client()
    messages = [
        {
            "role": "user",
            "content": f"Here is a dataset with columns: {list(df.columns)} and {len(df)} rows. Profile it.",
        }
    ]

    for _ in range(max_turns):
        response = client.messages.create(
            model=llm_client.MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = PROFILER_TOOLS[block.name]
            result = fn(df, **block.input) if block.input else fn(df)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "Agent did not finish within max_turns — consider raising the limit."


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "samples", "credit_risk_sample.csv"))
    print(run(df))
