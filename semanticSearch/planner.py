from datetime import date

import config
from llm import UTILITY_MODEL, ask_llm_json
from prompts import CRITIC_SCHEMA, critic_messages, planner_messages
from tools import SEARCH_TOOL, describe_tools, tool_names


def plan_schema() -> dict:
    """
    The tool name is an enum, so the planner cannot invent a tool that does
    not exist.
    """

    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": tool_names(),
                        },
                        "input": {
                            "type": "string",
                        },
                        "purpose": {
                            "type": "string",
                        },
                    },
                    "required": ["tool", "input", "purpose"],
                },
            },
        },
        "required": ["steps"],
    }


def fallback_plan(question: str) -> list[dict]:
    """
    What the planner should have said for an ordinary question.
    """

    return [
        {
            "tool": SEARCH_TOOL,
            "input": question,
            "purpose": "answer the question directly",
        },
    ]


# Wording that signals the question really does have several parts
MULTI_PART_MARKERS = (
    " and ",
    " also ",
    " versus ",
    " vs ",
    " compare",
    " difference",
    " both ",
    " then ",
    ";",
)


def looks_simple(question: str) -> bool:
    """
    A short single clause question decomposes into exactly the one search
    step plain retrieval would have run, so planning and criticising it costs
    two LLM calls and buys nothing.
    """

    normalised = f" {question.lower().strip()} "

    if len(question.split()) > config.AGENT_SIMPLE_MAX_WORDS:
        return False

    if question.count("?") > 1:
        return False

    return not any(marker in normalised for marker in MULTI_PART_MARKERS)


def make_plan(question: str, transcript: str = "") -> list[dict]:
    """
    Decompose the question into independent tool calls.
    Falls back to a single document search whenever the planner misbehaves.
    """

    messages = planner_messages(
        question,
        describe_tools(),
        config.AGENT_MAX_STEPS,
        date.today().isoformat(),
        transcript,
    )

    response = ask_llm_json(
        messages=messages,
        schema=plan_schema(),
        pref_model=UTILITY_MODEL,
        max_tokens=config.PLANNER_MAX_TOKENS,
    )

    if not response:
        return fallback_plan(question)

    steps = []
    known_tools = set(tool_names())

    for raw in response.get("steps") or []:
        if not isinstance(raw, dict):
            continue

        tool = (raw.get("tool") or "").strip()
        tool_input = (raw.get("input") or "").strip()

        if tool not in known_tools or not tool_input:
            continue

        steps.append({
            "tool": tool,
            "input": tool_input,
            "purpose": (raw.get("purpose") or "").strip(),
        })

    if not steps:
        return fallback_plan(question)

    return steps[:config.AGENT_MAX_STEPS]


def describe_findings(steps: list[dict], results: list[dict]) -> str:
    """
    The executed plan rendered for the critic. Retrieved text is summarised,
    not pasted, so the critic prompt stays small on a slow machine.
    """

    lines = []

    for number, (step, result) in enumerate(zip(steps, results), start=1):
        status = "ok" if result["ok"] else "FAILED"

        lines.append(
            f"Step {number} [{status}] {step['tool']}({step['input']!r})\n"
            f"  purpose: {step['purpose']}\n"
            f"  result: {result['output']}"
        )

    return "\n".join(lines)


def criticise(question: str, steps: list[dict], results: list[dict]) -> list[dict]:
    """
    Ask whether the executed plan covered the question. Returns the gaps
    worth retrying, each already validated against the plan.
    """

    if not config.AGENT_ALLOW_REPLAN:
        return []

    messages = critic_messages(question, describe_findings(steps, results))

    response = ask_llm_json(
        messages=messages,
        schema=CRITIC_SCHEMA,
        pref_model=UTILITY_MODEL,
        max_tokens=config.CRITIC_MAX_TOKENS,
    )

    if not response or response.get("complete"):
        return []

    gaps = []
    seen = set()

    for raw in response.get("gaps") or []:
        if not isinstance(raw, dict):
            continue

        number = raw.get("step")
        new_input = (raw.get("new_input") or "").strip()

        if not isinstance(number, int) or not 1 <= number <= len(steps):
            continue

        # Retrying with the input that just failed would burn a call to get
        # the same answer.
        if not new_input or new_input == steps[number - 1]["input"]:
            continue

        if number in seen:
            continue

        seen.add(number)
        gaps.append({
            "step": number,
            "reason": (raw.get("reason") or "").strip(),
            "new_input": new_input,
        })

    return gaps
