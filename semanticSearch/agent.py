import config
from planner import criticise, make_plan
from prompts import DOCUMENT_ROUTE
from retrieval import empty_trace
from router import EMPTY_TRANSCRIPT, route
from tools import SEARCH_TOOL, run_tool


def run_agent(question: str, transcript: str = EMPTY_TRANSCRIPT) -> tuple[list[dict], list[str], dict]:
    """
    Plan and execute, with one round of self correction.

    route -> plan -> execute every step -> critic -> re-execute only the
    steps the critic flagged -> collect evidence

    Returns (hits, tool_notes, trace). The answer itself is written by the
    responder from that evidence.
    """

    decision = route(question, transcript)

    trace = empty_trace()
    trace["route"] = decision["route"]
    trace["route_reason"] = decision["reason"]
    trace["llm_calls"] = 1
    trace["agent"] = True
    trace["plan"] = []
    trace["steps"] = []
    trace["replanned"] = []

    if decision["route"] != DOCUMENT_ROUTE:
        return [], [], trace

    trace["used"] = True

    steps = make_plan(question, transcript)
    trace["llm_calls"] += 1
    trace["plan"] = [dict(step) for step in steps]

    # Tools report their own LLM usage back through here, since a
    # search_documents step runs the whole retrieval pipeline.
    context = {
        "transcript": transcript,
        "llm_calls": 0,
    }

    results = [
        run_tool(step["tool"], step["input"], context)
        for step in steps
    ]

    gaps = criticise(question, steps, results)

    if config.AGENT_ALLOW_REPLAN:
        trace["llm_calls"] += 1

    for gap in gaps:
        index = gap["step"] - 1
        step = steps[index]

        retry = run_tool(step["tool"], gap["new_input"], context)

        trace["replanned"].append({
            "step": gap["step"],
            "reason": gap["reason"],
            "old_input": step["input"],
            "new_input": gap["new_input"],
            "recovered": retry["ok"],
        })

        # Keep whichever attempt actually produced something.
        if retry["ok"] or not results[index]["ok"]:
            results[index] = retry
            steps[index] = {**step, "input": gap["new_input"]}

    trace["llm_calls"] += context["llm_calls"]

    hits = collect_hits(results)
    tool_notes = collect_notes(steps, results)

    trace["steps"] = [
        {
            "tool": step["tool"],
            "input": step["input"],
            "purpose": step["purpose"],
            "ok": result["ok"],
            "output": result["output"],
        }
        for step, result in zip(steps, results)
    ]
    trace["reranked"] = len(hits)

    return hits, tool_notes, trace


def collect_hits(results: list[dict]) -> list[dict]:
    """
    Merge the chunks every search step found, keeping the best score when two
    steps land on the same chunk.
    """

    best = {}

    for result in results:
        for hit in result["hits"]:
            existing = best.get(hit["id"])

            if existing is None:
                best[hit["id"]] = hit
                continue

            if hit.get("rerank_score", 0) > existing.get("rerank_score", 0):
                best[hit["id"]] = hit

    merged = sorted(
        best.values(),
        key=lambda hit: hit.get("rerank_score", 0),
        reverse=True,
    )

    return merged[:config.AGENT_MAX_CHUNKS]


def collect_notes(steps: list[dict], results: list[dict]) -> list[str]:
    """
    Output of the non search tools, which is a computed fact rather than a
    document excerpt and so is handed to the answer separately.
    """

    notes = []

    for step, result in zip(steps, results):
        if step["tool"] == SEARCH_TOOL or not result["ok"]:
            continue

        notes.append(f"{step['tool']}: {result['output']}")

    return notes
