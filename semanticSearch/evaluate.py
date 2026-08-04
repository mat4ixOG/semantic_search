import argparse
import json
import time
from pathlib import Path

import config
import golden_set
import metrics
from agent import run_agent
from responder import answer_question
from retrieval import retrieve
from router import EMPTY_TRANSCRIPT

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"

# Each variant is a set of config overrides. Only pipeline flags belong here:
# the model names are read at import time, so overriding them would not take
# effect.
VARIANTS = {
    "baseline": {
        "USE_QUERY_REWRITE": False,
        "USE_MULTI_QUERY": False,
        "USE_HYDE": False,
        "USE_COMPRESSION": False,
        "USE_AGENT": False,
    },
    "rewrite": {
        "USE_QUERY_REWRITE": True,
        "USE_MULTI_QUERY": False,
        "USE_HYDE": False,
        "USE_COMPRESSION": False,
        "USE_AGENT": False,
    },
    "multi_query": {
        "USE_QUERY_REWRITE": True,
        "USE_MULTI_QUERY": True,
        "USE_HYDE": False,
        "USE_COMPRESSION": False,
        "USE_AGENT": False,
    },
    "hyde": {
        "USE_QUERY_REWRITE": True,
        "USE_MULTI_QUERY": True,
        "USE_HYDE": True,
        "USE_COMPRESSION": False,
        "USE_AGENT": False,
    },
    "compression": {
        "USE_QUERY_REWRITE": True,
        "USE_MULTI_QUERY": True,
        "USE_HYDE": True,
        "USE_COMPRESSION": True,
        "USE_AGENT": False,
    },
    "agent": {
        "USE_QUERY_REWRITE": True,
        "USE_MULTI_QUERY": True,
        "USE_HYDE": True,
        "USE_COMPRESSION": True,
        "USE_AGENT": True,
    },
    # Whatever config.py currently says
    "current": {},
}

REPORT_COLUMNS = [
    ("hit_rate", "hit@k"),
    ("mrr", "mrr"),
    ("context_precision", "ctx_prec"),
    ("context_recall", "ctx_rec"),
    ("similarity", "sim"),
    ("faithfulness", "faith"),
    ("correctness", "correct"),
    ("answer_relevance", "relevance"),
    ("llm_calls", "calls"),
    ("seconds", "secs"),
]


def apply_overrides(overrides: dict) -> dict:
    previous = {key: getattr(config, key) for key in overrides}

    for key, value in overrides.items():
        setattr(config, key, value)

    return previous


def answer_once(question: str) -> tuple[dict, list[dict], dict]:
    """
    One question through the pipeline, with no conversation history so that
    entries stay independent of each other.
    """

    if config.USE_AGENT:
        hits, tool_notes, trace = run_agent(question, EMPTY_TRANSCRIPT)
    else:
        hits, trace = retrieve(question, EMPTY_TRANSCRIPT)
        tool_notes = []

    answer = answer_question(question, hits, [], tool_notes)

    return answer, hits, trace


def score_entry(entry: dict, judge: bool) -> dict:
    started = time.perf_counter()
    answer, hits, trace = answer_once(entry["question"])
    elapsed = time.perf_counter() - started

    record = {
        "question": entry["question"],
        "ground_truth": entry["ground_truth"],
        "answer": answer["answer"],
        "answered": answer["answered"],
        "expected_pages": entry.get("expected_pages", []),
        "retrieved_pages": [hit["page"] for hit in hits],
        # +1 for the answer call itself
        "llm_calls": trace["llm_calls"] + 1,
        "seconds": round(elapsed, 2),
    }

    record.update(metrics.retrieval_metrics(hits, entry.get("expected_pages")))
    record["similarity"] = metrics.semantic_similarity(
        answer["answer"],
        entry["ground_truth"],
    )

    if judge:
        record.update(metrics.faithfulness(answer["answer"], hits))
        record.update(metrics.correctness(
            entry["question"],
            answer["answer"],
            entry["ground_truth"],
        ))
        record.update(metrics.relevance(entry["question"], answer["answer"]))

    return record


def summarise(records: list[dict]) -> dict:
    summary = {"questions": len(records)}

    for key, _ in REPORT_COLUMNS:
        summary[key] = metrics.average(record.get(key) for record in records)

    return summary


def run_variant(name: str, entries: list[dict], judge: bool) -> dict:
    overrides = VARIANTS[name]
    previous = apply_overrides(overrides)

    print(f"\n=== {name} ===")
    if overrides:
        print("  " + "  ".join(f"{key}={value}" for key, value in overrides.items()))

    records = []

    try:
        for number, entry in enumerate(entries, start=1):
            print(f"  [{number}/{len(entries)}] {entry['question'][:60]}",
                  end=" ", flush=True)

            record = score_entry(entry, judge)
            records.append(record)

            print(f"({record['seconds']}s, {record['llm_calls']} calls)")
    except KeyboardInterrupt:
        print("\n  interrupted, scoring what finished")
    finally:
        apply_overrides(previous)

    return {
        "variant": name,
        "overrides": overrides,
        "summary": summarise(records),
        "records": records,
    }


def format_cell(value) -> str:
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


def print_report(results: list[dict]):
    headers = ["variant", "n"] + [label for _, label in REPORT_COLUMNS]
    rows = [headers]

    for result in results:
        summary = result["summary"]

        rows.append([
            result["variant"],
            str(summary["questions"]),
            *[format_cell(summary[key]) for key, _ in REPORT_COLUMNS],
        ])

    widths = [
        max(len(row[column]) for row in rows)
        for column in range(len(headers))
    ]

    print()
    for index, row in enumerate(rows):
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))

        if index == 0:
            print("-" * (sum(widths) + 2 * len(widths)))


def main():
    parser = argparse.ArgumentParser(
        description="Score the retrieval pipeline against the golden set.",
    )
    parser.add_argument("--variants", default="current",
                        help=f"comma separated, or `all`. "
                             f"available: {', '.join(VARIANTS)}")
    parser.add_argument("--limit", type=int,
                        help="only score the first N questions")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM graded metrics, keeps it fast")
    parser.add_argument("--out", default=str(RESULTS_PATH),
                        help="where to write the full results")

    args = parser.parse_args()

    entries = golden_set.load()

    if args.limit:
        entries = entries[:args.limit]

    if not entries:
        print("golden set is empty")
        return

    if args.variants == "all":
        names = list(VARIANTS)
    else:
        names = [name.strip() for name in args.variants.split(",") if name.strip()]

    unknown = [name for name in names if name not in VARIANTS]

    if unknown:
        parser.error(f"unknown variant(s): {', '.join(unknown)}")

    judge = not args.no_judge

    print(f"{len(entries)} question(s), {len(names)} variant(s), "
          f"judge {'on' if judge else 'off'}")

    results = [run_variant(name, entries, judge) for name in names]

    print_report(results)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"\nfull results written to {args.out}")


if __name__ == "__main__":
    main()
