import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"

# Field names match RAGAS, so this file can be handed to the real library
# later without reshaping it.
REQUIRED_FIELDS = ("question", "ground_truth")


def load(path: Path = GOLDEN_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run `python generate_golden.py` to "
            f"bootstrap one from the indexed documents."
        )

    with path.open(encoding="utf-8") as handle:
        entries = json.load(handle)

    return [entry for entry in entries if is_valid(entry)]


def is_valid(entry) -> bool:
    return (
        isinstance(entry, dict)
        and all(entry.get(field) for field in REQUIRED_FIELDS)
    )


def save(entries: list[dict], path: Path = GOLDEN_PATH) -> int:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)

    return len(entries)


def make_entry(question: str, ground_truth: str, pages: list[int],
               document: str, chunk_id: str = None) -> dict:
    return {
        "question": question,
        "ground_truth": ground_truth,
        # Which pages a correct retrieval must surface
        "expected_pages": sorted(set(pages)),
        "document": document,
        "source_chunk": chunk_id,
    }
