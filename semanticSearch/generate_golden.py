import argparse
import random

import golden_set
from llm import MODEL_NAME, ask_llm_json
from prompts import GOLDEN_QUESTION_SCHEMA, golden_question_messages
from vector_store import get_collection

# Too short to hold a real answer, usually a heading or a page number
MIN_EXCERPT_LENGTH = 200


def sample_chunks(count: int, seed: int) -> list[dict]:
    """
    Spread the sample across the corpus instead of taking the first N chunks,
    which would only ever test chapter one.
    """

    stored = get_collection().get()

    candidates = [
        {
            "id": chunk_id,
            "text": text,
            "page": metadata["page"],
            "document": metadata["document"],
        }
        for chunk_id, text, metadata in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"],
        )
        if len(text.strip()) >= MIN_EXCERPT_LENGTH
    ]

    if not candidates:
        return []

    candidates.sort(key=lambda chunk: (chunk["document"], chunk["page"]))

    if count >= len(candidates):
        return candidates

    # Even stride across the sorted corpus, jittered so reruns with a
    # different seed pick different chunks.
    rng = random.Random(seed)
    stride = len(candidates) / count

    picked = []
    for index in range(count):
        position = int(index * stride + rng.random() * stride)
        picked.append(candidates[min(position, len(candidates) - 1)])

    return picked


def generate(count: int, seed: int) -> list[dict]:
    chunks = sample_chunks(count, seed)

    if not chunks:
        print("no chunks in the collection, run `python ingest.py` first")
        return []

    entries = []

    for number, chunk in enumerate(chunks, start=1):
        print(f"[{number}/{len(chunks)}] page {chunk['page']}", end=" ", flush=True)

        response = ask_llm_json(
            messages=golden_question_messages(chunk["text"]),
            schema=GOLDEN_QUESTION_SCHEMA,
            # The main model, a 1b would write sloppy questions and the
            # whole point of a golden set is that it is trustworthy.
            pref_model=MODEL_NAME,
        )

        if not response or not response.get("answerable"):
            print("skipped")
            continue

        question = (response.get("question") or "").strip()
        answer = (response.get("answer") or "").strip()

        if not question or not answer:
            print("skipped")
            continue

        print(f"-> {question[:60]}")

        entries.append(golden_set.make_entry(
            question=question,
            ground_truth=answer,
            pages=[chunk["page"]],
            document=chunk["document"],
            chunk_id=chunk["id"],
        ))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap an evaluation set from the indexed documents.",
    )
    parser.add_argument("-n", "--count", type=int, default=30,
                        help="how many chunks to sample (default 30)")
    parser.add_argument("--seed", type=int, default=7,
                        help="sampling seed")
    parser.add_argument("--append", action="store_true",
                        help="add to the existing golden set instead of replacing it")

    args = parser.parse_args()

    entries = generate(args.count, args.seed)

    if not entries:
        return

    if args.append and golden_set.GOLDEN_PATH.exists():
        entries = golden_set.load() + entries

    saved = golden_set.save(entries)

    print(f"\nwrote {saved} entries to {golden_set.GOLDEN_PATH.name}")
    print("read them before trusting them, generated questions are a draft")


if __name__ == "__main__":
    main()
