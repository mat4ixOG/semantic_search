from query import search

QUESTION = "How many annual leaves does an employee get?"

results = search(QUESTION)

print(f"Q: {QUESTION}\n")

if not results:
    print("No results — did you run `python ingest.py` first?")

for i, hit in enumerate(results, start=1):
    print(f"[{i}] {hit['document']} p.{hit['page']}  (distance {hit['distance']:.4f})")
    print(hit["text"].strip()[:400])
    print()
