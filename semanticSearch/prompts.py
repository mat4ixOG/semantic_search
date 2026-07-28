def build_prompts(question, hits):
    joined_context = "\n\n".join(hit["text"].strip() for hit in hits)

    return f"""You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present,
say "I don't know."

Context:
----------------

{joined_context}

----------------

Question:
{question}

Answer:"""
