def build_messages(question, hits: list[dict], historyChats) -> list[dict]:
    context = "\n\n".join(
        hit["text"].strip()
        for hit in hits
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant.\n"
                "Answer ONLY using the provided context.\n"
                "If the answer is not present in the context, "
                "respond with 'I don't know.'"
            ),
        },

        *historyChats,

        {
            "role": "user",
            "content": (
                f"Context:\n"
                f"----------------\n"
                f"{context}\n"
                f"----------------\n\n"
                f"Question:\n"
                f"{question}"
            ),
        },
    ]

def ignore_messages(question:str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a routing classifier.\n"
                "Determine whether the user's question requires "
                "retrieving information from a document.\n\n"
                "Return exactly one word:\n"
                "YES -> Retrieval is required.\n"
                "NO -> Retrieval is NOT required.\n\n"
                "Examples:\n"
                "- 'What is the leave policy?' -> YES\n"
                "- 'Hello' -> NO\n"
                "- 'What was my last question?' -> NO\n"
                "- 'Repeat your previous answer.' -> NO\n"
                "- 'Summarise chapter 2.' -> YES"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

