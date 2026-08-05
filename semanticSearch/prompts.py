JSON_ANSWER_RULES = (
    "\n\n"
    "Return a JSON object with exactly these keys:\n"
    '- "answer": your answer as a plain string.\n'
    '- "answered": true if you could answer, false otherwise.\n'
    '- "sources": the numbers of the excerpts you used, as a list of '
    "integers. Use an empty list when you used none."
)


def build_messages(question, hits: list[dict], historyChats, json_mode=True,
                   tool_notes: list[str] = None) -> list[dict]:
    context = "\n\n".join(
        f"[{number}] {hit['text'].strip()}"
        for number, hit in enumerate(hits, start=1)
    )

    if not context:
        context = "(no context available)"

    instructions = (
        "You are a helpful AI assistant.\n"
        "Answer ONLY using the provided context.\n"
        "Every context excerpt is prefixed with a number like [1].\n"
        "If the answer is not present in the context, "
        "respond with 'I don't know.'"
    )

    findings = ""

    if tool_notes:
        instructions += (
            "\nTool results are computed facts, trust them over the "
            "excerpts for dates and arithmetic."
        )

        joined = "\n".join(f"- {note}" for note in tool_notes)

        findings = (
            f"Tool results:\n"
            f"----------------\n"
            f"{joined}\n"
            f"----------------\n\n"
        )

    if json_mode:
        instructions += JSON_ANSWER_RULES

    return [
        {
            "role": "system",
            "content": instructions,
        },

        *historyChats,

        {
            "role": "user",
            "content": (
                f"Context:\n"
                f"----------------\n"
                f"{context}\n"
                f"----------------\n\n"
                f"{findings}"
                f"Question:\n"
                f"{question}"
            ),
        },
    ]


def build_chat_messages(question, historyChats, json_mode=True) -> list[dict]:
    """
    Used when the router decided the question needs no document lookup,
    so the answer comes from the conversation itself.
    """

    instructions = (
        "You are a helpful AI assistant.\n"
        "Answer using the conversation so far.\n"
        "Do not invent facts about any document."
    )

    if json_mode:
        instructions += JSON_ANSWER_RULES + (
            "\n\nThere are no excerpts here, so \"sources\" is always an "
            "empty list."
        )

    return [
        {
            "role": "system",
            "content": instructions,
        },

        *historyChats,

        {
            "role": "user",
            "content": question,
        },
    ]


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        },
        "answered": {
            "type": "boolean",
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "integer",
            },
        },
    },
    "required": ["answer", "answered", "sources"],
}


DOCUMENT_ROUTE = "document"
MEMORY_ROUTE = "memory"
DIRECT_ROUTE = "direct"

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": [DOCUMENT_ROUTE, MEMORY_ROUTE, DIRECT_ROUTE],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["route", "reason"],
}


def route_messages(question: str, transcript: str) -> list[dict]:
    """
    Three way routing. The conversation so far is handed to the classifier as
    plain text, so it can tell "what did I just ask?" (memory) apart from
    "what does the handbook say?" (document).
    """

    return [
        {
            "role": "system",
            "content": (
                "Route the question to where its answer comes from.\n\n"
                f"- {DOCUMENT_ROUTE}: needs the indexed documents. Any "
                "question about facts, policies, definitions, procedures or "
                "content, including follow ups asking for NEW facts. Use "
                "this whenever you are unsure.\n"
                f"- {MEMORY_ROUTE}: already answered in the conversation "
                "above. Asking what was said, or to repeat, shorten or "
                "rephrase your last answer.\n"
                f"- {DIRECT_ROUTE}: greetings, thanks, small talk.\n\n"
                "Resolve pronouns against the conversation first: 'how many "
                f"days of it do I get' is {DOCUMENT_ROUTE}, not "
                f"{MEMORY_ROUTE}.\n\n"
                "Give the route and a short reason."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n"
                f"----------------\n"
                f"{transcript}\n"
                f"----------------\n\n"
                f"Question to route:\n"
                f"{question}"
            ),
        },
    ]


QUERY_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten": {
            "type": "string",
        },
        "variations": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": ["rewritten", "variations"],
}


def query_expansion_messages(question: str, count: int, transcript: str = "") -> list[dict]:
    """
    Rewrite and multi query in a single call.

    Both stages read the same question and produce the same kind of output,
    so splitting them cost a whole extra round trip for nothing.
    """

    context_rule = ""

    if transcript:
        context_rule = (
            "- Resolve pronouns and follow ups using the conversation.\n"
        )

    return [
        {
            "role": "system",
            "content": (
                "You are an expert search query optimizer.\n"
                "Turn the user's question into queries that maximize "
                "document retrieval.\n\n"
                '"rewritten": one improved version of the question.\n'
                "- Preserve the original meaning.\n"
                f"{context_rule}"
                "- Make it standalone, it must make sense on its own.\n"
                "- Expand abbreviations if appropriate.\n"
                "- Use terminology likely to appear in documents.\n"
                "- Keep it concise.\n\n"
                f'"variations": {count} other search queries for the same '
                "information.\n"
                "- Each one approaches it from a different angle.\n"
                "- Vary the wording, the synonyms and the level of detail.\n\n"
                "Never answer the question, only rewrite it."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n"
                f"----------------\n"
                f"{transcript}\n"
                f"----------------\n\n"
                f"Question:\n"
                f"{question}"
                if transcript
                else question
            ),
        },
    ]


def hyde_messages(question: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You write short passages that look like they were taken "
                "from a reference document.\n"
                "Write the passage that would answer the user's question.\n\n"
                "Rules:\n"
                "- Write in the neutral, factual tone of a manual or "
                "a textbook.\n"
                "- Use the vocabulary such a document would use.\n"
                "- Keep it to one paragraph of about three or four sentences.\n"
                "- Invent plausible specifics instead of refusing.\n"
                "- Do NOT mention the question, yourself or your uncertainty.\n"
                "- Return ONLY the passage."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def planner_messages(question: str, tool_catalog: str, max_steps: int,
                     today: str, transcript: str = "") -> list[dict]:
    """
    Break a question into tool calls. The plan is made once, up front, so the
    steps must not depend on each other's output.
    """

    return [
        {
            "role": "system",
            "content": (
                "You plan how to answer a question about a set of indexed "
                "documents.\n"
                f"Today is {today}.\n\n"
                "Available tools:\n"
                f"{tool_catalog}\n\n"
                "Rules:\n"
                f"- Use at most {max_steps} steps, and as few as possible.\n"
                "- A simple question needs exactly one search_documents "
                "step.\n"
                "- Split into several steps only when the question really "
                "asks for several different things.\n"
                "- Steps run independently and all at once, so a step can "
                "NEVER use the result of another step. Write concrete "
                "inputs, never placeholders.\n"
                "- Anything about what the documents say must go through "
                "search_documents. You do not know their contents.\n"
                "- Match each input to the tool's input format exactly.\n"
                "- \"purpose\" says what the step contributes, in a few "
                "words.\n\n"
                "Return the steps as JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n"
                f"----------------\n"
                f"{transcript}\n"
                f"----------------\n\n"
                f"Question:\n"
                f"{question}"
                if transcript
                else question
            ),
        },
    ]


CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "complete": {
            "type": "boolean",
        },
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {
                        "type": "integer",
                    },
                    "reason": {
                        "type": "string",
                    },
                    "new_input": {
                        "type": "string",
                    },
                },
                "required": ["step", "reason", "new_input"],
            },
        },
    },
    "required": ["complete", "gaps"],
}


def critic_messages(question: str, findings: str) -> list[dict]:
    """
    One round of self correction: did the executed plan actually gather what
    the question needs, and if not, which step should be retried with what.
    """

    return [
        {
            "role": "system",
            "content": (
                "You check whether an executed plan gathered enough to "
                "answer a question.\n\n"
                "Set \"complete\" to true when the evidence covers the "
                "question, even partially. Only report a gap when a step "
                "clearly failed or came back with nothing relevant.\n\n"
                "For each gap give:\n"
                '- "step": the number of the step that failed.\n'
                '- "reason": what is missing, in a few words.\n'
                '- "new_input": a better input for that same tool. Rephrase '
                "with different wording, do not repeat the failed input.\n\n"
                "Report no gaps when the evidence is good enough. Retrying "
                "costs more than a slightly thin answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n"
                f"{question}\n\n"
                f"Executed plan:\n"
                f"----------------\n"
                f"{findings}\n"
                f"----------------"
            ),
        },
    ]


GOLDEN_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
        },
        "question": {
            "type": "string",
        },
        "answer": {
            "type": "string",
        },
    },
    "required": ["answerable", "question", "answer"],
}


def golden_question_messages(excerpt: str) -> list[dict]:
    """
    Bootstrap an evaluation pair from a real chunk, so the expected page is
    known without anyone labelling it by hand.
    """

    return [
        {
            "role": "system",
            "content": (
                "You write evaluation questions for a document search "
                "system.\n\n"
                "Read the excerpt and write ONE question a real user would "
                "ask, that this excerpt fully answers.\n\n"
                "Rules:\n"
                "- The question must stand on its own. Never write 'in this "
                "excerpt' or 'according to the passage'.\n"
                "- Ask about the substance, not about formatting or page "
                "numbers.\n"
                "- The answer must be short and taken from the excerpt.\n"
                "- If the excerpt is a heading, a table of contents or is "
                "too fragmentary to answer anything, set answerable to "
                "false and leave the other fields empty."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Excerpt:\n"
                f"----------------\n"
                f"{excerpt}\n"
                f"----------------"
            ),
        },
    ]


FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims_total": {
            "type": "integer",
        },
        "claims_supported": {
            "type": "integer",
        },
        "unsupported": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": ["claims_total", "claims_supported", "unsupported"],
}


def faithfulness_messages(answer: str, context: str) -> list[dict]:
    """
    Hallucination check: is every claim in the answer actually backed by the
    retrieved context.
    """

    return [
        {
            "role": "system",
            "content": (
                "You audit an answer against the context it was supposed to "
                "come from.\n\n"
                "Break the answer into individual factual claims, then "
                "count how many are supported by the context.\n\n"
                "Rules:\n"
                "- A claim is supported only if the context states it. "
                "Plausible is not supported.\n"
                "- Ignore filler like 'according to the document'.\n"
                "- List the unsupported claims verbatim.\n"
                "- An answer of 'I don't know' has zero claims."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n"
                f"----------------\n"
                f"{context}\n"
                f"----------------\n\n"
                f"Answer:\n"
                f"{answer}"
            ),
        },
    ]


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["correct", "partially_correct", "incorrect"],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["verdict", "reason"],
}


def correctness_messages(question: str, answer: str, expected: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You grade an answer against a reference answer.\n\n"
                '- "correct": same meaning as the reference, even if worded '
                "differently or with extra detail.\n"
                '- "partially_correct": some of the reference is there, but '
                "something important is missing or wrong.\n"
                '- "incorrect": contradicts the reference, or fails to '
                "answer, including 'I don't know'.\n\n"
                "Grade the meaning, not the wording."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Reference answer:\n{expected}\n\n"
                f"Given answer:\n{answer}"
            ),
        },
    ]


RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["score", "reason"],
}


def relevance_messages(question: str, answer: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You score how well an answer addresses the question that "
                "was asked. Judge relevance only, not whether it is true.\n\n"
                "5 - answers exactly what was asked, nothing off topic\n"
                "4 - answers it, with some padding\n"
                "3 - partially addresses it\n"
                "2 - mostly off topic\n"
                "1 - does not address it at all, or refuses"
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nAnswer:\n{answer}",
        },
    ]


def compression_messages(question: str, chunk: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You extract the parts of a document excerpt that help "
                "answer a question.\n\n"
                "Rules:\n"
                "- Copy the relevant sentences VERBATIM from the excerpt.\n"
                "- Never rewrite, summarise or add anything of your own.\n"
                "- Drop everything that does not help answer the question.\n"
                "- If nothing in the excerpt is relevant, return exactly "
                "NOT_RELEVANT.\n"
                "- Return ONLY the extracted sentences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n"
                f"{question}\n\n"
                f"Excerpt:\n"
                f"----------------\n"
                f"{chunk}\n"
                f"----------------"
            ),
        },
    ]
