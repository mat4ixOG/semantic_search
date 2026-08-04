import config


class ConversationMemory:
    def __init__(self , max_messages=None , history=None):
        self.max_messages = max_messages or config.MEMORY_MAX_MESSAGES
        self.history = list(history or [])
        self._trim()

    def add_user_message(self , message):
        self.history.append({
            "role" : "user",
            "content" : message
        })
        self._trim()

    def add_assistant_message(self , message):
        self.history.append({
            "role" : "assistant",
            "content" : message
        })
        self._trim()

    def add_conversation_message(self , question , answer):
        self.history.extend([
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ])
        self._trim()

    def get_history(self):
        return self.history.copy()

    def as_transcript(self, max_messages=6, max_chars=300):
        """
        The recent conversation as plain text, for prompts that reason ABOUT
        the conversation (routing, query rewriting) instead of continuing it.
        """

        if not self.history:
            return "(no conversation yet)"

        labels = {
            "user": "User",
            "assistant": "Assistant",
        }

        lines = []
        for message in self.history[-max_messages:]:
            content = message["content"].strip()

            if len(content) > max_chars:
                content = content[:max_chars].rstrip() + "..."

            lines.append(f"{labels[message['role']]}: {content}")

        return "\n".join(lines)

    def _trim(self):
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]