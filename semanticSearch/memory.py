class ConversationMemory:
    def __init__(self , max_messages=10):
        self.history = []
        self.max_messages = max_messages

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

    def _trim(self):
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]