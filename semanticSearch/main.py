import json

import config
import service
from memory import ConversationMemory

COMMANDS = {
    "json", "json on", "json off",
    "agent", "agent on", "agent off",
    "fast", "fast on", "fast off",
}


class Session:
    """
    CLI state. Kept in an object so nothing happens at import time and the
    module can be imported by tests without starting a REPL.
    """

    def __init__(self):
        self.memory = ConversationMemory()
        self.json_output = config.JSON_OUTPUT
        self.agent_mode = config.USE_AGENT
        self.fast_mode = False

    def banner(self):
        print("------ ENTERING WORLD OF AI-------")
        print("------ ACTIVATING NEURAL NETWORKS -------")
        print("------ JARVIS AT YOUR SERVICE-------")
        print("commands: `json on|off` | `agent on|off` | `fast on|off` | `exit`")
        self.print_modes()

    def print_modes(self):
        model = config.FAST_MODEL if self.fast_mode else config.MODEL_NAME

        print(f"output: {'json' if self.json_output else 'text'}"
              f"   agent: {'on' if self.agent_mode else 'off'}"
              f"   answers: {model}")

    def handle_command(self, command):
        """
        Returns True when the input was a command and not a question.
        """

        if command not in COMMANDS:
            return False

        name, _, state = command.partition(" ")
        value = state == "on" if state else None

        if name == "json":
            self.json_output = not self.json_output if value is None else value
        elif name == "agent":
            self.agent_mode = not self.agent_mode if value is None else value
        else:
            self.fast_mode = not self.fast_mode if value is None else value

        self.print_modes()

        return True

    def ask_json(self, question):
        payload = service.ask(
            question,
            self.memory.get_history(),
            self.agent_mode,
            self.fast_mode,
        )

        print(json.dumps(payload, indent=2))

        return payload["answer"]

    def ask_streaming(self, question):
        answer = ""

        for event in service.ask_stream(
                question,
                self.memory.get_history(),
                self.agent_mode,
                self.fast_mode,
        ):
            if event["type"] == "retrieval":
                self.print_trace(event["trace"])
                print("AI: ", end="", flush=True)
            elif event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "done":
                print()
                self.print_sources(event["payload"])
                answer = event["payload"]["answer"]

        return answer

    def print_trace(self, trace):
        print(f"[route] {trace['route']} — {trace['route_reason']}")

        for step in trace.get("steps", []):
            status = "ok" if step["ok"] else "FAILED"
            print(f"[step] {step['tool']}({step['input']!r}) -> {status}")

        for replan in trace.get("replanned", []):
            print(f"[replan] step {replan['step']}: {replan['reason']}")

        if trace["used"] and not trace.get("agent"):
            print(f"[queries] {' | '.join(trace['queries'])}")

    def print_sources(self, payload):
        citations = sorted({
            f"{source['document']} p.{source['page']}"
            for source in payload["sources"]
        })

        if citations:
            print(f"[sources] {', '.join(citations)}")

    def run(self):
        self.banner()

        while True:
            try:
                question = input("\nAI :: What would you like to ask? ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGood Bye!")
                return

            if not question:
                continue

            if question.lower() in {'exit', 'quit'}:
                print("Good Bye!")
                return

            if self.handle_command(question.lower()):
                continue

            try:
                if self.json_output:
                    answer = self.ask_json(question)
                else:
                    answer = self.ask_streaming(question)
            except service.PipelineError as error:
                print(f"AI :: the model backend failed: {error}")
                continue

            self.memory.add_conversation_message(question, answer)


def main():
    Session().run()


if __name__ == "__main__":
    main()
