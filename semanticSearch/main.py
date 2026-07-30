from query import search
from prompts import *
from llm import *
from memory import  ConversationMemory
from reranker import *

memory = ConversationMemory()

def banner():
    print("------ ENTERING WORLD OF AI-------")
    print("------ ACTIVATING NEURAL NETWORKS -------")
    print("------ JARVIS AT YOUR SERVICE-------")


banner()

while True:

    question = input("\nAI :: What would you like to ask? ").strip()

    if not question:
        continue

    if question.lower() in {'exit' , 'quit'}:
        print("Good Bye!")
        break

    results = search(question , top_k=20)
    results = rerank(question , results , top_k=5)
    print(f"Q: {question}\n")

    if not results:
        print("AI :: No results — did you run `python ingest.py` first?")

    messages = build_messages(question, results , memory.get_history())

    print("AI: ", end="", flush=True)

    answer = ""

    for chunk in stream_llm(messages):
        print(chunk, end="", flush=True)
        answer += chunk

    memory.add_conversation_message(question ,answer)

