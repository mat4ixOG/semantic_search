from llm import ask_llm
from prompts import ignore_messages

def should_retrieve(question:str)->bool:
    messages = ignore_messages(question)
    decision = ask_llm(messages).rstrip().upper()
    return decision == "YES"

