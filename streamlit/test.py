from backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "test-thread"}}

print("Graph stream test:")

for event in chatbot.stream(
    {"messages": [HumanMessage(content="hi")]},
    config=CONFIG
):
    print("\nEVENT:", event)

print("\nDONE")
