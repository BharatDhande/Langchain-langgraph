from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    #model='xiaomi/mimo-v2-flash:free',
    api_key=os.environ.get('OPEN_ROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
    streaming=True
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]

    full_response = ""

    for chunk in llm.stream(messages):
        if chunk.content:
            full_response += chunk.content
        # still stream to UI
        yield {"messages": [chunk]}

    # ✅ IMPORTANT: store final message in state
    final_msg = AIMessage(content=full_response)
    yield {"messages": [final_msg]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

all_threads = set()
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        tid = checkpoint.config.get("configurable", {}).get("thread_id")
        if tid:
            all_threads.add(tid)
    return list(all_threads)
