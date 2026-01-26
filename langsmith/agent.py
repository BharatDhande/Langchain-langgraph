import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

load_dotenv()

# -----------------------
# Tools
# -----------------------

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> dict:
    """Fetch current weather data for a given city"""
    url = "http://api.weatherstack.com/current"
    params = {
        "access_key": os.getenv("WEATHERSTACK_API_KEY"),
        "query": city
    }
    return requests.get(url, params=params, timeout=10).json()

tools = [search_tool, get_weather_data]

# -----------------------
# LLM with tools
# -----------------------

llm = ChatOpenAI(
    model="xiaomi/mimo-v2-flash:free",
    api_key=os.environ.get('OPEN_ROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
)

# -----------------------
# State
# -----------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -----------------------
# Prompt
# -----------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can search the web and get weather data."),
    ("placeholder", "{messages}")
])

# -----------------------
# Nodes
# -----------------------

def assistant(state: AgentState):
    chain = prompt | llm
    response = chain.invoke(
    {"messages": state["messages"]},
    config={"run_name": "assistant_node"}
)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -----------------------
# Graph
# -----------------------

graph = StateGraph(AgentState)

graph.add_node("assistant", assistant)
graph.add_node("tools", tool_node)

graph.set_entry_point("assistant")

graph.add_conditional_edges(
    "assistant",
    lambda state: "tools" if state["messages"][-1].tool_calls else END
)

graph.add_edge("tools", "assistant")

agent = graph.compile()

# -----------------------
# Run
# -----------------------

result = agent.invoke({
    "messages": [HumanMessage(content="Identify the birthplace city of Kalpana Chawla and give its current temperature.")]
})

print("\nFinal Answer:\n", result["messages"][-1].content)
