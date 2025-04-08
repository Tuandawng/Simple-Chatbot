import os
import getpass
from typing import Annotated
from dotenv import load_dotenv

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from IPython.display import Image, display

from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_contrib.llms.testing import FakeLLM

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("ANTHROPIC_API_KEY")
load_dotenv()
# print("Loaded API key:", os.getenv("ANTHROPIC_API_KEY"))
print("Loaded API key:", os.getenv("OPENAI_API_KEY"))
# print("Loaded API key:", os.getenv("TAVILY_API_KEY"))


class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def stream_graph_updates(use_input: str):
    for event in graph.stream({"messages": [{"role":"user","content": user_input}]}):
        for value in event.values():
            print("Assistant:",value["messages"][-1].content)

graph_builder = StateGraph(State)


llm = ChatOpenAI(model="gpt-3.5-turbo")
# llm = FakeLLM()

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

tool = TavilySearchResults(max_results=10)
tools = [tool]
tool.invoke("What's a 'node' in LangGraph?")

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break