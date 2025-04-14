import os
import re
from typing import Annotated
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import AIMessage
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool 

# Load environment variables
load_dotenv()
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY not loaded"

# Setup search tool
search_tool = TavilySearchResults()

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ReAct-style system prompt
system_prompt = """
You are an AI assistant with a knowledge cutoff in 2023, but you can access real-time information using the Search tool.

You MUST use Search if the question asks about:
- Events after 2023
- Current information
- Anything you're unsure about

Always respond using this step-by-step format:

Question: {input}
Thought: ...
Action: Search("...")
Observation: ...
Thought: ...
Final Answer: ...

You must use the Search tool if your knowledge may be outdated. Do not ask the user to search. Do it yourself.
""".strip()


# Setup LLM
llm = ChatOllama(model="llama3.2:1b-instruct-q4_K_M")

# ReAct parsing logic
# def parse_action(text: str):
#     match = re.search(r'Action:\s*(\w+)\((.*?)\)', text)
#     if match:
#         return match.group(1), match.group(2)
#     return None, None

# Main chatbot logic
def chatbot(state: State):
    history = state["messages"]

    # Get initial response from LLM
    result = llm.invoke(history)
    output = result.content
    # history.append({"role": "assistant", "content": f"Observation: {search_tool.run()}"})
    history.append(AIMessage(content=output))
    return {"messages": history}

    # Check for tool action in response
@tool
def search_function(query: str):
    """Use this tool to search the web for current or real-time information."""
    print(f"[TOOL CALLED] search_web({query})")
    return search_tool.run(query)

search_tool_function = Tool.from_function(
    func=search_function,
    name="Search",
    description="Use this tool to search the web for current or real-time information."
)

agent = initialize_agent([search_tool_function], llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
    
# Graph definition
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_node("search", search_node)

graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# graph_builder.add_conditional_edges("chatbot", route_logic, {
#     "search": "search",
#     "end": "chatbot",
# })
# graph_builder.add_edge("search", "chatbot")

graph = graph_builder.compile()

# Streaming graph output
def stream_graph_updates(user_input: str):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_input})

    for event in graph.stream({"messages": messages}):
        for value in event.values():
            last_message = value["messages"][-1]
            try:
                # Handles AIMessage or HumanMessage
                print("Assistant:", last_message.content)
            except AttributeError:
                # Handles dicts (fallback)
                print("Assistant:", last_message["content"])
    return state

# Interactive loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q", "goodbye"]:
            print("Assistant: Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print("Error:", e)
        break
