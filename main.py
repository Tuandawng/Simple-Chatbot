import os
import re
from typing import Annotated
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults

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
def parse_action(text: str):
    match = re.search(r"Action:\s*(\w+)\[(.+?)\]", text)
    if match:
        return match.group(1), match.group(2)
    return None, None

# Main chatbot logic
def chatbot(state: State):
    history = state["messages"]

    # Get initial response from LLM
    result = llm.invoke(history)
    output = result.content
    # history.append({"role": "assistant", "content": f"Observation: {search_tool.run()}"})
    history.append({"role": "assistant", "content": output})


    # Check for tool action in response
    action, argument = parse_action(output)

    if action == "Search":
        observation = search_tool.run(argument)
        history.append({"role": "tool", "content": f"Observation: {observation}"})

        # Continue reasoning with tool result
        followup = llm.invoke(history)
        history.append({"role": "assistant", "content": followup.content})

    return {"messages": history}

# Graph definition
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

# Streaming graph output
def stream_graph_updates(user_input: str):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_input})

    for event in graph.stream({"messages": messages}):
        for value in event.values():
            assistant_message = value["messages"][-1]["content"]
            print("Assistant:", assistant_message)

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
