import os
import re
from typing import Annotated, List, Union
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END # Import END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
# Use specific message types for clarity and potential downstream processing
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage

# Load environment variables
load_dotenv()
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY not loaded"

# --- Tool Setup ---
# Use the actual tool instance
search_tool = TavilySearchResults(max_results=3) # Limit results for brevity

# --- State Definition ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- System Prompt (Further Improved Prompt - Focus on Final Answer after Search) ---
system_prompt_content = """
You are a helpful AI assistant with a knowledge cutoff of 2023. When a user asks a question that requires information beyond 2023, or about current events (2025), or anything outside your internal knowledge, you MUST use the Search tool to get up-to-date information.

If the user's question is within your knowledge (before 2024), you can answer directly.

**Tool Use Instructions:**

1. **Identify Need for Search:** If the question is about current events, post-2023 information, or something you don't know, you MUST use the search tool. Do not say you cannot answer due to your knowledge cut-off without first trying to search.

2. **Action Output:**  When you need to use the search tool, output a single line that *exactly* matches this format:
   `Action: Search("your search query")`
   For example: `Action: Search("current president of the United States")`
   Make sure to put the query inside double quotes. **After outputting an Action, WAIT for the search results.**

3. **Process Search Results and Provide Final Answer:** **Once you receive the search results from the tool, you MUST formulate a "Final Answer" based on the information in the search results.**  Do not ask the user to search themselves or perform more searches unless absolutely necessary to clarify ambiguity in the *initial* query. Output your answer in a single line that *exactly* matches this format:
   `Final Answer: Your answer here.`
   For example: `Final Answer: Based on the search results, the current president of the United States in 2025 is likely to be [Name], as indicated by [cite source from search results briefly].`

**Important Notes:**

* Do not make up answers. Base your "Final Answer" **primarily on the search results.** If search results are inconclusive or contradictory, reflect that in your "Final Answer" (e.g., "Search results provide conflicting information...").
* Only use the "Search" tool. Do not use any other tools or functions.
* Stick to the "Action:" and "Final Answer:" formats strictly.
* Be concise and helpful in your "Final Answer". **Focus on answering the user's original question after searching.** Avoid getting sidetracked into related but unnecessary searches.
"""

# --- LLM Setup ---
# Using the same model as requested: llama3.2:1b-instruct-q4_K_M
llm = ChatOllama(model="llama3.2:1b-instruct-q4_K_M")

# --- Nodes ---

# 1. Chatbot Node (Invokes LLM)
def chatbot(state: State):
    print("---LLM INVOKED---")
    response = llm.invoke(state["messages"])
    print(f"---LLM Raw Response Content:---\n{response.content}\n--------------------") # Keep raw response print for debugging
    return {"messages": [response]}

# 2. Tool Execution Node
def call_tool(state: State):
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
         return {"messages": [SystemMessage(content="Error: No AIMessage found to process for tool call.")]}

    action_match = re.search(r"Action: Search\(\"(.*?)\"\)", last_message.content, re.IGNORECASE | re.DOTALL) # Robust regex

    if not action_match:
         print("---ERROR: Tool Node called but no Action found!---")
         return {"messages": [SystemMessage(content="Error: LLM output did not contain a valid Action: Search(...) structure.")]}

    query = action_match.group(1)
    print(f"---CALLING TOOL: Search('{query}')---")
    try:
        search_result = search_tool.invoke({"query": query})
        print(f"---TOOL RESULT (raw): {search_result}---") # Print raw result for debug

        tool_message = ToolMessage(
            content=str(search_result),
            name="Search",
            tool_call_id="react_search_call" # Dummy ID
        )

    except Exception as e:
        print(f"---TOOL ERROR: {e}---")
        tool_message = ToolMessage(
            content=f"Error during search: {e}",
            name="Search",
            tool_call_id="react_search_error" # Dummy ID for error case
        )

    return {"messages": [tool_message]}

def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    # If the last message is not from the AI, finish (error or initial state)
    if not isinstance(last_message, AIMessage):
        print("---DECISION: Last message not AI, finish---")
        return END

    # Check for "Action: Search(" to trigger tool use
    if "Action: Search(" in last_message.content:
        print("---DECISION: Action found, continue to tool---")
        return "continue"
    elif "Final Answer:" in last_message.content: # Check for Final Answer to end
        print("---DECISION: Final Answer found, finish---")
        return END
    else:
        # Assume it's an intermediate thought or error if neither Action nor Final Answer
        print("---DECISION: No Action or Final Answer, finish (fallback)---") # Modified decision
        return END # End if no action or final answer is clearly indicated

# --- Graph Definition ---
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("action", call_tool)

graph_builder.set_entry_point("chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "continue": "action",
        END: END
    }
)

graph_builder.add_edge("action", "chatbot")

graph = graph_builder.compile()

# --- Interaction Logic ---

initial_system_message = SystemMessage(content=system_prompt_content)
conversation_history: List[BaseMessage] = [initial_system_message]

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q", "goodbye"]:
            print("Assistant: Goodbye!")
            break

        current_human_message = HumanMessage(content=user_input)
        conversation_history.append(current_human_message)
        graph_input = {"messages": conversation_history}

        final_state_data = None # Initialize final_state here to None
        print("Assistant: ", end="", flush=True)

        full_response_content = ""
        final_state = None # Initialize final_state again right before the loop (redundancy for clarity)
        for event in graph.stream(graph_input):
             if END in event:
                 final_state = event[END]
                 break

        if final_state:
            final_ai_message = final_state["messages"][-1]

            if isinstance(final_ai_message, AIMessage):
                final_answer_match = re.search(r"Final Answer:\s*(.*)", final_ai_message.content, re.DOTALL | re.IGNORECASE)
                if final_answer_match:
                    print(final_answer_match.group(1).strip())
                    full_response_content = final_ai_message.content
                else:
                    print(final_ai_message.content) # Print full message if no Final Answer format
                    full_response_content = final_ai_message.content
            else:
                 print(f"[Assistant finished with non-AI message: {final_ai_message}]")
                 full_response_content = str(final_ai_message)

            conversation_history = final_state["messages"]

        else:
             print("Error: Graph did not reach a final state.")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        break