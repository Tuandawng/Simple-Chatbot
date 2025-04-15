# test.py
import networkx as nx
import matplotlib.pyplot as plt
# Create a directed graph
graph = nx.DiGraph()

# Add nodes
graph.add_node("chatbot")
graph.add_node("action")
graph.add_node("END")  # Assuming END is a logical end state

# Add edges
# Entry point
graph.add_edge("START", "chatbot")  # Assuming there's a conceptual START

# Conditional edges from "chatbot"
graph.add_edge("chatbot", "action", condition="continue")
graph.add_edge("chatbot", "END", condition="END")

# Edge from "action" to "chatbot"
graph.add_edge("action", "chatbot")

# Now the 'graph' object contains the structure
# graph_structure.py


# Draw the graph
pos = nx.spring_layout(graph)  # You can try different layout algorithms
nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)

# Add edge labels for the conditional edges (optional)
labels = {("chatbot", "action"): "continue", ("chatbot", "END"): "END"}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

plt.title("Chatbot Flow")
plt.show()