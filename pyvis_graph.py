from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

def graph_to_pyvis(G, title="Layer"):
    net = Network(height="500px", width="100%", directed=False)
    for node in G.nodes:
        net.add_node(node, label=node)
    for u, v, data in G.edges(data=True):
        weight = round(data.get("weight", 1), 3)
        label = data.get("label", f"{u}-{v}")
        net.add_edge(u, v, value=weight, title=label, label=label)
    net.set_options("""var options = { "physics": { "barnesHut": { "gravitationalConstant": -8000, "centralGravity": 0.3, "springLength": 95 }, "minVelocity": 0.75 } }""")
    return net

def show_pyvis_layers(layers):
    for i, G in enumerate(layers[:-1], 1):
        net = graph_to_pyvis(G, title=f"Layer {i}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            components.html(open(tmp.name, 'r', encoding='utf-8').read(), height=550)
            os.unlink(tmp.name)
    # Show combined graph
    combined_net = graph_to_pyvis(layers[-1], title="Combined Graph")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        combined_net.save_graph(tmp.name)
        components.html(open(tmp.name, 'r', encoding='utf-8').read(), height=550)
        os.unlink(tmp.name)
