from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

def graph_to_pyvis(G, title="Layer"):
    net = Network(height="500px", width="100%", directed=False, bgcolor="#f8f9fa")
    
    # Legg til noder med bedre styling
    for node in G.nodes:
        net.add_node(
            node, 
            label=G.nodes[node].get("label", str(node)),
            color='#4A90E2',  # Blå noder
            font={
                'size': 18, 
                'color': '#ffffff',  # Hvit tekst på nodene
                'bold': True
            },
            borderWidth=2,
            borderWidthSelected=4
        )
    
    # Legg til edges med bedre styling
    for u, v, data in G.edges(data=True):
        weight = round(data.get("weight", 1), 3)
        label = data.get("label", f"{u}-{v}")
        
        # Fargekode basert på vekt
        if weight > 0.5:
            edge_color = '#e74c3c'  # Rød for sterke koblinger
        elif weight > 0.3:
            edge_color = '#f39c12'  # Oransje for medium
        else:
            edge_color = '#95a5a6'  # Grå for svake
        
        net.add_edge(
            u, v, 
            value=weight * 5,  # Skalerer tykkelsen
            label=label, 
            color=edge_color,
            font={
                'size': 14, 
                'color': '#2c3e50',  # Mørk tekst på edges
                'background': '#ffffff',  # Hvit bakgrunn på edge-labels
                'strokeWidth': 0
            }
        )
    
    net.set_options("""
    var options = {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 95
            },
            "minVelocity": 0.75
        },
        "nodes": {
            "shape": "dot",
            "size": 20,
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.2)",
                "size": 10,
                "x": 2,
                "y": 2
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            }
        }
    }
    """)
    
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