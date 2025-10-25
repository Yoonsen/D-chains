from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

def graph_to_pyvis(G, words, title="Layer"):
    net = Network(height="600px", width="100%", directed=False, bgcolor="#f8f9fa")
    
    # Legg til noder med ord fra words-listen
    for node in G.nodes:
        # Node-ID starter på 1, words-liste starter på 0
        word_label = words[node - 1] if (node - 1) < len(words) and node > 0 else str(node)
        
        net.add_node(
            node,
            label=word_label,
            title=f"Position {node}: {word_label}",
            color='#4A90E2',
            shape="box",
            font={
                'size': 22,
                'color': '#ffffff',
                'bold': True,
                'face': 'Arial'
            },
            borderWidth=3,
            borderWidthSelected=5,
            size=25
        )
    
    # Legg til edges
    for u, v, data in G.edges(data=True):
        weight = round(data.get("weight", 1), 3)
        
        if weight > 0.5:
            edge_color = '#e74c3c'
            width = 4
        elif weight > 0.3:
            edge_color = '#f39c12'
            width = 3
        elif weight > 0.15:
            edge_color = '#3498db'
            width = 2
        else:
            edge_color = '#bdc3c7'
            width = 1
        
        net.add_edge(
            u, v,
            value=width * 2,
            label=str(weight),
            title=f"Weight: {weight}",
            color=edge_color,
            width=width,
            font={
                'size': 12, 
                'color': '#2c3e50',
                'background': 'rgba(255, 255, 255, 0.9)',
                'strokeWidth': 0,
                'align': 'middle'
            }
        )
    
    net.set_options("""
    var options = {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -10000,
                "centralGravity": 0.3,
                "springLength": 120,
                "damping": 0.5
            },
            "minVelocity": 0.75,
            "maxVelocity": 30
        },
        "nodes": {
            "shape": "dot",
            "size": 25,
            "font": {
                "size": 22,
                "color": "#ffffff",
                "bold": true,
                "face": "Arial"
            },
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.3)",
                "size": 10,
                "x": 3,
                "y": 3
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous",
                "roundness": 0.5
            },
            "arrows": {
                "to": {
                    "enabled": false
                }
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)
    
    return net

def show_pyvis_layers(layers, words):
    for i, G in enumerate(layers[:-1], 1):
        net = graph_to_pyvis(G, words, title=f"Layer {i}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            components.html(open(tmp.name, 'r', encoding='utf-8').read(), height=650)
            os.unlink(tmp.name)
    
    # Show combined graph
    combined_net = graph_to_pyvis(layers[-1], words, title="Combined Graph")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        combined_net.save_graph(tmp.name)
        components.html(open(tmp.name, 'r', encoding='utf-8').read(), height=650)
        os.unlink(tmp.name)