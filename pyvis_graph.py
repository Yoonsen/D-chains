from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os


def _blend_hex(c1, c2, t):
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def graph_to_pyvis(G, words, title="Layer", directed=False):
    net = Network(height="600px", width="100%", directed=directed, bgcolor="#f8f9fa")
    
    # Use node labels stored in graph nodes when available.
    for node in G.nodes:
        node_data = G.nodes[node] if node in G.nodes else {}
        word_label = node_data.get("label")
        if not word_label:
            word_label = str(node)
        entropy = node_data.get("entropy")
        entropy_visible = node_data.get("entropy_visible")
        entropy_text = f" | Entropy(full): {entropy:.3f}" if isinstance(entropy, (int, float)) else ""
        if isinstance(entropy_visible, (int, float)):
            entropy_text += f" | Entropy(visible): {entropy_visible:.3f}"
        
        net.add_node(
            node,
            label=word_label,
            title=f"Position {node}: {word_label}{entropy_text}",
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
    
    # Compute smooth edge scaling per layer to avoid abrupt width jumps.
    edge_weights = [float(data.get("weight", 0.0)) for _, _, data in G.edges(data=True)]
    if edge_weights:
        min_w = min(edge_weights)
        max_w = max(edge_weights)
    else:
        min_w = 0.0
        max_w = 1.0

    # Legg til edges
    for u, v, data in G.edges(data=True):
        weight = round(data.get("weight", 1), 3)
        edge_type = data.get("edge_type", "primary")

        if max_w > min_w:
            norm = (float(weight) - min_w) / (max_w - min_w)
        else:
            norm = 0.5

        width = 1.0 + 3.5 * (norm ** 0.85)
        edge_color = _blend_hex("#bdc3c7", "#e74c3c", norm)
        if edge_type == "secondary":
            # Secondary edges are exploratory links: keep them visible but lighter.
            width = max(0.8, width * 0.7)
            edge_color = _blend_hex("#a8e6a1", "#2e8b57", norm)
        
        net.add_edge(
            u, v,
            value=width * 2,
            label=str(weight),
            title=f"Weight: {weight} | Type: {edge_type}",
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
    
    arrows_to_enabled = "true" if directed else "false"
    options_js = """
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
                    "enabled": __ARROWS_TO_ENABLED__
                }
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """
    net.set_options(options_js.replace("__ARROWS_TO_ENABLED__", arrows_to_enabled))
    
    return net

def show_pyvis_layers(layers, words, directed=False):
    total_layers = len(layers)

    for i, G in enumerate(layers, 1):
        st.caption(f"Lag {i} av {total_layers}")
        net = graph_to_pyvis(G, words, title=f"Layer {i}", directed=directed)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            components.html(open(tmp.name, 'r', encoding='utf-8').read(), height=650)
            os.unlink(tmp.name)