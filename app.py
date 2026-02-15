import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from pyvis.network import Network
import streamlit.components.v1 as components

# --- 1. Funksjon for å hente data og beregne entropi ---
def get_attention_data(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    
    # attentions shape: (layers, batch, heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions).squeeze(1) 
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    return attentions, tokens

def calculate_entropy(attn_matrix):
    # attn_matrix: (heads, seq_len, seq_len)
    # Vi snitter over heads for å se lagets totale fokus
    avg_attn = attn_matrix.mean(dim=0).detach().numpy()
    epsilon = 1e-10
    entropy = -np.sum(avg_attn * np.log(avg_attn + epsilon), axis=-1)
    return entropy, avg_attn

# --- 2. Pyvis Visualisering ---
def create_pyvis_graph(tokens, avg_attn, entropy, layer_idx, hide_system_nodes=True):
    net = Network(height="400px", width="100%", notebook=False, directed=True)
    
    # Finn max/min entropi for fargeskalering
    max_e = np.max(entropy)
    min_e = np.min(entropy)
    
    for i, token in enumerate(tokens):
        if hide_system_nodes and token in ["[CLS]", "[SEP]", " ", " "]:
            continue
            
        # Fargekoding basert på entropi (Lav = Rød/Fokusert, Høy = Grå/Diffus)
        # Normaliserer til en verdi mellom 0 og 255
        val = int(255 * (entropy[i] - min_e) / (max_e - min_e + 1e-6))
        color = f"rgb({255-val}, {100}, {val})" # Går fra Rød mot Blå/Grå
        
        net.add_node(i, label=token, title=f"Entropy: {entropy[i]:.2f}", color=color)

    for i in range(len(tokens)):
        for j in range(len(tokens)):
            weight = avg_attn[i, j]
            if weight > 0.1: # Terskel for å unngå visuelt kaos
                if hide_system_nodes and (tokens[i] in ["[CLS]", "[SEP]"] or tokens[j] in ["[CLS]", "[SEP]"]):
                    continue
                net.add_edge(i, j, value=weight, title=f"Weight: {weight:.3f}")
    
    net.set_options("""
    var options = { "physics": { "barnesHut": { "gravitationalConstant": -2000, "centralGravity": 0.3, "springLength": 95 } } }
    """)
    return net

# --- 3. Streamlit App ---
st.set_page_config(layout="wide")
st.title("🧠 LLM Coreference & Entropy Visualizer")

col1, col2, col3 = st.columns([4, 3, 2])
with col1:
    text = st.text_input("Input sentence", "Ola sov. Han var trøtt.")
with col2:
    model_name = st.selectbox("Model", ["bert-base-multilingual-cased", "NbAiLab/nb-bert-base"])
with col3:
    hide_sys = st.checkbox("Skjul [CLS]/[SEP]", value=True)

if st.button("Analyser Lag for Lag"):
    attentions, tokens = get_attention_data(text, model_name)
    
    # Vi viser Lag 1 (Syntaks) og Lag 5 (Semantikk/Koreferanse)
    layers_to_show = [0, 2, 4] # Lag 1, 3 og 5
    
    cols = st.columns(len(layers_to_show))
    
    for idx, layer_num in enumerate(layers_to_show):
        with cols[idx]:
            st.subheader(f"Lag {layer_num + 1}")
            entropy, avg_attn = calculate_entropy(attentions[layer_num])
            
            # Lag graf
            net = create_pyvis_graph(tokens, avg_attn, entropy, layer_num, hide_sys)
            path = f"graph_l{layer_num}.html"
            net.save_graph(path)
            
            with open(path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=450)
            
            st.caption("🔴 Mørkerød = Fokusert | 🔵 Blå/Grå = Diffus")

    # Tabell for dypdykk i Lag 5
    st.write("### Nærbilde: Lag 5 Attention-vekter")
    final_entropy, final_attn = calculate_entropy(attentions[4])
    
    edges = []
    for i, t1 in enumerate(tokens):
        for j, t2 in enumerate(tokens):
            if t1 in ["[CLS]", "[SEP]"] or t2 in ["[CLS]", "[SEP]"]: continue
            if final_attn[i,j] > 0.05:
                edges.append({"Fra": t1, "Til": t2, "Vekt": final_attn[i,j], "Avstand": abs(i-j)})
    
    df = pd.DataFrame(edges).sort_values(by="Vekt", ascending=False)
    st.dataframe(df)

st.markdown("---")
st.markdown("**Test-case for koreferanse:** Prøv setningen `Ola slo Per fordi han var sint.` Se om `han` peker mest på `Ola` eller `Per` i lag 5!")
