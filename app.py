import streamlit as st
import numpy as np
import pandas as pd
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from pyvis.network import Network
import streamlit.components.v1 as components

# --- 1. UTILITIES ---
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>", "<mask()"}

def is_special(token):
    return token in SPECIAL_TOKENS or (token.startswith("<|") and token.endswith("|>"))

def format_token_label(token, index):
    clean = token.replace("##", "").replace("Ġ", "").replace(" ", "").strip()
    label = f"-{clean}" if token.startswith("##") else clean
    return f"{label}_{index}"

# --- 2. DATA ---
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

def get_metrics(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = torch.stack(outputs.attentions).squeeze(1) 
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return attentions, tokens

# --- 3. PYVIS (Vertikal/Stor font) ---
def create_styled_graph(tokens, avg_attn, entropy, hide_sys=True):
    # Bruker 600px høyde, men full bredde i vertikal layout
    net = Network(height="600px", width="100%", notebook=False, directed=True)
    
    e_min, e_max = float(np.min(entropy)), float(np.max(entropy))
    
    for i, token in enumerate(tokens):
        if hide_sys and is_special(token):
            continue
            
        display_label = format_token_label(token, i)
        e_val = float(entropy[i])
        norm_e = (e_val - e_min) / (e_max - e_min + 1e-6)
        node_color = f"rgb({int(255*(1-norm_e))}, 100, {int(255*norm_e)})"
        
        net.add_node(
            int(i), 
            label=display_label, 
            color=node_color,
            shape="box", 
            font={'size': 32, 'color': 'white', 'face': 'monospace'},
            margin=12
        )

    for i in range(len(tokens)):
        row = avg_attn[i]
        top_indices = np.argsort(row)[-3:] 
        for j in top_indices:
            weight = float(row[j])
            if weight > 0.05:
                if hide_sys and (is_special(tokens[i]) or is_special(tokens[j])):
                    continue
                if i == j: continue 
                
                net.add_edge(int(j), int(i), value=weight * 20, arrows='to')
    
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": { "gravitationalConstant": -200, "springLength": 200, "avoidOverlap": 1 },
        "solver": "forceAtlas2Based"
      },
      "interaction": { "navigationButtons": true, "zoomView": true }
    }
    """)
    return net

def get_phrases(tokens, avg_attn):
    G = nx.Graph()
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if avg_attn[i,j] > 0.15: G.add_edge(i, j)
    
    cliques = [c for c in nx.find_cliques(G) if len(c) >= 3]
    phrases = []
    for c in cliques:
        p = " + ".join([format_token_label(tokens[idx], idx) for idx in sorted(c) if not is_special(tokens[idx])])
        if p: phrases.append(p)
    return phrases

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="D-chains Vertical", layout="wide")
st.title("🔗 D-chains: Vertikal Analyse")

with st.sidebar:
    model_name = st.selectbox("Modell", ["bert-base-multilingual-cased", "NbAiLab/nb-bert-base"])
    hide_sys = st.checkbox("Skjul spesialtokens (Graf & Tabell)", value=True)
    st.write("🔴 Sikker | 🔵 Usikker")

text = st.text_input("Setning:", "Ola kjøpte et båthus. Han likte det.")

if text:
    tokenizer, model = load_model(model_name)
    attentions, tokens = get_metrics(text, tokenizer, model)
    num_layers = attentions.shape[0]
    layer_indices = [0, num_layers // 2, num_layers - 1]

    # --- VERTIKAL ORGANISERING ---
    for l_idx in layer_indices:
        st.write(f"## 🛠 Lag {l_idx + 1}")
        avg_attn, entropy = calculate_layer_stats = (lambda a: (
            a.mean(dim=0).detach().cpu().numpy(), 
            -np.sum(a.mean(dim=0).detach().cpu().numpy() * np.log(a.mean(dim=0).detach().cpu().numpy() + 1e-10), axis=-1)
        ))(attentions[l_idx])
        
        # 1. Graf
        net = create_styled_graph(tokens, avg_attn, entropy, hide_sys)
        path = f"graph_v{l_idx}.html"
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=620)
        
        # 2. Fraser
        phrases = get_phrases(tokens, avg_attn)
        if phrases:
            st.write("**Identifiserte frase-klynger:**")
            for p in phrases: st.info(p)
        else:
            st.write("*Ingen sterke fraser funnet i dette laget.*")
        st.write("---")

    # --- 5. DIN SISTE DATARAMME ---
    st.write("### 📊 Komplett Rådata (Siste Lag)")
    
    last_attn, last_entropy = (lambda a: (
        a.mean(dim=0).detach().cpu().numpy(), 
        -np.sum(a.mean(dim=0).detach().cpu().numpy() * np.log(a.mean(dim=0).detach().cpu().numpy() + 1e-10), axis=-1)
    ))(attentions[layer_indices[-1]])

    table_data = []
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            # Filtrer tabellen basert på samme checkbox som grafen
            if hide_sys and (is_special(tokens[i]) or is_special(tokens[j])):
                continue
            
            w = float(last_attn[i, j])
            if w > 0.001:
                table_data.append({
                    "Fra_Index": i, "Fra_Token": tokens[i],
                    "Til_Index": j, "Til_Token": tokens[j],
                    "Vekt": round(w, 5), "Entropi_Fra": round(float(last_entropy[i]), 3)
                })
    
    df = pd.DataFrame(table_data)
    if not df.empty:
        st.dataframe(df.sort_values(by="Vekt", ascending=False), use_container_width=True)
