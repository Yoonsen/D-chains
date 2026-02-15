import streamlit as st
import numpy as np
import pandas as pd
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from pyvis.network import Network
import streamlit.components.v1 as components

# --- 1. TOKEN VERKTØY ---
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>", "<mask()"}

def is_special(token):
    return token in SPECIAL_TOKENS or (token.startswith("<|") and token.endswith("|>"))

def format_token_label(token, index):
    clean = token.replace("##", "").replace("Ġ", "").replace(" ", "").strip()
    label = f"-{clean}" if token.startswith("##") else clean
    return f"{label}_{index}"

# --- 2. DATAHENTING ---
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

def calculate_layer_stats(attn_tensor):
    avg_attn = attn_tensor.mean(dim=0).detach().cpu().numpy()
    epsilon = 1e-10
    entropy = -np.sum(avg_attn * np.log(avg_attn + epsilon), axis=-1)
    return avg_attn, entropy

# --- 3. PYVIS VISUALISERING (Større font & Sentrerings-fix) ---
def create_styled_graph(tokens, avg_attn, entropy, hide_sys=True):
    # Høyere boks (700px) for mer plass
    net = Network(height="700px", width="100%", notebook=False, directed=True)
    
    e_min, e_max = float(np.min(entropy)), float(np.max(entropy))
    
    for i, token in enumerate(tokens):
        if hide_sys and is_special(token):
            continue
            
        display_label = format_token_label(token, i)
        e_val = float(entropy[i])
        norm_e = (e_val - e_min) / (e_max - e_min + 1e-6)
        node_color = f"rgb({int(255*(1-norm_e))}, 100, {int(255*norm_e)})"
        
        # Kvadratiske bokser, MYE større font og padding
        net.add_node(
            int(i), 
            label=display_label, 
            title=f"Full: {token} | Entropi: {e_val:.3f}", 
            color=node_color,
            shape="box", 
            font={'size': 28, 'color': 'white', 'face': 'monospace', 'multi': True},
            margin=15,
            borderWidth=2
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
                
                net.add_edge(
                    int(j), int(i), 
                    value=weight * 20, 
                    title=f"Vekt: {weight:.3f}",
                    color={'color': 'rgba(120, 120, 120, 0.4)', 'highlight': 'red'},
                    arrows='to'
                )
    
    # Layout-innstillinger optimert for trackpad og oversikt
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -250,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.08,
          "avoidOverlap": 1
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "enabled": true, "iterations": 100 }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true
      },
      "edges": { "smooth": { "type": "curvedArrow", "roundness": 0.2 } }
    }
    """)
    return net

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="D-chains Visualizer", layout="wide")
st.title("🔗 D-chains: Koreferanse-analyse")

with st.sidebar:
    st.header("Kontrollpanel")
    model_name = st.selectbox("Modell", ["bert-base-multilingual-cased", "NbAiLab/nb-bert-base"])
    hide_sys = st.checkbox("Skjul spesialtokens", value=True)
    
    # Knapp for å tvinge re-render (sentrering)
    recenter = st.button("🔄 Sentrer grafer")
    
    st.markdown("---")
    st.write("🔴 = Sikker | 🔵 = Usikker")

text = st.text_input("Setning:", "Ola kjøpte et båthus. Han likte det.")

if text:
    tokenizer, model = load_model(model_name)
    attentions, tokens = get_metrics(text, tokenizer, model)
    
    num_layers = attentions.shape[0]
    layer_indices = [0, num_layers // 2, num_layers - 1]
    
    # Lag 3 store kolonner
    cols = st.columns(len(layer_indices))
    final_layer_data = []

    for i, l_idx in enumerate(layer_indices):
        with cols[i]:
            st.subheader(f"Lag {l_idx + 1}")
            avg_attn, entropy = calculate_layer_stats(attentions[l_idx])
            
            net = create_styled_graph(tokens, avg_attn, entropy, hide_sys)
            
            # Navngi filen unikt slik at 'Sentrer' tvinger ny lasting
            path = f"graph_l{l_idx}_{'reset' if recenter else 'init'}.html"
            net.save_graph(path)
            
            with open(path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=720) # Økt høyde for iframe
            
            if i == len(layer_indices) - 1:
                for src_idx in range(len(tokens)):
                    for tgt_idx in range(len(tokens)):
                        w = float(avg_attn[src_idx, tgt_idx])
                        if w > 0.005:
                            final_layer_data.append({
                                "Fra_Index": src_idx, "Fra_Token": tokens[src_idx],
                                "Til_Index": tgt_idx, "Til_Token": tokens[tgt_idx],
                                "Vekt": round(w, 5), "Entropi_Fra": round(float(entropy[src_idx]), 3)
                            })

    # --- 5. RÅDATA TABELL ---
    st.write("---")
    st.write("### 📊 Rådata for Python-analyse (Siste lag)")
    
    df_final = pd.DataFrame(final_layer_data)
    if not df_final.empty:
        df_final = df_final[["Fra_Index", "Fra_Token", "Til_Index", "Til_Token", "Vekt", "Entropi_Fra"]]
        st.dataframe(df_final.sort_values(by="Vekt", ascending=False), use_container_width=True)
