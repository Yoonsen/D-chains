import streamlit as st
import numpy as np
import pandas as pd
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from pyvis.network import Network
import streamlit.components.v1 as components

# --- 1. UTVIDEDE VERKTØY FOR TOKEN-HÅNDTERING ---
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>", "<mask>"}

def is_special(token):
    return token in SPECIAL_TOKENS or (token.startswith("<|") and token.endswith("|>"))

def format_token_label(token, index):
    """Legger til indeks og håndterer sub-tokens visuelt."""
    is_subword = token.startswith("##") or "Ġ" not in token and index > 0 # Forenklet sjekk
    
    clean = token.replace("##", "").replace("Ġ", "").replace(" ", "").strip()
    
    # Hvis det er et sub-token, legg til bindestrek foran
    label = f"-{clean}" if token.startswith("##") else clean
    
    # Legg til indeks (f.eks. "han_5") for å skille tvetydige tokens
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

# --- 3. PYVIS VISUALISERING (Med indekser og sub-token logikk) ---
def create_styled_graph(tokens, avg_attn, entropy, hide_sys=True):
    net = Network(height="550px", width="100%", notebook=False, directed=True)
    
    e_min, e_max = float(np.min(entropy)), float(np.max(entropy))
    
    for i, token in enumerate(tokens):
        if hide_sys and is_special(token):
            continue
            
        # Bruker den nye formateringsfunksjonen
        display_label = format_token_label(token, i)

        e_val = float(entropy[i])
        norm_e = (e_val - e_min) / (e_max - e_min + 1e-6)
        node_color = f"rgb({int(255*(1-norm_e))}, 100, {int(255*norm_e)})"
        
        net.add_node(
            int(i), 
            label=display_label, 
            title=f"Fullt token: {token}\nIndeks: {i}\nEntropi: {e_val:.3f}", 
            color=node_color,
            shape="box", 
            font={'size': 20, 'color': 'white', 'face': 'monospace'},
            margin=10
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
                    value=weight * 15, 
                    title=f"Vekt: {weight:.3f}",
                    color={'color': 'rgba(120, 120, 120, 0.5)', 'highlight': 'red'},
                    arrows='to'
                )
    
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -150,
          "centralGravity": 0.01,
          "springLength": 160,
          "springConstant": 0.08,
          "avoidOverlap": 1
        },
        "solver": "forceAtlas2Based"
      },
      "edges": { "smooth": { "type": "curvedArrow", "roundness": 0.2 } }
    }
    """)
    return net

# --- 4. STREAMLIT APP ---
st.set_page_config(page_title="D-chains Visualizer", layout="wide")
st.title("🔗 D-chains: Sub-tokens & Indeks-mapping")

with st.sidebar:
    st.header("Innstillinger")
    model_name = st.selectbox("Modell", ["bert-base-multilingual-cased", "NbAiLab/nb-bert-base"])
    hide_sys = st.checkbox("Skjul spesialtokens", value=True)
    st.info("Noder merket med '-' er interne sub-tokens (f.eks. ##hus).")

text = st.text_input("Setning:", "Ola kjøpte et båthus. Han likte det.")

if text:
    tokenizer, model = load_model(model_name)
    attentions, tokens = get_metrics(text, tokenizer, model)
    
    num_layers = attentions.shape[0]
    layer_indices = [0, num_layers // 2, num_layers - 1] # Lag 1, 7, 12
    
    cols = st.columns(len(layer_indices))
    
    # Data for tabellen (bruker siste lag)
    final_layer_data = []

    for i, l_idx in enumerate(layer_indices):
        with cols[i]:
            st.subheader(f"Lag {l_idx + 1}")
            avg_attn, entropy = calculate_layer_stats(attentions[l_idx])
            
            net = create_styled_graph(tokens, avg_attn, entropy, hide_sys)
            path = f"graph_l{l_idx}.html"
            net.save_graph(path)
            
            with open(path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=550)
            
            # Lagre data hvis det er siste lag i løkken
            if i == len(layer_indices) - 1:
                for src_idx in range(len(tokens)):
                    for tgt_idx in range(len(tokens)):
                        w = float(avg_attn[src_idx, tgt_idx])
                        if w > 0.005: # Lav terskel for å få rikere rådata
                            final_layer_data.append({
                                "Fra_Index": src_idx,
                                "Fra_Token": tokens[src_idx],
                                "Til_Index": tgt_idx,
                                "Til_Token": tokens[tgt_idx],
                                "Vekt": round(w, 5),
                                "Entropi_Fra": round(float(entropy[src_idx]), 3)
                            })

    # --- 5. RÅDATA TABELL FOR PYTHON-ANALYSE ---
    st.write("---")
    st.write("### 📊 Rådata for Lag 12 (Klar for Python-analyse)")
    st.markdown("Bruk denne tabellen for å verifisere koreferanse (f.eks. om `han_7` peker på `Ola_1`).")
    
    df_final = pd.DataFrame(final_layer_data)
    if not df_final.empty:
        # Reorganiserer kolonner for logisk flyt
        df_final = df_final[["Fra_Index", "Fra_Token", "Til_Index", "Til_Token", "Vekt", "Entropi_Fra"]]
        st.dataframe(df_final.sort_values(by="Vekt", ascending=False), use_container_width=True)
        
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV for Python script", csv, "d_chains_data.csv", "text/csv")

st.caption("Fargekode: 🔴 Lav Entropi (Sikker) | 🔵 Høy Entropi (Usikker)")
