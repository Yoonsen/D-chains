import streamlit as st
import numpy as np
import pandas as pd
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from pyvis.network import Network
import streamlit.components.v1 as components

# --- 1. KONFIGURASJON OG SPESIALTOKENS ---
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>", "<mask>"}

def is_special(token):
    return token in SPECIAL_TOKENS or (token.startswith("<|") and token.endswith("|>"))

def normalize_label(token):
    return token.replace("##", "").replace("Ġ", "").replace(" ", "").strip()

# --- 2. DATAHENTING OG BEREGNING ---
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

def get_metrics(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Shape: (layers, heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions).squeeze(1) 
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return attentions, tokens

def calculate_layer_stats(attn_tensor):
    # Snitt over heads for å finne lagets "enighet"
    avg_attn = attn_tensor.mean(dim=0).detach().cpu().numpy()
    # Entropi: lav = fokusert, høy = diffus (usikker)
    epsilon = 1e-10
    entropy = -np.sum(avg_attn * np.log(avg_attn + epsilon), axis=-1)
    return avg_attn, entropy

# --- 3. PYVIS VISUALISERING (Gjenopprettet stil) ---
def create_styled_graph(tokens, avg_attn, entropy, hide_sys=True):
    net = Network(height="500px", width="100%", notebook=False, directed=True)
    
    # Skalering for farger (JSON-safe floats)
    e_min, e_max = float(np.min(entropy)), float(np.max(entropy))
    
    # Legg til noder
    for i, token in enumerate(tokens):
        if hide_sys and is_special(token):
            continue
            
        clean_token = normalize_label(token)
        if not clean_token: clean_token = token

        # Farge fra Rød (sikker) til Blå (usikker)
        e_val = float(entropy[i])
        norm_e = (e_val - e_min) / (e_max - e_min + 1e-6)
        node_color = f"rgb({int(255*(1-norm_e))}, 100, {int(255*norm_e)})"
        
        net.add_node(
            int(i), 
            label=clean_token, 
            title=f"Original: {token}\nEntropi: {e_val:.3f}", 
            color=node_color,
            font={'size': 18, 'color': 'black'}
        )

    # Legg til kanter basert på styrke
    for i in range(len(tokens)):
        # Finn top_k slik du gjorde i din originale kode
        row = avg_attn[i]
        top_indices = np.argsort(row)[-3:] # Vi viser de 3 sterkeste koblingene per ord
        
        for j in top_indices:
            weight = float(row[j])
            if weight > 0.05: # Minimum terskel
                if hide_sys and (is_special(tokens[i]) or is_special(tokens[j])):
                    continue
                if i == j: continue 
                
                # 'value' styrer tykkelsen på kanten i Pyvis
                net.add_edge(
                    int(j), int(i), # Retning: det ordet man ser på -> ordet som ser
                    value=weight * 15, 
                    title=f"Attention: {weight:.3f}",
                    color={'color': 'rgba(180, 180, 180, 0.7)', 'highlight': 'black'}
                )
    
    # Kraftig fysikk for å ligne på spring_layout
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -4000,
          "centralGravity": 0.2,
          "springLength": 180,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      },
      "nodes": { "borderWidth": 2, "shape": "dot" },
      "edges": { "smooth": { "type": "diagonalCross" } }
    }
    """)
    return net

# --- 4. KLYNGE-ANALYSE (3-cliques med selvtillit) ---
def find_cliques_safe(tokens, avg_attn, entropy):
    G = nx.Graph()
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if avg_attn[i,j] > 0.15: # Krav til styrke for å være i en klynge
                G.add_edge(i, j)
    
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["Klynge", "Confidence"])

    cliques = [c for c in nx.find_cliques(G) if len(c) >= 3]
    results = []
    
    for c in cliques:
        clean_words = [normalize_label(tokens[idx]) for idx in sorted(c) if not is_special(tokens[idx])]
        if len(clean_words) < 3: continue
        
        # Selvtillit = 1 / (1 + snitt_entropi)
        avg_e = np.mean([entropy[idx] for idx in c])
        confidence = 1.0 / (1.0 + avg_e)
        results.append({"Klynge": " + ".join(clean_words), "Confidence": round(float(confidence), 3)})
    
    if not results:
        return pd.DataFrame(columns=["Klynge", "Confidence"])
    return pd.DataFrame(results).sort_values(by="Confidence", ascending=False)

# --- 5. STREAMLIT APP ---
st.set_page_config(page_title="D-chains Visualizer", layout="wide")
st.title("🔗 D-chains: Attention & Entropi")

with st.sidebar:
    st.header("Modell-valg")
    model_name = st.selectbox("Modell", ["bert-base-multilingual-cased", "NbAiLab/nb-bert-base"])
    hide_sys = st.checkbox("Skjul system-noder ([CLS], [SEP])", value=True)
    st.markdown("---")
    st.write("**Fargeforklaring:**")
    st.write("🔴 **Rød:** Fokusert/Sikker kobling")
    st.write("🔵 **Blå:** Diffus attention (Usikkerhet)")

text = st.text_input("Skriv en setning for analyse:", "Ola sov. Han var trøtt.")

if text:
    tokenizer, model = load_model(model_name)
    attentions, tokens = get_metrics(text, tokenizer, model)
    
    # Vi plotter Lag 1, Lag 6 og Lag 12 (eller siste tilgjengelige)
    num_layers = attentions.shape[0]
    layer_indices = [0, num_layers // 2, num_layers - 1]
    
    cols = st.columns(len(layer_indices))
    
    for i, l_idx in enumerate(layer_indices):
        with cols[i]:
            st.subheader(f"Lag {l_idx + 1}")
            avg_attn, entropy = calculate_layer_stats(attentions[l_idx])
            
            # Opprett og lagre graf
            net = create_styled_graph(tokens, avg_attn, entropy, hide_sys)
            path = f"graph_layer_{l_idx}.html"
            net.save_graph(path)
            
            with open(path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=520)
            
            # Klyngevisning
            st.write("**3-Clique Klynger:**")
            df = find_cliques_safe(tokens, avg_attn, entropy)
            st.dataframe(df, hide_index=True, use_container_width=True)

st.caption("Tips: Se hvordan 'Ola' og 'Han' får sterkere kobling og lavere entropi i de dypere lagene.")
