import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx

# --- 1. Datatilkobling og Beregning ---

def get_attention_and_entropy(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Attentions: (layers, batch, heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions).squeeze(1) 
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    return attentions, tokens

def calculate_layer_metrics(attn_tensor):
    """Beregner gjennomsnittlig attention og entropi for et lag."""
    # Snitt over alle attention-hoder
    avg_attn = attn_tensor.mean(dim=0).detach().cpu().numpy()
    
    # Entropi per token: H = -sum(p * log(p))
    epsilon = 1e-10
    entropy = -np.sum(avg_attn * np.log(avg_attn + epsilon), axis=-1)
    
    return avg_attn, entropy

# --- 2. Graf-visualisering (Pyvis) ---

def create_pyvis_graph(tokens, avg_attn, entropy, hide_sys=True):
    net = Network(height="450px", width="100%", notebook=False, directed=True)
    
    # Finn min/max for fargeskalering (JSON-safe)
    e_min, e_max = float(np.min(entropy)), float(np.max(entropy))
    
    # Legg til noder
    for i, token in enumerate(tokens):
        if hide_sys and token in ["[CLS]", "[SEP]", " ", " "]:
            continue
            
        # Beregn farge basert på entropi (Rød = Lav/Sikker, Blå = Høy/Usikker)
        e_val = float(entropy[i])
        norm_e = (e_val - e_min) / (e_max - e_min + 1e-6)
        color = f"rgb({int(255*(1-norm_e))}, 100, {int(255*norm_e)})"
        
        net.add_node(
            int(i), 
            label=str(token), 
            title=f"Entropy: {e_val:.3f}", 
            color=color,
            shape="dot",
            size=20
        )

    # Legg til kanter (kun de sterkeste for å unngå JSON-støy)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            weight = float(avg_attn[i, j])
            if weight > 0.08:  # Terskel for synlighet
                if hide_sys and (tokens[i] in ["[CLS]", "[SEP]"] or tokens[j] in ["[CLS]", "[SEP]"]):
                    continue
                if i == j: continue # Hopp over self-attention for renere graf
                
                net.add_edge(
                    int(j), int(i), # Retning: kilde -> mål
                    value=weight, 
                    title=f"Weight: {weight:.3f}",
                    color="rgba(200, 200, 200, 0.5)"
                )
    
    net.set_options("""
    var options = {
      "physics": { "barnesHut": { "gravitationalConstant": -1500, "centralGravity": 0.3, "springLength": 120 } },
      "edges": { "smooth": { "type": "continuous" } }
    }
    """)
    return net

# --- 3. Klynge-analyse (D-chains) ---

def find_cliques_with_confidence(tokens, avg_attn, entropy):
    G = nx.Graph()
    # Bygg en midlertidig NetworkX-graf for klynge-deteksjon
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if avg_attn[i,j] > 0.15: # Sterkere terskel for klikker
                G.add_edge(i, j, weight=float(avg_attn[i,j]))
    
    cliques = list(nx.find_cliques(G))
    results = []
    
    for c in cliques:
        if len(c) >= 3: # Vi ser etter 3-cliques eller større
            words = [tokens[idx] for idx in sorted(c) if tokens[idx] not in ["[CLS]", "[SEP]"]]
            if len(words) < 3: continue
            
            # Beregn selvtillit (lav entropi = høy selvtillit)
            avg_e = np.mean([entropy[idx] for idx in c])
            conf = 1.0 / (1.0 + avg_e)
            results.append({"Klynge": " + ".join(words), "Confidence": round(float(conf), 3)})
            
    return pd.DataFrame(results).sort_values(by="Confidence", ascending=False)

# --- 4. Streamlit UI ---

st.set_page_config(page_title="D-chains Visualizer", layout="wide")
st.title("🔗 D-chains: Attention Hierarkier")
st.markdown("Visualiser hvordan modellen knytter ord sammen fra Lag 1 (syntaks) til Lag 12 (semantikk).")

with st.sidebar:
    st.header("Innstillinger")
    model_name = st.selectbox("Velg modell", ["bert-base-multilingual-cased", "NbAiLab/nb-bert-base"])
    hide_sys = st.checkbox("Skjul system-tokens ([CLS], [SEP])", value=True)
    st.info("Rød node = Fokusert/Sikker\nBlå node = Diffus/Usikker (f.eks. 'båthus')")

text = st.text_input("Input setning:", "Ola sov. Han var trøtt.")

if text:
    attentions, tokens = get_attention_and_entropy(text, model_name)
    
    # Velg ut tre representative lag
    layers_idx = [0, 5, 11] # Første, midtre, siste
    cols = st.columns(len(layers_idx))
    
    for i, l_idx in enumerate(layers_idx):
        with cols[i]:
            st.subheader(f"Lag {l_idx + 1}")
            avg_attn, entropy = calculate_layer_metrics(attentions[l_idx])
            
            # Lag og lagre Pyvis-graf
            net = create_pyvis_graph(tokens, avg_attn, entropy, hide_sys)
            html_path = f"layer_{l_idx}.html"
            net.save_graph(html_path)
            
            with open(html_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=470)
            
            # Vis klynger for dette laget
            st.write("**Identifiserte klynger:**")
            df_cliques = find_cliques_with_confidence(tokens, avg_attn, entropy)
            if not df_cliques.empty:
                st.dataframe(df_cliques, hide_index=True)
            else:
                st.write("*Ingen sterke klynger funnet.*")

st.markdown("---")
st.write("### Hvorfor 'all-over-the-place'?")
st.write("Hvis et ord har høy entropi (blå farge) og mange tynne kanter, betyr det at modellen er usikker. Dette skjer ofte med sjeldne ord eller ord som mangler kontekst.")
