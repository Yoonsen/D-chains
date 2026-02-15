import streamlit as st
import numpy as np
import pandas as pd
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from pyvis.network import Network
import streamlit.components.v1 as components
from collections import Counter

# --- 1. UTILITIES ---
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>", "<mask()"}

def is_special(token):
    return token in SPECIAL_TOKENS or (token.startswith("<|") and token.endswith("|>"))

def get_clean_label(token):
    return token.replace("##", "").replace("Ġ", "").replace(" ", "").strip()

def get_token_labels(tokens):
    """Genererer etiketter som kun har indeks ved tvetydighet."""
    clean_tokens = [get_clean_label(t) for t in tokens]
    counts = Counter(clean_tokens)
    labels = []
    
    for i, t in enumerate(tokens):
        clean = clean_tokens[i]
        prefix = "-" if t.startswith("##") else ""
        # Legg til indeks kun hvis ordet finnes flere ganger
        if counts[clean] > 1 and not is_special(t):
            labels.append(f"{prefix}{clean}_{i}")
        else:
            labels.append(f"{prefix}{clean}")
    return labels

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

def calculate_layer_stats(attn_tensor):
    avg_attn = attn_tensor.mean(dim=0).detach().cpu().numpy()
    epsilon = 1e-10
    entropy = -np.sum(avg_attn * np.log(avg_attn + epsilon), axis=-1)
    return avg_attn, entropy

# --- 3. PYVIS (Vertikal/Stor font) ---
def create_styled_graph(tokens, labels, avg_attn, entropy, hide_sys=True):
    net = Network(height="650px", width="100%", notebook=False, directed=True)
    e_min, e_max = float(np.min(entropy)), float(np.max(entropy))
    
    for i, token in enumerate(tokens):
        if hide_sys and is_special(token):
            continue
            
        display_label = labels[i]
        e_val = float(entropy[i])
        norm_e = (e_val - e_min) / (e_max - e_min + 1e-6)
        node_color = f"rgb({int(255*(1-norm_e))}, 100, {int(255*norm_e)})"
        
        net.add_node(
            int(i), 
            label=display_label, 
            color=node_color,
            shape="box", 
            font={'size': 48, 'color': 'white', 'face': 'monospace'},
            margin=15,
            title=f"Original: {token}\nIndeks: {i}\nEntropi: {e_val:.3f}"
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
                net.add_edge(int(j), int(i), value=weight * 22, arrows='to', color='rgba(150,150,150,0.5)')
    
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": { "gravitationalConstant": -250, "springLength": 250, "avoidOverlap": 1 },
        "solver": "forceAtlas2Based"
      },
      "interaction": { "navigationButtons": true, "zoomView": true }
    }
    """)
    return net

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="D-chains Coreference", layout="wide")
st.title("🔗 D-chains: Analyse av Koreferanse")

with st.sidebar:
    model_name = st.selectbox("Modell", ["bert-base-multilingual-cased", "NbAiLab/nb-bert-base"])
    hide_sys = st.checkbox("Skjul spesialtokens", value=True)
    st.info("💡 Indekser vises kun ved tvetydighet (f.eks. 'han_7').")

text = st.text_input("Setning for analyse:", "Ola elsket Marit. Han ga henne en ring.")

if text:
    tokenizer, model = load_model(model_name)
    attentions, tokens = get_metrics(text, tokenizer, model)
    labels = get_token_labels(tokens)
    
    num_layers = attentions.shape[0]
    layer_indices = [0, num_layers // 2, num_layers - 1]

    for l_idx in layer_indices:
        st.write(f"## 🛠 Lag {l_idx + 1}")
        avg_attn, entropy = calculate_layer_stats(attentions[l_idx])
        
        net = create_styled_graph(tokens, labels, avg_attn, entropy, hide_sys)
        path = f"graph_v{l_idx}.html"
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=670)
        
        st.write("---")

    # --- 5. RÅDATA ---
    st.write("### 📊 Rådata for Lag 12 (Siste Lag)")
    last_attn, last_entropy = calculate_layer_stats(attentions[layer_indices[-1]])

    table_data = []
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if hide_sys and (is_special(tokens[i]) or is_special(tokens[j])):
                continue
            w = float(last_attn[i, j])
            if w > 0.001:
                table_data.append({
                    "Fra": labels[i], "Til": labels[j],
                    "Vekt": round(w, 5), "Entropi_Fra": round(float(last_entropy[i]), 3)
                })
    
    df = pd.DataFrame(table_data)
    if not df.empty:
        st.dataframe(df.sort_values(by="Vekt", ascending=False), use_container_width=True)
