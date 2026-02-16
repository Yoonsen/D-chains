import streamlit as st
import attention_graphs as at
import pyvis_graph as pg
import os
import re

st.title("LLM Dependency Visualizer")
st.markdown("""
This app visualizes how transformer models build up sentence structure through attention. Source code on [Github repo D-chains](https://github.com/Yoonsen/D-chains). Run this app locally to experiment with larger models.

The graphs below show which words attend to each other — and what kinds of phrases emerge.

* There is one section with graphs
* One section with phrasal extraction
* One section where you can download the last layer as a table with attention weights
""")

st.sidebar.header("Local settings")
cache_dir = st.sidebar.text_input("Hugging Face cache directory", "./models/hf-cache")
use_cuda = st.sidebar.checkbox("Use CUDA if available", value=True)
include_all_tokens = st.sidebar.checkbox(
    "Include all tokens (special/boundary)",
    value=False,
    help="Off (default): cleaner token view. On: include <bos>, <eos>, and boundary subtokens.",
)
disambiguate_repeats = st.sidebar.checkbox(
    "Disambiguate repeated tokens",
    value=True,
    help="Adds subscripts for repeated tokens (e.g., at₁, at₂).",
)
directed_graph = st.sidebar.checkbox(
    "Directed graph (show arrows)",
    value=False,
    help="Show attention direction u -> v with arrows.",
)
show_secondary_edges = st.sidebar.checkbox(
    "Show secondary edges (green)",
    value=False,
    help="Adds next-best attention links as green edges.",
)
primary_top_k = st.sidebar.number_input(
    "Primary edges per node",
    min_value=1,
    max_value=8,
    value=3,
    step=1,
)
secondary_top_k = st.sidebar.number_input(
    "Secondary edges per node",
    min_value=0,
    max_value=5,
    value=1,
    step=1,
)
head_aggregation = st.sidebar.selectbox(
    "Head aggregation",
    ["mean", "max"],
    index=0,
    help="'mean' averages heads; 'max' keeps strongest head per token pair.",
)
layer_start = st.sidebar.number_input(
    "Start layer (inclusive)",
    min_value=0,
    max_value=128,
    value=4,
    step=1,
)
layer_end_inclusive = st.sidebar.number_input(
    "End layer (inclusive, -1 = last)",
    min_value=-1,
    max_value=128,
    value=-1,
    step=1,
)
phrase_method = st.sidebar.selectbox(
    "Phrase extraction",
    [
        "Attention chunks (recommended)",
        "Hierarchical bracketed cliques",
        "Clique propagation",
    ],
    index=0,
)


col1, col2, col3 = st.columns([5, 2, 2])
with col1:
    text = st.text_input("Input sentence", "who do you want to compete with?")
with col2:
    model = st.selectbox("Model", [
        "bert-base-multilingual-cased",
        "NbAiLab/nb-bert-base",
        "NbAiLab/borealis-270m-instruct-preview",
        "NbAiLab/borealis-1b-instruct-preview",
    ])
with col3:
    st.markdown(" ")
    run = st.button("Analyze")

if run:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    if layer_end_inclusive != -1 and layer_end_inclusive < layer_start:
        st.error("End layer must be >= start layer, or -1 for last layer.")
        st.stop()

    layer_range = (
        int(layer_start),
        None if layer_end_inclusive == -1 else int(layer_end_inclusive) + 1,
    )

    try:
        st.write("### Graph structures")
        layers, words = at.visualize_per_layer(
            text,
            model_name=model,
            layer_range=layer_range,
            top_k=int(primary_top_k),
            secondary_top_k=int(secondary_top_k),
            include_secondary_edges=show_secondary_edges,
            head_aggregation=head_aggregation,
            cache_dir=cache_dir or None,
            use_cuda=use_cuda,
            include_all_tokens=include_all_tokens,
            disambiguate_repeated_tokens=disambiguate_repeats,
        )
    except RuntimeError as exc:
        st.error(
            "Model run failed on this hardware/session. "
            "For the demo, use borealis-270m or borealis-1b."
        )
        st.exception(exc)
        st.stop()
    except Exception as exc:
        st.error("Unexpected error while analyzing this sentence.")
        st.exception(exc)
        st.stop()
    
    # DEBUG: Se hva vi faktisk har
        
    pg.show_pyvis_layers(layers, words, directed=directed_graph) 
    st.write("### Phrasal structures from graphs")
    if phrase_method == "Attention chunks (recommended)":
        st.caption("Built from strongest attention links per layer (connected components).")
    elif phrase_method == "Hierarchical bracketed cliques":
        st.caption("Built from strong cliques; each clique is linearized by a main path and connected recursively.")
    else:
        st.caption("Built from propagated 3-cliques.")
    for i, layer in enumerate(layers, 1):
        st.write(f"Layer {i}")
        phrase_layer = layer.copy()
        secondary_edges = [
            (u, v)
            for u, v, d in phrase_layer.edges(data=True)
            if d.get("edge_type") == "secondary"
        ]
        if secondary_edges:
            phrase_layer.remove_edges_from(secondary_edges)
        if phrase_method == "Attention chunks (recommended)":
            phrases = at.find_attention_phrase_chunks(phrase_layer, words, min_weight_quantile=0.6)
            for j, phrase in enumerate(phrases, 1):
                st.write(f"{j}. {' '.join(phrase)}")
        elif phrase_method == "Hierarchical bracketed cliques":
            structures = at.find_hierarchical_clique_brackets(phrase_layer, words, min_weight_quantile=0.4, min_clique_size=3)
            if not structures:
                st.write("No stable clique structure in this layer.")
            else:
                st.write("1. " + " | ".join(structures))
        else:
            phrases = at.find_3clique_clusters(phrase_layer.to_undirected(), words)
            for j, phrase in enumerate(phrases, 1):
                st.write(f"{j}. {' '.join(phrase)}")

            
    st.write("### As table with weights for last layer")
    combined = layers[-1] #at.combine_graphs_weighted(layers)
    subscript_suffix_re = re.compile(r"[₀₁₂₃₄₅₆₇₈₉]+$")

    def node_label(n):
        data = combined.nodes[n] if n in combined.nodes else {}
        return data.get("label", str(n))

    def clean_display_label(label):
        return subscript_suffix_re.sub("", str(label))

    edges = [
            {
                "Fra": clean_display_label(node_label(u)),
                "Til": clean_display_label(node_label(v)),
                "Fra_id": u,
                "Til_id": v,
                "Vekt": round(d.get("weight", 0), 3),
                "Etikett": d.get("label", ""),
                "Type": d.get("edge_type", "primary"),
            }
            for u, v, d in combined.edges(data=True)
        ]
    st.dataframe(edges)

    st.write("### Node entropy (last layer)")
    node_rows = []
    for n, data in combined.nodes(data=True):
        node_rows.append(
            {
                "Node": clean_display_label(data.get("label", str(n))),
                "Node_id": n,
                "Entropy_full": round(float(data.get("entropy", 0.0)), 4),
                "Entropy_visible": round(float(data.get("entropy_visible", 0.0)), 4),
            }
        )
    node_rows = sorted(node_rows, key=lambda x: x["Node_id"])
    st.dataframe(node_rows)
