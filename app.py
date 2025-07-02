import streamlit as st
import attention_graphs as at
import pyvis_graph as pg

st.title("LLM Dependency Visualizer")
st.markdown("""
This app visualizes how transformer models build up sentence structure through attention. Source code on [Github repo D-chains](https://github.com/Yoonsen/D-chains). Run this app locally to experiment with larger models.

The graphs below show which words attend to each other — and what kinds of phrases emerge.

* There is one section with graphs
* One section with phrasal extraction
* One section where you can download the last layer as a table with attention weights
""")


col1, col2, col3 = st.columns([5, 2, 2])
with col1:
    text = st.text_input("Input sentence", "who do you want to compete with?")
with col2:
    model = st.selectbox("Model", [
        "bert-base-multilingual-cased",
        "NbAiLab/nb-bert-base",
    ])
with col3:
    st.markdown(" ")
    run = st.button("Analyze")

if run:
    st.write("### Graph structures")
    layers, words = at.visualize_per_layer(text, model_name=model)
    pg.show_pyvis_layers(layers)

    st.write("### Phrasal structures from graphs")
    for i, layer in enumerate(layers, 1):
        st.write(f"Layer {i}")
        phrases = at.find_3clique_clusters(layer.to_undirected(), words)
        for j, phrase in enumerate(phrases, 1):
            st.write(f"{j}. {' '.join(phrase)}")

            
    st.write("### As table with weights for last layer")
    combined = layers[-1] #at.combine_graphs_weighted(layers)
    edges = [
            {"Fra": u, "Til": v, "Vekt": round(d.get("weight", 0), 3), "Etikett": d.get("label", "")}
            for u, v, d in combined.edges(data=True)
        ]
    st.dataframe(edges)