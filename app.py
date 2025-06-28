import streamlit as st
import attention_graphs as at
import pyvis_graph as pg

st.title("LLM Dependency Visualizer")

# Input
text = st.text_input("Input sentence or phrase", "who do you want to compete with? ")
model = st.selectbox("Model", [
    "NbAiLab/nb-bert-base",
    "ltgoslo/norbert3-small",
    "bert-base-multilingual-cased"
])

if st.button("Analyze"):
    layers = at.visualize_per_layer(text, model_name=model)
    combined = at.combine_graphs_weighted(layers)
    pg.show_pyvis_layers(layers + [combined])


if st.checkbox("Show as table"):
    edges = [
        {"Fra": u, "Til": v, "Vekt": round(d.get("weight", 0), 3), "Etikett": d.get("label", "")}
        for u, v, d in combined.edges(data=True)
    ]
    st.dataframe(edges)