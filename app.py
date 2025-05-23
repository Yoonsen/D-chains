import streamlit as st
import attention_graphs as at
import pyvis_graph as pg

st.title("BERT Soft Dependency Visualizer")

# Input
text = st.text_input("Skriv inn en setning", "Hvem trodde du at ble syk")
model = st.selectbox("Velg modell", [
    "NbAiLab/nb-bert-base",
    "ltgoslo/norbert3-small",
    "bert-base-multilingual-cased"
])

if st.button("Kjør analyse"):
    layers = at.visualize_per_layer(text, model_name=model)
    combined = at.combine_graphs_weighted(layers)
    pg.show_pyvis_layers(layers + [combined])



