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


st.header("📚 Eksempler på syntaktiske mønstre")

examples = {
    "Subjekt + finittverb": "Han spiser epler",
    "Hjelpeverb + partisipp": "Han blir lest",
    "Verb + objekt": "Hun skrev brevet",
    "Preposisjon + komplement": "Boken ligger på bordet",
    "Relativbinding": "Mannen som stod der, smilte"
}

selected_example = st.selectbox("Velg et eksempel", list(examples.keys()))

if st.button("Vis eksempel"):
    sentence = examples[selected_example]
    st.markdown(f"**Setning:** {sentence}")
    layers = at.visualize_per_layer(sentence, model_name=model)
    combined = at.combine_graphs_weighted(layers)
    pg.show_pyvis_layers(layers + [combined])

if st.checkbox("📊 Vis koblinger som tabell"):
    edges = [
        {"Fra": u, "Til": v, "Vekt": round(d.get("weight", 0), 3), "Etikett": d.get("label", "")}
        for u, v, d in combined.edges(data=True)
    ]
    st.dataframe(edges)