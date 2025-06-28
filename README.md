# 🔍 LLM Attention Graph Explorer

Visualiser attention-strukturer fra transformer-baserte språkmodeller som grafnettverk.
Skriv inn en setning, velg modell, og få opp både lagvise og aggregerte attention-grafer – direkte i Streamlit med Pyvis-interaktivitet.

## 🚀 Kom i gang

1. Installer avhengigheter:
```bash
pip install streamlit transformers torch networkx matplotlib pyvis
```

2. Kjør appen:
```bash
streamlit run app.py
```

## ✨ Funksjoner

- Visualiser top-k attention-forbindelser i hvert lag
- Se samlet graf med vektet lagkombinasjon
- Utforsk subword-fragmentering
- Støtte for flere Hugging Face-modeller

## 🧠 Relatert arbeid og inspirasjon

- [**BertViz**](https://github.com/jessevig/bertviz)
- [**exBERT**](https://exbert.net)
- [**Ecco**](https://github.com/allenai/ecco)
- [**TransformerLens**](https://github.com/neelnanda-io/TransformerLens)
- [**Captum**](https://github.com/pytorch/captum)
- [**Language Interpretability Tool**](https://pair-code.github.io/lit/)

## 🛡️ Lisens

MIT © Lars G Bagøien Johnsen
