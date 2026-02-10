# D-chains Manifest

This app connects to language models that expose attention weights and uses PyVis to visualize token-level attention graphs, where tokens are nodes and edges are weighted by attention strength.

The app inspects multiple layers of an LLM (or BERT) and displays how an input sentence is represented and transformed across layers.

The purpose is to demonstrate the highly hierarchical internal representations in LLMs, in contrast to the simplistic "next token predictor" meme.
