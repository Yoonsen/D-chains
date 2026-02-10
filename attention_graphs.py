
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import lru_cache


SPECIAL_TOKENS = {
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[UNK]",
    "<bos>",
    "<eos>",
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
}


def _is_special_token(token):
    if token in SPECIAL_TOKENS:
        return True
    # Covers model-specific placeholders like <|...|>
    if token.startswith("<|") and token.endswith("|>"):
        return True
    return False


def _normalize_token_label(token):
    # WordPiece continuation marker.
    normalized = token.replace("##", "-")
    # SentencePiece / GPT-style word boundary markers.
    normalized = normalized.replace("▁", "").replace("Ġ", "")
    return normalized.strip()

def extract_soft_dependency_tree(text, model_name="bert-base-cased", layer_range=(4, 9)):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = outputs.attentions
    start, end = layer_range
    avg_attention = torch.stack(attentions[start:end]).mean(dim=0)[0].mean(dim=0).numpy()

    G = nx.Graph()
    G.add_nodes_from(tokens)

    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        top_indices = np.argsort(avg_attention[i])[-3:]
        for j in top_indices:
            source = tokens[j]
            if source not in ['[CLS]', '[SEP]']:
                weight = float(avg_attention[i][j])
                G.add_edge(token, source, weight=weight)

    # Build spanning tree using maximum attention
    T = nx.maximum_spanning_tree(G, weight='weight')
    return tokens, T

# Visualize dependency as graph with role labels and weights

def visualize_soft_dependencies(text, model_name="bert-base-cased", layer_range=(4, 9), top_k=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = outputs.attentions
    start, end = layer_range
    avg_attention = torch.stack(attentions[start:end]).mean(dim=0)[0].mean(dim=0).numpy()

    G = nx.Graph()
    G.add_nodes_from(tokens)

    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        top_indices = np.argsort(avg_attention[i])[-top_k:]
        for j in top_indices:
            source = tokens[j]
            if source not in ['[CLS]', '[SEP]']:
                label = f"({source}, {token})"
                weight = round(float(avg_attention[i][j]), 3)
                G.add_edge(source, token, label=label, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'label')
    weights = [G[u][v]['weight'] * 5 for u,v in G.edges()]

    plt.figure(figsize=(12, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=weights, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkgreen')
    plt.title(f"Typed Soft Dependency Graph for: '{text}'")
    plt.show()



import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

@lru_cache(maxsize=8)
def _load_model_bundle(model_name, cache_dir=None, use_cuda=True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    load_kwargs = {
        "output_attentions": True,
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }

    model = None
    load_errors = []

    for loader in (AutoModelForCausalLM, AutoModel):
        try:
            model = loader.from_pretrained(model_name, **load_kwargs)
            break
        except TypeError:
            # Some models do not accept attn_implementation on load.
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            try:
                model = loader.from_pretrained(model_name, **fallback_kwargs)
                break
            except Exception as exc:  # noqa: PERF203
                load_errors.append(f"{loader.__name__}: {exc}")
        except Exception as exc:  # noqa: PERF203
            load_errors.append(f"{loader.__name__}: {exc}")

    if model is None:
        joined_errors = "\n".join(load_errors)
        raise RuntimeError(f"Could not load model '{model_name}'.\n{joined_errors}")

    if hasattr(model, "config"):
        model.config.output_attentions = True
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def visualize_per_layer(
    text,
    model_name="bert-base-cased",
    layer_range=(4, 9),
    top_k=3,
    group_subwords=True,
    cache_dir=None,
    use_cuda=True,
):
    tokenizer, model, device = _load_model_bundle(
        model_name=model_name,
        cache_dir=cache_dir,
        use_cuda=use_cuda,
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Filter special tokens and empty spacer tokens, but keep original positions.
    tokens_filtered = []
    word_order = [None] * len(tokens)
    for i, token in enumerate(tokens):
        if _is_special_token(token):
            continue
        clean_label = _normalize_token_label(token)
        if not clean_label:
            continue
        tokens_filtered.append((i, clean_label))
        word_order[i] = clean_label

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

    if attentions is None:
        raise ValueError(
            f"Model '{model_name}' did not return attention tensors. "
            "Try another model or update transformers."
        )

    num_layers = len(attentions)
    start, end = layer_range
    start = max(0, min(start, num_layers - 1))
    end = max(start + 1, min(end, num_layers))
    Gs = []

    for layer in range(start, end):
        attn = attentions[layer][0].mean(dim=0).detach().cpu().numpy()
        G = nx.DiGraph()

        for i, label in tokens_filtered:
            G.add_node(i, label=label)

        for i, _ in tokens_filtered:
            scores = np.zeros(len(tokens))
            for j in range(len(tokens)):
                scores[j] = attn[i][j]

            top_indices = np.argsort(scores)[-top_k:]
            for j in top_indices:
                if j == i or _is_special_token(tokens[j]):
                    continue
                source_label = _normalize_token_label(tokens[j])
                target_label = _normalize_token_label(tokens[i])
                label = f"({source_label}, {target_label})"
                weight = round(float(attn[i][j]), 3)
                G.add_edge(j, i, label=label, weight=weight)

        Gs.append(G)

    return Gs, word_order

def find_3clique_clusters(G, word_order=None):
    cliques = [set(c) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    merged = []

    while cliques:
        base = cliques.pop(0)
        changed = True
        while changed:
            changed = False
            to_merge = []
            for c in cliques:
                if len(base & c) >= 2:
                    base |= c
                    to_merge.append(c)
                    changed = True
            for c in to_merge:
                cliques.remove(c)
        merged.append(base)

    if word_order:
        merged = [
            [word_order[n] for n in sorted(cluster) if n < len(word_order) and word_order[n]]
            for cluster in merged
        ]
    else:
        merged = [sorted(list(cluster), key=lambda w: w) for cluster in merged]

    return merged  # list of sorted clusters only


# Define the function to analyze subword fragmentation
def analyze_subword_fragmentation(text, model_name="bert-base-cased", layer=-1, top_k=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = outputs.attentions[layer][0].mean(dim=0).numpy()

    G = nx.Graph()
    G.add_nodes_from(tokens)

    # Build the attention-based graph
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        top_indices = np.argsort(attentions[i])[-top_k:]
        for j in top_indices:
            source = tokens[j]
            if source not in ['[CLS]', '[SEP]']:
                weight = float(attentions[i][j])
                G.add_edge(source, token, weight=weight)

    # Group subwords into words
    words = []
    current_word = ""
    word_map = []
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                words.append(current_word)
            current_word = token
        word_map.append((current_word, token))
    if current_word:
        words.append(current_word)

    # Track token locations in the graph
    word_to_tokens = defaultdict(list)
    for word, token in word_map:
        word_to_tokens[word].append(token)

    # Detect fragmented words
    fragmented_words = {}
    for word, sub_tokens in word_to_tokens.items():
        subgraph = G.subgraph(sub_tokens)
        components = list(nx.connected_components(subgraph))
        if len(components) > 1:
            fragmented_words[word] = components

    return fragmented_words


# In[60]:


def show_graph(G, title=""):
        pos = nx.spring_layout(G, seed=42)
        edge_labels = nx.get_edge_attributes(G, 'label')
        weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        plt.figure(figsize=(12, 7))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=weights, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkgreen')
        plt.title(title)
        plt.show()

def show_layers(G):
    for i, g in enumerate(G,1):
        show_graph(g, f"layer {i}") 
    show_graph(combine_graphs_weighted(G), "All layers combined")



def combine_graphs_weighted(layers):
    combined_weights = defaultdict(float)
    total_layers = len(layers)
    total_weight = sum(range(1, total_layers + 1))

    for i, G in enumerate(layers, 1):
        weight_factor = i / total_weight
        for u, v, d in G.edges(data=True):
            key = tuple(sorted((u, v)))
            combined_weights[key] += d['weight'] * weight_factor

    # Build the final graph
    combined_graph = nx.Graph()
    for (u, v), weight in combined_weights.items():
        combined_graph.add_edge(u, v, weight=round(weight, 3), label=f"{u}-{v}-{round(weight,3)}")

    return combined_graph




