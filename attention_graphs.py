
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


def _normalize_token_label(
    token,
    is_first_content_token=False,
    assume_unmarked_is_inner=False,
):
    # WordPiece continuation marker.
    if token.startswith("##"):
        return f"-{token[2:]}".strip()

    # SentencePiece / GPT-style word boundary markers (start of token/word).
    if token.startswith("▁") or token.startswith("Ġ"):
        return token[1:].strip()

    normalized = token.strip()
    # For boundary-marker tokenizers (e.g. SentencePiece), unmarked non-first
    # tokens are usually inner subtokens. Mark them with '-' like BERT output.
    if (
        assume_unmarked_is_inner
        and not is_first_content_token
        and normalized
        and any(ch.isalnum() for ch in normalized)
        and not normalized.startswith("-")
    ):
        normalized = f"-{normalized}"
    return normalized


def _looks_like_subtoken_join(left_label, right_label):
    """
    Heuristic for fragmented word pieces that should stay connected.
    Examples: app + -le, at + -e, break + -fast
    """
    if not left_label or not right_label:
        return False
    if right_label.startswith("-"):
        return True
    if left_label.endswith("-"):
        return True
    return False


def _to_subscript(n):
    mapping = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(mapping)


def _disambiguate_duplicate_labels(labels):
    totals = defaultdict(int)
    for label in labels:
        totals[label] += 1

    seen = defaultdict(int)
    out = []
    for label in labels:
        if totals[label] <= 1:
            out.append(label)
            continue
        seen[label] += 1
        out.append(f"{label}{_to_subscript(seen[label])}")
    return out

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
    layer_range=(4, None),
    top_k=3,
    secondary_top_k=0,
    include_secondary_edges=False,
    head_aggregation="mean",
    group_subwords=True,
    cache_dir=None,
    use_cuda=True,
    include_all_tokens=False,
    disambiguate_repeated_tokens=True,
):
    tokenizer, model, device = _load_model_bundle(
        model_name=model_name,
        cache_dir=cache_dir,
        use_cuda=use_cuda,
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    boundary_marker_tokenizer = any(
        (t.startswith("▁") or t.startswith("Ġ"))
        for t in tokens
        if not _is_special_token(t)
    )

    # Build token labels while optionally filtering special/boundary tokens.
    token_positions = []
    raw_labels = []
    content_count = 0
    for i, token in enumerate(tokens):
        if include_all_tokens:
            clean_label = token
        else:
            if _is_special_token(token):
                continue
            clean_label = _normalize_token_label(
                token,
                is_first_content_token=(content_count == 0),
                assume_unmarked_is_inner=boundary_marker_tokenizer,
            )
            if not clean_label:
                continue

        token_positions.append(i)
        raw_labels.append(clean_label)
        content_count += 1

    if disambiguate_repeated_tokens:
        display_labels = _disambiguate_duplicate_labels(raw_labels)
    else:
        display_labels = raw_labels

    tokens_filtered = list(zip(token_positions, display_labels))
    word_order = [None] * len(tokens)
    for pos, label in tokens_filtered:
        word_order[pos] = label

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
    if start is None:
        start = 0
    if end is None:
        end = num_layers
    start = max(0, min(start, num_layers - 1))
    end = max(start + 1, min(end, num_layers))
    Gs = []

    for layer in range(start, end):
        layer_attn = attentions[layer][0]
        if head_aggregation == "max":
            attn = layer_attn.max(dim=0).values.detach().cpu().numpy()
        else:
            attn = layer_attn.mean(dim=0).detach().cpu().numpy()
        G = nx.DiGraph()
        valid_positions = [idx for idx, _ in tokens_filtered]
        full_positions = list(range(len(tokens)))

        for i, label in tokens_filtered:
            # Full entropy: uses all attention targets from model output, including
            # edges that are not shown in the pruned graph.
            if len(full_positions) <= 1:
                entropy_full = 0.0
            else:
                probs_full = np.array([max(float(attn[i][j]), 0.0) for j in full_positions], dtype=float)
                total_full = float(probs_full.sum())
                if total_full <= 0:
                    entropy_full = 0.0
                else:
                    p_full = probs_full / total_full
                    p_full = np.clip(p_full, 1e-12, 1.0)
                    raw_entropy_full = float(-(p_full * np.log(p_full)).sum())
                    entropy_full = raw_entropy_full / float(np.log(len(p_full))) if len(p_full) > 1 else 0.0

            # Visible entropy: only over currently displayed token set.
            if len(valid_positions) <= 1:
                entropy_visible = 0.0
            else:
                probs_visible = np.array([max(float(attn[i][j]), 0.0) for j in valid_positions], dtype=float)
                total_visible = float(probs_visible.sum())
                if total_visible <= 0:
                    entropy_visible = 0.0
                else:
                    p_visible = probs_visible / total_visible
                    p_visible = np.clip(p_visible, 1e-12, 1.0)
                    raw_entropy_visible = float(-(p_visible * np.log(p_visible)).sum())
                    entropy_visible = raw_entropy_visible / float(np.log(len(p_visible))) if len(p_visible) > 1 else 0.0

            G.add_node(
                i,
                label=label,
                entropy=round(float(entropy_full), 4),
                entropy_visible=round(float(entropy_visible), 4),
            )

        for i, _ in tokens_filtered:
            scores = np.zeros(len(tokens))
            for j in range(len(tokens)):
                scores[j] = attn[i][j]

            sorted_desc = np.argsort(scores)[::-1]
            primary_added = 0
            secondary_added = 0

            for j in sorted_desc:
                if j == i:
                    continue
                if (not include_all_tokens) and _is_special_token(tokens[j]):
                    continue

                source_label = word_order[j] if j < len(word_order) else None
                target_label = word_order[i] if i < len(word_order) else None
                if not source_label:
                    source_label = (
                        _normalize_token_label(
                            tokens[j],
                            is_first_content_token=False,
                            assume_unmarked_is_inner=boundary_marker_tokenizer,
                        )
                        if not include_all_tokens
                        else tokens[j]
                    )
                if not target_label:
                    target_label = (
                        _normalize_token_label(
                            tokens[i],
                            is_first_content_token=False,
                            assume_unmarked_is_inner=boundary_marker_tokenizer,
                        )
                        if not include_all_tokens
                        else tokens[i]
                    )

                label = f"({source_label}, {target_label})"
                weight = round(float(attn[i][j]), 3)

                if primary_added < top_k:
                    G.add_edge(j, i, label=label, weight=weight, edge_type="primary")
                    primary_added += 1
                    continue

                if include_secondary_edges and secondary_added < secondary_top_k:
                    if G.has_edge(j, i):
                        continue
                    G.add_edge(j, i, label=label, weight=weight, edge_type="secondary")
                    secondary_added += 1

                if primary_added >= top_k and (not include_secondary_edges or secondary_added >= secondary_top_k):
                    break

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


def find_attention_phrase_chunks(
    G,
    word_order=None,
    min_weight_quantile=0.6,
    min_size=2,
    force_subtoken_links=True,
):
    """
    Extract phrase-like chunks from a layer graph using strong attention links.

    Strategy:
    1) Keep only edges above a per-layer weight quantile.
    2) Build undirected connected components from those strong edges.
    3) Map node positions back to token order.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return []

    weights = np.array([float(d.get("weight", 0.0)) for _, _, d in edges], dtype=float)
    threshold = float(np.quantile(weights, min_weight_quantile))

    H = nx.Graph()
    for u, v, d in edges:
        w = float(d.get("weight", 0.0))
        if w < threshold:
            continue
        if H.has_edge(u, v):
            H[u][v]["weight"] = max(H[u][v]["weight"], w)
        else:
            H.add_edge(u, v, weight=w)

    # Keep fragmented subtokens connected even when their attention link is
    # just below threshold, so phrase chunks stay word-like.
    if force_subtoken_links and word_order is not None:
        valid_positions = [idx for idx, label in enumerate(word_order) if label]
        for pos, next_pos in zip(valid_positions, valid_positions[1:]):
            left_label = word_order[pos]
            right_label = word_order[next_pos]
            if _looks_like_subtoken_join(left_label, right_label):
                if H.has_edge(pos, next_pos):
                    H[pos][next_pos]["weight"] = max(H[pos][next_pos]["weight"], 1.0)
                else:
                    H.add_edge(pos, next_pos, weight=1.0)

    chunks = []
    for comp in nx.connected_components(H):
        if len(comp) < min_size:
            continue
        ordered_nodes = sorted(comp)
        if word_order is not None:
            labels = [
                word_order[n]
                for n in ordered_nodes
                if n < len(word_order) and word_order[n]
            ]
        else:
            labels = [str(n) for n in ordered_nodes]
        if labels:
            chunks.append((ordered_nodes[0], labels))

    chunks.sort(key=lambda x: x[0])
    return [labels for _, labels in chunks]


def _to_undirected_weighted_graph(G):
    U = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        if U.has_edge(u, v):
            U[u][v]["weight"] = max(U[u][v]["weight"], w)
        else:
            U.add_edge(u, v, weight=w)
    return U


def _build_strong_undirected_graph(
    G,
    min_weight_quantile=0.6,
    word_order=None,
    force_subtoken_links=False,
):
    edges = list(G.edges(data=True))
    if not edges:
        return nx.Graph()

    weights = np.array([float(d.get("weight", 0.0)) for _, _, d in edges], dtype=float)
    threshold = float(np.quantile(weights, min_weight_quantile))

    U = _to_undirected_weighted_graph(G)
    H = nx.Graph()
    for u, v, d in U.edges(data=True):
        w = float(d.get("weight", 0.0))
        if w >= threshold:
            H.add_edge(u, v, weight=w)

    if force_subtoken_links and word_order is not None:
        valid_positions = [idx for idx, label in enumerate(word_order) if label]
        for pos, next_pos in zip(valid_positions, valid_positions[1:]):
            left_label = word_order[pos]
            right_label = word_order[next_pos]
            if not _looks_like_subtoken_join(left_label, right_label):
                continue
            base_weight = 1.0
            if U.has_edge(pos, next_pos):
                base_weight = max(base_weight, float(U[pos][next_pos].get("weight", 0.0)))
            if H.has_edge(pos, next_pos):
                H[pos][next_pos]["weight"] = max(H[pos][next_pos]["weight"], base_weight)
            else:
                H.add_edge(pos, next_pos, weight=base_weight)
    return H


def _weighted_path_diameter_nodes(T):
    if T.number_of_nodes() == 0:
        return None, None
    if T.number_of_nodes() == 1:
        n = next(iter(T.nodes()))
        return n, n

    def inv_weight(_, __, data):
        return 1.0 / (float(data.get("weight", 0.0)) + 1e-9)

    start = next(iter(T.nodes()))
    d1 = nx.single_source_dijkstra_path_length(T, start, weight=inv_weight)
    u = max(d1, key=d1.get)
    d2 = nx.single_source_dijkstra_path_length(T, u, weight=inv_weight)
    v = max(d2, key=d2.get)
    return u, v


def _clique_main_path_labels(H, clique_nodes, word_order=None):
    C = H.subgraph(clique_nodes).copy()
    if C.number_of_nodes() == 0:
        return []
    if C.number_of_nodes() == 1:
        only = next(iter(C.nodes()))
        if word_order is None:
            return [str(only)]
        return [word_order[only] if only < len(word_order) and word_order[only] else str(only)]

    T = nx.maximum_spanning_tree(C, weight="weight")
    u, v = _weighted_path_diameter_nodes(T)
    if u is None or v is None:
        ordered = sorted(clique_nodes)
    else:
        def inv_weight(_, __, data):
            return 1.0 / (float(data.get("weight", 0.0)) + 1e-9)
        ordered = nx.shortest_path(T, source=u, target=v, weight=inv_weight)
    if len(ordered) >= 2 and ordered[0] > ordered[-1]:
        ordered = list(reversed(ordered))

    if word_order is None:
        return [str(n) for n in ordered]
    out = []
    for n in ordered:
        if n < len(word_order) and word_order[n]:
            out.append(word_order[n])
        else:
            out.append(str(n))
    return out


def _max_link_between_cliques(H, c1, c2):
    best = 0.0
    for u in c1:
        for v in c2:
            if H.has_edge(u, v):
                best = max(best, float(H[u][v].get("weight", 0.0)))
    return best


def _linearize_component_nodes(H_sub):
    if H_sub.number_of_nodes() == 0:
        return []
    if H_sub.number_of_nodes() == 1:
        return [next(iter(H_sub.nodes()))]
    T = nx.maximum_spanning_tree(H_sub, weight="weight")
    u, v = _weighted_path_diameter_nodes(T)
    if u is None or v is None:
        ordered = sorted(H_sub.nodes())
    else:
        def inv_weight(_, __, data):
            return 1.0 / (float(data.get("weight", 0.0)) + 1e-9)
        ordered = nx.shortest_path(T, source=u, target=v, weight=inv_weight)
    if len(ordered) >= 2 and ordered[0] > ordered[-1]:
        ordered = list(reversed(ordered))
    return ordered


def _format_linear_tokens_with_subtoken_brackets(labels):
    if not labels:
        return ""
    groups = [[labels[0]]]
    for label in labels[1:]:
        prev = groups[-1][-1]
        if _looks_like_subtoken_join(prev, label):
            groups[-1].append(label)
        else:
            groups.append([label])
    parts = []
    for group in groups:
        if len(group) >= 2:
            parts.append(f"[{' '.join(group)}]")
        else:
            parts.append(group[0])
    return " ".join(parts)


def find_hierarchical_clique_brackets(
    G,
    word_order=None,
    min_weight_quantile=0.4,
    min_clique_size=3,
    attach_singletons=True,
):
    """
    Build recursive bracketed phrase structures:
    - Detect strong-edge cliques
    - Linearize each clique by a strongest internal path
    - Connect cliques via strongest inter-clique edges
    """
    H = _build_strong_undirected_graph(
        G,
        min_weight_quantile=min_weight_quantile,
        word_order=word_order,
        force_subtoken_links=True,
    )
    if H.number_of_edges() == 0:
        return []

    cliques = [set(c) for c in nx.find_cliques(H) if len(c) >= min_clique_size]

    # Fallback: when few/no larger cliques exist, use smaller phrase chunks.
    if not cliques:
        chunk_lists = find_attention_phrase_chunks(
            G,
            word_order=word_order,
            min_weight_quantile=min_weight_quantile,
            min_size=2,
            force_subtoken_links=True,
        )
        return [f"[{' '.join(chunk)}]" for chunk in chunk_lists if chunk]

    clique_labels = []
    for c in cliques:
        path_labels = _clique_main_path_labels(H, c, word_order=word_order)
        clique_labels.append(path_labels)

    CG = nx.Graph()
    for idx, c in enumerate(cliques):
        CG.add_node(idx, clique=c)

    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            w = _max_link_between_cliques(H, cliques[i], cliques[j])
            if w > 0:
                CG.add_edge(i, j, weight=w)

    # Attach residual (non-clique) components to nearest clique when possible.
    U = _to_undirected_weighted_graph(G)
    attached_prefix = defaultdict(list)  # clique_idx -> [(pos, text)]
    attached_suffix = defaultdict(list)  # clique_idx -> [(pos, text)]
    free_components = []  # [(min_pos, text)]
    if attach_singletons and word_order is not None:
        clique_nodes = set().union(*cliques) if cliques else set()
        R = H.copy()
        R.remove_nodes_from(clique_nodes)
        for comp in nx.connected_components(R):
            comp_nodes = set(comp)
            # For free residual text, keep original token order for readability.
            ordered_nodes = sorted(comp_nodes)
            labels = [
                word_order[n] if n < len(word_order) and word_order[n] else str(n)
                for n in ordered_nodes
            ]
            comp_text = _format_linear_tokens_with_subtoken_brackets(labels)
            if not comp_text:
                continue

            best_idx = None
            best_w = 0.0
            for idx, c in enumerate(cliques):
                w = _max_link_between_cliques(U, comp_nodes, c)
                if w > best_w:
                    best_w = w
                    best_idx = idx

            comp_min = min(comp_nodes)
            if best_idx is None or best_w <= 0:
                free_components.append((comp_min, comp_text))
                continue

            cmin = min(cliques[best_idx])
            cmax = max(cliques[best_idx])
            if comp_min < cmin:
                attached_prefix[best_idx].append((comp_min, comp_text))
            elif comp_min > cmax:
                attached_suffix[best_idx].append((comp_min, comp_text))
            else:
                attached_suffix[best_idx].append((comp_min, comp_text))

    # If clique graph is disconnected, return one recursive expression per component.
    outputs = []
    for component in nx.connected_components(CG):
        sub = CG.subgraph(component).copy()
        tree = nx.maximum_spanning_tree(sub, weight="weight")
        root = min(component, key=lambda idx: min(cliques[idx]))

        def build_expr(node, parent=None):
            pref = [text for _, text in sorted(attached_prefix[node], key=lambda x: x[0])]
            suff = [text for _, text in sorted(attached_suffix[node], key=lambda x: x[0])]
            self_parts = pref + [f"[{' '.join(clique_labels[node])}]"] + suff
            self_bracket = " ".join(self_parts)
            children = [n for n in tree.neighbors(node) if n != parent]
            children.sort(key=lambda n: min(cliques[n]))
            if not children:
                return self_bracket
            child_exprs = [build_expr(ch, node) for ch in children]
            return "(" + " ".join([self_bracket] + child_exprs) + ")"

        outputs.append(build_expr(root))

    outputs = outputs + [text for _, text in sorted(free_components, key=lambda x: x[0])]
    return outputs


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




