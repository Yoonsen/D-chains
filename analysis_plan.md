# Analysis Plan: Pronouns, Control, and Argument Structure in Attention Graphs

## Goal

Investigate whether attention-graph structure reveals systematic differences in:
- pronoun-antecedent behavior,
- control constructions (matrix verb + infinitive),
- subject/object representation and links to verbal morphology.

The main comparison is across:
- `bert-base-multilingual-cased`
- `NbAiLab/nb-bert-base`
- `NbAiLab/borealis-270m-instruct-preview`
- `NbAiLab/borealis-1b-instruct-preview`


## Core Research Questions

1. Do pronouns connect to plausible antecedents directly (primary edges) or indirectly (secondary paths)?
2. In control constructions, does the controller token connect to both matrix and infinitival domains?
3. Is subject representation more tied to verb + verbal morphology than object representation?
4. Do rare/fragmented words increase dispersion and weaken role-relevant links?


## Hypotheses

- H1 (Pronoun): mBERT/nb-BERT will show stronger direct pronoun-antecedent clustering than Borealis under identical pruning settings.
- H2 (Pronoun): Borealis may encode antecedent links more indirectly (secondary edges / multi-hop paths).
- H3 (Control): Controller subjects will show dual-domain connectivity (matrix verb + infinitive) across layers.
- H4 (Arguments): Subject tokens will have stronger links to verbal morphology-bearing subtokens than objects.
- H5 (Lexical stability): Rare or fragmented words show higher `Entropy_full` and weaker re-cohesion.
- H6 (Aggregation sensitivity): `head=max` recovers specialized links that are diluted under `head=mean`.


## Experimental Design

### 1) Pronoun-Antedecent Set

Use ambiguity and disambiguation minimal pairs:
- `Eva sa at hun kom fordi hun var trøtt.`
- `Ola ga Per boka fordi han var sen.`
- `Kari ringte Anne da hun kom hjem.`
- Controlled variants with gender/number cues where possible.

Track:
- direct edge existence `pronoun -> antecedent` and `antecedent -> pronoun`
- top-1/top-2 rank of antecedent among pronoun outgoing links
- whether link is `primary` or only appears as `secondary`
- earliest layer where stable link appears


### 2) Control/Sub-Verb Set

Templates:
- Subject control: `Han prøvde å sykle.`
- Subject control: `Hun lovet å komme.`
- Object control contrast: `Hun ba ham dra.`
- Raising/control-like contrasts where natural.

Track:
- controller links to matrix verb subtokens
- controller links to infinitive subtokens
- shared-cluster membership over layers
- divergence between subject-control and object-control patterns


### 3) Subject vs Object + Morphology Set

Minimal pairs:
- `Han spiste eplet.` vs `Eplet ble spist av ham.`
- `Han fant en sykkel i båthuset.` vs `Han fant en sykkel i parken.`
- Verb inflection variants where possible.

Track:
- subject/object links to verb stem subtokens
- subject/object links to verb inflection subtokens (e.g., `-te`, `-de`, etc.)
- asymmetry score (subject-minus-object) per layer
- entropy profile by role


## Metrics (Operational)

### A) Antecedent Rank

For each pronoun token at layer `l`, rank all outgoing targets by attention weight.
- `rank=1` if antecedent is top target.
- Report mean rank and hit@k across examples/models.


### B) Direct vs Indirect Coreference Signal

- Direct signal: edge exists between pronoun and antecedent (either direction).
- Indirect signal: no direct edge, but short path length (1-2 hops) via high-weight nodes.


### C) Control Dual-Link Score

For controller token `c`:
- `S_matrix = max link(c, matrix-verb-subtokens)`
- `S_inf = max link(c, infinitive-subtokens)`
- Dual-link score: harmonic mean of `S_matrix` and `S_inf`.


### D) Role-Morphology Coupling

For subject/object token `x`:
- `M_stem(x)` = mean/max links to verb stem subtokens
- `M_morph(x)` = mean/max links to inflection subtokens
- Compare `(M_stem, M_morph)` for subject vs object.


### E) Entropy-Based Dispersion

Already available in app:
- `Entropy_full`: entropy over full attention distribution.
- `Entropy_visible`: entropy over visible (pruned) graph.

Use:
- `Gap = Entropy_visible - Entropy_full` as pruning-loss indicator.
- Higher gap suggests important mass lies outside visualized edges.


## Recommended App Settings for Analysis Runs

Baseline settings:
- `Primary edges per node = 1`
- `Secondary edges per node = 1`
- `Show secondary edges = On`
- `Head aggregation = max` (then repeat with `mean`)
- `Directed graph = On` for pronoun direction tests
- layer range sweeps:
  - early: `0..3`
  - mid: `4..8`
  - late: `9..last` (adjust per model depth)

Run each set with identical settings before changing one factor.


## Data Logging Template

Create one row per sentence x model x layer-window:

- `sentence_id`
- `phenomenon` (`pronoun`, `control`, `role_morphology`)
- `model`
- `head_aggregation`
- `primary_k`
- `secondary_k`
- `layer_start`
- `layer_end`
- `pronoun_token`
- `candidate_antecedents`
- `best_antecedent`
- `antecedent_rank`
- `direct_edge` (0/1)
- `indirect_path_len`
- `control_dual_link_score`
- `subject_morph_score`
- `object_morph_score`
- `entropy_full_mean`
- `entropy_visible_mean`
- `notes`


## Analysis Workflow

1. Build sentence sets (30-50 examples per phenomenon).
2. Run all models with fixed baseline settings.
3. Export edge/node tables from app (last layer) and capture screenshots for qualitative cases.
4. Re-run with `head=mean` and compare deltas.
5. Summarize:
   - quantitative table (hit@k, rank, entropy gaps),
   - qualitative case studies (2-4 per phenomenon),
   - failure/anomaly section.


## Expected Contributions

- A graph-based probe showing model differences in pronoun/coreference behavior.
- Evidence on control structure representation under different aggregation regimes.
- Role-sensitive morphology coupling patterns (subject vs object).
- Practical methodological insight: how pruning and head aggregation can hide or reveal linguistic structure.
