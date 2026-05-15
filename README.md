# CiteClaw

**Transparent, high-fidelity literature acquisition for autonomous scientific discovery.**

CiteClaw is a composable snowballing agent that constructs systematic research corpora
from a handful of seed papers. It traverses the global citation graph with a circuit-style
pipeline of rule-based, similarity-based, and LLM-based filter blocks — tiering cheap
checks before expensive LLM screening to deliver survey-grade collections at remarkably
low cost, with full provenance at every step.

In a recent benchmark mapping the evolution of AI4Biology, CiteClaw screened **18,000+
candidates** in 30 minutes and extracted **400+ milestone papers** (2% acceptance rate)
with high precision and recall — at a total cost of **$0.70** using Gemini 3 Flash-Lite.
[Visualization](https://drive.google.com/file/d/14nNVAOmLy8FvEwKcRBwOVRu7tSpZeyFL/view?usp=sharing).

---

## Why CiteClaw

Autonomous scientific discovery with LLM agents hinges on rigorous understanding of
frontier literature. Today, building survey-grade research corpora is a punishing manual
bottleneck, while existing LLM-based tools trade rigor for efficiency, producing opaque,
hallucination-prone knowledge bases that scientists cannot trust for high-stakes work.

CiteClaw is built to be the trustworthy substrate for the next generation of autonomous
research — from hypothesis generation and experiment planning to fully self-driving labs.

Key design principles:

- **Circuit-style composition** — `nn.Sequential`-inspired pipeline of pure signal
  transformers, with a `Parallel` branch primitive for independent exploration.
- **Tiered filtering** — YearFilter → CitationFilter → SimilarityFilter → LLMFilter,
  so expensive LLM calls only see pre-screened candidates.
- **Full provenance** — every paper's acceptance path is tracked; run state, shape
  summaries, and rejection counts are persisted to disk.
- **Community-aware reranking** — graph- or embedding-based clustering can drive
  diversity-aware top-K selection.
- **S2 SPECTER2 out of the box** — semantic similarity and topic modeling read
  Semantic Scholar's precomputed vectors; no embedding backend setup required.
- **Cost-conscious** — batched LLM dispatch, read-through SQLite cache, and
  cheap-first filter ordering push per-paper cost to cents.

---

## Installation

```bash
pip install -e .

# optional: UMAP+HDBSCAN topic modeling
pip install -e '.[topic_model]'

# optional: dev tooling (ruff, mypy, pytest)
pip install -e '.[dev]'
```

Requires Python 3.11+.

### Environment

Set your API keys in a `.env` file (or export them directly):

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...   # optional, improves S2 rate limits
```

---

## Quick start

1. Write a `configs/config.yaml` describing reusable filter blocks and a linear
   list of pipeline steps. The repository ships with a worked biology / ML example
   at `configs/config.yaml` — copy it and adapt the topic description, seed papers,
   and screening prompts to your own domain. All runs write outputs under `runs/`.

2. Run CiteClaw:

```bash
python -m citeclaw -c configs/config.yaml
```

3. Continue a prior run (adds another expansion generation on top of the existing
   collection):

```bash
python -m citeclaw -c configs/config.yaml --continue-from runs/data/
```

4. Annotate the resulting citation graph with LLM-generated node summaries:

```bash
python -m citeclaw annotate runs/data/citation_network.graphml -c configs/config.yaml
```

5. Bulk-download open-access PDFs for a finished run:

```bash
python -m citeclaw fetch-pdfs runs/data/
```

6. Rebuild the citation/collaboration graphs from an existing collection:

```bash
python -m citeclaw rebuild-graph runs/data/
```

CLI flags: `--topic`, `--seed`, `--data-dir`, `--max-papers`, `--model`, `-v`,
`--continue-from`. `fetch-pdfs` flags: `--workers`, `--overwrite`,
`--no-refresh-cache`, `--no-update-cache`.

---

## Architecture at a glance

Two compositional layers, mirroring `nn.Sequential` plus a parallel branch.

| Layer            | Operates on                 | Composition primitives                                                    |
| ---------------- | --------------------------- | ------------------------------------------------------------------------- |
| Pipeline (steps) | `signal: list[PaperRecord]` | top-level Sequential, `Parallel`                                          |
| Filter blocks    | one paper → bool            | `Sequential` (AND), `Any` (OR), `Not` (invert), `Route` (if/elif/else)    |

Every step is a pure signal transformer:

```python
class BaseStep(Protocol):
    name: str
    def run(self, signal: list[PaperRecord], ctx: Context) -> StepResult: ...
```

`ctx.collection` is the cumulative union of every paper ever accepted; only `ReScreen`
removes from it. `Rerank` is **non-destructive** — it filters the signal but never
touches `ctx.collection`. This invariant lets `Parallel` work: one branch can
rerank-then-forward while another sees the original input untouched.

### Pipeline steps

| Step                | Purpose                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| `LoadSeeds`         | Initialise `ctx.collection` from `seed_papers`. Emits the seeds as the first signal.                 |
| `ResolveSeeds`      | Convert mixed `{title: ...}` / `{paper_id: ...}` seed entries to canonical S2 IDs via `search_match`; optionally pull preprint↔published siblings via `external_ids`. |
| `ExpandForward`     | For every paper in the signal, fetch citers, screen them, add the survivors to collection + signal.  |
| `ExpandBackward`    | Same as ExpandForward but follows references instead of citers.                                       |
| `ExpandBySearch`    | **Expansion family**: iterative meta-LLM search agent designs targeted database queries from a topic + anchor papers, refines per turn, and screens the hits. |
| `ExpandBySemantics` | **Expansion family**: anchor on the input signal, fetch SPECTER2 nearest neighbours via S2 Recommendations API, and screen the hits. |
| `ExpandByAuthor`    | **Expansion family**: rank authors in the input signal by h-index / citation count / collab-graph degree, pull each top-K author's papers, and screen them. |
| `Rerank`            | Score-based top-K (with optional cluster-aware diversity). Non-destructive — only filters the signal. |
| `ReScreen`          | Apply a screener block to the entire `ctx.collection` (minus seeds), removing rejected papers.        |
| `Cluster`           | Run a clusterer over the signal once, store the `ClusterResult` in `ctx.clusters[<store_as>]`.        |
| `MergeDuplicates`   | Detect and merge preprint↔published duplicates via DOI/ArXiv ID + title sim + SPECTER2 cosine. **Auto-injected before `Finalize` if you don't list it explicitly** — listing it manually only matters if you want dedup to run earlier in the pipeline. |
| `HumanInTheLoop`    | Opt-in interactive screener-quality check: balanced sample by rejection category, per-filter agreement report, can `stop_pipeline`. |
| `Parallel`          | Broadcast the signal to N branches, run each independently, union outputs by `paper_id`.             |
| `Finalize`          | Write `literature_collection.json` / `.bib`, `citation_network.graphml`, `run_state.json`.            |

**Expansion family.** `ExpandBySearch`, `ExpandBySemantics`, and `ExpandByAuthor` compose at the same level as `ExpandForward` / `ExpandBackward` — users mix all five freely in YAML pipelines. Each ExpandBy* step is anchored on its input *signal* (not an upstream citation edge), so the source-less `FilterContext` they pass to the screener carries `source=None`. All built-in atoms and measures tolerate this case; insert a `Rerank` (with diversity) before any ExpandBy* step to control which papers the agent uses as anchors.

### Filter blocks

| Type                | Purpose                                                                                |
| ------------------- | -------------------------------------------------------------------------------------- |
| `Sequential`        | AND of `layers:` (plural). Short-circuits on first reject.                             |
| `Any`               | OR of `layers:` (plural). Short-circuits on first pass.                                |
| `Not`               | Invert a single child block specified as `layer:` (singular).                          |
| `Route`             | if/elif/else dispatch over `routes:`.                                                  |
| `SimilarityFilter`  | Max of normalized scores from `measures:` list (RefSim / CitSim / SemanticSim).        |
| `YearFilter`        | Pass if `year` is in `[min, max]`.                                                     |
| `CitationFilter`    | Pass if citation count is high enough relative to `beta` and paper age.                |
| `LLMFilter`         | Batched LLM screening; `scope:` is `title` / `title_abstract` / `venue` / `full_text`. Single-prompt or Boolean formula mode. `full_text` reads parsed PDFs from the `paper_full_text` cache. |
| `TitleKeywordFilter`    | Plain string search over the paper's **title**. Single `keyword:` or Boolean `formula:` over named `keywords:` (operators `& | !`). Knobs: `case_sensitive`, `match: substring \| whole_word \| starts_with`. |
| `AbstractKeywordFilter` | Same DSL as `TitleKeywordFilter`, applied to the paper's **abstract**. Missing/None abstracts are treated as empty (negations like `!survey` still pass). |
| `VenueKeywordFilter`    | Same DSL as `TitleKeywordFilter`, applied to the paper's **venue**. Use `match: starts_with` for hard journal allow-lists like "venue starts with Nature / Science / Cell" — accepts `Cell Reports`, rejects `Cellulose`, `Stem Cell Reports`, and `Royal Society Open Science`. |

### Clusterers

| Type          | Mechanism                                                                              |
| ------------- | -------------------------------------------------------------------------------------- |
| `walktrap`    | igraph `community_walktrap` over the citation graph; targets a fixed `n_communities`.  |
| `louvain`     | igraph `community_multilevel` (modularity maximisation); auto-determines count.        |
| `topic_model` | UMAP + HDBSCAN over S2 SPECTER2 embeddings. BERTopic-inspired, no bertopic dep.        |

Cluster naming (label / summary / keywords / representative papers) is filled in by an
algorithm-agnostic post-processor using c-TF-IDF and/or LLM calls — the same code labels
a walktrap community and a topic_model topic identically.

---

## Example config

```yaml
screening_model: "gemini-2.5-flash-lite"
search_model: "gemini-3-pro"            # optional override for ExpandBySearch agent

# Per-model registry — aliases not in the registry fall through to
# Gemini detection / global llm_base_url / OpenAI SaaS.
models:
  gemma-4-31b:
    base_url: "https://you--citeclaw-vllm-gemma-serve.modal.run/v1"
    served_model_name: "google/gemma-4-31B-it"
    api_key_env: "CITECLAW_VLLM_API_KEY"
    reasoning_parser: "gemma4"

data_dir: "runs/data_bio"
topic_description: "..."
max_papers_total: 500
seed_papers:
  - paper_id: "DOI:10.1038/s41586-021-03819-2"

blocks:
  year_layer:   {type: YearFilter, min: 2018, max: 2026}
  cit_base:     {type: CitationFilter, beta: 30}

  # Cheap zero-cost keyword pre-filters: drop obviously off-topic papers
  # before they reach similarity / LLM screening.
  abstract_topic_kw:
    type: AbstractKeywordFilter
    formula: "(ml | bio) & !erratum"
    keywords:
      ml:      "machine learning"
      bio:     "biology"
      erratum: "erratum"

  similarity:
    type: SimilarityFilter
    threshold: 0.025
    measures:
      - {type: RefSim}
      - {type: CitSim, pass_if_cited_at_least: 200}
      - {type: SemanticSim, embedder: s2}

  topic_llm:
    type: LLMFilter
    scope: title_abstract
    formula: "(q_ml | q_stats) & !q_survey"
    queries:
      q_ml:     "the paper proposes a new ML/DL method"
      q_stats:  "the paper proposes a new statistical method"
      q_survey: "the paper is a pure survey or review"

  forward_screener:
    type: Sequential
    layers: [year_layer, cit_base, abstract_topic_kw, similarity, topic_llm]

pipeline:
  - step: LoadSeeds
  - step: ExpandForward
    max_citations: 30
    screener: forward_screener
  - step: ExpandBackward
    screener: forward_screener
  - step: Cluster
    store_as: topics
    algorithm: {type: topic_model, min_cluster_size: 5}
    naming: {mode: both, n_keywords: 10, n_representative: 5}
  - step: Rerank
    metric: pagerank
    k: 100
    diversity: {cluster: topics}
  - step: Finalize
```

See `config.yaml` for a full production configuration with `Route`, `Parallel`,
and `Not` in use.

---

## Output artifacts

Every run writes into `data_dir/`:

- `literature_collection.json` (+ `.expN` for continuation runs)
- `literature_collection.bib`
- `citation_network.graphml` — rich node/edge attributes, cluster assignments, and
  LLM-generated cluster labels (ready for Gephi color-coding by topic)
- `collaboration_network.graphml` — undirected author co-authorship graph
- `run_state.json` — full run state for continuation
- `cache.db` — SQLite read-through cache for S2 metadata
- `shape_summary.txt` — PyTorch-summary-style pipeline shape table

---

## Module layout

```
src/citeclaw/
  config.py            Settings, SeedPaper, ModelEndpoint, BudgetTracker
  context.py           Context — collection / seen / rejected / expanded_* / clusters / rejection_ledger
  pipeline.py          run_pipeline(ctx, *, event_sink=None)
  event_sink.py        EventSink Protocol + NullEventSink + RecordingEventSink
  cache.py             SQLite read-through cache (paper_metadata, paper_references, paper_citations,
                       paper_embeddings, author_metadata, author_papers, search_queries,
                       paper_full_text, llm_response_cache)
  models.py            PaperRecord (with external_ids, source, fields_of_study, ...) + exceptions
  network.py           igraph wrappers (build_citation_graph, compute_pagerank)
  author_graph.py      build_author_graph + export_author_graphml
  dedup.py             detect_duplicate_clusters + merge_cluster
  fetch_pdfs.py        bulk PDF fetcher CLI
  annotate.py          LLM graph annotation driver
  progress.py          live Rich mission-control dashboard
  logging_config.py    structured console + file logging setup
  search/query_engine.py    apply_local_query — pure regex/range filter over PaperRecord lists
  agents/iterative_search.py    AgentConfig + AgentTurn + run_iterative_search (ExpandBySearch core)
  prompts/             screening / annotation / topic_naming / search_refine prompt templates
  filters/
    base.py            Filter Protocol, FilterContext, FilterOutcome
    builder.py         block dict → Filter (+ predicate registry: venue_in, cit_at_least, year_at_least)
    runner.py          apply_block(papers, block, fctx) → (passed, rejected); record_rejections → ledger
    blocks/            sequential, any_block, not_block, route, similarity
    atoms/             year, citation, llm_query, predicates
    measures/          ref_sim, cit_sim, semantic_sim + MEASURE_TYPES
  screening/
    formula.py         BooleanFormula DSL
    llm_runner.py      batched concurrent LLM dispatch
    schemas.py         screening_json_schema + openai_response_format
  cluster/
    base.py            Clusterer Protocol + ClusterResult + ClusterMetadata
    walktrap.py / louvain.py / topic_model.py
    representation.py  c-TF-IDF + select_representative_papers + name_topics_via_llm
  steps/
    base.py            BaseStep + StepResult
    __init__.py        STEP_REGISTRY + build_step()
    _expand_helpers.py shared ExpandBy* pipeline helpers
    load_seeds / resolve_seeds / expand_forward / expand_backward
    expand_by_search / expand_by_semantics / expand_by_author
    rerank / rescreen / cluster / merge_duplicates / human_in_the_loop / parallel / finalize
    shape_log.py       PyTorch-summary-style table
    checkpoint.py      --continue-from loader
  rerank/
    metrics.py         compute_metric(name, signal, ctx)
    diversity.py       cluster_diverse_top_k (floor-then-proportional)
  clients/
    s2/                SemanticScholarClient — search_bulk / search_match /
                       fetch_recommendations / fetch_author_papers;
                       split: api.py / http.py / cache_layer.py / converters.py
    llm/               base.py (LLMClient Protocol + LLMResponse), caching.py (CachingLLMClient),
                       factory.py (build_llm_client), openai_client.py, gemini.py, stub.py
    embeddings/        base.py (EmbeddingClient Protocol), factory.py (build_embedder),
                       voyage.py, local.py
    pdf.py             PdfFetcher — download + parse PDFs for full_text screening
  output/              json / bibtex / graphml writers
```

---

## Development

```bash
pytest                 # run the full test suite
ruff check src tests   # lint
mypy src               # type-check
```

Tests that exercise optional `topic_model` extras are skipped if the extras aren't
installed.

---

## pdfclaw — standalone PDF fetcher

Sister package at `src/pdfclaw/`. Downloads and parses PDFs for any CiteClaw
checkpoint directory. Communicates via `literature_collection.json` + `cache.db`
— never imports from `citeclaw`.

```bash
python -m pdfclaw list  <checkpoint>              # DOI coverage + fetch progress
python -m pdfclaw login                            # one-time SSO (opens Chrome)
python -m pdfclaw fetch <checkpoint>               # download + parse PDFs
python -m pdfclaw fetch <checkpoint> --filter-doi 10.1038/  # only Nature
python -m pdfclaw fetch <checkpoint> --filter-recipe nature_browser --max 5
```

### 5-layer fallback chain (per paper)

| Layer | Recipes | Cost |
|---|---|---|
| HTTP API | Unpaywall, OpenAlex, EuropePMC, arXiv direct, eLife XML, Wiley TDM, Elsevier TDM | free, fast, no browser |
| PDF URL template | ACS `/doi/pdf/`, Science `/doi/pdf/`, Wiley `/doi/pdfdirect/`, Springer, PNAS | free, fast, uses SSO cookies |
| Browser selectors | Nature, Elsevier, Oxford, RSC, IEEE, ACM, Springer, IOP, EMBO, bioRxiv, MDPI, etc. | needs Chrome + SSO profile |
| LLM finder | Gemini Flash Lite or Modal Gemma. Multi-turn reasoning agent. | ~600 tokens/paper |
| Sci-Hub | Opt-in via `PDFCLAW_ENABLE_SCIHUB=1`. | grey area |

ArXiv fallback: if primary DOI chain fails and paper has an arXiv ID in S2's
`externalIds`, automatically retries with the arXiv DOI.

---

## Supported LLM providers

`screening_model:`, `search_model:`, and any per-block `model:` override
all accept the same alias surface. Routing is decided by
`citeclaw.clients.llm.factory.build_llm_client` in this order:

| Alias prefix / value          | Provider                                           | Required env var                                                    |
| ----------------------------- | -------------------------------------------------- | ------------------------------------------------------------------- |
| `stub` (case-insensitive)     | Deterministic offline stub (tests / dev runs)      | none                                                                |
| any key in `models:` registry | OpenAI-compatible endpoint at the registered URL   | the entry's `api_key_env` (e.g. `CITECLAW_VLLM_API_KEY`, `XAI_API_KEY`) |
| `gemini-*`                    | Gemini 2.5 / 3.x via the `google-genai` SDK        | `GEMINI_API_KEY`                                                    |
| `o1*`, `o3*`, `o4*`           | OpenAI o-series reasoning models                   | `OPENAI_API_KEY`                                                    |
| `gpt-*`                       | OpenAI chat models                                 | `OPENAI_API_KEY`                                                    |
| `claude-*`                    | Routed via OpenAI SaaS proxy by default            | `OPENAI_API_KEY` (override with a `models:` entry for native API)   |
| (anything else)               | Falls through to OpenAI SaaS                       | `OPENAI_API_KEY`                                                    |

`SEMANTIC_SCHOLAR_API_KEY` (or `S2_API_KEY`) is **always required** —
the pre-flight validator refuses to start without it.

The pre-flight validator walks the pipeline + every filter block and
reports each missing env var up-front, so a stub-only run never
demands an OpenAI key and a Gemma-via-vLLM run never demands a
Gemini key.

### `reasoning_effort:`

Top-level setting + per-filter override on `LLMFilter` (and
per-step override on `ExpandBySearch.agent`). Accepted values:

| Value      | Meaning                                          | Sent as (per provider)                                                              |
| ---------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `""` (default) | No thinking — model returns content only        | nothing on the wire                                                                  |
| `low`      | Short thinking trace                             | OpenAI / xAI Grok / Mistral: `reasoning_effort="low"`. Gemini: `thinking_level="low"`. vLLM: `enable_thinking=True` + `thinking_budget=4096`. |
| `medium`   | Default for capable models                       | same shape, `medium` / `thinking_budget=8192`                                       |
| `high`     | Long thinking trace                              | same shape, `high` / `thinking_budget=16384`                                        |
| `minimal`  | Gemini-only — minimal reasoning tokens          | Gemini `thinking_level="minimal"`. Other providers treat as empty.                  |
| `off` / `none` / `disabled` | Explicitly turn thinking off       | nothing on the wire (forces a non-thinking call)                                    |

Set `thinking_budget: <int>` on a `models:` registry entry to override
the effort-based default for vLLM endpoints (Gemma 4 / Qwen3 / DeepSeek-R1).

### xAI Grok 3 / 4

xAI's API mirrors OpenAI's chat-completions surface (with native
`reasoning_effort`), so Grok plugs in via the `models:` registry — no
new code required. Example:

```yaml
models:
  grok-4:
    base_url: "https://api.x.ai/v1"
    served_model_name: "grok-4-latest"
    api_key_env: "XAI_API_KEY"
    reasoning_parser: "grok"      # use native reasoning_effort kwarg

screening_model: "grok-4"
reasoning_effort: "medium"
```

`reasoning_parser: "grok"` (also `"xai"`) selects the native
`reasoning_effort` wire shape rather than the vLLM
`chat_template_kwargs.enable_thinking` path used for self-hosted
OSS models. The same registry mechanism covers Mistral Magistral
(`reasoning_parser: "mistral"`) and any other OpenAI-compat
provider that supports `reasoning_effort` natively.

---

## Self-hosted LLM (Modal vLLM)

`modal_vllm_server.py` (project root) wraps `vllm serve` with an
OpenAI-compatible endpoint on Modal. Wire it via the YAML `models:` registry:

```bash
pip install modal && modal setup
CITECLAW_VLLM_APP_NAME=citeclaw-vllm-gemma \
CITECLAW_VLLM_MODEL=google/gemma-4-31B-it \
CITECLAW_VLLM_GPU=H200 CITECLAW_VLLM_GPU_COUNT=1 \
CITECLAW_VLLM_API_KEY=<bearer> \
modal deploy modal_vllm_server.py
```

Then reference the alias in any filter block with `model: gemma-4-31b`.

---

## Web UI

A lightweight FastAPI + React (Vite + TypeScript + Tailwind) dashboard lives in
`web/`. It provides config browsing, run monitoring, and citation graph
visualization (sigma.js / graphology).

```bash
# backend
cd web/backend && uvicorn main:app --port 9999

# frontend
cd web/frontend && pnpm dev
```
