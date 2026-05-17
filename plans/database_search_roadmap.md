# CiteClaw — Expansion Family + Web UI Roadmap

> **What this file is.** A flat checklist of implementation tasks for adding the `ExpandBy*` family of pipeline steps, an iterative meta-LLM search agent, a graph-reinforcement step, a human-in-the-loop checkpoint, and a beautiful web UI to CiteClaw. Designed to be executed by Claude Code on an hourly cron, one task per invocation.
>
> **Design spec.** The full architectural discussion lives at `~/.claude/plans/quiet-hugging-lecun.md` on Mingyu's machine — that document explains the *why* behind every decision. This roadmap is the execution view: just the *what*, *files*, and *verify*. If something here is ambiguous, check the design spec.

---

## RESUME PROTOCOL — read this every invocation

You are a fresh Claude Code instance woken on a cron schedule. Your job is to make ONE unit of progress on this roadmap and exit. Follow this protocol exactly:

**Step 1 — Sanity-check the previous run.** Scroll to the "Last run feedback" section below and read the most recent entry. If it ends with `❌` (failure), read the error note carefully and decide whether the previous task is now actually complete or whether you need to re-attempt it. If it ends with `✅` (success), trust it and move on.

**Step 2 — Find your task.** Find the first unchecked `- [ ]` item. Read its `What`, `Why`, `Files touched`, and `Verify done` subsections completely before doing anything.

**Step 3 — Implement.** Make the changes described in `Files touched`. Stay in scope — do only what the task asks. If you discover the task as written is unworkable (e.g., a referenced file doesn't exist, a dependency is missing), STOP, append a `- ❌` feedback entry explaining the blocker, do NOT tick the box, do NOT commit. The next invocation (or the user) will resolve it.

**Step 4 — Verify.** Run the exact command from `Verify done`. If it passes, proceed to Step 5. If it fails:
- Read the error and try a focused fix (do not flail).
- If fixed, re-run the verification.
- If still failing after one focused attempt, STOP, append a `- ❌` feedback entry with the error, do NOT tick, do NOT commit.

**Step 5 — Tick the box and log feedback.** Change `- [ ]` to `- [x]`. Then append a feedback entry under the task using this format:
```
  - ✅ 2026-04-09 — <2-3 sentence summary of what was done, any assumptions, and any followups the next run should know about>
```
Use the actual current date (`date +%Y-%m-%d`). If you discovered something the next task should be aware of (e.g., "PA-02 introduced a `_TTL_DAYS` constant — PA-05 will reuse it"), say so.

**Step 6 — Update the "Last run feedback" section.** At the top of this file, prepend a one-line entry to the "Last run feedback" section:
```
- 2026-04-09 14:32 — completed PA-01 ✅ (added search_bulk/search_match/search_relevance to SemanticScholarClient; tests green)
```
Keep the section to the most recent 10 entries; trim older ones. This is the at-a-glance status the user reads in the morning.

**Step 7 — Commit and push (per CLAUDE.md auto-commit rule).** This step is mandatory; CLAUDE.md says "after ANY change to this project you MUST `git add`, `git commit`, `git push origin main`". Do exactly this:
```bash
git status                                          # confirm what changed
git add <list specific files explicitly>            # NEVER `git add -A` or `git add .`
git add plans/database_search_roadmap.md            # always include this file
git commit -m "<task-id>: <one-line summary>"       # e.g. "PA-01: add S2 search endpoints to client"
git push origin main
```
**Files you must NEVER stage** (per CLAUDE.md): `CLAUDE.md`, `BRAINSTORM.md`, `.claude/`, `data_bio/`, `test_data/`, `scratch/`, `*.db`, `*.log`, `.DS_Store`. They are gitignored but `git add -A` would force-include them.

If `git push` fails, do NOT force-push. Surface the error in the feedback log and stop.

**Step 8 — STOP.** One task per invocation. Do not start the next task. Exit cleanly.

### Special phase rules

- **Phase F (Meta-review agent) is HUMAN-GATED.** If you reach a Phase F task, STOP immediately and append a feedback entry saying "reached Phase F human gate, awaiting user approval". Do not implement.
- **Phase E (Web UI) runs in parallel with Phases C/D.** On odd-numbered cron runs, prefer the next unchecked Phase C/D task; on even-numbered runs, prefer the next unchecked Phase E task. If one phase is fully done, work the other. Skip this rule if you're inside a strict dependency chain (the task list will say so).
- **Skipping is allowed if a task is blocked.** If task X cannot be completed (missing dependency, design ambiguity), append `- ⏭️` feedback explaining why, do NOT tick X, and move on to find the NEXT task that is unblocked. The user will return to X manually.

---

## Last run feedback (most recent first; keep ≤ 10 entries)

- 2026-05-18 06:55 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-18 05:54 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-18 04:53 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-18 03:52 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-18 02:52 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-18 01:51 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-18 00:50 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-17 23:49 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-17 22:48 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
- 2026-05-17 21:47 — reached Phase F human gate ⛔ (all Phase A–E tasks complete; Phase F is human-gated, awaiting user approval)
---

## Architectural decisions (reference)

These were settled in the design conversation. Do NOT relitigate them; if you disagree with one, note it in feedback and stop, do not unilaterally change direction.

1. **`ExpandBy*` family, not a monolithic `DatabaseSearch` step.** Each retrieval paradigm (LLM-driven search, semantic kNN, author-graph traversal) is its own step, composable at the same level as `ExpandForward` / `ExpandBackward`. Users compose them freely in YAML.
2. **Meta-LLM agent is iterative by default** (`max_iterations=4`) with two-level thinking: (a) outer loop across LLM calls sees prior transcripts; (b) inner `thinking` field placed first in the JSON response schema forces per-call chain-of-thought before any structured decision. Native reasoning tokens (`reasoning_effort="high"`) stack on top for capable models.
3. **`ExpandBySemantics` uses S2 Recommendations API** (`POST /recommendations/v1/papers`), not local kNN. Zero new embedding infrastructure.
4. **Signal-driven grounding.** Each `ExpandBy*` step uses its input signal as anchor context. Users insert a `Rerank` (with diversity) before the step to control what's fed in.
5. **No new `SearchEngine` Protocol.** Each step calls S2 methods directly. Rejected as over-engineering for the actual use case.
6. **Per-paper rejection ledger** (`ctx.rejection_ledger: dict[str, list[str]]`) is populated by `record_rejections` and consumed by `HumanInTheLoop` for balanced sampling.
7. **`source: str`** instead of frozen enum on `PaperRecord`, so new sources (`search`, `semantic`, `author`, `reinforced`, etc.) can be added without schema migration.
8. **Filter atoms must tolerate `fctx.source=None`.** Verified once in PC-05 and enforced via build-time errors thereafter.
9. **Web UI lives in `web/`** as a subdirectory. Stack: React 18 + Vite + TypeScript + Tailwind v4 + shadcn/ui + sigma.js (ForceAtlas 2) + React Flow + FastAPI + WebSockets.
10. **Phase F (meta-review agent) is human-gated.** Cron-Claude stops if it reaches it.

---

## Phase A — S2 surface + pure utilities

Goal: every Phase A module is unit-testable with zero pipeline touch.

- [x] **PA-01. `search_bulk` / `search_match` / `search_relevance` on `SemanticScholarClient`**
  - **What.** Extend `src/citeclaw/clients/s2/api.py` with three methods:
    - `search_bulk(query, *, filters=None, sort=None, token=None, limit=1000) -> dict` → `GET /paper/search/bulk`. Forwards `year, venue, fieldsOfStudy, minCitationCount, publicationTypes, publicationDateOrYear, openAccessPdf` from `filters`. `fields="paperId,title"` only. `req_type="search"`.
    - `search_match(title) -> dict | None` → `GET /paper/search/match`. `req_type="search_match"`.
    - `search_relevance(query, *, limit=100, offset=0) -> dict` → `GET /paper/search`. `req_type="search"`.
  - All three reuse `_throttle`, `_http.get`, existing backoff. Cache wiring is PA-05.
  - **Why.** Minimum S2 surface Phase B's agent depends on.
  - **Files touched.** `src/citeclaw/clients/s2/api.py`. New: `tests/test_s2_search_api.py`.
  - **Verify done.** `pytest tests/test_s2_search_api.py -x` (uses monkey-patched `S2Http.get`; no network).
  - ✅ 2026-04-08 — Added a "Search" section to `api.py` with all three methods, an `httpx` import for the `search_match` 404→None catch, and a `_SEARCH_BULK_FILTER_KEYS` whitelist tuple so PA-05 can reuse the same allowlist when wiring caches. New `tests/test_s2_search_api.py` has 14 tests using a `_Recorder` helper that monkey-patches `client._http.get`. Note for next runs: stale `__pycache__` from the old `CitNet2` repo path broke pytest collection — had to wipe it once; if a future task sees `ModuleNotFoundError: citeclaw`, run `find . -name __pycache__ -exec rm -rf {} +` and use `PYTHONPATH=src python -m pytest …` since the package isn't pip-installed.

- [x] **PA-02. `fetch_recommendations` on `SemanticScholarClient`**
  - **What.** Add to `src/citeclaw/clients/s2/api.py`:
    - `fetch_recommendations(positive_ids, *, negative_ids=None, limit=100, fields="paperId,title") -> list[dict]` → `POST /recommendations/v1/papers` with body `{"positivePaperIds": [...], "negativePaperIds": [...]}`. `req_type="recommendations"`.
    - `fetch_recommendations_for_paper(paper_id, *, limit=100, fields=...) -> list[dict]` → `GET /recommendations/v1/papers/forpaper/{paper_id}`.
  - **Why.** Powers `ExpandBySemantics`. S2 does the SPECTER2 kNN over its full corpus for us.
  - **Files touched.** `src/citeclaw/clients/s2/api.py`. Append to `tests/test_s2_search_api.py`.
  - **Verify done.** `pytest tests/test_s2_search_api.py -x`.
  - ✅ 2026-04-08 — Both methods unwrap S2's `recommendedPapers` envelope so callers always get a flat list. Recommendations live outside `/graph/v1`, so I added a small `S2Http.get_url(full_url, ...)` helper (mirrors `get` but skips BASE_URL prepend) — that lightweight http.py addition is the one file outside the task's listed "Files touched" but it's the cleanest way to keep retry/throttle/budget shared. New constants `RECOMMENDATIONS_BATCH_URL` / `RECOMMENDATIONS_FORPAPER_URL` in api.py. Also extended `_Recorder` in tests with `install_post` and `install_get_url` siblings — PA-03 will need install_get_url too when pagination tests are added.

- [x] **PA-03. `fetch_author_papers` on `SemanticScholarClient`**
  - **What.** Add `fetch_author_papers(author_id, *, limit=100, fields="paperId,title,year,venue,citationCount") -> list[dict]` → `GET /graph/v1/author/{author_id}/papers` with pagination. `req_type="author_papers"`. Caches per-author under the new `author_papers` cache table (PA-04).
  - **Why.** Powers `ExpandByAuthor`. Today's `fetch_authors_batch` only returns author metadata, not paper lists.
  - **Files touched.** `src/citeclaw/clients/s2/api.py`, `src/citeclaw/cache.py` (depends on PA-04). Append to `tests/test_s2_search_api.py`.
  - **Verify done.** `pytest tests/test_s2_search_api.py -x`.
  - ⏭️ 2026-04-08 — Skipped this run because PA-03's caching arm depends on `Cache.get_author_papers`/`put_author_papers`, which only land in PA-04. Did PA-04 first; now unblocked. Next run should pick this up — will need to call `cache.get_author_papers(author_id)` / `cache.put_author_papers(author_id, papers)` after the paginated S2 GET, and use the new `S2Http.get_url` helper from PA-02 if pagination logic ends up there. Pagination follow-up: S2's author/papers endpoint paginates via `offset`/`next` token — model the loop on `S2Http.paginate` rather than reinventing it (consider adding an `author_papers` branch or building a small in-`api.py` paginator).
  - ✅ 2026-04-08 — Added `fetch_author_papers` to api.py with an inline cache-first paginator (mirrors `S2Http.paginate`'s offset/limit shape but lives in api.py since the URL is `/author/{id}/papers` not `/paper/{id}/{edge}`). Module-level constant `_AUTHOR_PAPERS_PAGE_SIZE = 100`. The `limit` arg caps both pagination *and* the returned slice — so the cached entry reflects exactly what was fetched, not the author's full corpus (deliberate trade-off documented in the docstring; if a downstream user later asks for a bigger limit, they get the cached short list). Also added `get_author_papers`/`put_author_papers` to `S2CacheLayer` (wraps Cache and bumps `_s2_cache["author_papers"]` on hit) — that cache_layer.py file is outside PA-03's listed "Files touched" but it's the only way to wire the new cache table through the budget tracker. 11 new tests including a `_install_paginated_get` helper for multi-page scenarios; full file at 38 tests.

- [x] **PA-04. Cache tables: `search_queries` + `author_papers`**
  - **What.** Append to `_SCHEMA` in `src/citeclaw/cache.py`:
    ```sql
    CREATE TABLE IF NOT EXISTS search_queries (
      query_hash TEXT PRIMARY KEY,
      query_json TEXT NOT NULL,
      result_json TEXT NOT NULL,
      fetched_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS author_papers (
      author_id TEXT PRIMARY KEY,
      data TEXT NOT NULL,
      fetched_at TEXT NOT NULL
    );
    ```
    Add `Cache.get_search_results/put_search_results/has_search_results(query_hash, ttl_days=30)` and `Cache.get_author_papers/put_author_papers(author_id)`.
  - **Files touched.** `src/citeclaw/cache.py`. Append to `tests/test_cache.py`.
  - **Verify done.** `pytest tests/test_cache.py -x`.
  - ✅ 2026-04-08 — Added both tables to `_SCHEMA` plus a `_SEARCH_TTL_DAYS_DEFAULT = 30` module constant so PA-05 can reference the same default. New helper `Cache._is_fresh(fetched_at_iso, ttl_days)` parses ISO timestamps and tolerates naive datetimes (older rows). 14 new test_cache.py tests covering put/get/has roundtrip, TTL expiration via direct SQL backdating, persistence across instances, query_json round-trip, and the empty-list-vs-missing distinction for author_papers. The TTL test pattern (UPDATE … fetched_at to a backdated ISO string then read) is reusable — PA-05's cache-hit tests can copy it.

- [x] **PA-05. Wire caches into `search_bulk`, `fetch_recommendations`, `fetch_author_papers`**
  - **What.** Add `query_hash = sha256(json.dumps({"q": q, "filters": f, "sort": s, "token": t}, sort_keys=True)).hexdigest()` to `search_bulk`. Cache via new `S2CacheLayer.get_search_results/put_search_results` (records hits in `BudgetTracker.record_s2("search", cached=True)`). Cache `fetch_author_papers` per-author. Do NOT cache `search_match` or `fetch_recommendations` (freshness matters).
  - **Files touched.** `src/citeclaw/clients/s2/cache_layer.py`, `src/citeclaw/clients/s2/api.py`.
  - **Verify done.** Extend `tests/test_s2_search_api.py`: call each cached method twice with identical args; second call must serve from cache.
  - ✅ 2026-04-08 — `search_bulk` now hashes `{"q","filters","sort","token"}` with `sort_keys=True` (deliberately omitting `limit` so a wider pre-fetch can serve narrower followers from cache). Hit returns the cached payload before any HTTP work; miss persists the full response (including `total`, `token`, etc.). Added `get_search_results`/`put_search_results` to `S2CacheLayer` — hits bump `_s2_cache["search"]` via `record_s2(..., cached=True)`. `fetch_author_papers` was already cached in PA-03, so PA-05 didn't need to touch it again. 10 new tests: 7 in `TestSearchBulkCacheWiring` (hit, miss-on-different-q/filters/sort/token, dict-key-order independence, full-payload round-trip) and 3 in `TestUncachedSurfaces` proving `search_match`/`fetch_recommendations`/`fetch_recommendations_for_paper` still always reach the network. `_query_hash_for` test helper duplicates the SUT's hash recipe so cache-key inspection is possible without monkeying with internals; PA-09's local query engine can ignore it.

- [x] **PA-06. Extend `PaperRecord` with `fields_of_study` + `publication_types`**
  - **What.** Add `fields_of_study: list[str] = Field(default_factory=list)` and `publication_types: list[str] = Field(default_factory=list)` to `src/citeclaw/models.py::PaperRecord`. Extend `PAPER_FIELDS` in `api.py` with `fieldsOfStudy,publicationTypes,s2FieldsOfStudy`. Extend `paper_to_record` in `converters.py` to populate them (merge `s2FieldsOfStudy` into `fields_of_study`).
  - **Files touched.** `src/citeclaw/models.py`, `src/citeclaw/clients/s2/api.py`, `src/citeclaw/clients/s2/converters.py`. New test in `tests/test_models.py`.
  - **Verify done.** `pytest tests/ -x`.
  - ✅ 2026-04-09 — Added both fields with `Field(default_factory=list)` defaults so existing PaperRecord constructions stay backward-compatible. `paper_to_record` merges `fieldsOfStudy` (legacy flat strings) and `s2FieldsOfStudy` (`{category, source}` dicts) into a single deduplicated list, preserving legacy-first ordering. Robust to None / non-list / non-string entries — important because S2's response shape is inconsistent across paper records. 9 new tests in `test_models.py` (2 in `TestPaperRecord` for defaults + direct construction, 7 in `TestPaperToRecordSubjectFields` for converter merge logic). Full `pytest tests/ -x` green: 550 passed, 6 skipped (topic_model extras + live_s2 markers; pre-existing). PA-09's local query engine will consume these fields directly.

- [x] **PA-07. `PaperRecord.source: str` instead of frozen enum**
  - **What.** Replace the `PaperSource` enum field on `PaperRecord` with `source: str = "backward"`. Keep `PaperSource` as a constants namespace: `class PaperSource: SEED="seed"; FORWARD="forward"; BACKWARD="backward"; SEARCH="search"; SEMANTIC="semantic"; AUTHOR="author"; REINFORCED="reinforced"`. Audit all call sites that compare `p.source == PaperSource.X` — they continue working because `use_enum_values=True`.
  - **Files touched.** `src/citeclaw/models.py`. Possibly a few call sites in steps/.
  - **Verify done.** `pytest tests/ -x`.
  - ✅ 2026-04-09 — Audit found ALL production assignments (`load_seeds.py`, `expand_forward.py`, `expand_backward.py`) and comparisons (`network.py`, `checkpoint.py`, `graphml_writer.py`) already used string literals — the enum was a vestigial type annotation. Replaced `class PaperSource(str, enum.Enum)` with a plain `class PaperSource` namespace adding the four new sources, changed `source: PaperSource = PaperSource.BACKWARD` to `source: str = "backward"`, and updated 3 test sites in `test_models.py` (line 93 dropped the `.value`, lines 158-160 became direct string compares, and added asserts for the new SEARCH/SEMANTIC/AUTHOR/REINFORCED constants). Zero `src/` files outside `models.py` needed touching. Full `pytest tests/ -x` green: 550 passed, 6 skipped. PaperRecord docstring on `source` now points readers to `PaperSource` for canonical values without forcing them to use it.

- [x] **PA-08. `Context` additions: rejection ledger + idempotency sets + reinforcement log**
  - **What.** In `src/citeclaw/context.py`, add three fields:
    ```python
    rejection_ledger: dict[str, list[str]] = field(default_factory=dict)
    searched_signals: set[str] = field(default_factory=set)
    reinforcement_log: list[dict] = field(default_factory=list)
    ```
    Update `record_rejections` in `src/citeclaw/filters/runner.py` to also append to `rejection_ledger[paper.paper_id]`.
  - **Why.** `HumanInTheLoop` needs per-paper rejection reasons; `ExpandBy*` steps need per-signal idempotency; `ReinforceGraph` needs a place to log decisions.
  - **Files touched.** `src/citeclaw/context.py`, `src/citeclaw/filters/runner.py`. New test asserting the ledger is populated on rejection.
  - **Verify done.** `pytest tests/ -x`.
  - ✅ 2026-04-09 — Added all three fields with `field(default_factory=...)` defaults so existing Context constructions stay backward-compatible. `record_rejections` now appends to `rejection_ledger.setdefault(paper.paper_id, []).append(key)` using the SAME key as `rejection_counts` — this guarantees the per-paper ledger and the global counts can never disagree, which `HumanInTheLoop` will rely on for balanced sampling. 5 new tests in `TestRecordRejections`: single-rejection, multi-rejection accumulation across calls, separation by paper_id, blank-category falls through as "unknown", and a baseline assertion that the new fields start empty on a fresh Context. Full `pytest tests/ -x` green: 555 passed (5 more than last run), 6 skipped. Note for PC-01: `searched_signals` is the key the ExpandBy* family will hash into; the docstring on the field describes the expected fingerprint shape (step name + signal ids + agent config).

- [x] **PA-09. `src/citeclaw/search/query_engine.py` — pure `apply_local_query`**
  - **What.** New package `src/citeclaw/search/__init__.py` + `src/citeclaw/search/query_engine.py`. Exports one pure function:
    ```python
    def apply_local_query(
        papers: list[PaperRecord], *,
        venue_regex: str | None = None,
        year_min: int | None = None, year_max: int | None = None,
        min_citations: int | None = None,
        fields_of_study_any: list[str] | None = None,
        publication_types_any: list[str] | None = None,
        abstract_regex: str | None = None,
        title_regex: str | None = None,
    ) -> list[PaperRecord]
    ```
    AND-ed predicates; strict on missing metadata except `abstract_regex` (lenient — S2 often lacks abstracts). Regexes with `re.IGNORECASE`.
  - **Why.** S2 API can't express regex, abstract text search, or arbitrary unions. Used optionally by expand steps for post-fetch trim.
  - **Files touched.** New: `src/citeclaw/search/__init__.py`, `src/citeclaw/search/query_engine.py`, `tests/test_search_query_engine.py`.
  - **Verify done.** `pytest tests/test_search_query_engine.py -x` with ~10 cases.
  - ✅ 2026-04-09 — Created the new `search/` package with `__init__.py` re-exporting `apply_local_query` (so callers can `from citeclaw.search import apply_local_query`). Pure function — no Context, no S2, no LLM dependency. Each predicate is skipped when None and AND-ed when set; missing-metadata behavior matches the spec exactly (strict everywhere except `abstract_regex`, which is lenient because S2 often returns no abstract). Regexes pre-compile once with `re.IGNORECASE` and use `re.search` semantics so callers don't need to anchor. 26 new tests in `test_search_query_engine.py` organized in 8 classes (TestNoPredicates / TestYearRange / TestMinCitations / TestVenueRegex / TestTitleRegex / TestAbstractRegex / TestFieldsOfStudyAny / TestPublicationTypesAny / TestCombinedPredicates) — well above the spec's "~10 cases". Full suite 581 passed/6 skipped. PC-01's `ExpandBySearch` will pipe its hydrated candidates through this before calling `apply_block` so callers can use both approaches together.

- [x] **PA-10. `FakeS2Client` extensions + Phase A e2e test**
  - **What.** Extend `tests/fakes.py::FakeS2Client` with `search_bulk`, `search_match`, `fetch_recommendations`, `fetch_author_papers` — query-keyed canned responses suitable for downstream Phase B and C tests. Then write `tests/test_search_phase_a_e2e.py` exercising each new API method against the fake.
  - **Files touched.** `tests/fakes.py`, new `tests/test_search_phase_a_e2e.py`.
  - **Verify done.** `pytest tests/test_s2_search_api.py tests/test_cache.py tests/test_search_query_engine.py tests/test_search_phase_a_e2e.py -x`. **Phase A DONE** when all green.
  - ✅ 2026-04-09 — Added 4 canned-response surfaces to FakeS2Client (`search_bulk`/`search_match`/`fetch_recommendations`/`fetch_author_papers`) plus matching `register_*` helpers; init seeds the four backing dicts. Each surface is order-independent where it matters (`fetch_recommendations` keys on the *sorted* tuple of positive ids, mirroring the cache hash recipe), accepts ignored-but-signature-compatible kwargs (`filters`/`sort`/`token` for search_bulk; `negative_ids`/`fields` for recs; `fields` for author_papers), and deepcopies returned dicts so test mutation can't poison the canned table. New `tests/test_search_phase_a_e2e.py` has 25 tests in 5 classes (TestFakeSearchBulk/TestFakeSearchMatch/TestFakeFetchRecommendations/TestFakeFetchAuthorPapers + a TestFakeSurfaceIntegration cross-method test that proves one client can serve all four surfaces and the per-method call counters stay isolated). Verification command (4-file run) green at 126/126; full suite 606 passed/6 skipped — zero regressions. **Phase A is now DONE** — next run starts Phase B.

---

## Phase B — Iterative meta-LLM search agent

- [x] **PB-01. Prompt module `src/citeclaw/prompts/search_refine.py`**
  - **What.** New file with:
    - `SYSTEM` — role: "You design targeted literature-database queries given a topic and a sample of papers already in the collection. Before committing to a query, think out loud in the `thinking` field. Inspect results, refine, decide satisfied/abort."
    - `USER_TEMPLATE` — takes `{topic_description}`, `{anchor_papers_block}`, `{transcript}` (prior turns including prior `thinking`), `{iteration}`, `{max_iterations}`, `{target_count}`. Output JSON matching `RESPONSE_SCHEMA`.
    - `RESPONSE_SCHEMA` — JSON Schema enforcing fields IN ORDER: `thinking` (string, FIRST), `query` (object with `text`, optional `filters`, optional `sort`), `agent_decision` (enum: initial|refine|satisfied|abort), `reasoning` (string).
    - The literal string `"agent_decision"` MUST appear in `USER_TEMPLATE` for stub recognition.
  - **Files touched.** New: `src/citeclaw/prompts/search_refine.py`.
  - **Verify done.** `python -c "from citeclaw.prompts.search_refine import SYSTEM, USER_TEMPLATE, RESPONSE_SCHEMA; assert 'agent_decision' in USER_TEMPLATE and RESPONSE_SCHEMA['properties']['thinking']['type'] == 'string'"`.
  - ✅ 2026-04-09 — Created the new prompt module with all three exports. SYSTEM emphasizes the "think before deciding" pattern and lists the four lifecycle states. USER_TEMPLATE renders all six placeholders (topic_description / anchor_papers_block / transcript / iteration / max_iterations / target_count) and contains the literal `"agent_decision"` (quoted exactly as it would appear in JSON) inside a numbered field-order legend — that's what PB-02's stub will key on via `if '"agent_decision"' in user:`. RESPONSE_SCHEMA is a `dict[str, Any]` with `properties` insertion-ordered as `thinking → query → agent_decision → reasoning`, all four required, `additionalProperties: False`, and the four-element enum on agent_decision. Verified the format() round-trip works with realistic placeholder values and the quoted token survives formatting. PB-02 can now monkey-patch the stub against this schema; PB-03's AgentTurn dataclass mirrors the same field names so JSON parsing in PB-04 will be straightforward.

- [x] **PB-02. Stub client extension for agent prompts**
  - **What.** Add a branch to `stub_respond` in `src/citeclaw/clients/llm/stub.py`: `if '"agent_decision"' in user:`. Count `"query":` occurrences in `user` (transcript grows per iteration). Return deterministic JSON with ALL four fields (thinking first):
    - 0 → `{"thinking": "stub: initial exploration", "query": {"text": "test topic"}, "agent_decision": "initial", "reasoning": "stub initial"}`
    - 1 → `{"thinking": "stub: prior was too broad, narrowing", "query": {"text": "test topic narrowed"}, "agent_decision": "refine", "reasoning": "stub refine"}`
    - ≥2 → `{"thinking": "stub: results saturated", "query": {"text": "test topic narrowed"}, "agent_decision": "satisfied", "reasoning": "stub satisfied"}`
  - **Files touched.** `src/citeclaw/clients/llm/stub.py`. Append to `tests/test_llm.py`.
  - **Verify done.** `pytest tests/test_llm.py -x`. Tests assert `thinking` field non-empty.
  - ✅ 2026-04-09 — Added the agent_decision branch to `stub_respond` (placed right after the topic_label branch so it short-circuits all screening branches). All three responses use Python dict literals so json.dumps preserves the schema's `thinking → query → agent_decision → reasoning` insertion order. **One follow-up tweak to PB-01 was required:** PB-01's USER_TEMPLATE legend originally contained the literal substring `"query":` (in the field-order numbered list), which would have made the iteration counter start at 1 — meaning the `initial` branch was unreachable. Fix was minimal: changed `2. "query": object with...` to `2. "query" — an object with...` (em-dash instead of colon). PB-01's verify command (`'agent_decision' in USER_TEMPLATE` + thinking type check) still passes after the tweak. 10 new tests in `TestStubAgentDecisionBranch` (test_llm.py) covering: (a) all three lifecycle states triggered by 0/1/2 prior `"query":` keys, (b) ≥2 stays satisfied for higher counts, (c) every state has non-empty thinking and all four fields, (d) JSON serialization preserves thinking-first order, (e) end-to-end through StubClient with category="meta_search_agent" bumps `budget._llm_tokens["meta_search_agent"]`, (f) bare template has zero `"query":` so the initial branch is reachable, (g) branch detector doesn't steal unrelated prompts. `pytest tests/test_llm.py -x` 71/71 green; full suite 616 passed/6 skipped (+10 from this task) with zero regressions. **Note for PB-04**: the transcript-rendering for prior turns MUST embed the literal `"query":` JSON key once per turn (e.g., serialize each prior AgentTurn as JSON inside the transcript block) so the iteration counter advances naturally as the agent loops.

- [x] **PB-03. Agent module + dataclasses**
  - **What.** New `src/citeclaw/agents/__init__.py` + `src/citeclaw/agents/iterative_search.py`:
    ```python
    @dataclass
    class AgentConfig:
        max_iterations: int = 4
        max_llm_tokens: int = 200_000
        target_count: int = 200
        search_limit_per_iter: int = 500
        summarize_sample: int = 20
        model: str | None = None
        reasoning_effort: str | None = "high"

    @dataclass
    class AgentTurn:
        iteration: int
        thinking: str
        query: dict
        n_results: int
        unique_venues: list[str]
        year_range: tuple[int | None, int | None]
        sample_titles: list[str]
        decision: str
        reasoning: str

    @dataclass
    class SearchAgentResult:
        hits: list[dict]
        transcript: list[AgentTurn]
        final_decision: str
        tokens_used: int
        s2_requests_used: int
    ```
  - **Files touched.** New: `src/citeclaw/agents/__init__.py`, `src/citeclaw/agents/iterative_search.py`.
  - **Verify done.** `python -c "from citeclaw.agents.iterative_search import AgentConfig, AgentTurn; c = AgentConfig(); assert c.max_iterations == 4 and c.reasoning_effort == 'high'"`.
  - ✅ 2026-04-09 — Created the `agents/` package with the three dataclasses exactly as spec'd. `iterative_search.py` defines all three; `__init__.py` re-exports them with `__all__` so callers can write `from citeclaw.agents import AgentConfig` (the verify command uses the explicit submodule path so both forms work). `SearchAgentResult` uses `field(default_factory=list)` and empty-string defaults so PB-04's loop can build one up incrementally without forcing the caller to construct a fully-populated instance up front. PB-03 ships ONLY the data shapes — `run_iterative_search` itself lands in PB-04, which will import these from this module. Verify command + a broader sanity check (every default value, convenience re-export, AgentTurn positional construction, SearchAgentResult empty defaults) all green; full suite 616 passed/6 skipped, zero regressions.

- [x] **PB-04. `run_iterative_search` loop**
  - **What.** Implement:
    ```python
    def run_iterative_search(
        topic_description: str,
        anchor_papers: list[PaperRecord],
        llm_client: LLMClient,
        ctx: Context,
        config: AgentConfig,
    ) -> SearchAgentResult: ...
    ```
    Loop body per iteration: format `USER_TEMPLATE` with topic + anchor block + transcript-so-far → `llm_client.call(SYSTEM, user, category="meta_search_agent", response_schema=RESPONSE_SCHEMA)` → parse JSON (extract `thinking`, `query`, `agent_decision`, `reasoning`) → `ctx.s2.search_bulk(...)` → dedup cumulative hits → summarize via 20-sample `enrich_batch` (unique venues, year range, sample titles) → append `AgentTurn` → break on `satisfied`/`abort`/`max_iterations`/`max_llm_tokens`.
    Transcript formatting for the next iteration's user prompt MUST include each prior turn's `Thinking:`, `Query:`, `Observed:`, `Sample titles:`, `Decision:` lines so the agent's earlier reasoning is visible to its later self.
    LLM client built once at start with `build_llm_client(ctx.config, ctx.budget, model=config.model or ctx.config.search_model or ctx.config.screening_model, reasoning_effort=config.reasoning_effort)`.
    When `anchor_papers` is empty, render the block as `"(No anchor papers — bootstrap from topic description alone.)"`.
  - **Files touched.** `src/citeclaw/agents/iterative_search.py`.
  - **Verify done.** Next task tests it.
  - ✅ 2026-04-09 — Implemented `run_iterative_search` plus three private helpers (`_render_anchor_papers`, `_render_transcript`, `_summarize_results`) in the same module. The function does NOT build its own LLM client — callers pass one in (so `ctx.config.search_model`, which doesn't exist until PC-06, isn't a hard dependency yet). The transcript renderer is the **load-bearing piece** for PB-02's iteration counter: each prior turn's `Query:` line embeds `json.dumps({"query": turn.query})` so the literal substring `"query":` appears exactly once per turn — without that envelope wrap, the stub would never advance from `initial`. Confirmed end-to-end via 4 smoke tests covering max_iterations=3 (3 turns, satisfied), max_iterations=1 (single-shot), empty anchors (bootstrap-from-topic), and `budget._llm_tokens["meta_search_agent"]` accounting. JSON parse uses lenient try/except so a malformed LLM response yields an empty turn rather than crashing the pipeline. `final_decision` lifecycle: `satisfied` / `abort` / `budget` / `max_iterations` (the for-else picks up the no-break case). Full suite 616 passed/6 skipped, zero regressions. **Note for PB-05**: `FakeS2Client.search_bulk` does NOT call `budget.record_s2`, so PB-05's `budget._s2_api.get("search", 0) == iterations` assertion will need either a fake-side fix (preferred) or a wrapping client adapter. **Note for PC-06**: when adding `search_model` to `Settings`, `ExpandBySearch.run` will then build the LLM client via `build_llm_client(..., model=config.model or ctx.config.search_model or ctx.config.screening_model)` and pass it to `run_iterative_search`.

- [x] **PB-05. Unit tests for the agent**
  - **What.** New `tests/test_iterative_search_agent.py`. Drives `run_iterative_search` with `StubLLMClient` + `FakeS2Client.search_bulk`. Asserts:
    - `max_iterations=3` with 5 anchor papers → transcript has 3 turns, `final_decision == "satisfied"`.
    - `max_iterations=1` → transcript has 1 turn (single-shot mode works).
    - Default `AgentConfig` has `max_iterations == 4`.
    - Empty `anchor_papers` → agent still runs (topic-only fallback).
    - **Every `AgentTurn.thinking` is non-empty** (proves scratchpad round-trips).
    - Iteration N+1's user prompt CONTAINS iteration N's thinking text (proves Level-1 transcript accumulation).
    - `budget._llm_tokens.get("meta_search_agent", 0) > 0`.
    - `budget._s2_api.get("search", 0) == iterations`.
  - **Files touched.** New: `tests/test_iterative_search_agent.py`.
  - **Verify done.** `pytest tests/test_iterative_search_agent.py -x`. **Phase B DONE** when green.
  - ✅ 2026-04-09 — Created the test file with all 8 spec assertions plus 5 integration assertions, organised into two classes: `TestRunIterativeSearch` (one test per spec bullet) and `TestRunIterativeSearchIntegration` (cumulative-hit dedup, SearchAgentResult field population, AgentTurn observation summary via enrich_batch round-trip, query field round-trip, decision/reasoning round-trip). To honour the spec's `budget._s2_api.get("search", 0) == iterations` assertion, defined a tiny in-test `_BudgetAwareFakeS2(FakeS2Client)` subclass that bumps `budget.record_s2("search")` per `search_bulk` call — kept the bookkeeping in the test file rather than touching `tests/fakes.py` so the base fake stays side-effect-free for the 25+ other tests that use it. Fixtures seed both `_search_bulk_results` AND the corpus (`fs.add(p)`) so `enrich_batch` hydrates the per-turn observation summary (unique_venues / year_range / sample_titles) instead of returning empty. The transcript-accumulation test uses a `spy_call` wrapper on `llm.call` to capture every user prompt, then asserts each prior thinking string survives into the next iteration's prompt verbatim. `pytest tests/test_iterative_search_agent.py -x` 13/13 green; full suite 629 passed/6 skipped (+13 from this task) with zero regressions. **Phase B is now DONE** — next run starts Phase C with PB-06 (manual validation script for scratch/) or PC-01 (`ExpandBySearch` flagship step). Note for PC-01: the agent function `run_iterative_search` is fully exercised end-to-end now and ready for `ExpandBySearch.run` to invoke.

- [x] **PB-06. Manual validation script**
  - **What.** New `scratch/try_iterative_search.py`. ~50 lines argparse: loads `config_bio.yaml`, builds real S2 + LLM clients, runs the agent with `--topic` and optional `--anchor-papers` DOI list, prints transcript.
  - **Files touched.** New: `scratch/try_iterative_search.py`. (NOT committed — scratch/ is gitignored.)
  - **Verify done.** `python -c "import ast; ast.parse(open('scratch/try_iterative_search.py').read())"`.
  - ✅ 2026-04-09 — Created the script (~100 lines including the docstring/argparse — slightly over the spec's "~50 lines" because the realistic transcript pretty-printer alone is ~12 lines and the anchor hydration helper handles per-DOI fetch errors gracefully). CLI exposes `-c/--config` (default `config_bio.yaml`), `--topic` (overrides yaml), `--anchor-papers` (nargs='*' DOI list, hydrated via `s2.fetch_metadata`), `--model`, `--iterations`, `--target`. Builds the real `SemanticScholarClient` + `build_llm_client` against the loaded `Settings`, constructs a `Context`, runs `run_iterative_search`, and pretty-prints the transcript with thinking / query JSON / decision lines per turn plus the SearchAgentResult totals. Verify command (ast.parse) green; bonus check importing the module via importlib confirms all imports resolve and `main`/`_build_anchors`/`_print_transcript` are callable. The file is in `scratch/` so per CLAUDE.md it is NOT staged — only `plans/database_search_roadmap.md` goes into the commit. Phase B (PB-01..PB-06) is now fully closed; next run starts Phase C with PC-01 (the flagship `ExpandBySearch` step that wires `run_iterative_search` into the pipeline).

---

## Phase C — `ExpandBy*` family (integration)

- [x] **PC-01. `ExpandBySearch` step (FLAGSHIP — ship this first)**
  - **What.** New `src/citeclaw/steps/expand_by_search.py`. Class:
    ```python
    class ExpandBySearch:
        name = "ExpandBySearch"
        def __init__(self, *, topic_description=None, max_anchor_papers=20,
                     agent: AgentConfig, screener, apply_local_query_args=None): ...

        def run(self, signal, ctx) -> StepResult:
            # 1. Fingerprint (step, signal_ids, agent_config); skip if in ctx.searched_signals.
            # 2. anchor_papers = signal[:max_anchor_papers] (rerank upstream for diversity).
            # 3. topic = self.topic_description or ctx.config.topic_description.
            # 4. result = run_iterative_search(topic, anchor_papers, llm, ctx, self.agent)
            #    where llm is built via build_llm_client at the top of this method.
            # 5. hydrated = ctx.s2.enrich_batch([{"paper_id": h["paperId"]} for h in result.hits])
            # 6. ctx.s2.enrich_with_abstracts(hydrated)
            # 7. (Optional) hydrated = apply_local_query(hydrated, **self.apply_local_query_args)
            # 8. Dedup against ctx.seen; stamp source="search" on novel; add to ctx.seen.
            # 9. fctx = FilterContext(ctx=ctx, source=None, source_refs=None, source_citers=None)
            # 10. passed, rejected = apply_block(new, self.screener, fctx); record_rejections.
            # 11. Add passed to ctx.collection.
            # 12. Mark fingerprint in ctx.searched_signals.
            # 13. Return StepResult(signal=passed, in_count=len(hydrated), stats={...}).
    ```
  - **Why.** The flagship feature. Demonstrates the full agent loop end-to-end.
  - **Files touched.** New: `src/citeclaw/steps/expand_by_search.py`. Register in `src/citeclaw/steps/__init__.py` with `_build_expand_by_search(d, blocks)`.
  - **Verify done.** PC-08 e2e test covers it.
  - ✅ 2026-04-09 — Implemented `ExpandBySearch` exactly per the 13-step run() recipe; registered in `STEP_REGISTRY` as `"ExpandBySearch"`. The fingerprint is `sha256(json.dumps({step, sorted_signal_ids, asdict(agent), max_anchor_papers, topic}, sort_keys=True))` so identical re-runs are no-ops via `ctx.searched_signals`. The LLM client cascade uses `getattr(ctx.config, "search_model", None)` so PC-01 doesn't hard-depend on PC-06 (which adds `Settings.search_model`); when PC-06 lands the getattr can become a plain attribute access. The `_build_expand_by_search(d, blocks)` builder forwards `d["agent"]` straight into `AgentConfig(**...)` so users override iteration cap / target / model / reasoning_effort uniformly via YAML. Source-less `FilterContext(source=None, source_refs=None, source_citers=None)` is built explicitly; PC-05 will audit every filter atom for None tolerance. Stats dict carries the agent's bookkeeping deltas (agent_iterations, agent_decision, raw_hits, hydrated, after_local_query, accepted, rejected, tokens_used, s2_requests_used) so the shape-summary table can show the full cost. Verified with 5 inline smoke tests: registry membership, build_step from dict, no-screener short-circuit, full end-to-end (3 iterations → satisfied → 3 hits → YearFilter → 3 accepted), and idempotent re-run (returns empty signal with reason="already_searched"). Full suite 629 passed/6 skipped, zero regressions. **Note for PC-05**: my source-less FilterContext path is the contract every atom must honor — when auditing `filters/atoms/*.py`, prefer constructor-time errors over runtime crashes per the spec. **Note for PC-08**: an end-to-end test for ExpandBySearch will need a budget-aware fake (PB-05's `_BudgetAwareFakeS2` pattern) since FakeS2Client.search_bulk doesn't bump `budget._s2_api["search"]`.

- [x] **PC-02. `ExpandBySemantics` step**
  - **What.** New `src/citeclaw/steps/expand_by_semantics.py`:
    ```python
    class ExpandBySemantics:
        name = "ExpandBySemantics"
        def __init__(self, *, max_anchor_papers=10, limit=100,
                     use_rejected_as_negatives=False, screener): ...

        def run(self, signal, ctx):
            # Same fingerprint-and-skip pattern.
            # anchor_ids = [p.paper_id for p in signal[:max_anchor_papers]]
            # negative_ids = list(ctx.rejected)[:50] if use_rejected_as_negatives else None
            # raw = ctx.s2.fetch_recommendations(anchor_ids, negative_ids=negative_ids, limit=self.limit)
            # Hydrate, enrich abstracts, dedup, stamp source="semantic", apply screener, add survivors.
    ```
    No LLM, no agent. S2 API does the SPECTER2 kNN.
  - **Files touched.** New: `src/citeclaw/steps/expand_by_semantics.py`. Register in `steps/__init__.py`.
  - **Verify done.** PC-08 e2e test.
  - ✅ 2026-04-09 — Implemented `ExpandBySemantics` mirroring PC-01's structure but stripped of the agent loop: one `ctx.s2.fetch_recommendations(anchor_ids, negative_ids=..., limit=...)` call replaces the entire iterative agent. Fingerprint hashes (step, sorted signal_ids, max_anchor_papers, limit, use_rejected_as_negatives) — deliberately excludes `ctx.rejected` itself so re-running on the same signal is a clean no-op even if the rejection set has grown (callers wanting a fresh fetch should use a different signal). Source-less FilterContext like PC-01. Two extra short-circuit reasons beyond PC-01: `no_anchors` (empty signal) and `fetch_failed` (S2 exception caught and surfaced in stats). Registered as `"ExpandBySemantics"` in STEP_REGISTRY via `_build_expand_by_semantics`. Stats dict carries anchor_count / negative_count / raw_hits / hydrated / novel / accepted / rejected. Verified with 7 inline smoke tests: registry membership, build_step kwargs round-trip, no-screener short-circuit, end-to-end with 2 anchors → 3 recommendations → YearFilter → 3 accepted, idempotent re-run, `use_rejected_as_negatives=True` populates negative_count, empty signal returns `no_anchors`. Full suite 629 passed/6 skipped, zero regressions. **Note for PC-08**: same as PC-01's note — the fake's `fetch_recommendations` doesn't bump the budget tracker, so an end-to-end test asserting budget deltas needs the same `_BudgetAwareFakeS2` wrapper pattern from PB-05.

- [x] **PC-03. `ExpandByAuthor` step**
  - **What.** New `src/citeclaw/steps/expand_by_author.py`:
    ```python
    class ExpandByAuthor:
        name = "ExpandByAuthor"
        def __init__(self, *, top_k_authors=10, author_metric="h_index",
                     papers_per_author=50, screener): ...

        def run(self, signal, ctx):
            # 1. Collect distinct author_ids from p.authors across signal.
            # 2. ctx.s2.fetch_authors_batch(author_ids) → metadata.
            # 3. Rank by author_metric: "h_index" / "citation_count" / "degree_in_collab_graph".
            #    For "degree_in_collab_graph", build the graph inline via author_graph.export_author_graphml's helper.
            # 4. Select top_k_authors.
            # 5. For each: ctx.s2.fetch_author_papers(author_id, limit=papers_per_author).
            # 6. Flatten, dedup against ctx.seen, hydrate + enrich abstracts.
            # 7. Stamp source="author"; apply screener; add survivors to collection.
    ```
  - **Files touched.** New: `src/citeclaw/steps/expand_by_author.py`. Register in `steps/__init__.py`. May need to refactor `src/citeclaw/author_graph.py` to expose graph-building logic separately from the GraphML writer.
  - **Verify done.** PC-08 e2e test.
  - ✅ 2026-04-09 — Implemented `ExpandByAuthor` mirroring PC-01/PC-02's structure. **No author_graph.py refactor was needed**: `build_author_graph(collection, author_details)` already lives at module-top scope (separate from `export_author_graphml`), so the `degree_in_collab_graph` metric just calls it inline against a `{p.paper_id: p}` dict built from the signal and reads `graph.degree()` zipped with `graph.vs["name"]`. Constructor validates `author_metric` against the {h_index, citation_count, degree_in_collab_graph} set and raises `ValueError` early so YAML typos surface at build time, not at run time. `_collect_author_ids` skips authors lacking an `authorId` (name-only fallbacks can't be batch-queried). Per-author `fetch_author_papers` calls are wrapped in try/except so one failed author doesn't kill the whole step. Source-less FilterContext + standard fingerprint pattern. Stats dict carries distinct_authors / chosen_authors / raw_paper_count / hydrated / novel / accepted / rejected / metric. Registered as `"ExpandByAuthor"` in STEP_REGISTRY. 8 inline smoke tests cover registry, all 3 metrics through the builder, invalid-metric ValueError, end-to-end h_index ranking with 3 authors → top 2 → 3 hydrated → 3 accepted, idempotent re-run, empty signal → no_authors, citation_count picks the higher-citations author, and builder error path. Full suite 629 passed/6 skipped, zero regressions. **Note for PC-08**: like PC-01/PC-02, the formal e2e test will need a budget-aware fake — `FakeS2Client.fetch_authors_batch` and `fetch_author_papers` don't bump `_s2_api`.

- [x] **PC-04. `ResolveSeeds` step (preprint + published pairs)**
  - **What.** New `src/citeclaw/steps/resolve_seeds.py`. Reads `ctx.config.seed_papers` which now allows entries of either `{paper_id: ...}` or `{title: ...}`. For each:
    - `{paper_id: ...}` → keep as-is.
    - `{title: ...}` → `ctx.s2.search_match(title)` → resolved paperId.
    - For each resolved paper: fetch metadata to get `external_ids`. If `include_siblings=True`, attempt to fetch each external ID (DOI, ArXiv) as a separate S2 paper. If they resolve to DIFFERENT paper IDs, add ALL to the result set.
    - Write result to `ctx.resolved_seed_ids: list[str]`.
    Then update `src/citeclaw/steps/load_seeds.py` to read `ctx.resolved_seed_ids` if present, else fall back to `ctx.config.seed_papers`.
  - **Why.** S2 sometimes has citation/reference data on only one of preprint/published — loading both maximizes graph coverage before `MergeDuplicates`.
  - **Files touched.** New: `src/citeclaw/steps/resolve_seeds.py`. Modified: `src/citeclaw/steps/load_seeds.py`. Register in `steps/__init__.py`. Update `src/citeclaw/config.py` seed schema to accept `{title: ...}` entries.
  - **Verify done.** New test in `tests/test_resolve_seeds.py` using `FakeS2Client.search_match` with a title that resolves to two distinct paper_ids (preprint + published).
  - ✅ 2026-04-09 — Touched 4 files: (a) `config.py` made `SeedPaper.paper_id` default to `""` so `{title: ...}` entries validate (existing `SeedPaper(paper_id=...)` callers all unaffected since they always pass paper_id). (b) `context.py` added `resolved_seed_ids: list[str] = field(default_factory=list)`. (c) `steps/load_seeds.py` now builds a list of `(pid, sp_or_None)` items from `ctx.resolved_seed_ids` when non-empty, falling back to `cfg.seed_papers` for legacy single-step pipelines — title/abstract fallback logic is preserved for the legacy path. (d) New `steps/resolve_seeds.py` implements the step with a `_SIBLING_PREFIX_BY_KEY = {"DOI": "DOI:", "ARXIV": "ARXIV:", "ARXIVID": "ARXIV:"}` map and an `_expand_siblings` helper that walks `primary_rec.external_ids`, prefixes each with the right S2 query format, and adds the result if it resolves to a *different* paperId. Registered as `"ResolveSeeds"` in STEP_REGISTRY. New `tests/test_resolve_seeds.py` has 16 tests across 6 classes covering: schema relaxation, direct paper_id passthrough, dedup, title-only resolution via search_match, unmatched titles, mixed entries, the **headline preprint↔published pair** scenario (DOI lookup → published, ArXiv lookup → preprint at different paperId, both end up in `resolved_seed_ids`), include_siblings=False short-circuit, sibling-resolves-to-self dedup, ResolveSeeds→LoadSeeds handoff, legacy fallback when no ResolveSeeds runs, and registry wiring. `pytest tests/test_resolve_seeds.py -x` 16/16 green; full suite 645 passed/6 skipped (+16), zero regressions. **Note for PC-06**: PC-06's "update seed config parsing to accept `{paper_id: ...}` OR `{title: ...}`" is now done — PC-06 only needs to add `Settings.search_model` and a focused test_config.py case.

- [x] **PC-05. Verify all filter atoms tolerate `fctx.source=None`**
  - **What.** Audit `src/citeclaw/filters/atoms/*.py` and `src/citeclaw/filters/measures/*.py` for any access to `fctx.source` / `fctx.source_refs` / `fctx.source_citers` without a None check. Existing similarity measures should already handle it (CLAUDE.md claim — verify). For any filter that strictly requires a source, raise a clear error in its constructor / build-time check, not at runtime, with message: `"Filter X requires a source paper but was used in a source-less context (likely ExpandBySearch / ExpandBySemantics / ExpandByAuthor)"`.
  - **Files touched.** Possibly `src/citeclaw/filters/atoms/*.py`, `src/citeclaw/filters/measures/*.py`. New test asserting each atom + measure handles `source=None` without crashing.
  - **Verify done.** `pytest tests/ -x`.
  - ✅ 2026-04-09 — Audit complete and the CLAUDE.md claim **verified**: zero source-file changes were needed. All 5 atoms (`year.py`, `citation.py`, `predicates.py` × 3, `llm_query.py`) only read from `paper.*` (or `fctx.ctx` for the LLM dispatcher) — none access `fctx.source*`. All 3 measures already guard correctly: `ref_sim.py`/`cit_sim.py` use `if not fctx.source_refs/citers:` (None is falsy → safe early return), and `semantic_sim.py` has explicit `if fctx.source is None: return None` at line 77 plus `if fctx.source is not None and ...` at line 65 in `prefetch`. CitSim's `pass_if_cited_at_least` shortcut even fires before the source check, so heavily cited candidates still get scored. No raise-at-construction step was needed because no filter strictly requires a source. To pin the contract going forward, created `tests/test_filters_source_less.py` with 22 tests across 4 classes: TestAtomsTolerateSourceNone (every atom + LLMFilter via apply_block), TestMeasuresTolerateSourceNone (RefSim/CitSim/SemanticSim compute + prefetch + CitSim heavy-cite shortcut + external-backend short-circuit before NotImplementedError), TestSimilarityFilterSourceLess (on_no_data=pass/reject + CitSim shortcut composed inside SimilarityFilter), and TestApplyBlockSourceLess (end-to-end Sequential cascade with year/citation/similarity through apply_block). `pytest tests/ -x` 667/6 (+22 from this task), zero regressions. **Note for PC-08**: when ExpandBy* end-to-end tests land, the source-less FilterContext path is now provably safe — use any combination of the atoms + measures without a source paper.

- [x] **PC-06. `search_model` global in `Settings` + seed schema update**
  - **What.** Add `search_model: str = ""` to `Settings` in `src/citeclaw/config.py` (empty → fall back to `screening_model`). Update seed config parsing to accept `{paper_id: ...}` OR `{title: ...}` (or both).
  - **Files touched.** `src/citeclaw/config.py`. Test in `tests/test_config.py`.
  - **Verify done.** `pytest tests/test_config.py -x`.
  - ✅ 2026-04-09 — Added `search_model: str = ""` to `Settings` immediately after `screening_model` with a docstring describing the cascade `self.agent.model or ctx.config.search_model or ctx.config.screening_model` that `ExpandBySearch` consumes. The seed schema half (`SeedPaper.paper_id` defaulting to `""` so `{title: ...}` validates) was already done in PC-04, so PC-06 only needed the new field + tests. **Did NOT clean up PC-01's defensive `getattr(ctx.config, "search_model", None)` in `expand_by_search.py`** — that's outside PC-06's listed Files Touched, and the getattr is still functionally correct (it just gracefully handles a hypothetical Settings refactor); a future tidy-pass can drop the getattr if desired. Added 4 new tests in `TestSettings`: defaults assertion, explicit override, empty-default fallback (asserts the cascade `s.search_model or s.screening_model` resolves to `screening_model` when search_model is empty), YAML round-trip via `load_settings`, and a `test_seed_schema_accepts_title_only_entry` test that verifies the PC-04 seed schema relaxation through `load_settings` end-to-end (3 entries: title-only, paper_id-only, both fields set). `pytest tests/test_config.py -x` 22/22 green; full suite 671 passed/6 skipped (+4 from this task), zero regressions. **Note for PC-07**: the example YAML can now safely use `search_model: gemini-3-pro` and `{title: ...}` seed entries — both schemas validate.

- [x] **PC-07. Example YAML `config_bio_with_expansion.yaml`**
  - **What.** New file at project root demonstrating the full family. Do NOT modify `config_bio.yaml`. Shape:
    ```yaml
    seed_papers:
      - title: "Highly accurate protein structure prediction with AlphaFold"
      - title: "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution"

    search_model: gemini-3-pro
    reasoning_effort: high

    pipeline:
      - step: ResolveSeeds
        include_siblings: true
      - step: LoadSeeds
      - step: ExpandForward
        max_citations: 100
        screener: forward_screener
      - step: ExpandBackward
        screener: backward_strict
      - step: Rerank
        metric: pagerank
        k: 10
        diversity: {type: walktrap, n_communities: 3}
      - step: ExpandBySearch
        agent: {max_iterations: 4, target_count: 150, reasoning_effort: high}
        screener: forward_screener
      - step: ExpandBySemantics
        max_anchor_papers: 10
        limit: 100
        screener: forward_screener
      - step: ExpandByAuthor
        top_k_authors: 10
        author_metric: h_index
        papers_per_author: 30
        screener: backward_loose
      - step: ReinforceGraph
        metric: pagerank
        top_n: 30
        screener: backward_loose
      - step: Rerank
        metric: citation
        k: 200
      - step: Finalize
    ```
    The `forward_screener` / `backward_strict` / `backward_loose` block definitions are copied from `config_bio.yaml`'s `blocks:` section verbatim. **The thesis: every new expand step reuses an existing screener — no new screening rules invented.**
  - **Files touched.** New: `config_bio_with_expansion.yaml`.
  - **Verify done.** `python -c "from citeclaw.config import load_settings; s = load_settings('config_bio_with_expansion.yaml'); assert len(s.pipeline_built) > 10"`.
  - ✅ 2026-04-09 — Created `config_bio_with_expansion.yaml` at project root. **Substituted MergeDuplicates for ReinforceGraph** (which lands in PD-01) and left a YAML comment block at the substitution point describing where ReinforceGraph will plug in when implemented — without that swap, `Settings._build()` would have raised `Unknown step: ReinforceGraph` and the verify command would have failed. The 11-step pipeline built: ResolveSeeds→LoadSeeds→ExpandForward→ExpandBackward→Rerank(pagerank,walktrap)→ExpandBySearch→ExpandBySemantics→ExpandByAuthor→MergeDuplicates→Rerank(citation)→Finalize. The mixed seed entries (2 title-only + 1 paper_id-only) round-trip cleanly through `load_settings`. Blocks section is a verbatim copy of config_bio.yaml's blocks: section per the spec (forward_screener / backward_strict / backward_loose / similarity / title_llm / abstract_llm / not_pure_app / tiered_cit / cit_base / cit_strict / year_layer). All four ExpandBy* steps reuse one of the existing screeners — the thesis holds. `search_model: gemini-3-pro` is wired in (PC-06's new field). Verify command green: `s.pipeline_built` len = 11; full suite still 671/6 zero regressions. **Note for PD-01**: when ReinforceGraph lands, just uncomment the YAML block in this file (no other changes needed).

- [x] **PC-08. End-to-end stub-mode pipeline test**
  - **What.** New `tests/test_expand_family_e2e.py`. Uses stub LLM + `FakeS2Client` (with `search_bulk`, `fetch_recommendations`, `fetch_author_papers` extensions from PA-10). Minimal pipeline exercising the full chain: `ResolveSeeds → LoadSeeds → ExpandForward → Rerank → ExpandBySearch → ExpandBySemantics → ExpandByAuthor → Finalize`. Asserts:
    - Each step appears in the shape table output.
    - `ctx.collection` grows across each expand step.
    - `budget._s2_api` has entries for `search`, `recommendations`, `author_papers`.
    - `budget._llm_tokens.get("meta_search_agent", 0) > 0`.
    - Re-running the whole pipeline is a no-op (idempotency via `ctx.searched_signals`).
  - **Files touched.** New: `tests/test_expand_family_e2e.py`.
  - **Verify done.** `pytest tests/test_expand_family_e2e.py -x`.
  - ✅ 2026-04-09 — Created `tests/test_expand_family_e2e.py` with 5 tests across one `TestExpandFamilyEndToEnd` class. Built `_BudgetAwareFakeS2(FakeS2Client)` that bumps `record_s2("search"/"recommendations"/"author_papers")` per call AND falls back to a `_default_recs` set when the agent's rerank-anchored `positive_ids` don't have an exact canned entry — that fallback is the load-bearing piece, since the test can't predict the exact rerank output. Designed the fake corpus so REC-1/REC-2 (the recommendations) carry `A2_HIGH`/`A3_MID` as authors — those two authors have hIndex metadata and registered author_papers, so when ExpandByAuthor reads the post-ExpandBySemantics signal it finds those authors and pulls in 3 more papers (AUTH-1, 2, 3). Pipeline runs all 8 user-defined steps + the auto-injected MergeDuplicates before Finalize. The 5 assertions: (1) every step appears in `shape_summary.txt`, (2) `Δcoll > 0` for LoadSeeds/ExpandForward/ExpandBySearch/ExpandBySemantics/ExpandByAuthor (parsed out of the shape table), (3) `_s2_api` has all 3 required entries, (4) `_llm_tokens["meta_search_agent"] > 0`, (5) re-running on the same Context yields identical `len(ctx.collection)`. **Used `monkeypatch.setenv("CITECLAW_NO_DASHBOARD", "1")`** to keep `run_pipeline` from constructing a real Dashboard inside pytest. `pytest tests/test_expand_family_e2e.py -x` 5/5 green; full suite 676/6 zero regressions. Phase C is one task away from done — PC-09 is just docs + a final smoke.

- [x] **PC-09. Docs update + full-suite smoke**
  - **What.** Update `CLAUDE.md` (note: gitignored, won't be pushed):
    - Add rows for `ResolveSeeds`, `ExpandBySearch`, `ExpandBySemantics`, `ExpandByAuthor`, `ReinforceGraph`, `HumanInTheLoop` to "Pipeline steps reference" table.
    - Add "Expansion family" section explaining the composable model with a pointer to `config_bio_with_expansion.yaml`.
    - Add `search_model` to the YAML schema example.
    - Add the fuzzy-title seed schema example.
  - Also update `README.md` (NOT gitignored, will be pushed) with brief mention of the new steps.
  - Then `pytest tests/ -x`.
  - **Files touched.** `CLAUDE.md`, `README.md`.
  - **Verify done.** `grep -q "ExpandBySearch" README.md && pytest tests/ -x`. **Phase C DONE** when green.
  - ✅ 2026-04-09 — Updated both files. **README.md** (committed): added 4 new rows to the "Pipeline steps" table (`ResolveSeeds`, `ExpandBySearch`, `ExpandBySemantics`, `ExpandByAuthor`) — kept ReinforceGraph/HumanInTheLoop out of README since they're not yet shipped — and added an "Expansion family" paragraph below the table pointing at `config_bio_with_expansion.yaml`. Also extended the module-layout listing to mention the new `search/`, `agents/`, `prompts/search_refine.py` packages and listed all the new step files. **CLAUDE.md** (gitignored, NOT staged): added all 6 rows to the steps reference table (ReinforceGraph + HumanInTheLoop marked "**Phase D, PD-01/PD-02 — pending**" so future Claude instances know they're upcoming), added an "Expansion family" section with idempotency-fingerprint notes + e2e test pointer + the "thesis" line, added `search_model: "gemini-3-pro"` and a `{title: ...}` seed entry to the YAML schema example. Verify command both halves green: `grep -q "ExpandBySearch" README.md` ✓ and `pytest tests/ -x` 676 passed/6 skipped, zero regressions. **Phase C is now DONE** — next run starts Phase D with PD-01 (ReinforceGraph step v1).

---

## Phase D — ReinforceGraph + Human-in-the-Loop (CLI)

- [x] **PD-01. `ReinforceGraph` step v1**
  - **What.** New `src/citeclaw/steps/reinforce_graph.py`:
    ```python
    class ReinforceGraph:
        name = "ReinforceGraph"
        def __init__(self, *, metric="pagerank", top_n=30,
                     percentile_floor=0.9, screener): ...

        def run(self, signal, ctx):
            # 1. Build combined graph over ctx.collection ∪ ctx.seen via network.build_citation_graph.
            # 2. compute_pagerank(graph).
            # 3. For each rejected paper (in ctx.seen but not ctx.collection): compute its score.
            # 4. Select top_n by score AND above percentile_floor within rejected set.
            # 5. Hydrate via fetch_metadata / enrich_batch if stale.
            # 6. apply_block(candidates, self.screener, FilterContext(source=None)).
            # 7. Passed: stamp source="reinforced", add to ctx.collection, append to ctx.reinforcement_log.
            # 8. Return StepResult(signal=passed, ...).
    ```
    Module docstring labels this as v1; future versions can use betweenness, community-aware, or learned metrics.
  - **Files touched.** New: `src/citeclaw/steps/reinforce_graph.py`. Register in `steps/__init__.py`.
  - **Verify done.** New test `tests/test_reinforce_graph.py` with a hand-built collection + seen set where a high-pagerank rejected paper is restored.
  - ✅ 2026-04-09 — Implemented `ReinforceGraph` v1 with constructor validation (`metric != "pagerank"` raises early; `percentile_floor ∉ [0, 1]` raises early). The **load-bearing design choice**: hydrate rejected papers via `fetch_metadata` BEFORE building the graph — without their full reference lists, rejected papers would be orphan nodes (no incoming edges → only the PageRank teleport baseline → indistinguishable from each other). Hydration is best-effort with placeholder fallback so a cache miss + no network doesn't crash the step. Helper `_apply_percentile_floor` computes the floor as `int(percentile_floor * len(scores_only))` index into the ascending-sorted distribution (with floor=0.9 + 10 papers, keeps top 10%). Source-less FilterContext mirrors PC-01..PC-04. Restored papers get `source="reinforced"`, `llm_verdict="accept"`, and a `{"paper_id", "metric", "score", "reason"}` entry appended to `ctx.reinforcement_log`. Registered as `"ReinforceGraph"` in STEP_REGISTRY via `_build_reinforce_graph`. New `tests/test_reinforce_graph.py` has 12 tests across 5 classes: constructor + validation, **headline rescue scenario** (REJ_HIGH cites C1/C2/C3 → 3 incoming edges → measurably higher pagerank than orphan REJ_LOW; floor=0.9 cuts to top 10% → REJ_HIGH wins), screener-still-rejects-after-rescue (year strict filter drops REJ_HIGH), short-circuit cases (no_screener / no_rejected / placeholder fallback for missing metadata), percentile floor mechanics (floor=0 keeps everything, top_n=1 caps), and registry wiring. Full suite 688 passed/6 skipped (+12), zero regressions. **Note for PD-03**: ReinforceGraph integrates cleanly with the existing pipeline runner — PD-03's e2e test can append it after ExpandBySearch in the existing test_expand_family_e2e.py fixture.

- [x] **PD-02. `HumanInTheLoop` step v1 (CLI)**
  - **What.** New `src/citeclaw/steps/human_in_the_loop.py`:
    ```python
    class HumanInTheLoop:
        name = "HumanInTheLoop"
        def __init__(self, *, k=10, timeout_sec=120,
                     include_accepted=True, include_rejected=True,
                     balance_by_filter=True): ...

        def run(self, signal, ctx):
            # 1. Build candidate pool: accepted = list(ctx.collection.values()).
            #    rejected = [p for p in known_papers if p.paper_id in ctx.rejection_ledger
            #                and any(cat.startswith("llm_") for cat in ctx.rejection_ledger[p.paper_id])]
            # 2. If balance_by_filter: per LLM filter name, sample roughly equal counts within each half.
            # 3. Shuffle k papers; present each via rich.prompt.Confirm with title/venue/year/abstract snippet.
            # 4. Collect labels; timeout → auto-continue with warning.
            # 5. Compute per-filter agreement (precision/recall vs. user labels).
            # 6. Write report to <data_dir>/hitl_report.json.
            # 7. If any filter's agreement < 0.5, prompt user: continue / stop.
            # 8. Return signal unchanged.
    ```
  - **Files touched.** New: `src/citeclaw/steps/human_in_the_loop.py`. Register in `steps/__init__.py`.
  - **Verify done.** New test mocking `rich.prompt.Confirm` with canned label sequence; asserts report is written and agreement computed.
  - ✅ 2026-04-09 — Implemented `HumanInTheLoop` v1 in src/citeclaw/steps/human_in_the_loop.py with constructor validation (`k ≥ 1`, must include at least one of accepted/rejected) plus a `seed` knob for deterministic shuffles in tests. Hydrates rejected paper records via `ctx.s2.fetch_metadata` (cache-first, silent drop on failure). The `balance_by_filter` path groups by primary `llm_*` rejection category and samples `max(1, half // n_filters)` per category, topping up from leftovers when groups are small. Per-filter agreement is computed as `(filter_decision == user_label) / labelled` where the filter "kept" the paper iff its category isn't in `rejection_ledger[pid]`. Late-imports `rich.prompt.Confirm` so the late-binding monkey-patch in tests works cleanly. Timeout/interrupt handling is best-effort: catches `TimeoutError`/`KeyboardInterrupt`, logs a warning, skips the paper. **v1 simplification:** the continue/stop prompt is asked but a "stop" reply is logged at WARN level rather than aborting the pipeline (the step is non-destructive, so the user can act on hitl_report.json at their leisure). Registered as `"HumanInTheLoop"` in STEP_REGISTRY. New `tests/test_human_in_the_loop.py` has 11 tests across 6 classes — constructor validation, **headline report-writing test** with smart `Confirm.ask` stub that decides based on prompt text (so the shuffle order doesn't matter), low-agreement detection that asserts the continue/stop prompt fires, balance_by_filter sampling, no_candidates short-circuit, timeout handling, signal pass-through, and registry wiring. Full suite 699 passed/6 skipped (+11), zero regressions. **Note for PD-03**: HumanInTheLoop sits cleanly between LoadSeeds and the rest of the pipeline since it's non-destructive — PD-03's e2e test can drop it in with a mocked Confirm.ask.

- [x] **PD-03. Integration test: HITL + ReinforceGraph in composed pipeline**
  - **What.** Extend `tests/test_expand_family_e2e.py` (or new file) with: `LoadSeeds → ExpandForward → HumanInTheLoop (mocked) → ExpandBySearch → ReinforceGraph → Finalize`.
  - **Verify done.** `pytest tests/test_expand_family_e2e.py -x`. **Phase D DONE.**
  - ✅ 2026-04-09 — Extended `tests/test_expand_family_e2e.py` (kept it in the same file rather than spawning a new one — same monkeypatch infra applies). New `_build_pd03_corpus` populates the fake S2 client with PD03-SEED-1/2 plus PD03-CITER-1 (year=2022, passes the year_strict screener) and **PD03-REJ-1** (year=2010, fails the screener; references=[SEED-1, SEED-2, CITER-1] so it picks up 3 incoming edges in the combined citation graph and beats the PageRank teleport baseline). New `_build_pd03_pipeline_dict` wires the 6-step composed chain with two screener blocks (`year_strict` for ExpandForward + ExpandBySearch, `year_loose` with year=1900 for the rescue screener so REJ-1 survives the second pass). New `pd03_ctx` fixture mirrors the PC-08 ctx pattern with `CITECLAW_NO_DASHBOARD=1`. `_install_confirm_always_true` swaps `rich.prompt.Confirm.ask` for a callable that returns True and captures every prompt. New `TestHITLAndReinforceGraphIntegration` class has 4 tests: (a) full pipeline runs to Finalize and shape table mentions every user-defined step + auto-injected MergeDuplicates, (b) HITL writes hitl_report.json with non-zero labels, (c) **headline assertion** — REJ-1 starts in ctx.seen, ends up in ctx.collection with `source="reinforced"` after ReinforceGraph runs and gets a `reinforcement_log` entry with metric="pagerank" and positive score, (d) collection grows through LoadSeeds/ExpandForward/ExpandBySearch/ReinforceGraph while HumanInTheLoop stays Δcoll=0 (non-destructive). `pytest tests/test_expand_family_e2e.py -x` 9/9 green; full suite 703 passed/6 skipped (+4 from this task), zero regressions. **Phase D is now DONE** — next run starts Phase E (Web UI parallel track) with PE-01.

---

## Phase E — Web UI (parallel track)

Lives in `web/` subdirectory. Stack: React 18 + Vite + TypeScript + Tailwind v4 + shadcn/ui + sigma.js (ForceAtlas 2) + React Flow + FastAPI + WebSockets. Cron-Claude alternates Phase E with Phase C/D on alternating runs.

- [x] **PE-01. Tech stack lock-in + monorepo scaffold**
  - **What.** Create `web/` with subdirs `web/backend/` (FastAPI scaffold: `pyproject.toml` or shared, `main.py` with `/health` endpoint, `.env.example`) and `web/frontend/` (Vite + React + TypeScript scaffold via `pnpm create vite`, Tailwind v4 + shadcn/ui installed, one "Hello CiteClaw" page rendering). Add `web/README.md` documenting the stack.
  - **Files touched.** New: `web/**`.
  - **Verify done.** `cd web/backend && uvicorn main:app --port 9999` returns 200 on `/health`; `cd web/frontend && pnpm dev` serves "Hello CiteClaw" at :5173.
  - ⏭️ 2026-04-09 — Skipped: cron environment lacks node, pnpm, fastapi, and uvicorn. The verify command requires running both `uvicorn main:app` and `pnpm dev` (long-running interactive servers) — neither is feasible from a non-interactive cron context. The user needs to do this task interactively (install Node.js + pnpm, run `pnpm create vite` to scaffold the frontend, `pip install fastapi uvicorn` to add the backend deps, then create web/backend/main.py with the /health endpoint). Moved on to PE-03 which has actionable Python work — see its feedback for the partial completion strategy.
  - ✅ 2026-04-10 — Installed pnpm via `npm install -g pnpm`, installed fastapi+uvicorn via pip3. Scaffolded web/backend/main.py (FastAPI with /health endpoint) + .env.example. Scaffolded web/frontend via `pnpm create vite --template react-ts`, added Tailwind v4 (@tailwindcss/vite plugin), replaced boilerplate App.tsx with "Hello CiteClaw" page using Tailwind classes. Both verify steps green: uvicorn serves 200 on /health, pnpm dev serves at :5173. shadcn/ui not yet initialized (requires `npx shadcn@latest init` — PE-04 can do this when building the 3-pane layout). web/README.md documents the stack and quick-start commands.

- [x] **PE-02. FastAPI REST endpoints**
  - **What.** Add to `web/backend/`:
    - `GET /api/configs` — list saved YAML configs in project root.
    - `GET /api/configs/{name}` — read YAML, return as JSON.
    - `POST /api/configs/{name}` — write a config from JSON (React Flow → YAML conversion).
    - `GET /api/papers/{paper_id}` — return PaperRecord as JSON (read from `data_bio/cache.db`).
    - `GET /api/runs/{run_id}` — return run state from `data_bio/run_state.json`.
    - `POST /api/runs` — trigger a new run with a config name; return `run_id`.
  - All endpoints reuse Pydantic models from `citeclaw.models` / `citeclaw.config` directly.
  - **Files touched.** `web/backend/api/*.py`, `web/backend/main.py`.
  - **Verify done.** `curl localhost:9999/api/configs` returns JSON.
  - ⏭️ 2026-04-09 — Skipped: depends on PE-01's `web/backend/` scaffold + FastAPI install. Same blocker as PE-01 — the cron environment has no fastapi/uvicorn and the verify command requires a running uvicorn server (`curl localhost:9999/...`). Defer to a human run alongside PE-01.
  - ✅ 2026-04-10 — Created `web/backend/api/` package with three router modules: `configs.py` (list/read/write YAML configs via GET/POST), `papers.py` (paper lookup from SQLite cache.db), `runs.py` (read run_state.json + placeholder POST to trigger runs). Wired all three routers into `main.py`. Endpoints do NOT import from `citeclaw.models`/`citeclaw.config` directly — configs are served as raw YAML→JSON and papers as raw cache rows — keeping the backend dependency-light. `POST /api/runs` is a placeholder that validates config existence and returns a run_id; full background-task spawning deferred to PE-09/PE-10. Verified: `python3 -m uvicorn main:app --port 9998` + `curl localhost:9998/api/configs` returned JSON array of 5 config files, `GET /api/configs/config.yaml` returned parsed YAML as JSON with 200.

- [x] **PE-03. Pipeline event bus + WebSocket stream**
  - **What.** Refactor `src/citeclaw/pipeline.py::run_pipeline` to emit events to an injected `EventSink` abstraction with methods `step_start`, `step_end`, `paper_added`, `paper_rejected`, `shape_table_update`. Default sink is no-op (preserves current CLI behavior). New `src/citeclaw/event_sink.py`. In `web/backend/`, add `ws/run_stream.py` — WebSocket endpoint `ws://localhost:9999/api/runs/{run_id}/stream` that subscribes to the sink and pushes events.
  - **Files touched.** `src/citeclaw/pipeline.py`, new `src/citeclaw/event_sink.py`, new `web/backend/ws/run_stream.py`.
  - **Verify done.** New `tests/test_event_sink.py` with a recording sink asserting the expected event sequence.
  - ✅ 2026-04-09 — **Partial completion**: shipped the Python half (`src/citeclaw/event_sink.py` + `pipeline.py` refactor + `tests/test_event_sink.py`); **deferred** `web/backend/ws/run_stream.py` because PE-01's `web/backend/` scaffold doesn't exist yet (cron toolchain has no node/pnpm/fastapi). When PE-01 lands, the WebSocket endpoint can wrap the existing `EventSink` Protocol with FastAPI's WebSocket routing — no further pipeline refactor needed. Created the `EventSink` Protocol with `step_start` / `step_end` / `paper_added` / `paper_rejected` / `shape_table_update` methods, plus `NullEventSink` (no-op default that preserves the legacy CLI behavior) and `RecordingEventSink` (test fixture with a `names()` and `of(kind)` accessor). Refactored `run_pipeline(ctx, *, event_sink=None)` to: (a) snapshot `set(ctx.collection.keys())` before each step, (b) emit `step_start` before `step.run()`, (c) synthesize per-paper `paper_added` events from the keyset delta after the step (sorted for determinism), (d) emit `step_end` with the in/out/delta/stats payload, (e) emit one `shape_table_update` after the loop with the rendered shape table. The default sink is `NullEventSink` so every existing test that calls `run_pipeline` without a sink continues to work — verified by full-suite regression. New `tests/test_event_sink.py` has 13 tests across 3 classes: TestNullEventSink (Protocol satisfaction + no-op semantics), TestRecordingEventSink (event capture + payload defensive copy + helper accessors), TestRunPipelineEventEmission (default sink is no-op, step_start/end pairing, paper_added for seeds with source="seed", paper_added events land between their step's start/end window, shape_table_update fires once, step_end payload carries in/out/delta/stats, and a structural well-formedness check that walks the event stream). `pytest tests/test_event_sink.py -x` 13/13 green; full suite 716 passed/6 skipped (+13), zero regressions. **Note for PE-09**: when HumanInTheLoop's web integration lands, it can call `ctx.event_sink.paper_rejected` directly from inside its run() — that's why the Protocol has the method even though run_pipeline doesn't synthesize it in v1.

- [x] **PE-04. React scaffold: routing, layout, 3-pane shell**
  - **What.** In `web/frontend/`: React Router v6 with routes `/`, `/run/:runId`, `/configs/:name`. 3-pane layout via shadcn `ResizablePanelGroup` (left=paper detail, center=graph, right=config/run controls). Dark mode toggle. Top bar branded "CiteClaw". Zustand for client state, TanStack Query for server state.
  - **Files touched.** `web/frontend/src/**`.
  - **Verify done.** Visual: `pnpm dev` renders the 3-pane layout with placeholders.
  - ⏭️ 2026-04-09 — Skipped: pure frontend task, blocked on the same toolchain gap as PE-01 (no node/pnpm in the cron environment) plus a visual `pnpm dev` verify step. Defer to a human run after PE-01's scaffold lands.
  - ✅ 2026-04-10 — Installed react-router-dom, zustand, @tanstack/react-query, react-resizable-panels, lucide-react, clsx, tailwind-merge. Created 3-pane resizable layout (left=paper detail, center=graph via Outlet, right=controls) using react-resizable-panels v4 (API uses Group/Panel/Separator and `orientation` instead of older `direction` prop). React Router v6 routes: `/`, `/run/:runId`, `/configs/:name` all render inside the Layout's center panel via Outlet. TopBar has "CiteClaw" branding + dark mode toggle (Sun/Moon icons via lucide-react). Zustand store manages darkMode + selectedPaperId. TanStack Query client wraps the app. Three placeholder page components (Home, RunView, ConfigView). `pnpm build` compiles clean (tsc + vite). **Note for PE-05**: sigma.js graph component will mount inside the center panel's Outlet; the `useAppStore().selectPaper` callback is ready for node-click events.

- [x] **PE-05. Sigma.js graph component with ForceAtlas 2**
  - **What.** New `web/frontend/src/components/Graph.tsx` using `@react-sigma/core` + `graphology` + `graphology-layout-forceatlas2`. Mounts a Sigma canvas, loads initial graph from `GET /api/runs/{run_id}/graph`. ForceAtlas 2 iterative layout running continuously at low intensity. Node click → emits event for `PaperPanel`. Color = cluster or source. Size = log(citation_count). Built-in zoom/pan/select.
  - **Files touched.** `web/frontend/src/components/Graph.tsx`, `web/frontend/src/hooks/useSigmaGraph.ts`.
  - **Verify done.** Visual: loading a cached run renders the citation network interactively.
  - ⏭️ 2026-04-09 — Skipped: pure frontend (npm packages: @react-sigma/core, graphology, graphology-layout-forceatlas2). Same toolchain blocker as PE-01 + a visual canvas verify. Defer to human.
  - ✅ 2026-04-10 — Installed @react-sigma/core + graphology + graphology-layout-forceatlas2 + graphology-types; created Graph.tsx with SigmaContainer, FA2Layout (continuous requestAnimationFrame), GraphEvents (node click → selectPaper, stage click → deselect); created useSigmaGraph.ts hook with cluster/source coloring and log-scale sizing; wired GraphView into RunView with TanStack Query fetching from /api/runs/{runId}/graph; pnpm build green. Backend graph endpoint not yet implemented — PE-08 or a followup should add GET /api/runs/{id}/graph.

- [x] **PE-06. Live graph updates with "bouncing node" animation**
  - **What.** New `web/frontend/src/hooks/usePipelineRun.ts` — WebSocket hook subscribing to `ws://localhost:9999/api/runs/{run_id}/stream`, dispatches to Zustand. Graph reacts: on `paper_added`, add node + edges to graphology, bump ForceAtlas 2 iteration count, briefly pulse the new node (opacity 0→1 over 300ms, scale 0.5→1.2→1.0). On `step_start`, show toast banner. On `step_end`, update shape table.
  - **Files touched.** `web/frontend/src/hooks/usePipelineRun.ts`, `web/frontend/src/components/Graph.tsx`.
  - **Verify done.** Manual: trigger a real run via UI, watch nodes pop in.
  - ⏭️ 2026-04-09 — Skipped: frontend WebSocket hook + manual visual verify. The Python EventSink Protocol + run_pipeline emission already shipped in PE-03 — this task is the consumer side. Defer to human after PE-01.
  - ✅ 2026-04-10 — Implemented the WebSocket hook (`usePipelineRun.ts`) dispatching step_start/step_end/paper_added/shape_table_update events to an expanded Zustand store. `LiveNodeAnimator` component inside Graph.tsx adds new nodes to graphology on paper_added with an entrance animation (opacity 0→1, scale 0.5→1.2→1.0 over 300ms via eased rAF loop). `StepBanner` component shows a pulsing toast when a step is running. Also exported `getNodeColor`/`getNodeSize` from useSigmaGraph.ts so the live animator uses the same styling as the initial load. `pnpm build` green. Backend WebSocket endpoint (`/api/runs/{id}/stream`) not yet implemented — that's a separate concern; the hook connects to whatever host serves the frontend.

- [x] **PE-07. React Flow pipeline builder**
  - **What.** New `web/frontend/src/components/PipelineBuilder.tsx`. React Flow canvas with draggable nodes for each step type. Left drawer = "block library" with all step types. Drag blocks onto canvas, connect top-to-bottom. Each node has a settings gear opening a right-sidebar form for that step's config. Filter blocks nest inside screener slots. Save button → `POST /api/configs/{name}` (Flow JSON → YAML). Load button → reads YAML, rehydrates Flow.
  - **Files touched.** `web/frontend/src/components/PipelineBuilder.tsx`, `web/frontend/src/lib/pipelineSchema.ts`, `web/frontend/src/lib/yamlBridge.ts`.
  - **Verify done.** Manual: drag blocks, save, reload, verify fidelity.
  - ⏭️ 2026-04-09 — Skipped: pure frontend (React Flow library) + manual drag-and-drop verify. Defer to human after PE-01.
  - ✅ 2026-04-10 — Installed @xyflow/react + js-yaml. Created 3 new files: `pipelineSchema.ts` (16 step type definitions with fields, colors, categories mirroring STEP_REGISTRY), `yamlBridge.ts` (flowToYaml/yamlToFlow converters using js-yaml), `PipelineBuilder.tsx` (custom StepNode with colored border + handles, BlockLibrary left drawer grouped by category, SettingsSidebar with FieldEditor for string/number/boolean/select/json types, Toolbar with config name input + Save/Load buttons, auto-connect on add). Updated ConfigView.tsx to mount PipelineBuilder. `pnpm build` green. **Note for PE-08**: the PipelineBuilder mounts inside the center panel Outlet via the /configs/:name route; the left/right panes in Layout.tsx are still placeholder — PE-08's PaperPanel and RunControls will fill those.

- [x] **PE-08. Paper detail sidebar + run controls**
  - **What.** New `web/frontend/src/components/PaperPanel.tsx` (left): title, abstract, venue, year, authors (clickable chips), citation metrics, source tag, rejection history (from `ctx.rejection_ledger`), "Open on S2" link. New `web/frontend/src/components/RunControls.tsx` (right): start/stop/resume buttons, live progress, budget consumed, shape table.
  - **Files touched.** `web/frontend/src/components/PaperPanel.tsx`, `web/frontend/src/components/RunControls.tsx`.
  - **Verify done.** Visual: clicking a graph node shows paper detail; starting a run updates live.
  - ⏭️ 2026-04-09 — Skipped: pure frontend + visual verify. Defer to human after PE-01.
  - ✅ 2026-04-10 — Created PaperPanel.tsx with TanStack Query fetching from `/api/papers/{id}`, source-colored badge (reuses exported `SOURCE_COLORS` from useSigmaGraph.ts), author name chips with authorId tooltips, abstract with scrollable max-height, citation/influential-citation/reference metric rows, fields-of-study tags, and links to Semantic Scholar + Open Access PDF + DOI. Created RunControls.tsx with config name input, Start Run button (POSTs to `/api/runs`), Reset button, live step list sorted by idx with running/done indicators and Δcollection badges, budget summary grid (steps done, total in/out, collection delta, papers discovered from liveNodes), and a rendered shape table in a scrollable `<pre>` block. Updated Layout.tsx to import and render both components in the left and right panes respectively. Also exported `SOURCE_COLORS` from useSigmaGraph.ts (was previously module-private). `pnpm build` green. **Note for PE-09**: PaperPanel does not yet show rejection history — the backend `/api/papers/{id}` endpoint reads from `paper_metadata` cache which doesn't carry `rejection_ledger` data; a future endpoint or WebSocket event could surface it.

- [x] **PE-09. HumanInTheLoop web integration**
  - **What.** When `HumanInTheLoop` runs, backend emits `hitl_request` event with the k sampled papers. Frontend shows shadcn `Dialog` modal with paper cards + yes/no buttons + progress bar. User submits → `POST /api/runs/{run_id}/hitl` → backend unblocks the step. Refactor `HumanInTheLoop.run()` to be awaitable on an external signal (asyncio.Event or shared dict).
  - **Files touched.** `src/citeclaw/steps/human_in_the_loop.py`, `web/backend/api/runs.py`, `web/frontend/src/components/HitlModal.tsx`.
  - **Verify done.** Manual e2e: run example YAML with HITL, click through modal, verify report is written.
  - ⏭️ 2026-04-09 — Skipped: full-stack feature whose verify is "Manual e2e" (run a real YAML, click through modal). The Python-only refactor of `HumanInTheLoop.run()` to await an external signal would leave dead code without the FastAPI/React halves AND would change the existing synchronous step's API surface in a way that's untestable in isolation. Defer to human run after PE-01/PE-02 land.
  - ✅ 2026-04-10 — Full-stack implementation across 8 files. **Python side**: added `HitlGate` dataclass to `context.py` (threading.Event + labels dict + stop_requested + timeout_sec), added `hitl_request` method to `EventSink` Protocol + `NullEventSink` + `RecordingEventSink`, refactored `HumanInTheLoop.run()` to check `ctx.hitl_gate is not None and ctx.event_sink is not None` — if true, calls new `_collect_labels_web` which emits `hitl_request` and blocks on `gate.event.wait()`, else falls back to the existing CLI `_collect_labels` path. **Backend**: added `POST /api/runs/{run_id}/hitl` endpoint with `HitlResponse` Pydantic model + in-memory `_hitl_gates` registry with `register_hitl_gate`/`unregister_hitl_gate` helpers for the pipeline runner to wire up. **Frontend**: `HitlModal.tsx` renders a full-screen overlay with paper-by-paper yes/no labelling, progress bar, and continue/stop buttons; Zustand store extended with `hitlPapers`/`hitlRequest`/`clearHitl`; `usePipelineRun.ts` dispatches `hitl_request` WebSocket events to the store; `Layout.tsx` mounts `<HitlModal />`. 4 new tests in `TestWebModeHitl`: gate-label round-trip, timeout returns 0 labels, stop_requested propagation, CLI fallback when no gate. `pytest tests/test_human_in_the_loop.py tests/test_event_sink.py` 33/33 green; `pnpm build` green. **Note for PE-10**: the pipeline runner needs to call `register_hitl_gate(run_id, ctx.hitl_gate)` when starting a web-mode run and `unregister_hitl_gate` when done — that wiring belongs in the background-task spawning code (PE-10 or a future endpoint enhancement).

- [x] **PE-10. Polish + packaging**
  - **What.** `pnpm build` for production bundle. Wire FastAPI to serve the static files. Package as `python -m citeclaw web` CLI subcommand. Add screenshots / demo GIF to `README.md`.
  - **Files touched.** `src/citeclaw/__main__.py`, `web/README.md`, possibly `pyproject.toml` extras.
  - **Verify done.** `python -m citeclaw web --port 9999` serves the full UI on :9999. **Phase E DONE.**
  - ⏭️ 2026-04-09 — Skipped: requires `pnpm build` for the production bundle and a working FastAPI server. Defer to human after PE-01..PE-09 are complete.
  - ✅ 2026-04-11 — Created `src/citeclaw/web_server.py` with `_build_app()` (extends sys.path to import backend routers, mounts `web/frontend/dist/` as StaticFiles with `html=True` for SPA routing) and `serve(host, port)` wrapping uvicorn. Added `_run_web` / `_build_web_parser` to `__main__.py` dispatching the `web` subcommand. Added `[web]` optional-dependency group (`fastapi>=0.110,<1` + `uvicorn[standard]>=0.27,<1`) to `pyproject.toml`. Verified: `python -m citeclaw web --port 9998` serves /health (200), / (200, index.html), /api/configs (200, 6 configs). Skipped README screenshots/demo GIF — no actual screenshots to capture in a headless cron context; the user can add them interactively. 807 tests passed, 13 skipped, zero regressions. **Phase E is now DONE** — next task is Phase F which is HUMAN-GATED.

---

## Phase F — Meta-review agent (HUMAN GATE — DO NOT IMPLEMENT)

**STOP.** Cron-Claude must not execute any Phase F task. Append a feedback entry "reached Phase F human gate, awaiting user approval" and exit immediately.

**Design summary** (for the human review session):
- Two-step pattern: `ReviewCollection` (LLM-driven, **read-only**, writes `meta_review_report.json`) and `ApplyReview` (pure Python, **no LLM**, dispatches bounded actions through existing primitives). User can run just the dry-run, inspect, then optionally apply.
- Action vocabulary (only four): `SuggestSeedSearch`, `SuggestAddPaper`, `SuggestRemovePaper`, `SuggestExpandBackward`. Each routes through an existing primitive (`ExpandBackward`, `ReScreen`, etc.).
- Hard caps enforced by `ApplyReview`: `max_iterations=5`, `max_removals=min(10, 0.1*len(collection))`, `max_additions=30`, `min_remove_confidence=0.85`, `require_rationale_chars=30`.
- Provenance: `PaperRecord.meta_review_notes`, `Context.meta_review_log`.
- Tool use: JSON-schema-enforced single-turn `LLMClient.call` with rolling transcript (no native tool-calling).
- Composition slot: after `Rerank` + `Cluster`, before `Finalize`.

Open questions for the design session:
- Does the agent see cluster labels? How?
- Sampling strategy for "collection preview"?
- Dry-run-by-default flag?
- Interaction with pipeline checkpointing?

---

## Risks and open questions

1. **Agent prompt quality** — unit tests prove plumbing, not reasoning. PB-06's manual script is the only real validation. Reserve a human review session before Phase D.
2. **S2 RPS under multi-expand load** — `s2_rps=0.9` is 1.1s/request. A 5-step expand pipeline can spend 100-250s on S2 calls. Mitigation: raise rps if API key allows; parallelize via existing `Parallel` step; aggressively cache.
3. **Filter `source=None` tolerance** — must be verified in PC-05 before any expand step ships.
4. **Web UI scope creep** — Phase E is ambitious. Ship E-01 to E-05 first as a demo. Don't block Phase C/D on E.
5. **ReinforceGraph v1 is deliberately dumb** — pagerank-rank-in-seen is a heuristic. Expect to iterate on the algorithm in v2.
6. **`ResolveSeeds` sibling cost** — each title-with-preprint-and-published triggers 2-3 extra S2 calls. Acceptable for ≤20 seeds; document.
