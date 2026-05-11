"""Post-pipeline meta-review: extract structured info from each paper, then synthesize.

A standalone CLI step run *after* a CiteClaw pipeline has finished.
Given a run directory (``literature_collection.json`` + ``cache.db``)
and a user instruction, the module:

  1. Walks every accepted paper.
  2. Fetches its parsed PDF text via :class:`PdfClawBridge` (cache hit
     when ``fetch-pdfs`` ran first, live HTTP / pdfclaw fallback when
     not).
  3. Feeds the text to an *extractor* LLM with the instruction,
     producing one short markdown block per paper.  These per-paper
     extractions run concurrently through a :class:`ThreadPoolExecutor`.
  4. Concatenates the per-paper extractions with numbered metadata
     headers and feeds the bundle to a *meta* LLM with a separate
     synthesis prompt.  The meta LLM produces a unified report with
     numeric in-line citations (``[1]``, ``[2]``, …) and a References
     section at the end.
  5. Writes the final report to ``<run_dir>/meta_review.md`` and a
     ``meta_review_extractions.json`` sidecar so re-running the meta
     step (e.g. with a different model or instruction tweak) can skip
     re-extraction.

The extractor and meta LLMs are independently configurable — the same
alias-from-registry pattern as the rest of CiteClaw, so any model
wired into the YAML's ``models:`` registry is selectable here.  This
lets the user pick a fast model for the per-paper pass and a stronger,
more deliberative model for the meta synthesis, or run both with the
same alias when budget is tight.

CLI::

    python -m citeclaw meta-review runs/data_rna \\
        -c configs/config_rna.yaml \\
        --instruction "Extract model architecture, n parameters, training objective, datasets, hyperparameters." \\
        --extractor-model mimo-v2.5 \\
        --meta-model mimo-v2.5 \\
        --meta-reasoning high \\
        --output meta_review.md
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citeclaw.budget import BudgetTracker
    from citeclaw.cache import Cache
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.config import Settings

log = logging.getLogger("citeclaw.meta_review")

# Per-paper extraction text budget. Picked so 60 papers × this budget
# still fit in a 128K-context meta model with reasoning headroom.
_DEFAULT_PER_PAPER_MAX_CHARS = 2000

# PDF body text fed to the extractor LLM. Generous because the
# extractor truncates middle-out internally via extract_from_text.
_DEFAULT_EXTRACTOR_INPUT_CHARS = 60_000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PaperExtraction:
    """One paper's worth of LLM-extracted information."""

    paper_id: str
    title: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    extraction: str = ""
    """Markdown block produced by the extractor LLM."""
    reasoning: str = ""
    """Optional reasoning trace from the extractor LLM (DEBUG diagnostics)."""
    error: str = ""
    """Non-empty when fetch_text or LLM call failed for this paper."""


@dataclass
class MetaReviewResult:
    """End-to-end result returned by :func:`run_meta_review`."""

    instruction: str
    extractor_model: str
    meta_model: str
    extractions: list[PaperExtraction]
    report_markdown: str
    n_papers_total: int
    n_papers_extracted: int
    n_papers_skipped: int


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EXTRACTOR_SYSTEM = (
    "You are a careful academic literature analyst.  The user will give "
    "you the full text of one research paper and ask you to extract "
    "specific information from it.\n\n"
    "Output rules:\n"
    "  - Return ONLY a concise markdown block (no preamble, no JSON, no "
    "    code fences around the whole block).\n"
    "  - Use short markdown sections (### Headings) for each piece of "
    "    information the user asked for.\n"
    "  - Quote exact numbers and names from the paper.  Do not "
    "    paraphrase or speculate.  If a piece of information is genuinely "
    "    absent, write `Not stated`.\n"
    "  - Keep the entire response under {max_chars} characters."
)

_EXTRACTOR_USER_TEMPLATE = (
    "## What to extract\n"
    "{instruction}\n\n"
    "## Paper\n"
    "Title: {title}\n"
    "{paper_text}\n"
)

_META_SYSTEM = (
    "You are a senior researcher writing a comparative meta-review of "
    "a set of papers.  The user has already extracted topic-specific "
    "information from each paper individually; your job is to weave "
    "those per-paper extractions into a single coherent report.\n\n"
    "Rules:\n"
    "  1. Use numeric in-line citations in the form ``[N]`` where N is "
    "     the paper number given in the per-paper section headings.  "
    "     Cite EVERY factual claim you make.\n"
    "  2. Compare and contrast — group papers by approach, identify "
    "     trends, call out outliers and disagreements.  Do NOT just "
    "     list one-paragraph-per-paper.\n"
    "  3. Stay strictly within what the per-paper extractions actually "
    "     say.  Do not invent details that aren't in the input.  If the "
    "     extractions disagree or are silent on a point, say so.\n"
    "  4. End the report with a ``## References`` section listing each "
    "     cited paper in the format ``[N] <title>. <venue> (<year>). <paper_id>``.\n"
    "  5. Use markdown headings (``#``, ``##``, ``###``) to organise the "
    "     report.  Pick the section structure that best fits the "
    "     instruction — do not impose a fixed template."
)

_META_USER_TEMPLATE = (
    "## Original question\n"
    "{instruction}\n\n"
    "## Per-paper extractions\n"
    "{corpus}\n\n"
    "## Your task\n"
    "Write a comparative meta-review markdown report that synthesises the "
    "per-paper extractions above into a unified analysis answering the "
    "original question.  Use numeric in-line citations matching the paper "
    "numbers, and include a ``## References`` section listing every cited paper."
)


# ---------------------------------------------------------------------------
# Per-paper extraction
# ---------------------------------------------------------------------------


def _read_parsed_text_from_disk(run_dir: Path, paper_id: str) -> str | None:
    """Read pre-parsed body text from ``<run_dir>/parsed/<paper_id>.json``.

    The ``fetch-pdfs`` CLI writes one JSON per paper into ``parsed/`` —
    its schema includes a ``body_text`` field carrying the extracted
    PDF body.  When the run directory has this artefact we prefer it
    over hitting :class:`PdfClawBridge`: the cache table may carry a
    stale ``error="no_pdf"`` row from the original pipeline run that
    fired *before* ``fetch-pdfs`` populated the disk, which causes the
    bridge to return ``None`` even though the text is in fact sitting
    on disk.  Returns ``None`` when the file is missing or empty.
    """
    path = run_dir / "parsed" / f"{paper_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    text = (data.get("body_text") or "").strip()
    return text or None


def _extract_one_paper(
    paper: dict[str, Any],
    bridge,
    extractor_llm: "LLMClient",
    instruction: str,
    *,
    per_paper_max_chars: int,
    pdf_input_max_chars: int,
    run_dir: Path,
) -> PaperExtraction:
    """Worker body: fetch text, call extractor LLM, return :class:`PaperExtraction`.

    Pure function modulo network calls.  Catches every failure mode so
    one bad paper never poisons the corpus — broken papers come back
    with ``error`` populated and ``extraction=""``.
    """
    pid = paper.get("paper_id") or ""
    title = (paper.get("title") or "").strip()
    year = paper.get("year")
    venue = paper.get("venue") or ""
    authors_raw = paper.get("authors") or []
    authors: list[str] = []
    for a in authors_raw[:3]:
        if isinstance(a, dict):
            authors.append(str(a.get("name") or ""))
        else:
            authors.append(str(a))
    authors = [a for a in authors if a]

    if not pid:
        return PaperExtraction(
            paper_id="", title=title, year=year, authors=authors,
            venue=venue, error="missing paper_id",
        )

    # ------- fetch ----------------------------------------------------
    # Disk-first: when ``fetch-pdfs`` has been run, the parsed/<id>.json
    # file is the authoritative source and bypasses any stale "no_pdf"
    # cache row that the original pipeline may have written before the
    # PDF was downloadable.  The bridge fallback covers papers added
    # after fetch-pdfs (or runs where fetch-pdfs was never run).
    text = _read_parsed_text_from_disk(run_dir, pid)
    if text is None:
        try:
            from citeclaw.models import PaperRecord

            pdf_url = paper.get("pdf_url") or ""
            external_ids = paper.get("external_ids") or {}
            rec = PaperRecord(
                paper_id=pid,
                title=title,
                year=year,
                external_ids=external_ids,
                pdf_url=pdf_url,
            )
            text = bridge.fetch_text(rec)
        except Exception as exc:  # noqa: BLE001
            log.debug("meta-review: fetch failed for %s: %s", pid[:20], exc)
            return PaperExtraction(
                paper_id=pid, title=title, year=year, authors=authors,
                venue=venue, error=f"fetch failed: {exc}",
            )

    if not text:
        return PaperExtraction(
            paper_id=pid, title=title, year=year, authors=authors,
            venue=venue, error="no PDF text available",
        )

    # ------- extract --------------------------------------------------
    from citeclaw.extraction import _truncate_middle_out  # noqa: PLC0415

    truncated = _truncate_middle_out(text, pdf_input_max_chars)
    sys_prompt = _EXTRACTOR_SYSTEM.format(max_chars=per_paper_max_chars)
    user_prompt = _EXTRACTOR_USER_TEMPLATE.format(
        instruction=instruction,
        title=title or "(untitled)",
        paper_text=truncated,
    )

    try:
        resp = extractor_llm.call(
            sys_prompt,
            user_prompt,
            category="meta_review_extraction",
            response_schema=None,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("meta-review: extractor LLM failed for %s: %s", pid[:20], exc)
        return PaperExtraction(
            paper_id=pid, title=title, year=year, authors=authors,
            venue=venue, error=f"LLM call failed: {exc}",
        )

    extraction_text = (resp.text or "").strip()
    if not extraction_text:
        return PaperExtraction(
            paper_id=pid, title=title, year=year, authors=authors,
            venue=venue, error="empty extractor output",
        )

    # Hard cap: trim the extraction to the per-paper budget so the meta
    # prompt stays predictably sized.  The LLM is *asked* to stay under
    # this in the system prompt but we enforce it here too.
    if len(extraction_text) > per_paper_max_chars:
        extraction_text = (
            extraction_text[:per_paper_max_chars]
            + "\n\n[... extraction truncated to per-paper budget ...]"
        )

    return PaperExtraction(
        paper_id=pid,
        title=title,
        year=year,
        authors=authors,
        venue=venue,
        extraction=extraction_text,
        reasoning=getattr(resp, "reasoning_content", "") or "",
    )


# ---------------------------------------------------------------------------
# Corpus assembly + meta synthesis
# ---------------------------------------------------------------------------


def _build_corpus_markdown(extractions: list[PaperExtraction]) -> str:
    """Format successful extractions into the numbered markdown corpus."""
    parts: list[str] = []
    for idx, ex in enumerate((e for e in extractions if e.extraction), start=1):
        author_str = ", ".join(ex.authors[:3]) if ex.authors else ""
        if author_str and len(ex.authors) > 3:
            author_str += " et al."
        meta_bits: list[str] = []
        if author_str:
            meta_bits.append(author_str)
        if ex.year:
            meta_bits.append(str(ex.year))
        if ex.venue:
            meta_bits.append(ex.venue)
        meta_bits.append(f"paper_id: {ex.paper_id}")
        header = f"### [{idx}] {ex.title or '(untitled)'}\n_{' · '.join(meta_bits)}_"
        parts.append(header + "\n\n" + ex.extraction)
    return "\n\n---\n\n".join(parts)


def _build_reference_list(extractions: list[PaperExtraction]) -> list[dict]:
    """Mapping the meta LLM and the JSON sidecar need: number → paper metadata."""
    out: list[dict[str, Any]] = []
    idx = 0
    for ex in extractions:
        if not ex.extraction:
            continue
        idx += 1
        out.append({
            "n": idx,
            "paper_id": ex.paper_id,
            "title": ex.title,
            "year": ex.year,
            "venue": ex.venue,
            "authors": ex.authors,
        })
    return out


def _run_meta_synthesis(
    extractions: list[PaperExtraction],
    meta_llm: "LLMClient",
    instruction: str,
) -> str:
    """Single LLM call: corpus + instruction → markdown meta-review."""
    corpus = _build_corpus_markdown(extractions)
    if not corpus.strip():
        return (
            "# Meta-Review\n\n"
            "_No per-paper extractions were available — every paper failed to "
            "fetch or extract.  Check the run log for the per-paper error reasons._\n"
        )

    user_prompt = _META_USER_TEMPLATE.format(
        instruction=instruction,
        corpus=corpus,
    )

    try:
        resp = meta_llm.call(
            _META_SYSTEM,
            user_prompt,
            category="meta_review_synthesis",
            response_schema=None,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("meta-review: meta synthesis LLM call failed: %s", exc)
        return (
            "# Meta-Review\n\n"
            f"_The meta synthesis call failed: {exc}.  Per-paper extractions "
            "are preserved in the JSON sidecar — you can retry with a different "
            "meta model._\n"
        )

    return (resp.text or "").strip() or (
        "# Meta-Review\n\n_The meta LLM returned an empty response._\n"
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_meta_review(
    run_dir: Path,
    config: "Settings",
    *,
    instruction: str,
    extractor_model: str,
    meta_model: str,
    extractor_reasoning: str | None = "medium",
    meta_reasoning: str | None = "high",
    max_workers: int = 4,
    max_papers: int | None = None,
    per_paper_max_chars: int = _DEFAULT_PER_PAPER_MAX_CHARS,
    pdf_input_max_chars: int = _DEFAULT_EXTRACTOR_INPUT_CHARS,
    parser: str = "pymupdf",
    parser_kwargs: dict | None = None,
    headless: bool = True,
    output_path: Path | None = None,
    from_extractions: Path | None = None,
) -> MetaReviewResult:
    """Run the full meta-review pipeline on a finished CiteClaw run directory.

    See the module docstring for an overview.  This function is the
    library entry point; the CLI subcommand (``_run_meta_review`` in
    :mod:`citeclaw.__main__`) is a thin wrapper around it.
    """
    from citeclaw.budget import BudgetTracker
    from citeclaw.cache import Cache
    from citeclaw.clients.llm.factory import build_llm_client
    from citeclaw.clients.pdfclaw_bridge import PdfClawBridge

    # ---- short-circuit: re-meta from a prior sidecar --------------------
    # When ``from_extractions`` is set, we skip the entire per-paper pass
    # and re-feed an existing sidecar into the meta LLM.  Useful for
    # iterating on the synthesis prompt or recovering from a meta-side
    # failure (rate limit, balance hiccup) without spending another
    # 60+ extractor calls.  ``max_papers`` and ``per_paper_max_chars``
    # apply here too: they let the caller shrink the corpus to fit a
    # tighter-context meta model than was originally used.
    if from_extractions is not None:
        sidecar = json.loads(from_extractions.read_text(encoding="utf-8"))
        extractions = [
            PaperExtraction(
                paper_id=e.get("paper_id", ""),
                title=e.get("title", ""),
                year=e.get("year"),
                authors=e.get("authors") or [],
                venue=e.get("venue", "") or "",
                extraction=(e.get("extraction") or "")[:per_paper_max_chars],
                error=e.get("error", "") or "",
            )
            for e in sidecar.get("extractions", [])
        ]
        if max_papers is not None:
            # Keep the successful extractions first so the cap doesn't
            # silently waste room on skipped papers (which contribute
            # nothing to the corpus anyway).
            extractions.sort(key=lambda e: 0 if e.extraction else 1)
            extractions = extractions[:max_papers]
        # Use the instruction from the sidecar when the CLI didn't
        # supply one — keeps the meta-prompt consistent with the
        # per-paper pass that produced these extractions.
        instr = instruction or sidecar.get("instruction") or ""
        budget = BudgetTracker()
        cache_path = run_dir / "cache.db"
        cache = Cache(cache_path) if cache_path.exists() else None
        meta_llm = build_llm_client(
            config, budget,
            model=meta_model,
            reasoning_effort=meta_reasoning,
            cache=cache,
        )
        log.info(
            "meta-review: loaded %d extractions from %s — meta synthesis only",
            len(extractions), from_extractions,
        )
        report_md = _run_meta_synthesis(extractions, meta_llm, instr)
        output_path = output_path or (run_dir / "meta_review.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_md, encoding="utf-8")
        log.info(
            "meta-review: wrote %s (%d chars), %s",
            output_path, len(report_md), budget.summary(),
        )
        n_extracted = sum(1 for e in extractions if e.extraction)
        return MetaReviewResult(
            instruction=instr,
            extractor_model=sidecar.get("extractor_model", "(from sidecar)"),
            meta_model=meta_model,
            extractions=extractions,
            report_markdown=report_md,
            n_papers_total=len(extractions),
            n_papers_extracted=n_extracted,
            n_papers_skipped=len(extractions) - n_extracted,
        )

    # ---- load collection --------------------------------------------------
    collection_path = run_dir / "literature_collection.json"
    if not collection_path.exists():
        raise FileNotFoundError(
            f"meta-review needs a finished run; "
            f"{collection_path} does not exist."
        )
    data = json.loads(collection_path.read_text(encoding="utf-8"))
    papers: list[dict[str, Any]] = (
        data["papers"] if isinstance(data, dict) else data
    )
    if max_papers is not None:
        papers = papers[:max_papers]
    log.info(
        "meta-review: %d papers selected from %s",
        len(papers), collection_path,
    )

    # ---- shared dependencies --------------------------------------------
    cache_path = run_dir / "cache.db"
    cache = Cache(cache_path) if cache_path.exists() else None
    if cache is None:
        log.info(
            "meta-review: no cache.db found at %s — every PDF fetch will go live",
            cache_path,
        )

    budget = BudgetTracker()
    extractor_llm = build_llm_client(
        config, budget,
        model=extractor_model,
        reasoning_effort=extractor_reasoning,
        cache=cache,
    )
    meta_llm = build_llm_client(
        config, budget,
        model=meta_model,
        reasoning_effort=meta_reasoning,
        cache=cache,
    )

    bridge = PdfClawBridge(
        cache,
        headless=headless,
        parser=parser,
        parser_kwargs=parser_kwargs or {},
    )

    # ---- per-paper extraction (concurrent) ------------------------------
    extractions: list[PaperExtraction] = []
    try:
        with ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="meta-review",
        ) as pool:
            futures = {
                pool.submit(
                    _extract_one_paper, p, bridge, extractor_llm, instruction,
                    per_paper_max_chars=per_paper_max_chars,
                    pdf_input_max_chars=pdf_input_max_chars,
                    run_dir=run_dir,
                ): p
                for p in papers
            }
            for fut in as_completed(futures):
                paper = futures[fut]
                try:
                    ex = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "meta-review: worker raised for %s: %s",
                        (paper.get("paper_id") or "?")[:20], exc,
                    )
                    ex = PaperExtraction(
                        paper_id=paper.get("paper_id") or "",
                        title=paper.get("title") or "",
                        year=paper.get("year"),
                        error=f"worker exception: {exc}",
                    )
                extractions.append(ex)
                # Log a short per-paper progress line — useful when
                # tailing a long-running corpus.
                tag = "✓" if ex.extraction else "✗"
                log.info(
                    "meta-review: [%s] %s %s",
                    tag, ex.paper_id[:12], (ex.title or "")[:60],
                )
    finally:
        bridge.close()

    n_extracted = sum(1 for e in extractions if e.extraction)
    n_skipped = len(extractions) - n_extracted
    log.info(
        "meta-review: per-paper pass done — %d extracted, %d skipped",
        n_extracted, n_skipped,
    )

    # ---- meta synthesis --------------------------------------------------
    log.info("meta-review: calling meta LLM (%s) for synthesis", meta_model)
    report_md = _run_meta_synthesis(extractions, meta_llm, instruction)

    # ---- write outputs --------------------------------------------------
    output_path = output_path or (run_dir / "meta_review.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8")
    log.info("meta-review: wrote %s (%d chars)", output_path, len(report_md))

    sidecar_path = output_path.with_name(
        output_path.stem + "_extractions.json"
    )
    sidecar = {
        "instruction": instruction,
        "extractor_model": extractor_model,
        "meta_model": meta_model,
        "n_papers_total": len(extractions),
        "n_papers_extracted": n_extracted,
        "n_papers_skipped": n_skipped,
        "references": _build_reference_list(extractions),
        "extractions": [
            # Drop the full reasoning trace from the sidecar by default
            # — they're useful for DEBUG but bloat the JSON and contain
            # essentially the same content as the extraction text.
            {**asdict(ex), "reasoning": ""}
            for ex in extractions
        ],
    }
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("meta-review: wrote %s", sidecar_path)

    # ---- budget summary --------------------------------------------------
    log.info("meta-review: %s", budget.summary())

    return MetaReviewResult(
        instruction=instruction,
        extractor_model=extractor_model,
        meta_model=meta_model,
        extractions=extractions,
        report_markdown=report_md,
        n_papers_total=len(extractions),
        n_papers_extracted=n_extracted,
        n_papers_skipped=n_skipped,
    )
