"""CLI entry point — ``python -m citeclaw [subcommand] ...``.

Subcommands, dispatched in :func:`main` by the first positional arg:

* (no arg or anything not below) → :func:`_run_snowball` — the default
  pipeline run that reads ``-c config.yaml`` + flags, validates the
  config, builds a Context, runs the pipeline, and finalises.
* ``annotate <graph>`` → :func:`_run_annotate` — the LLM-driven
  graph-node-labelling subcommand (see :mod:`citeclaw.annotate`).
* ``rebuild-graph <data_dir>`` → :func:`_run_rebuild_graph` —
  re-emit citation_network / collaboration_network GraphML from an
  existing run's literature_collection.json + cache.db (no S2 calls).
* ``fetch-pdfs <data_dir>`` → :func:`_run_fetch_pdfs` — the bulk PDF
  download CLI (see :mod:`citeclaw.fetch_pdfs`).
* ``mainpath <graph>`` → :func:`_run_mainpath` — extract the main path
  subnetwork from a citation GraphML (see :mod:`citeclaw.mainpath`).
* ``extract-info <pdf-or-text>`` → :func:`_run_extract_info` —
  generic LLM extraction: paper text + free-form instruction +
  optional JSON schema → structured JSON
  (see :mod:`citeclaw.extraction`).
* ``meta-review <run_dir>`` → :func:`_run_meta_review` — post-pipeline
  corpus analysis: per-paper LLM extraction with a user-supplied
  instruction, then a meta LLM that synthesises every extraction into
  a single markdown report with numeric in-line citations
  (see :mod:`citeclaw.meta_review`).
* ``web`` → :func:`_run_web` — launch the FastAPI + React web UI
  (see :mod:`citeclaw.web_server`).

API keys are intentionally never read from YAML — :func:`_validate_config`
walks the configured pipeline + filter blocks, computes the set of
required env vars via :func:`citeclaw.preflight.find_missing_api_keys`,
and exits with a clear error before any LLM/S2 spend if any are unset.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from citeclaw.config import SeedPaper, load_settings
from citeclaw.logging_config import setup_logging
from citeclaw.models import BudgetExhaustedError, CiteClawError, S2OutageError
from citeclaw.pipeline import build_context, finalize_partial, run_pipeline
from citeclaw.preflight import find_missing_api_keys
from citeclaw.steps.checkpoint import load_checkpoint
from citeclaw.steps.finalize import write_graphs

log = logging.getLogger("citeclaw")


def _build_run_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw")
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument("--topic", type=str, default=None)
    p.add_argument("--seed", type=str, nargs="+", default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--continue-from", type=Path, default=None, dest="continue_from")
    p.add_argument("-v", "--verbose", action="store_true")
    # Deprecated, accepted but ignored
    p.add_argument("--max-depth", type=int, default=None, help="(deprecated)")
    p.add_argument("--citation-beta", type=float, default=None, help="(deprecated)")
    return p


def _build_annotate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw annotate")
    p.add_argument("graph", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("-i", "--instruction", type=str, default=None)
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    return p


def _build_rebuild_graph_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw rebuild-graph")
    p.add_argument("data_dir", type=Path, help="Data directory from a previous run")
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite the original citation_network.graphml / "
             "collaboration_network.graphml instead of writing .regen variants.",
    )
    return p


def _build_mainpath_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw mainpath")
    p.add_argument(
        "graph", type=Path,
        help="Input GraphML — a CiteClaw citation network "
             "(e.g. <data_dir>/citation_network.graphml).",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output GraphML path (default: "
             "<stem>_mainpath_<search>_<weight>.graphml next to input). "
             "A sibling .json summary is always written too.",
    )
    p.add_argument(
        "-w", "--weight", default="spc",
        choices=["spc", "splc", "spnp"],
        help="Traversal weight (default: spc). "
             "spc follows Kirchhoff's conservation; splc treats every "
             "paper as a knowledge source; spnp treats every paper as "
             "both source and destination.",
    )
    p.add_argument(
        "-s", "--search", default="key-route",
        choices=["local-forward", "local-backward", "global",
                 "key-route", "multi-local"],
        help="Main-path extraction variant (default: key-route). "
             "key-route guarantees the highest-weighted arc is on the "
             "path; local-forward / local-backward are the classical "
             "priority-first search from sources / sinks; global is "
             "the critical (max-sum-weight) path; multi-local is "
             "local-forward with per-vertex tolerance relaxation.",
    )
    p.add_argument(
        "--cycle", default="shrink",
        choices=["shrink", "preprint"],
        help="How to handle strongly connected components "
             "(default: shrink). shrink collapses each cycle into its "
             "oldest representative paper (Liu, Lu & Ho 2019); "
             "preprint is Batagelj's preprint transform which "
             "preserves SCC members as individual vertices.",
    )
    p.add_argument(
        "--key-routes", type=int, default=1, dest="key_routes",
        help="Number of top-weighted arcs to seed as key routes "
             "when --search=key-route (default: 1).",
    )
    p.add_argument(
        "--tolerance", type=float, default=0.2,
        help="Per-vertex tolerance when --search=multi-local: arcs "
             "with weight >= (1-tolerance) * per_vertex_max are "
             "included (default: 0.2, per Liu & Lu 2012's example).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _build_fetch_pdfs_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw fetch-pdfs")
    p.add_argument(
        "data_dir", type=Path,
        help="CiteClaw data directory containing literature_collection.json",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent download workers (default: 4)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and re-parse PDFs even if they already exist on disk.",
    )
    p.add_argument(
        "--no-refresh-cache", action="store_true",
        help="Skip the refresh of pdf_url from the local S2 cache.db. "
             "By default, papers missing pdf_url in the JSON get rechecked "
             "against cache.db's paper_metadata table.",
    )
    p.add_argument(
        "--no-update-cache", action="store_true",
        help="Skip writing parse outcomes back into cache.db's "
             "paper_full_text table.",
    )
    p.add_argument(
        "--parser",
        type=str,
        default="pymupdf",
        choices=("pymupdf", "docling", "grobid"),
        help="PDF parser engine (default: pymupdf). See pdfclaw.parsers.",
    )
    p.add_argument(
        "--parser-kwarg",
        action="append",
        dest="parser_kwargs",
        metavar="KEY=VALUE",
        help='Engine kwarg (repeatable). Example: --parser-kwarg base_url=https://...',
    )
    return p


def _validate_config(config) -> None:
    """Pre-flight check: seeds + pipeline + every required env var.

    Exits with status 1 + a structured error log if any check fails so
    the user sees actionable errors before any LLM / S2 spend. API keys
    are never read from YAML — :func:`citeclaw.preflight.find_missing_api_keys`
    walks the built pipeline + filter blocks to compute the env-var set
    that the configured providers will actually need at runtime.
    """
    errors: list[str] = []
    if not config.seed_papers:
        errors.append("At least one seed paper is required.")
    if not config.pipeline:
        errors.append("'pipeline' section is required.")
    errors.extend(find_missing_api_keys(config))
    if errors:
        for e in errors:
            log.error("Config error: %s", e)
        log.error(
            "Set the missing env vars and re-run. "
            "API keys are intentionally never read from YAML.",
        )
        sys.exit(1)


def _run_snowball(argv: list[str]) -> None:
    parser = _build_run_parser()
    args = parser.parse_args(argv)

    if args.max_depth is not None:
        log.warning("--max-depth is deprecated and ignored")
    if args.citation_beta is not None:
        log.warning("--citation-beta is deprecated and ignored")

    overrides: dict = {}
    if args.topic:
        overrides["topic_description"] = args.topic
    if args.seed:
        overrides["seed_papers"] = [SeedPaper(paper_id=s) for s in args.seed]
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.max_papers is not None:
        overrides["max_papers_total"] = args.max_papers
    if args.model:
        overrides["screening_model"] = args.model

    config = load_settings(args.config, overrides)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir=config.data_dir, level=log_level)
    _validate_config(config)

    ctx, s2, cache = build_context(config)

    if args.continue_from is not None:
        try:
            load_checkpoint(ctx, args.continue_from)
        except FileNotFoundError as exc:
            log.error("Checkpoint load failed: %s", exc)
            sys.exit(1)

    try:
        run_pipeline(ctx)
    except BudgetExhaustedError as exc:
        log.warning("Budget exhausted: %s", exc)
        finalize_partial(ctx)
    except S2OutageError as exc:
        log.error(
            "S2 API appears to be down or rate-limiting hard — %s. "
            "Saving partial collection and exiting.", exc,
        )
        finalize_partial(ctx)
        sys.exit(1)
    except CiteClawError as exc:
        log.error("CiteClaw error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Interrupted — saving partial")
        finalize_partial(ctx)
    finally:
        s2.close()
        cache.close()


def _run_annotate(argv: list[str]) -> None:
    from citeclaw.annotate import annotate_graph

    parser = _build_annotate_parser()
    args = parser.parse_args(argv)
    setup_logging(log_dir=None, level=logging.INFO)
    output = args.output or args.graph.with_name(args.graph.stem + "_annotated.graphml")
    annotate_graph(
        graph_path=args.graph,
        output_path=output,
        instruction=args.instruction,
        config_path=args.config,
        api_key=args.api_key,
        model_override=args.model,
        limit=args.limit,
    )


def _run_rebuild_graph(argv: list[str]) -> None:
    """Rebuild the citation + collaboration graphs for an existing data dir.

    Useful when the original graphs have been modified in place during
    downstream analysis and the user wants a fresh copy regenerated from
    the same underlying run state (``literature_collection*.json`` +
    ``cache.db``) without re-running the whole pipeline.
    """
    parser = _build_rebuild_graph_parser()
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    # Load settings. If a config file is provided, honor it; otherwise fall
    # back to a minimal Settings with data_dir set so Cache/S2 point at the
    # right place.
    overrides = {"data_dir": data_dir}
    config = load_settings(args.config, overrides)

    setup_logging(log_dir=config.data_dir, level=logging.INFO)

    ctx, s2, cache = build_context(config)
    try:
        try:
            load_checkpoint(ctx, data_dir)
        except FileNotFoundError as exc:
            log.error(
                "Data directory %s does not contain a valid CiteClaw run: %s",
                data_dir, exc,
            )
            sys.exit(1)

        # load_checkpoint advances ``iteration`` to (prior+1) so continuation
        # runs don't clobber existing artifacts. For a rebuild we want the
        # *existing* iteration number — so rewind by one.
        ctx.iteration = max(1, ctx.iteration - 1)
        # Rebuild is a read-only regeneration: nothing was newly accepted,
        # so clear the continuation-only ``new_seed_ids`` trail.
        ctx.new_seed_ids = []

        suffix = "" if args.force else ".regen"
        write_graphs(ctx, suffix=suffix)
        log.info(
            "Rebuilt graphs in %s (suffix=%r, %d papers)",
            data_dir, suffix, len(ctx.collection),
        )
    finally:
        s2.close()
        cache.close()


def _run_mainpath(argv: list[str]) -> None:
    """Run main path analysis on a CiteClaw citation GraphML.

    Thin CLI adapter around :func:`citeclaw.mainpath.run_mpa`.
    See :mod:`citeclaw.mainpath` for the algorithmic layer.
    """
    from citeclaw.mainpath import run_mpa

    parser = _build_mainpath_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir=None, level=log_level)

    if not args.graph.exists():
        log.error("Graph file not found: %s", args.graph)
        sys.exit(1)

    output = args.output or args.graph.with_name(
        f"{args.graph.stem}_mainpath_{args.search}_{args.weight}.graphml",
    )
    try:
        run_mpa(
            graph_path=args.graph,
            output_path=output,
            weight=args.weight,
            search=args.search,
            cycle=args.cycle,
            key_routes=args.key_routes,
            tolerance=args.tolerance,
        )
    except ValueError as exc:
        log.error("mainpath failed: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        log.error("mainpath failed: %s", exc)
        sys.exit(1)


def _run_fetch_pdfs(argv: list[str]) -> None:
    """Bulk-download open-access PDFs for a finished CiteClaw run.

    Loads ``literature_collection.json`` from the given data directory
    and writes ``<data_dir>/PDFs/<paper_id>.pdf`` (raw) and
    ``<paper_id>.txt`` (parsed body) for every accepted paper that has
    an ``openAccessPdf.url`` in S2. See :mod:`citeclaw.fetch_pdfs` for
    the implementation.
    """
    from citeclaw.fetch_pdfs import run_fetch_pdfs

    parser = _build_fetch_pdfs_parser()
    args = parser.parse_args(argv)

    setup_logging(log_dir=None, level=logging.INFO)

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    try:
        run_fetch_pdfs(
            data_dir,
            max_workers=args.workers,
            overwrite=args.overwrite,
            refresh_from_cache=not args.no_refresh_cache,
            update_cache=not args.no_update_cache,
            parser=args.parser,
            parser_kwargs=_parse_kwarg_pairs(args.parser_kwargs),
        )
    except FileNotFoundError as exc:
        log.error("fetch-pdfs failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)


def _build_extract_info_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw extract-info")
    p.add_argument(
        "input",
        type=Path,
        help="Path to a PDF or text file to extract from",
    )
    p.add_argument(
        "--instruction",
        "-i",
        type=str,
        required=True,
        help='Instruction telling the LLM what to extract '
             '(e.g. "list all dataset names mentioned").',
    )
    p.add_argument(
        "--schema-file",
        type=Path,
        default=None,
        help="Optional JSON Schema file constraining the output shape.",
    )
    p.add_argument(
        "--paper-title",
        type=str,
        default="",
        help="Optional paper title (included in the prompt for context).",
    )
    p.add_argument(
        "--max-input-chars",
        type=int,
        default=80_000,
        help="Truncation budget for the paper text (default: 80000).",
    )
    p.add_argument(
        "--parser",
        type=str,
        default="pymupdf",
        choices=("pymupdf", "docling", "grobid"),
        help="PDF parser engine to use when input is a PDF (default: pymupdf). "
             "Ignored for plain-text input.",
    )
    p.add_argument(
        "--parser-kwarg",
        action="append",
        dest="parser_kwargs",
        metavar="KEY=VALUE",
        help='Engine kwarg (repeatable). Example: --parser-kwarg base_url=https://...',
    )
    p.add_argument(
        "--model",
        type=str,
        default="grok-4-1-fast-non-reasoning",
        help="Model alias (default: grok-4-1-fast-non-reasoning, reads XAI_API_KEY).",
    )
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Optional YAML config providing the models registry. "
             "When omitted, an inline registry is built for grok-4-1-fast-non-reasoning.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the extracted JSON to this path instead of stdout.",
    )
    return p


def _load_input_text(
    path: Path,
    *,
    max_chars: int,
    parser: str = "pymupdf",
    parser_kwargs: dict | None = None,
) -> str:
    """Read the input file as text (parsing if it's a PDF).

    PDFs go through the configured engine in :mod:`pdfclaw.parsers`;
    plain text files are read verbatim and the *parser* / *parser_kwargs*
    arguments are ignored.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pdfclaw.parsers import parse as parse_pdf
        result = parse_pdf(path, parser=parser, **(parser_kwargs or {}))
        return result.body_text
    return path.read_text(encoding="utf-8", errors="ignore")


def _build_extract_settings(model_alias: str):
    """Inline-build a Settings object for the extract-info CLI when no YAML.

    The default points at xAI's OpenAI-compatible endpoint with
    ``reasoning_parser="none"`` so no thinking kwargs are sent (the
    grok-4-1-fast-non-reasoning model rejects them).
    """
    from citeclaw.config import ModelEndpoint, Settings

    if model_alias == "grok-4-1-fast-non-reasoning":
        endpoint = ModelEndpoint(
            base_url="https://api.x.ai/v1",
            served_model_name="grok-4-1-fast-non-reasoning",
            api_key_env="XAI_API_KEY",
            reasoning_parser="none",
        )
        return Settings(
            screening_model=model_alias,
            models={model_alias: endpoint},
        )
    # For any other alias, fall back to plain SaaS routing — caller is
    # responsible for setting OPENAI_API_KEY / GEMINI_API_KEY / etc.
    return Settings(screening_model=model_alias)


def _parse_kwarg_pairs(raw_pairs: list[str] | None) -> dict[str, str]:
    """Turn ``--parser-kwarg key=value`` repeats into a dict.

    Values are kept as strings — individual parsers cast as needed,
    matching the convention used by ``pdfclaw.cli``.
    """
    out: dict[str, str] = {}
    for raw in raw_pairs or []:
        if "=" not in raw:
            log.warning("ignoring --parser-kwarg %r (expected key=value)", raw)
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _check_extract_info_llm_key(config, model_alias: str) -> str | None:
    """Verify only the LLM key for the chosen model is set.

    The pipeline-level ``find_missing_api_keys`` is too strict for the
    ``extract-info`` CLI because it also requires ``S2_API_KEY`` (S2 is
    used everywhere else in CiteClaw). Extraction only needs the LLM,
    so we walk the registry directly.
    """
    import os

    entry = config.models.get(model_alias)
    if entry is not None and entry.api_key_env:
        if not os.environ.get(entry.api_key_env):
            return entry.api_key_env
        return None
    # Fallback: rely on env-var probing — Gemini, OpenAI SaaS, etc.
    if model_alias.lower().startswith("gemini") and not os.environ.get("GEMINI_API_KEY"):
        return "GEMINI_API_KEY"
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("CITECLAW_OPENAI_API_KEY")):
        return "OPENAI_API_KEY"
    return None


def _run_extract_info(argv: list[str]) -> None:
    """Extract structured information from a paper text or PDF."""
    import json as _json

    from citeclaw.budget import BudgetTracker
    from citeclaw.clients.llm.factory import build_llm_client
    from citeclaw.extraction import extract_from_text

    parser = _build_extract_info_parser()
    args = parser.parse_args(argv)
    setup_logging(log_dir=None, level=logging.INFO)

    if not args.input.is_file():
        log.error("Input not found: %s", args.input)
        sys.exit(2)

    parser_kwargs = _parse_kwarg_pairs(args.parser_kwargs)
    text = _load_input_text(
        args.input,
        max_chars=args.max_input_chars,
        parser=args.parser,
        parser_kwargs=parser_kwargs,
    )
    if not text.strip():
        log.error("Input file %s contains no extractable text", args.input)
        sys.exit(1)

    schema = None
    if args.schema_file is not None:
        schema = _json.loads(Path(args.schema_file).read_text(encoding="utf-8"))

    if args.config is not None:
        config = load_settings(args.config, {})
        if not config.screening_model:
            config.screening_model = args.model
    else:
        config = _build_extract_settings(args.model)

    missing_env = _check_extract_info_llm_key(config, args.model)
    if missing_env:
        log.error(
            "Missing env var '%s' for model %r. "
            "Export it in your shell before running extract-info.",
            missing_env, args.model,
        )
        sys.exit(1)

    llm = build_llm_client(config, BudgetTracker(), model=args.model)

    log.info(
        "Extracting from %s (%d chars) with model=%s, schema=%s",
        args.input, len(text), args.model,
        "yes" if schema else "no",
    )

    result = extract_from_text(
        text,
        args.instruction,
        llm=llm,
        schema=schema,
        paper_title=args.paper_title,
        max_input_chars=args.max_input_chars,
    )

    if result.extraction_failed:
        log.error("Extraction failed: %s", result.error)
        log.info("Raw LLM output (first 500 chars): %s", result.raw_text[:500])
        sys.exit(1)

    out_json = _json.dumps(result.output, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out_json, encoding="utf-8")
        log.info("Wrote extraction to %s", args.out)
    else:
        print(out_json)


def _build_meta_review_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw meta-review")
    p.add_argument(
        "run_dir",
        type=Path,
        help="Finished CiteClaw run directory (contains literature_collection.json + cache.db).",
    )
    p.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="YAML config providing the models registry + base settings "
             "(reused so extractor/meta-model aliases route through the "
             "same factory as the pipeline did).",
    )
    p.add_argument(
        "--instruction", "-i",
        type=str,
        default=None,
        help='What to extract from each paper (free-form, e.g. '
             '"Extract model architecture, n parameters, training objective, datasets"). '
             'Required unless --from-extractions is used.',
    )
    p.add_argument(
        "--extractor-model",
        type=str,
        default=None,
        help="Model alias for the per-paper extractor (must exist in the config's models registry). "
             "Required unless --from-extractions is used.",
    )
    p.add_argument(
        "--from-extractions",
        type=Path,
        default=None,
        help="Path to an existing meta_review_extractions.json sidecar. When set, the per-paper "
             "extraction pass is skipped and the meta LLM is called directly on the cached "
             "extractions — useful for iterating on the meta prompt without re-running 60+ paper "
             "extractions, or for recovering from a meta-side failure.",
    )
    p.add_argument(
        "--extractor-reasoning",
        type=str,
        default="medium",
        help="Reasoning effort for the per-paper extractor (default: medium).",
    )
    p.add_argument(
        "--meta-model",
        type=str,
        required=True,
        help="Model alias for the meta-synthesis LLM.",
    )
    p.add_argument(
        "--meta-reasoning",
        type=str,
        default="high",
        help="Reasoning effort for the meta-synthesis LLM (default: high).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent papers in flight during per-paper extraction (default: 4).",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap how many papers to process (default: every paper in the collection).",
    )
    p.add_argument(
        "--per-paper-max-chars",
        type=int,
        default=2000,
        help="Per-paper extraction character budget (default: 2000). Keeps the meta prompt sized.",
    )
    p.add_argument(
        "--pdf-input-max-chars",
        type=int,
        default=60000,
        help="Per-paper PDF text fed to the extractor LLM (default: 60000).",
    )
    p.add_argument(
        "--parser",
        type=str,
        default="pymupdf",
        choices=("pymupdf", "docling", "grobid"),
        help="PDF parser engine for any papers that need fetching (default: pymupdf).",
    )
    p.add_argument(
        "--parser-kwarg",
        action="append",
        dest="parser_kwargs",
        metavar="KEY=VALUE",
        help="Parser-engine kwarg (repeatable).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the markdown report (default: <run_dir>/meta_review.md).",
    )
    return p


def _run_meta_review(argv: list[str]) -> None:
    """Post-pipeline meta-review.  See :mod:`citeclaw.meta_review`."""
    from citeclaw.meta_review import run_meta_review

    parser = _build_meta_review_parser()
    args = parser.parse_args(argv)
    setup_logging(log_dir=None, level=logging.INFO)

    if not args.run_dir.exists():
        log.error("Run directory not found: %s", args.run_dir)
        sys.exit(1)
    if not (args.run_dir / "literature_collection.json").exists():
        log.error(
            "Run directory %s has no literature_collection.json — "
            "did the pipeline finish?", args.run_dir,
        )
        sys.exit(1)

    config = load_settings(args.config, {})

    # Validate the meta-only vs full-run argument set.
    if args.from_extractions is not None:
        if not args.from_extractions.exists():
            log.error("--from-extractions: file not found: %s", args.from_extractions)
            sys.exit(1)
        # extractor_model / instruction become optional in this mode —
        # they'll be filled in from the sidecar's metadata.
    else:
        if not args.instruction:
            log.error(
                "--instruction is required unless --from-extractions is set"
            )
            sys.exit(2)
        if not args.extractor_model:
            log.error(
                "--extractor-model is required unless --from-extractions is set"
            )
            sys.exit(2)

    # Pre-flight: check api_key_env for each model alias that will
    # actually be invoked.  When --from-extractions is set, only the
    # meta model gets called.
    aliases_to_check: list[str] = [args.meta_model]
    if args.from_extractions is None and args.extractor_model:
        aliases_to_check.append(args.extractor_model)
    for alias in aliases_to_check:
        missing = _check_extract_info_llm_key(config, alias)
        if missing:
            log.error(
                "Missing env var '%s' required by model %r. "
                "Export it in your shell before running meta-review.",
                missing, alias,
            )
            sys.exit(1)

    try:
        result = run_meta_review(
            args.run_dir,
            config,
            instruction=args.instruction or "",
            extractor_model=args.extractor_model or "",
            extractor_reasoning=args.extractor_reasoning,
            meta_model=args.meta_model,
            meta_reasoning=args.meta_reasoning,
            max_workers=args.max_workers,
            max_papers=args.max_papers,
            per_paper_max_chars=args.per_paper_max_chars,
            pdf_input_max_chars=args.pdf_input_max_chars,
            parser=args.parser,
            parser_kwargs=_parse_kwarg_pairs(args.parser_kwargs),
            from_extractions=args.from_extractions,
        )
    except FileNotFoundError as exc:
        log.error("meta-review failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)

    log.info(
        "meta-review complete: %d/%d papers extracted, report at %s",
        result.n_papers_extracted, result.n_papers_total,
        args.output or (args.run_dir / "meta_review.md"),
    )


def _build_web_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw web")
    p.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    p.add_argument(
        "--port", type=int, default=9999,
        help="Port to listen on (default: 9999)",
    )
    return p


def _run_web(argv: list[str]) -> None:
    """Launch the CiteClaw web UI (FastAPI + React frontend)."""
    from citeclaw.web_server import serve

    parser = _build_web_parser()
    args = parser.parse_args(argv)
    setup_logging(log_dir=None, level=logging.INFO)
    serve(host=args.host, port=args.port)


# Subcommand dispatch table. Order here only matters for docs / --help;
# ``main`` matches the first positional arg against the keys and falls
# through to ``_run_snowball`` (the default pipeline run) when nothing
# matches. Each handler parses the remaining argv tail itself.
_SUBCOMMANDS: dict[str, "Callable[[list[str]], None]"] = {
    "annotate": lambda argv: _run_annotate(argv),
    "rebuild-graph": lambda argv: _run_rebuild_graph(argv),
    "fetch-pdfs": lambda argv: _run_fetch_pdfs(argv),
    "mainpath": lambda argv: _run_mainpath(argv),
    "extract-info": lambda argv: _run_extract_info(argv),
    "meta-review": lambda argv: _run_meta_review(argv),
    "web": lambda argv: _run_web(argv),
}


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv[0]`` as a subcommand name and dispatch to its handler.

    When no match is found (or argv is empty), falls through to
    :func:`_run_snowball`, the default pipeline run.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _SUBCOMMANDS:
        _SUBCOMMANDS[argv[0]](argv[1:])
        return
    _run_snowball(argv)


if __name__ == "__main__":
    main()
