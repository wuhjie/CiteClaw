#!/usr/bin/env python3
"""
annotate_graph.py
-----------------
Read a CiteClaw GraphML file, use an LLM to generate a concise label for each
node from paper metadata (title + abstract), and write the annotated graph back.

Usage:
    python annotate_graph.py runs/data_bio/citation_network.graphml \
        -c configs/config_bio.yaml \
        -i "use the 1-word model name for the label (e.g., AlphaFold3, RNAErine, ProGen2, DNABERT)"

    # instruction is optional — if omitted, labels default to paper titles
    python annotate_graph.py runs/data_bio/citation_network.graphml -c configs/config_bio.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Regex to strip any leftover <think>...</think> blocks (safety net in case
# the OSS server isn't running with --reasoning-parser).
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    cleaned = _THINK_TAG_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _custom_reasoning_kwargs(reasoning_effort: str) -> dict[str, Any]:
    """Map unified ``reasoning_effort`` to OSS thinking controls.

    - ``""``: no overrides (server default)
    - ``"off"`` / ``"none"``: disable thinking
    - ``"low"`` / ``"medium"`` / ``"high"`` / ``"minimal"``: enable thinking +
      forward effort level to servers that honor it natively.

    PH-09: also sets ``skip_special_tokens: False`` so vLLM keeps the
    ``<|channel>...<channel|>`` thinking-block delimiters visible in the
    response text — without this, the gemma4 reasoning parser can't find
    the markers and the entire thinking trace leaks into
    ``message.content`` as a ``thought\\n...`` blob. Same fix as
    citeclaw.clients.llm.openai_client._custom_endpoint_reasoning_kwargs.
    """
    e = (reasoning_effort or "").strip().lower()
    if not e:
        return {}
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False,
    }
    if e in ("off", "none", "false", "disable", "disabled"):
        return {"extra_body": extra_body}
    extra_body["chat_template_kwargs"]["enable_thinking"] = True
    return {
        "reasoning_effort": e,
        "extra_body": extra_body,
    }


def _normalize_base_url(url: str) -> str:
    """Ensure the base_url ends with '/v1' so the OpenAI SDK targets vLLM correctly."""
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url

import igraph as ig
import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are labelling a paper in a citation network.\n"
    "Generate a concise label based on the instruction.\n"
    "Reply with ONLY the label text, nothing else — no quotes, no explanation."
)


def _build_user_message(
    *,
    instruction: str,
    title: str,
    abstract: str,
    full_text: str | None,
    use_title: bool,
    use_abstract: bool,
    use_full_text: bool,
    full_text_max_chars: int,
) -> str:
    """Assemble the user-message for the annotator LLM from the enabled fields.

    Any of ``use_title`` / ``use_abstract`` / ``use_full_text`` can be toggled
    independently. A "missing" source (e.g. ``use_full_text=True`` but the PDF
    failed to download) is silently skipped — the LLM still sees whichever
    fields did resolve. If every source is disabled or missing the function
    still produces a prompt header, so the caller never has to special-case it.
    """
    parts: list[str] = [f"Instruction: {instruction}", ""]
    if use_title:
        parts.append(f"Title: {title or '(no title)'}")
    if use_abstract:
        parts.append(f"Abstract: {abstract or '(no abstract)'}")
    if use_full_text:
        if full_text:
            body = full_text[:full_text_max_chars]
            if len(full_text) > full_text_max_chars:
                body = body + "\n[...truncated...]"
            parts.append("Full text:")
            parts.append(body)
        else:
            parts.append("Full text: (unavailable — no open-access PDF)")
    parts.append("")
    parts.append("Label:")
    return "\n".join(parts)


import threading

_total_input_tokens = 0
_total_output_tokens = 0
_total_reasoning_tokens = 0
_total_calls = 0
# Lock protects the four global token counters above and also serializes
# the per-paper progress print in the concurrent loop below, so output
# doesn't get interleaved when multiple workers finish at once.
_stats_lock = threading.Lock()

_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4")


def _is_openai_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in _OPENAI_REASONING_PREFIXES)


_gemini_client = None  # reused across calls


def _get_gemini_client(api_key: str):
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _call_llm_gemini(
    api_key: str,
    model: str,
    system: str,
    user: str,
    *,
    reasoning_effort: str = "",
    max_retries: int = 3,
) -> str:
    from google.genai import types

    global _total_input_tokens, _total_output_tokens, _total_reasoning_tokens, _total_calls
    client = _get_gemini_client(api_key)
    for attempt in range(max_retries):
        try:
            gen_config: dict[str, Any] = {
                "temperature": 0.0,
                "system_instruction": system,
            }
            if reasoning_effort:
                gen_config["thinking_config"] = types.ThinkingConfig(
                    thinking_level=reasoning_effort,
                )
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(**gen_config),
            )
            parts = (resp.candidates[0].content.parts if resp.candidates else []) or []
            text_parts = [
                p.text for p in parts
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            ]
            text = "\n".join(text_parts) if text_parts else (getattr(resp, "text", "") or "")
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                with _stats_lock:
                    _total_input_tokens += getattr(um, "prompt_token_count", 0) or 0
                    _total_output_tokens += getattr(um, "candidates_token_count", 0) or 0
                    _total_reasoning_tokens += getattr(um, "thinking_token_count", 0) or 0
                    _total_calls += 1
            return text.strip()
        except Exception as e:
            console.print(f"[red]gemini call failed (attempt {attempt+1}): {type(e).__name__}: {e}[/]")
            time.sleep(2 ** attempt)
    return ""


def _call_llm_openai(
    client: openai.OpenAI,
    model: str,
    system: str,
    user: str,
    *,
    reasoning_effort: str = "",
    is_custom: bool = False,
    max_retries: int = 3,
) -> str:
    global _total_input_tokens, _total_output_tokens, _total_reasoning_tokens, _total_calls
    is_reasoning = _is_openai_reasoning(model)
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            # OpenAI o-series reasoning models don't support temperature.
            # Everything else (including OSS via vLLM) is fine with temp=0.
            if not is_reasoning:
                kwargs["temperature"] = 0.0
            if is_custom:
                # Custom endpoint: map reasoning_effort to OSS thinking controls
                kwargs.update(_custom_reasoning_kwargs(reasoning_effort))
            elif reasoning_effort and is_reasoning:
                kwargs["reasoning_effort"] = reasoning_effort
            resp = client.chat.completions.create(**kwargs)
            usage = resp.usage
            if usage:
                with _stats_lock:
                    _total_input_tokens += usage.prompt_tokens
                    _total_output_tokens += usage.completion_tokens
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        _total_reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
                    _total_calls += 1
            text = (resp.choices[0].message.content or "").strip()
            if is_custom:
                text = _strip_think_tags(text)
            return text
        except openai.RateLimitError:
            time.sleep(2**attempt)
    return ""


def label_paper(
    *,
    model: str,
    instruction: str,
    title: str,
    abstract: str,
    full_text: str | None,
    use_title: bool,
    use_abstract: bool,
    use_full_text: bool,
    full_text_max_chars: int,
    api_key: str,
    reasoning_effort: str = "",
    openai_client: openai.OpenAI | None = None,
    is_custom: bool = False,
) -> str:
    """Generate a concise label for one paper.

    ``use_title`` / ``use_abstract`` / ``use_full_text`` control which
    fields are shown to the LLM. ``full_text`` may be None — set it when
    the paper has no open-access PDF or the parse failed. See
    :func:`_build_user_message` for prompt assembly rules.
    """
    user_msg = _build_user_message(
        instruction=instruction,
        title=title,
        abstract=abstract,
        full_text=full_text,
        use_title=use_title,
        use_abstract=use_abstract,
        use_full_text=use_full_text,
        full_text_max_chars=full_text_max_chars,
    )
    # Custom endpoints (vLLM, Modal, etc.) go through the OpenAI SDK path
    # even if the model name happens to start with "gemini-".
    if model.startswith("gemini-") and not is_custom:
        label = _call_llm_gemini(api_key, model, _SYSTEM, user_msg, reasoning_effort=reasoning_effort)
    else:
        assert openai_client is not None
        label = _call_llm_openai(
            openai_client,
            model,
            _SYSTEM,
            user_msg,
            reasoning_effort=reasoning_effort,
            is_custom=is_custom,
        )
    # Strip quotes if the model wraps the label
    label = label.strip('"\'').strip()
    return label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def annotate(
    graph_path: Path,
    output_path: Path,
    instruction: str | None,
    api_key: str,
    model: str = "gpt-5.4-nano",
    reasoning_effort: str = "",
    base_url: str = "",
    request_timeout: float = 60.0,
    limit: int | None = None,
    *,
    use_title: bool = True,
    use_abstract: bool = True,
    use_full_text: bool = False,
    full_text_max_chars: int = 30_000,
    data_dir: Path | None = None,
) -> None:
    console.print(f"[bold]Loading graph:[/] {graph_path}")
    g = ig.Graph.Read_GraphML(str(graph_path))
    console.print(f"  {g.vcount()} nodes, {g.ecount()} edges")

    if not instruction:
        # No instruction — use title as label (truncated)
        console.print("  No instruction provided — using paper titles as labels")
        g.vs["label"] = [
            (v["title"][:40] if "title" in v.attributes() else v.get("label", "?"))
            for v in g.vs
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g.write_graphml(str(output_path))
        console.print(f"[bold green]✓[/] Graph saved to {output_path}")
        return

    is_custom = bool(base_url)

    # Create client:
    #   - custom endpoint (vLLM/Modal/etc.) → OpenAI SDK with base_url
    #   - Gemini                            → native google-genai SDK (no OpenAI client needed)
    #   - everything else                   → standard OpenAI
    openai_client: openai.OpenAI | None = None
    if is_custom:
        normalized = _normalize_base_url(base_url)
        openai_client = openai.OpenAI(
            api_key=api_key or "none",
            base_url=normalized,
            timeout=request_timeout,
        )
        console.print(f"[dim]  Using custom endpoint: {normalized}[/]")
    elif not model.startswith("gemini-"):
        openai_client = openai.OpenAI(api_key=api_key)

    total = g.vcount()
    n_to_label = min(limit, total) if limit else total
    if limit and limit < total:
        console.print(
            f"[bold]Labelling first {n_to_label}/{total} papers[/] (--limit active; remaining keep titles)"
        )
    else:
        console.print(f"[bold]Labelling {total} papers[/] (instruction: {instruction[:60]})")
    if reasoning_effort:
        console.print(f"[dim]  reasoning_effort = {reasoning_effort}[/]")
    enabled = [
        name for name, on in (
            ("title", use_title),
            ("abstract", use_abstract),
            ("full-text", use_full_text),
        ) if on
    ]
    console.print(
        f"[dim]  fields shown to LLM: {', '.join(enabled) if enabled else '(none — instruction only)'}[/]"
    )

    # Pre-extract (paper_id, pdf_url, title, abstract) for every node so the
    # worker threads don't touch igraph vertex objects concurrently (igraph
    # isn't thread-safe for reads during this kind of access pattern).
    nodes: list[tuple[str, str, str, str]] = []
    for v in g.vs:
        paper_id = v["paper_id"] if "paper_id" in v.attributes() else ""
        pdf_url = v["pdf_url"] if "pdf_url" in v.attributes() else ""
        title = v["title"] if "title" in v.attributes() else v.get("label", "")
        abstract = v["abstract"] if "abstract" in v.attributes() else ""
        nodes.append((paper_id, pdf_url or "", title, abstract))

    # Optionally prefetch full-text PDFs for every node so workers can read
    # body text locally from the cache. The PdfFetcher is cache-aware: if the
    # upstream pipeline already populated ``paper_full_text`` (or if
    # ``use_full_text`` was enabled on a previous run), this call is a no-op.
    full_text_by_id: dict[str, str | None] = {}
    if use_full_text:
        # Lazy imports so a user who only labels titles+abstracts doesn't
        # need the CiteClaw package on PYTHONPATH at annotate time.
        from citeclaw.cache import Cache
        from citeclaw.clients.pdf import PdfFetcher
        from citeclaw.models import PaperRecord

        resolved_dir = (data_dir or graph_path.parent).resolve()
        cache_path = resolved_dir / "cache.db"
        console.print(
            f"[dim]  Full-text enabled — using cache at {cache_path}[/]"
        )
        cache = Cache(cache_path)
        fetcher = PdfFetcher(cache)
        stub_records = [
            PaperRecord(paper_id=pid, title=title, pdf_url=(purl or None))
            for (pid, purl, title, _abs) in nodes
            if pid
        ]
        try:
            full_text_by_id = fetcher.prefetch(stub_records, max_workers=4)
        except Exception as exc:
            console.print(
                f"[yellow]  PDF prefetch failed: {type(exc).__name__}: {exc} — "
                f"full-text will be skipped for every node[/]"
            )
            full_text_by_id = {}
        finally:
            fetcher.close()
        n_with_text = sum(1 for v in full_text_by_id.values() if v)
        console.print(
            f"[dim]  Full-text ready for {n_with_text}/{len(stub_records)} papers"
            f" (cache + fresh downloads combined)[/]"
        )

    # Results indexed by vertex id so we can write them back in order at
    # the end. Pre-populate with fallback labels (truncated titles) for
    # nodes past the limit OR nodes that fail to label. ``nodes`` items
    # are ``(paper_id, pdf_url, title, abstract)`` 4-tuples — index 2 is
    # the title.
    labels: list[str] = [((node[2] or "?")[:40]) for node in nodes]

    # Concurrency: the Modal vLLM server is configured for up to 256
    # concurrent inputs (`@modal.concurrent(max_inputs=256)`), and measured
    # KV cache on 2× H200 with Qwen3.5-122B-A10B-FP8 fits ~370 parallel
    # 16k-context sequences. 64 workers gives good batch utilization
    # without risking HTTP client timeouts from queueing. If you see
    # 429/timeouts, drop this to 32 or 16.
    max_workers = 64

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _label_one(idx: int) -> tuple[int, str]:
        paper_id, _purl, title, abstract = nodes[idx]
        ftext = full_text_by_id.get(paper_id) if use_full_text else None
        lbl = label_paper(
            model=model,
            instruction=instruction,
            title=title,
            abstract=abstract,
            full_text=ftext,
            use_title=use_title,
            use_abstract=use_abstract,
            use_full_text=use_full_text,
            full_text_max_chars=full_text_max_chars,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            openai_client=openai_client,
            is_custom=is_custom,
        )
        return idx, lbl

    done_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_label_one, i) for i in range(n_to_label)]
        for fut in as_completed(futures):
            idx, lbl = fut.result()
            labels[idx] = lbl
            title = nodes[idx][0]
            with _stats_lock:
                done_count += 1
                console.print(
                    f"  [dim][{done_count}/{n_to_label}][/] {title[:60]}  →  [bold]{lbl}[/]"
                )

    # Preserve original title, set label
    if "title" in g.vs.attributes():
        g.vs["original_title"] = g.vs["title"]
    g.vs["label"] = labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.write_graphml(str(output_path))
    console.print(f"[bold green]✓[/] Annotated graph saved to {output_path}")

    # Show sample
    console.print("\n[bold]Sample labels:[/]")
    ranked = sorted(
        range(g.vcount()),
        key=lambda i: g.vs[i]["citation_count"] if "citation_count" in g.vs.attributes() else 0,
        reverse=True,
    )
    for i in ranked[:10]:
        v = g.vs[i]
        orig = v["original_title"][:45] if "original_title" in v.attributes() else "?"
        console.print(f"  [bold]{v['label']:<25}[/]  [dim]← {orig}[/]")

    # Token usage summary
    total = _total_input_tokens + _total_output_tokens
    reasoning_part = f", {_total_reasoning_tokens:,} reasoning" if _total_reasoning_tokens else ""
    console.print(
        f"\n[dim]Token usage: {total / 1_000_000:.3f}M total "
        f"({_total_input_tokens:,} in + {_total_output_tokens:,} out{reasoning_part}, "
        f"{_total_calls} calls)[/]"
    )


def main():
    parser = argparse.ArgumentParser(description="Annotate citation graph nodes with LLM-generated labels.")
    parser.add_argument("graph", type=Path, help="Input GraphML file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output GraphML (default: <input>_annotated.graphml)")
    parser.add_argument("-i", "--instruction", type=str, default=None,
                        help='Labelling instruction (optional). If omitted, uses paper titles.')
    parser.add_argument("-c", "--config", type=Path, default=None,
                        help="Config YAML (for API key and model)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (overrides config)")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only label the first N nodes (for testing). Remaining nodes keep their titles as labels.",
    )
    # ------------------------------------------------------------------
    # Annotator input toggles: choose which fields the LLM sees.
    #
    # Defaults preserve the pre-existing behaviour (title+abstract on,
    # full-text off). The BooleanOptionalAction flavour gives every flag
    # a matching ``--no-...`` negation so the CLI overrides YAML cleanly.
    # YAML equivalents (read via ``graph_label_use_*`` / ``graph_label_full_text_max_chars``):
    #   graph_label_use_title:  true
    #   graph_label_use_abstract: true
    #   graph_label_use_full_text: false
    #   graph_label_full_text_max_chars: 30000
    # ``--data-dir`` points the full-text PDF cache at a specific folder
    # (defaults to the graph file's parent, which matches Finalize).
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use-title",
        dest="use_title",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the paper title to the annotator LLM (default: on).",
    )
    parser.add_argument(
        "--use-abstract",
        dest="use_abstract",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the abstract to the annotator LLM (default: on).",
    )
    parser.add_argument(
        "--use-full-text",
        dest="use_full_text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Show the parsed PDF body to the annotator LLM (default: off). "
            "Uses citeclaw.clients.pdf.PdfFetcher + the cache.db in --data-dir "
            "to fetch / parse / cache open-access PDFs."
        ),
    )
    parser.add_argument(
        "--full-text-max-chars",
        dest="full_text_max_chars",
        type=int,
        default=None,
        help="Character cap for the full-text section in each prompt (default: 30000).",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=None,
        help="CiteClaw data_dir (used to locate cache.db for full-text). Defaults to the graph file's parent.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible endpoint for the annotator (e.g. a Modal vLLM "
            "URL ending in /v1). Overrides graph_label_base_url / llm_base_url "
            "in the YAML. Self-hosted endpoints default their API key to "
            "CITECLAW_VLLM_API_KEY (or 'none' if unset)."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        type=str,
        default=None,
        help=(
            "Reasoning effort for the annotator LLM "
            "(low | medium | high | minimal | off). Overrides "
            "graph_label_reasoning_effort / reasoning_effort in the YAML."
        ),
    )
    args = parser.parse_args()

    # Resolve API key, model, and reasoning effort
    api_key = args.api_key
    model = args.model or "gpt-5.4-nano"
    reasoning_effort = ""
    base_url = ""
    request_timeout = 60.0

    instruction = args.instruction

    # Annotator field toggles — YAML defaults, CLI overrides below.
    cfg_use_title: bool | None = None
    cfg_use_abstract: bool | None = None
    cfg_use_full_text: bool | None = None
    cfg_full_text_max_chars: int | None = None
    cfg_data_dir: Path | None = None

    if args.config and args.config.exists():
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        # API keys MUST NOT appear in YAML — reject loudly if they do.
        _forbidden = {
            "openai_api_key", "gemini_api_key", "s2_api_key", "llm_api_key",
            "OPENAI_API_KEY", "GEMINI_API_KEY", "S2_API_KEY",
            "SEMANTIC_SCHOLAR_API_KEY", "LLM_API_KEY",
        }
        _leaked = [k for k in cfg if k in _forbidden]
        if _leaked:
            console.print(
                "[red]Error:[/] API keys must not be set in config YAML "
                f"(found: {', '.join(sorted(_leaked))}). Remove these fields "
                "and set environment variables instead (OPENAI_API_KEY, "
                "GEMINI_API_KEY, S2_API_KEY)."
            )
            sys.exit(1)
        # Annotator-specific overrides win over the pipeline-wide fields.
        # This is what lets config_rna.yaml keep screening_model on Gemini
        # for the pipeline while pointing the annotator at Gemma on Modal.
        if not args.model:
            model = cfg.get("graph_label_model") or cfg.get("screening_model", model)
        base_url = (
            cfg.get("graph_label_base_url")
            or cfg.get("llm_base_url", "")
            or ""
        )
        request_timeout = float(cfg.get("llm_request_timeout", 60.0))
        if not instruction:
            instruction = cfg.get("graph_label_instruction", "")
        reasoning_effort = (
            cfg.get("graph_label_reasoning_effort")
            or cfg.get("reasoning_effort", "")
            or ""
        )
        # Annotator field toggles from YAML (None means "unset, use default").
        if "graph_label_use_title" in cfg:
            cfg_use_title = bool(cfg.get("graph_label_use_title"))
        if "graph_label_use_abstract" in cfg:
            cfg_use_abstract = bool(cfg.get("graph_label_use_abstract"))
        if "graph_label_use_full_text" in cfg:
            cfg_use_full_text = bool(cfg.get("graph_label_use_full_text"))
        if "graph_label_full_text_max_chars" in cfg:
            cfg_full_text_max_chars = int(cfg.get("graph_label_full_text_max_chars"))
        if cfg.get("data_dir"):
            cfg_data_dir = Path(cfg["data_dir"])

    # CLI beats YAML for endpoint + reasoning effort.
    if args.base_url is not None:
        base_url = args.base_url
    if args.reasoning_effort is not None:
        reasoning_effort = args.reasoning_effort

    # CLI overrides YAML; YAML overrides defaults.
    use_title = (
        args.use_title if args.use_title is not None
        else (cfg_use_title if cfg_use_title is not None else True)
    )
    use_abstract = (
        args.use_abstract if args.use_abstract is not None
        else (cfg_use_abstract if cfg_use_abstract is not None else True)
    )
    use_full_text = (
        args.use_full_text if args.use_full_text is not None
        else (cfg_use_full_text if cfg_use_full_text is not None else False)
    )
    full_text_max_chars = (
        args.full_text_max_chars if args.full_text_max_chars is not None
        else (cfg_full_text_max_chars if cfg_full_text_max_chars is not None else 30_000)
    )
    data_dir = args.data_dir or cfg_data_dir

    if not (use_title or use_abstract or use_full_text):
        console.print(
            "[yellow]Warning:[/] all annotator field toggles are disabled — "
            "the LLM will see only the instruction. Enable at least one of "
            "--use-title / --use-abstract / --use-full-text."
        )

    # API keys come from env vars only.
    if not api_key:
        if base_url:
            # Self-hosted vLLM endpoints (Modal, etc.) usually need a bearer
            # token. Honour CITECLAW_VLLM_API_KEY (the project's canonical
            # variable, matching config.ModelEndpoint.resolved_api_key) and
            # fall back to "none" for servers that accept any string.
            api_key = (
                os.environ.get("CITECLAW_VLLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or "none"
            )
        elif model.startswith("gemini-"):
            api_key = os.environ.get("GEMINI_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key and instruction:
        console.print(
            "[red]Error:[/] API key needed for LLM labelling. Pass --api-key, -c config.yaml, or set the appropriate env var."
        )
        sys.exit(1)

    output = args.output or args.graph.with_name(args.graph.stem + "_annotated.graphml")

    annotate(
        graph_path=args.graph,
        output_path=output,
        instruction=instruction or None,
        api_key=api_key or "",
        model=model,
        reasoning_effort=reasoning_effort,
        base_url=base_url,
        request_timeout=request_timeout,
        limit=args.limit,
        use_title=use_title,
        use_abstract=use_abstract,
        use_full_text=use_full_text,
        full_text_max_chars=full_text_max_chars,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()
