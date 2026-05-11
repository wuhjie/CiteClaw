"""ExpandBackward step — for each paper in signal, fetch references and screen.

When ``pdf_references=True``, papers whose S2 reference list is empty
(not indexed, too new, or grey literature) get a PDF-based fallback:
the step fetches the paper's PDF and extracts references, then resolves
them through ``ctx.s2.fetch_metadata`` (DOI-first) and ``ctx.s2.search_match``
(title fallback).  Two parser modes:

  * ``parser="grobid"``: download raw PDF bytes (cache → S2 HTTP →
    pdfclaw recipes), POST to the deployed GROBID server, consume the
    structured ``<biblStruct>`` references directly.  Each ref entry
    is a cleanly-extracted bibliography line with the title, authors,
    year and (when present) DOI preserved.
  * default ``parser="pymupdf"``: use the legacy regex heuristic over
    body text — coarser but no GROBID dependency, kept for backward
    compatibility.

S2 references are always tried first; the PDF fallback only fires when
both S2 and OpenAlex come back empty.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.network import saturation_for_paper
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_backward")

# ---------------------------------------------------------------------------
# Lightweight reference-list parser (no LLM, no GROBID)
# ---------------------------------------------------------------------------

_REF_HEADING_RE = re.compile(
    r"^\s*(?:References|Bibliography|Works\s+Cited|Literature\s+Cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Matches common bibliography entry patterns:
#   [1] Author, ...  Title. Journal ...
#   1. Author, ...   Title. Journal ...
_REF_ENTRY_RE = re.compile(
    r"^\s*\[?\d{1,4}\]?\.?\s+",
    re.MULTILINE,
)


def _extract_all_ref_titles(text: str) -> list[str]:
    """Best-effort title extraction from a raw reference list.

    Heuristic: each numbered reference entry starts with ``[N]`` or
    ``N.``.  The title is typically the first sentence-like phrase after
    the author block (which ends with a period or colon after a year).
    We extract a candidate title by taking the text between the author
    block's terminating punctuation and the next period.

    This is intentionally rough — it's a fallback for the rare case
    where S2 has no reference data at all.
    """
    # Find the reference section.
    matches = list(_REF_HEADING_RE.finditer(text))
    if not matches:
        return []
    last = matches[-1]
    if last.start() < len(text) * 0.4:
        return []
    ref_section = text[last.end():]

    # Split into individual entries.
    entries = _REF_ENTRY_RE.split(ref_section)
    titles: list[str] = []
    for entry in entries:
        entry = entry.strip()
        if not entry or len(entry) < 20:
            continue
        title = _guess_title(entry)
        if title:
            titles.append(title)
    return titles


def _guess_title(entry: str) -> str | None:
    """Extract a plausible title from a single bibliography entry.

    Strategy: look for the pattern ``Author(s). Title. Journal/venue``
    and take the second sentence (the title).  Falls back to taking
    everything before the first period that's followed by a venue-like
    token (a capitalised word or a journal abbreviation).
    """
    # Common pattern: "Author, A., Author, B.: Title. Journal ..."
    # or "Author, A., Author, B. Title. Journal ..."
    # Try splitting on ". " and taking the first segment that looks
    # like a title (starts with a capital, > 15 chars, < 300 chars).
    parts = re.split(r"\.\s+", entry)
    for part in parts:
        part = part.strip()
        if len(part) < 15 or len(part) > 300:
            continue
        # Skip parts that look like author lists (contain ", " heavily).
        if part.count(",") > 3 and len(part) < 80:
            continue
        # Skip parts that look like journal names (short, all caps / mixed).
        if len(part) < 30 and part.count(" ") < 3:
            continue
        # Accept the first part that starts with a capital letter.
        if part[0].isupper():
            # Clean up trailing year/volume markers.
            cleaned = re.sub(r"\s*\(\d{4}\).*$", "", part).strip()
            if len(cleaned) >= 15:
                return cleaned
    return None


class ExpandBackward:
    name = "ExpandBackward"

    # Strict DOI pattern mirroring :mod:`citeclaw.models` — used to lift
    # DOIs out of GROBID-formatted reference strings so we can hit
    # ``s2.fetch_metadata`` (deterministic) before falling back to
    # ``search_match`` (fuzzy title query).
    _DOI_EXTRACT_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;)\]]+)")

    def __init__(
        self,
        *,
        screener=None,
        pdf_references: bool = False,
        pdf_model: str | None = None,
        parser: str = "pymupdf",
        parser_kwargs: dict | None = None,
        headless: bool = True,
        openalex_references: bool = True,
    ) -> None:
        self.screener = screener
        self.pdf_references = pdf_references
        self.pdf_model = pdf_model
        # Parser engine for the PDF fallback.  ``"grobid"`` takes the
        # structured-TEI path (best reference quality, requires the
        # ``PDFCLAW_GROBID_URL`` env var pointing at a running server);
        # anything else takes the legacy regex-on-body-text heuristic.
        self.parser = parser
        self.parser_kwargs = parser_kwargs or {}
        self.headless = headless
        # OpenAlex reference fallback — used when S2 returns empty refs
        # AND the paper has a DOI in external_ids. Cheap (one OpenAlex
        # call + N single-DOI S2 resolves) and strictly improves recall
        # for fresh preprints S2 hasn't fully ingested. Defaults to True
        # because the network cost is small and the payoff is real; set
        # False to disable.
        self.openalex_references = openalex_references

    def run(self, signal: list[PaperRecord], ctx: Any) -> StepResult:
        if self.screener is None:
            return StepResult(signal=[], in_count=len(signal), stats={"reason": "no screener"})

        dash = ctx.dashboard
        dash.enable_outer_bar(total=len(signal), description="source papers")

        accepted: list[PaperRecord] = []
        pdf_fallback_count = 0
        openalex_fallback_count = 0

        for source in signal:
            if source.paper_id in ctx.expanded_backward:
                dash.advance_outer(1)
                continue
            ctx.expanded_backward.add(source.paper_id)

            # Indeterminate bar — ref counts aren't known upfront and
            # S2 pagination runs at ~1 rps, so any paper with >100 refs
            # would stall a 1/1 bar for seconds. Callback bumps the
            # displayed count after every page.
            dash.begin_phase("fetch refs", total=None)
            try:
                ref_records = ctx.s2.fetch_references(
                    source.paper_id, progress_cb=dash.tick_inner,
                )
            except Exception as exc:
                log.warning("backward: failed for %s: %s", source.paper_id[:20], exc)
                dash.advance_outer(1)
                continue
            dash.complete_phase()

            # OpenAlex fallback: when S2 returns no references and the
            # paper has a DOI, consult OpenAlex's referenced_works. Tried
            # before the PDF fallback because it's O(refs) network calls
            # (cheap) vs O(PDF fetch + LLM parse) for the PDF path.
            if not ref_records and self.openalex_references:
                oa_refs = self._openalex_fallback(source, ctx, dash=dash)
                if oa_refs:
                    ref_records = oa_refs
                    openalex_fallback_count += 1

            # PDF fallback: when S2 AND OpenAlex both miss and the user
            # opted in, extract reference titles from the paper's PDF.
            if not ref_records and self.pdf_references:
                pdf_refs = self._pdf_fallback(source, ctx)
                if pdf_refs:
                    ref_records = pdf_refs
                    pdf_fallback_count += 1

            source.references = [r.paper_id for r in ref_records if r.paper_id]

            cands: list[PaperRecord] = []
            for r in ref_records:
                if not r.paper_id or r.paper_id in ctx.seen:
                    continue
                ctx.seen.add(r.paper_id)
                r.depth = source.depth + 1
                r.source = "backward"
                r.supporting_papers = [source.paper_id]
                cands.append(r)

            if not cands:
                dash.advance_outer(1)
                continue
            dash.note_candidates_seen(len(cands))

            # Size the bar to the cands that actually need a live
            # fetch (local cache hits complete instantly). The callback
            # ticks once per S2 batch (or per singleton when the batch
            # path fails and fallback to per-paper GETs kicks in).
            n_miss = sum(1 for r in cands if not r.abstract)
            dash.begin_phase("enrich · abstracts", total=max(1, n_miss))
            ctx.s2.enrich_with_abstracts(cands, progress_cb=dash.tick_inner)
            dash.complete_phase()

            fctx = FilterContext(ctx=ctx, source=source)
            passed, rejected = apply_block(cands, self.screener, fctx)
            record_rejections(rejected, fctx)
            for p in passed:
                p.llm_verdict = "accept"
                ctx.collection[p.paper_id] = p
                accepted.append(p)
                dash.paper_accepted(p, saturation=saturation_for_paper(p, ctx))

            dash.advance_outer(1)

        stats: dict[str, Any] = {"accepted": len(accepted)}
        if self.pdf_references:
            stats["pdf_fallback_used"] = pdf_fallback_count
        if self.openalex_references:
            stats["openalex_fallback_used"] = openalex_fallback_count
        return StepResult(
            signal=accepted, in_count=len(signal),
            stats=stats,
        )

    def _openalex_fallback(
        self,
        source: PaperRecord,
        ctx: Any,
        *,
        dash: Any = None,
    ) -> list[PaperRecord]:
        """Fetch references via OpenAlex for a paper S2 has no refs for.

        Only runs when the paper carries a DOI in ``external_ids``.
        OpenAlex's ``referenced_works`` is keyed by DOI (via the
        ``/works/doi:...`` path); we resolve each returned DOI back to
        an S2 ``PaperRecord`` via ``ctx.s2.fetch_metadata("DOI:...")``.
        Failures (network, missing work, non-DOI'd records) are silently
        skipped so a partially-broken OpenAlex response doesn't kill
        the run.
        """
        doi = (source.external_ids or {}).get("DOI")
        if not doi:
            return []
        try:
            from citeclaw.clients.openalex import OpenAlexClient
        except ImportError:  # pragma: no cover
            return []
        client = OpenAlexClient(ctx.config)
        try:
            ref_dois = client.fetch_references_by_doi(doi)
        except Exception as exc:  # noqa: BLE001 — best effort
            log.info("OpenAlex references lookup failed for %s: %s",
                     source.paper_id[:20], exc)
            client.close()
            return []
        finally:
            # ``client.close`` is safe to call twice; ``try/finally``
            # guarantees it runs even on the happy path. DEBUG-log the
            # second-close failure (audit silent-failure flag) since
            # the close-on-error path inside the try-block already
            # logged the underlying exception at INFO.
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("openalex client double-close failed: %s", exc)

        if not ref_dois:
            return []

        # Per-DOI resolve via S2 at ~1 rps — large OpenAlex responses
        # otherwise leave the inner bar idle for minutes. (The OpenAlex
        # call itself happened above in one shot; this phase is S2
        # resolving the DOIs OpenAlex returned.)
        if dash is not None:
            dash.begin_phase(
                f"s2: resolve {len(ref_dois)} openalex DOIs",
                total=len(ref_dois),
            )

        records: list[PaperRecord] = []
        for ref_doi in ref_dois:
            try:
                rec = ctx.s2.fetch_metadata(f"DOI:{ref_doi}")
            except Exception as exc:  # noqa: BLE001 — skip unresolvable DOIs
                # Per-DOI miss is the common case (S2 doesn't have every
                # OpenAlex-cited DOI). DEBUG-log so the diagnostic trail
                # exists without spamming WARNING on a known-tolerable path.
                log.debug("openalex fallback: S2 fetch_metadata(DOI:%s) failed: %s",
                          ref_doi, exc)
                if dash is not None:
                    dash.tick_inner(1)
                continue
            if rec is not None:
                records.append(rec)
            if dash is not None:
                dash.tick_inner(1)

        log.info(
            "openalex fallback: %d DOIs → %d resolved refs for %s",
            len(ref_dois), len(records), source.paper_id[:20],
        )
        return records

    def _pdf_fallback(
        self,
        source: PaperRecord,
        ctx: Any,
    ) -> list[PaperRecord]:
        """Extract references from the source paper's PDF, then resolve via S2.

        Two paths depending on ``self.parser``:

        * ``"grobid"``: download PDF bytes, hand them to GROBID for TEI
          parsing, and consume the structured ``<biblStruct>`` references
          directly.  Each ref string is a clean bibliography line; the
          resolver tries a DOI regex first, then ``s2.search_match`` on
          the whole string.
        * other values (default ``"pymupdf"``): use the legacy regex
          heuristic over body text.  Kept for backward compatibility and
          for environments without a GROBID server.

        Returns the list of resolved ``PaperRecord``s (may be empty).
        """
        from citeclaw.clients.pdfclaw_bridge import PdfClawBridge

        bridge = PdfClawBridge(
            ctx.cache,
            headless=self.headless,
            parser=self.parser,
            parser_kwargs=self.parser_kwargs,
        )
        try:
            if self.parser == "grobid":
                ref_strings = self._grobid_ref_strings(bridge, source)
            else:
                text = bridge.fetch_text(source)
                ref_strings = _extract_all_ref_titles(text) if text else []
        finally:
            bridge.close()

        if not ref_strings:
            log.debug(
                "pdf_references fallback (%s): no refs for %s",
                self.parser, source.paper_id[:20],
            )
            return []

        log.info(
            "pdf_references fallback (%s): extracted %d references from %s",
            self.parser, len(ref_strings), source.paper_id[:20],
        )

        records: list[PaperRecord] = []
        for ref_text in ref_strings:
            rec = self._resolve_ref_string(ref_text, ctx)
            if rec:
                records.append(rec)

        log.info(
            "pdf_references fallback (%s): resolved %d / %d for %s",
            self.parser, len(records), len(ref_strings), source.paper_id[:20],
        )
        return records

    def _grobid_ref_strings(
        self,
        bridge: Any,
        source: PaperRecord,
    ) -> list[str]:
        """Fetch PDF bytes, run GROBID, return the structured-ref list.

        On any GROBID-side failure (server down, parser error, malformed
        TEI) falls back to PyMuPDF body text + the regex heuristic so the
        whole step still makes a best-effort attempt rather than dropping
        the paper.
        """
        body = bridge.fetch_pdf_bytes(source)
        if not body:
            return []

        from pdfclaw.parsers import parse as parse_pdf

        try:
            result = parse_pdf(body, parser="grobid", **self.parser_kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "pdf_references: GROBID failed for %s (%s) — falling back to heuristic",
                source.paper_id[:20], exc,
            )
            try:
                result = parse_pdf(body, parser="pymupdf")
            except Exception as exc2:  # noqa: BLE001
                log.debug(
                    "pdf_references: pymupdf fallback also failed for %s: %s",
                    source.paper_id[:20], exc2,
                )
                return []
            return _extract_all_ref_titles(result.body_text or "")

        return list(result.references or [])

    def _resolve_ref_string(
        self,
        ref_text: str,
        ctx: Any,
    ) -> PaperRecord | None:
        """Resolve one bibliography string to an S2 ``PaperRecord``.

        Cascade:

          1. Regex-extract a DOI; ``s2.fetch_metadata("DOI:<doi>")`` is
             deterministic when it returns something.
          2. Fall back to ``s2.search_match`` on the whole ref string.
             S2's search is fuzzy enough that feeding it ``"[67] Smith,
             J. et al. Title. Journal vol(issue):pages."`` typically
             still locks onto the title.

        Returns ``None`` when both paths fail (logged at DEBUG so the
        diagnostic trail exists without spam at INFO).
        """
        ref_text = ref_text.strip()
        if len(ref_text) < 12:
            return None

        # DOI-first path.
        m = self._DOI_EXTRACT_RE.search(ref_text)
        if m:
            doi = m.group(1).rstrip(".,;)]")
            try:
                rec = ctx.s2.fetch_metadata(f"DOI:{doi}")
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "pdf_references: DOI lookup failed for %r: %s",
                    doi, exc,
                )
            else:
                if rec is not None:
                    return rec

        # Title / full-line fuzzy search.
        try:
            match = ctx.s2.search_match(ref_text)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "pdf_references: search_match failed for %r: %s",
                ref_text[:60], exc,
            )
            return None
        if match is None:
            return None
        pid = match.get("paperId")
        if not pid:
            return None

        from citeclaw.clients.s2.converters import paper_to_record

        return paper_to_record(match)
