"""``ExpandForward`` step — expand the signal forward through citers.

For each source paper in ``signal``: fetch its citers from S2, take
the top ``max_citations`` by citation count, hydrate them, and screen
through the configured filter block. Survivors get
``depth = source.depth + 1`` / ``source = "forward"`` /
``supporting_papers = [source.paper_id]`` and are added to
``ctx.collection``.

Idempotency lives on ``ctx.expanded_forward`` (a set of source paper
ids) so re-running the step on the same source skips rather than
re-fetching. This is the bottom-half twin of
:class:`~citeclaw.steps.expand_backward.ExpandBackward` (references
instead of citers).

The audit's "no silent failure" rule: per-source citer-fetch failures
log at WARNING + skip the source; the source-references fetch
(which only feeds the SimilarityFilter ``RefSim`` measure) is allowed
to silently fall back to an empty set so a screener without ref-based
similarity still works — DEBUG-logged.
"""

from __future__ import annotations

import logging

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.network import saturation_for_paper
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_forward")


class ExpandForward:
    """Per-source forward citation expansion + screening."""

    name = "ExpandForward"

    def __init__(self, *, max_citations: int = 100, screener=None) -> None:
        self.max_citations = max_citations
        self.screener = screener

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        """Expand each source paper through up to ``max_citations`` citers."""
        if self.screener is None:
            return StepResult(signal=[], in_count=len(signal), stats={"reason": "no screener"})

        dash = ctx.dashboard
        dash.enable_outer_bar(total=len(signal), description="source papers")

        accepted: list[PaperRecord] = []
        for source in signal:
            if source.paper_id in ctx.expanded_forward:
                dash.advance_outer(1)
                continue
            ctx.expanded_forward.add(source.paper_id)

            # Use the source's known citation_count as the inner-bar
            # total so the user sees e.g. "300 / 5348" while paginating
            # citers. Falls back to 1 when S2 hasn't reported a count.
            cit_total = source.citation_count or 0
            dash.begin_phase("fetch citers", total=cit_total or 1)
            try:
                citers = ctx.s2.fetch_citation_ids_and_counts(
                    source.paper_id,
                    progress_cb=(dash.tick_inner if cit_total else None),
                )
            except Exception as exc:
                log.warning("forward: failed to fetch citers for %s: %s", source.paper_id[:20], exc)
                dash.advance_outer(1)
                continue
            dash.complete_phase()

            citers.sort(key=lambda x: x.get("citation_count") or 0, reverse=True)
            citers = citers[: self.max_citations]
            unseen = [c for c in citers if c.get("paper_id") and c["paper_id"] not in ctx.seen]
            if not unseen:
                dash.advance_outer(1)
                continue
            for c in unseen:
                ctx.seen.add(c["paper_id"])
            dash.note_candidates_seen(len(unseen))

            dash.begin_phase("fetch source refs", total=None)
            try:
                source_refs = set(ctx.s2.fetch_reference_ids(source.paper_id))
            except Exception as exc:  # noqa: BLE001
                # Source-refs feed only the SimilarityFilter RefSim path —
                # a screener without ref-based similarity works without
                # them. DEBUG-log so the failure leaves a diagnostic trail
                # without spamming WARNING on a known-tolerable path.
                log.debug(
                    "forward: source-refs fetch for %s failed: %s — using empty set",
                    source.paper_id[:20], exc,
                )
                source_refs = set()
            dash.complete_phase()

            source_citers = {c.get("paper_id") for c in citers if c.get("paper_id")}

            dash.begin_phase("enrich · batch", total=None)
            records = ctx.s2.enrich_batch(unseen)
            for rec in records:
                rec.depth = source.depth + 1
                rec.source = "forward"
                rec.supporting_papers = [source.paper_id]
                if source.paper_id not in rec.references:
                    rec.references.append(source.paper_id)
            dash.complete_phase()

            # Need abstracts for title_abstract LLMFilters. S2's batch
            # endpoint handles up to 500 papers per POST, so any normal
            # enrich call is one atomic round-trip — a fake N-paper
            # total just sits at 0 until the response lands and then
            # snaps to N. Use indeterminate so the bar pulses while the
            # request is in flight. The per-chunk tick still bumps the
            # internal counter, which complete_phase snaps to total at
            # end. Falls back to the per-paper slow path only on batch
            # failure (logged at WARNING in _batch_fetch).
            dash.begin_phase("enrich · abstracts", total=None)
            ctx.s2.enrich_with_abstracts(records, progress_cb=dash.tick_inner)
            dash.complete_phase()

            fctx = FilterContext(
                ctx=ctx, source=source, source_refs=source_refs, source_citers=source_citers,
            )
            # apply_block drives the inner bar through the screener cascade.
            passed, rejected = apply_block(records, self.screener, fctx)
            record_rejections(rejected, fctx)
            for p in passed:
                p.llm_verdict = "accept"
                ctx.collection[p.paper_id] = p
                accepted.append(p)
                # Saturation: cache-only ref lookup so we never trigger
                # surprise S2 calls for the metric. Will be None for any
                # paper whose references aren't already cached (e.g.
                # because the screener didn't include a CitSim/RefSim).
                dash.paper_accepted(p, saturation=saturation_for_paper(p, ctx))

            dash.advance_outer(1)

        return StepResult(
            signal=accepted, in_count=len(signal),
            stats={"accepted": len(accepted)},
        )
