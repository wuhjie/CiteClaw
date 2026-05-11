"""Walk a Filter (atom or compositor) over a list of papers.

:func:`apply_block` is the single entry point. It dispatches by block
type — Sequential / Any / Not / Route / LLMFilter / generic atom — and
each branch handles batching at the right level so a deeply-nested
tree gets correct LLM batching at every leaf, dashboard phase
bookkeeping at every Sequential step, and per-paper Route partitioning
without re-running cheap atoms across cases.

:func:`record_rejections` is the companion sink: pass it the
``(paper, FilterOutcome)`` rejections returned by :func:`apply_block`
to flow them into ``ctx.rejected``, ``ctx.rejection_counts``, and
``ctx.rejection_ledger``.
"""

from __future__ import annotations

import logging
from typing import Any

from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.filters.base import FilterContext, FilterOutcome
from citeclaw.filters.blocks.any_block import Any_
from citeclaw.filters.blocks.not_block import Not_
from citeclaw.filters.blocks.route import Route
from citeclaw.filters.blocks.sequential import Sequential
from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.filters.runner")

RejectionList = list[tuple[PaperRecord, FilterOutcome]]


def _is_llm_layer(block: Any) -> bool:
    """True iff ``block`` is an LLMFilter (or a Not_-wrapped one).

    Sequential's dashboard branch uses this to decide whether the
    inner-bar tick comes from the dispatcher (LLM, async, multi-vote)
    or from apply_block itself in one shot (cheap synchronous filters).
    """
    if isinstance(block, LLMFilter):
        return True
    if isinstance(block, Not_) and isinstance(block.layer, LLMFilter):
        return True
    return False


# ---------------------------------------------------------------------------
# Per-block-type helpers
# ---------------------------------------------------------------------------


def _apply_sequential(
    papers: list[PaperRecord], block: Sequential, fctx: FilterContext,
) -> tuple[list[PaperRecord], RejectionList]:
    passed = list(papers)
    rejected: RejectionList = []
    dash = getattr(fctx.ctx, "dashboard", None)
    for layer in block.layers:
        if not passed:
            break
        in_count = len(passed)
        # Tell the dashboard which filter is running and how many papers
        # it's about to chew on. LLM filters run as atomic batched calls
        # — futures complete all-at-once per batch, so a fake N-paper
        # total just sits at 0 until the LLM responds and then snaps to
        # N. Pulse instead; complete_phase snaps total at end. Cheap
        # synchronous filters keep the determinate N-paper bar (they
        # tick per paper or in one shot after returning).
        if dash is not None:
            layer_name = getattr(layer, "name", type(layer).__name__)
            total: int | None = None if _is_llm_layer(layer) else in_count
            dash.begin_phase(layer_name, total=total)
        p, r = apply_block(passed, layer, fctx)
        passed = p
        rejected.extend(r)
        # Clamp the inner bar to its total regardless of who drove it.
        # Prevents A>B overshoot when a nested dispatcher (LLM / nested
        # Sequential / Any) has already ticked beyond the layer total.
        if dash is not None:
            dash.complete_phase()
    return passed, rejected


def _apply_any(
    papers: list[PaperRecord], block: Any_, fctx: FilterContext,
) -> tuple[list[PaperRecord], RejectionList]:
    decided_pass: list[PaperRecord] = []
    undecided = list(papers)
    last_rej: dict[str, FilterOutcome] = {}
    for layer in block.layers:
        if not undecided:
            break
        p, r = apply_block(undecided, layer, fctx)
        decided_pass.extend(p)
        for rec, outcome in r:
            last_rej[rec.paper_id] = outcome
        passed_ids = {rec.paper_id for rec in p}
        undecided = [x for x in undecided if x.paper_id not in passed_ids]
    rejected = [
        (
            x,
            last_rej.get(
                x.paper_id, FilterOutcome(False, "any: all failed", "any"),
            ),
        )
        for x in undecided
    ]
    return decided_pass, rejected


def _apply_not(
    papers: list[PaperRecord], block: Not_, fctx: FilterContext,
) -> tuple[list[PaperRecord], RejectionList]:
    inner_passed, _inner_rejected = apply_block(papers, block.layer, fctx)
    inner_passed_ids = {p.paper_id for p in inner_passed}
    new_passed = [p for p in papers if p.paper_id not in inner_passed_ids]
    new_rejected: RejectionList = [
        (
            p,
            FilterOutcome(
                False,
                f"not({block.layer.name})",
                f"not_{block.layer.name}",
            ),
        )
        for p in inner_passed
    ]
    return new_passed, new_rejected


def _apply_route(
    papers: list[PaperRecord], block: Route, fctx: FilterContext,
) -> tuple[list[PaperRecord], RejectionList]:
    groups: dict[int, tuple[Any, list[PaperRecord]]] = {}
    rejected: RejectionList = []
    for paper in papers:
        target = block.select(paper, fctx)
        if target is None:
            rejected.append(
                (paper, FilterOutcome(False, "no_route_match", "no_route_match"))
            )
            continue
        key = id(target)
        groups.setdefault(key, (target, []))[1].append(paper)
    passed: list[PaperRecord] = []
    for _, (target, plist) in groups.items():
        p, r = apply_block(plist, target, fctx)
        passed.extend(p)
        rejected.extend(r)
    return passed, rejected


def _apply_llm(
    papers: list[PaperRecord], block: LLMFilter, fctx: FilterContext,
) -> tuple[list[PaperRecord], RejectionList]:
    # Imported here so the lightweight filter atoms don't pull the
    # screening runner (and its async / token-budget machinery) at
    # module import time.
    from citeclaw.screening.llm_runner import dispatch_batch

    verdicts = dispatch_batch(papers, block, fctx.ctx)
    # Per-filter screening trace for HumanInTheLoop. ``screened`` is
    # every paper this filter actually ran on (so HITL only counts
    # toward agreement on papers the filter saw). ``accepted`` is the
    # per-filter accept set so HITL can sample LLM-accepted papers
    # specifically (rather than papers that survived only on cheap
    # rules upstream).
    cat = f"llm_{block.name}"
    ctx_obj = fctx.ctx
    screened_set = ctx_obj.papers_screened_by_filter.setdefault(cat, set())
    accepted_set = ctx_obj.papers_accepted_by_filter.setdefault(cat, set())
    for p in papers:
        screened_set.add(p.paper_id)
        if verdicts.get(p.paper_id, False):
            accepted_set.add(p.paper_id)
    passed = [p for p in papers if verdicts.get(p.paper_id, False)]
    rejected: RejectionList = [
        (p, FilterOutcome(False, f"llm:{block.name}", f"llm_{block.name}"))
        for p in papers if not verdicts.get(p.paper_id, False)
    ]
    return passed, rejected


def _apply_atom(
    papers: list[PaperRecord], block: Any, fctx: FilterContext,
) -> tuple[list[PaperRecord], RejectionList]:
    dash = getattr(fctx.ctx, "dashboard", None)

    # Bulk prefetch hook — e.g. SimilarityFilter warms embedding /
    # citation caches for all papers in one S2 call so per-paper
    # check() is cheap. Errors are non-fatal: per-paper compute falls
    # back to one-by-one fetches.
    prefetch = getattr(block, "prefetch", None)
    if callable(prefetch):
        try:
            prefetch(papers, fctx)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "filter %r prefetch failed: %s",
                getattr(block, "name", type(block).__name__), exc,
            )

    # If the atom offers a self-contained ``check_batch`` AND no prefetch
    # hook (so we can't interleave progress), fall through to batch mode.
    check_batch = getattr(block, "check_batch", None)
    if callable(check_batch) and not callable(prefetch):
        outcomes = check_batch(papers, fctx)
        passed = [p for p, o in zip(papers, outcomes) if o.passed]
        rejected: RejectionList = [
            (p, o) for p, o in zip(papers, outcomes) if not o.passed
        ]
        if dash is not None:
            dash.tick_inner(len(papers))
        return passed, rejected

    # Per-paper check with live inner-bar ticking. Used when the atom
    # exposes a prefetch hook, or when it only defines ``check``.
    passed: list[PaperRecord] = []
    rejected: RejectionList = []
    for paper in papers:
        outcome = block.check(paper, fctx)
        if outcome.passed:
            passed.append(paper)
        else:
            rejected.append((paper, outcome))
        if dash is not None:
            dash.tick_inner(1)
    return passed, rejected


# ---------------------------------------------------------------------------
# Dispatch entry points
# ---------------------------------------------------------------------------


def apply_block(
    papers: list[PaperRecord],
    block: Any,
    fctx: FilterContext,
) -> tuple[list[PaperRecord], RejectionList]:
    """Walk ``block`` over ``papers``; return ``(passed, rejected)``.

    Empty input short-circuits to ``([], [])``. Recursion is unbounded
    in principle but the YAML schema effectively caps it at ~5 levels
    of nesting (Sequential of Sequential of …) which is well under the
    default Python recursion limit.
    """
    if not papers:
        return [], []
    if isinstance(block, Sequential):
        return _apply_sequential(papers, block, fctx)
    if isinstance(block, Any_):
        return _apply_any(papers, block, fctx)
    if isinstance(block, Not_):
        return _apply_not(papers, block, fctx)
    if isinstance(block, Route):
        return _apply_route(papers, block, fctx)
    if isinstance(block, LLMFilter):
        return _apply_llm(papers, block, fctx)
    return _apply_atom(papers, block, fctx)


def record_rejections(
    rejected: RejectionList,
    fctx: FilterContext,
) -> None:
    """Flow ``rejected`` into the three context-level rejection sinks.

    Updates ``ctx.rejected`` (set of all rejected paper ids),
    ``ctx.rejection_counts`` (Counter by category for the dashboard),
    and ``ctx.rejection_ledger`` (per-paper category list so HITL can
    sample by reason).
    """
    ctx = fctx.ctx
    for paper, outcome in rejected:
        ctx.rejected.add(paper.paper_id)
        key = outcome.category or "unknown"
        ctx.rejection_counts[key] += 1
        ctx.rejection_ledger.setdefault(paper.paper_id, []).append(key)
