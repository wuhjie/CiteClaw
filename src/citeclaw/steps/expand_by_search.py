"""ExpandBySearch — agent-driven S2 bulk-search expansion (1-shot mode).

The full multi-iteration / supervisor agent is parked; this revival
wires a minimal 1-turn worker so the step is functional end-to-end:

  1. Resolve the topic description (step override -> top-level config).
  2. Build the LLM client (``self.agent.model`` -> ``config.search_model``
     -> ``config.screening_model``).
  3. Call :func:`citeclaw.agents.iterative_search.run_iterative_search`
     with ``max_iterations`` clamped per ``self.agent`` (default 1).
  4. Feed the deduped paperIds through the shared ExpandBy* screening
     pipeline (``screen_expand_candidates``).

The verbatim multi-iteration prompts live in
:mod:`citeclaw.prompts.search_refine` so flipping the iteration cap up
in the agent backend turns the longer flow back on without prompt
changes.
"""

from __future__ import annotations

import logging
from typing import Any

from citeclaw.agents.iterative_search import (
    AgentConfig,
    AgentTurn,
    run_iterative_search,
)
from citeclaw.clients.llm.factory import build_llm_client
from citeclaw.models import PaperRecord
from citeclaw.search.query_engine import apply_local_query
from citeclaw.steps._expand_helpers import (
    fingerprint_signal,
    screen_expand_candidates,
)
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_by_search")


# Cap the agent at one turn for the shipped V3 revival. The multi-turn
# loop and the supervisor stay in prompts/search_refine.py for the next
# iteration of this step but are deliberately not exercised today —
# tuning the refinement chain regressed retrieval quality in practice,
# so 1-shot ships first.
_DEFAULT_MAX_ITERATIONS = 1


class ExpandBySearch:
    """Meta-LLM search expansion (1-shot worker)."""

    name = "ExpandBySearch"

    def __init__(
        self,
        *,
        agent: dict[str, Any] | None = None,
        screener: Any = None,
        topic_description: str | None = None,
        max_anchor_papers: int = 20,
        apply_local_query_args: dict[str, Any] | None = None,
    ) -> None:
        self.agent = dict(agent or {})
        self.screener = screener
        self.topic_description = topic_description
        self.max_anchor_papers = max_anchor_papers
        self.apply_local_query_args = apply_local_query_args or None

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        # ExpandBySearch is an *augmentation* step, not a *traversal* one
        # (unlike ExpandForward / ExpandBackward, which consume their
        # input as the set of papers to expand from). Search uses
        # topic_description, NOT signal, to look up new candidates — so
        # every early-exit path must pass ``signal`` through unchanged.
        # Otherwise a 0-hit search (or a rejected agent reply / cached
        # fingerprint) erases the seeds and kills the downstream
        # snowball with an empty input.
        if self.screener is None:
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "no screener"},
            )

        fp = self._fingerprint(signal)
        if fp in ctx.searched_signals:
            log.info(
                "%s: signal fingerprint already searched, passing input through",
                self.name,
            )
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "already_searched", "fingerprint": fp[:12]},
            )

        topic = self._resolve_topic(ctx)
        if not topic.strip():
            log.warning(
                "ExpandBySearch: topic_description is empty (step override + "
                "config.topic_description both blank) — skipping",
            )
            ctx.searched_signals.add(fp)
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "no_topic_description"},
            )

        agent_cfg = self._build_agent_config(ctx)
        llm = build_llm_client(
            ctx.config,
            ctx.budget,
            model=agent_cfg.model,
            reasoning_effort=agent_cfg.reasoning_effort,
            cache=getattr(ctx, "cache", None),
        )

        aggregate_ids, turns = run_iterative_search(
            topic=topic,
            ctx=ctx,
            llm=llm,
            config=agent_cfg,
        )

        extra_stats: dict[str, Any] = {
            "agent_iterations": len(turns),
            "anchor_count": min(len(signal), self.max_anchor_papers),
            "agent_paperids": len(aggregate_ids),
        }
        if turns:
            extra_stats["last_query"] = turns[-1].query_s2[:200]

        if not aggregate_ids:
            ctx.searched_signals.add(fp)
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "no_results", **extra_stats},
            )

        return self._screen_and_finalize(
            aggregate_ids=aggregate_ids,
            signal=signal,
            ctx=ctx,
            fp=fp,
            extra_stats=extra_stats,
        )

    # ---- Config helpers ---------------------------------------------

    def _build_agent_config(self, ctx) -> AgentConfig:
        """Translate the YAML ``agent:`` dict into an :class:`AgentConfig`.

        Resolution cascade for the model alias:
        ``self.agent['model']`` -> ``ctx.config.search_model`` ->
        ``ctx.config.screening_model``.

        Unknown keys in ``self.agent`` are silently ignored.
        """
        model = (
            self.agent.get("model")
            or getattr(ctx.config, "search_model", "")
            or getattr(ctx.config, "screening_model", "")
            or None
        )
        reasoning_effort = (
            self.agent.get("reasoning_effort")
            or getattr(ctx.config, "reasoning_effort", "")
            or None
        )
        max_iter = int(self.agent.get("max_iterations", _DEFAULT_MAX_ITERATIONS))
        # Hard-cap iterations regardless of YAML so a stale config doesn't
        # accidentally turn the still-experimental multi-turn loop back on.
        max_iter = max(1, min(max_iter, _DEFAULT_MAX_ITERATIONS))
        return AgentConfig(
            max_iterations=max_iter,
            max_papers_per_iteration=int(
                self.agent.get("max_papers_per_iteration", 10_000),
            ),
            max_llm_tokens=int(self.agent.get("max_llm_tokens", 50_000)),
            model=model,
            reasoning_effort=reasoning_effort,
            sort=self.agent.get("sort") or None,
        )

    def _fingerprint(self, signal: list[PaperRecord]) -> str:
        return fingerprint_signal(
            self.name, signal,
            agent=self.agent,
            max_anchor_papers=self.max_anchor_papers,
            topic=self.topic_description or "",
        )

    def _resolve_topic(self, ctx) -> str:
        return (
            self.topic_description
            or getattr(ctx.config, "topic_description", "")
            or ""
        )

    def _screen_and_finalize(
        self,
        *,
        aggregate_ids: list[str],
        signal: list[PaperRecord],
        ctx,
        fp: str,
        extra_stats: dict[str, Any] | None = None,
    ) -> StepResult:
        raw_hits = [{"paperId": pid} for pid in aggregate_ids]
        post_trim = (
            (lambda recs: apply_local_query(recs, **self.apply_local_query_args))
            if self.apply_local_query_args
            else None
        )
        screened = screen_expand_candidates(
            raw_hits=raw_hits,
            source_label="search",
            screener=self.screener,
            ctx=ctx,
            post_hydrate_fn=post_trim,
        )
        ctx.searched_signals.add(fp)
        # Pass input signal through unchanged and APPEND the newly-screened
        # search hits — see the comment in run() for why augmentation steps
        # must not consume their input.
        return StepResult(
            signal=list(signal) + screened.passed,
            in_count=len(screened.hydrated),
            stats={**screened.base_stats, **(extra_stats or {})},
        )
