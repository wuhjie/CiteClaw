"""Iterative-search agent backend for ``ExpandBySearch`` (1-shot mode).

The full multi-iteration agent (diagnose → plan → write-next) is parked.
This shipped version runs **one** worker turn:

  1. Render :data:`citeclaw.prompts.search_refine.WORKER_SYSTEM` with the
     parent topic description as the sub-topic, plus
     :data:`WORKER_PROPOSE_FIRST` as the user message.
  2. Parse the worker's JSON ``{"query": "..."}`` reply.
  3. Translate the natural-language ``AND`` / ``OR`` / ``NOT`` operators
     to the S2 bulk-search syntax (``space`` / ``|`` / ``-``).
  4. Send one ``GET /paper/search/bulk`` request and collect ``paperId``s.

The verbatim multi-turn / supervisor prompts stay in
:mod:`citeclaw.prompts.search_refine` so the iterative path can be turned
back on without re-tuning prompts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from citeclaw.clients.llm.base import LLMClient
from citeclaw.prompts.search_refine import (
    WORKER_PROPOSE_FIRST,
    WORKER_SYSTEM,
)

log = logging.getLogger("citeclaw.agents.iterative_search")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Runtime knobs for :func:`run_iterative_search`.

    Built from the YAML ``agent:`` sub-dict on the ``ExpandBySearch`` step.
    Unknown keys are silently ignored so a richer future schema (used by
    the multi-turn supervisor) doesn't fail-build today's 1-shot pipeline.

    ``max_papers_per_iteration`` is the *total* cap on candidates per
    iteration, NOT the per-call limit (S2's ``/paper/search/bulk`` caps
    each call at 1000; we paginate using the ``token`` field until we
    hit this cap or exhaust the query's matches). Setting this above
    10K is fine but each extra page costs one S2 request, so 10K is a
    reasonable default for a single broad query.
    """

    max_iterations: int = 1
    max_papers_per_iteration: int = 10_000
    max_llm_tokens: int = 50_000
    model: str | None = None
    reasoning_effort: str | None = None
    sort: str | None = None  # None / "paperId" / "publicationDate[:asc|desc]" / "citationCount[:asc|desc]"


@dataclass
class AgentTurn:
    """One worker turn's record — kept for diagnostics / logging."""

    iteration: int
    raw_response: str
    query_natural: str
    query_s2: str
    total: int
    new_ids: list[str]


# ---------------------------------------------------------------------------
# Query-string translation: natural-language AND/OR/NOT → S2 bulk syntax
# ---------------------------------------------------------------------------


_KEYWORD_AND = re.compile(r"\bAND\b", re.IGNORECASE)
_KEYWORD_OR = re.compile(r"\bOR\b", re.IGNORECASE)
# NOT eats the following whitespace so ``NOT survey`` becomes ``-survey``
# (S2 requires ``-token`` with no space).
_KEYWORD_NOT = re.compile(r"\bNOT\s+", re.IGNORECASE)
_REPEATED_WS = re.compile(r"\s+")


def translate_query(query: str) -> str:
    """Translate the worker's natural-language Boolean query to S2 bulk syntax.

    S2 ``/paper/search/bulk`` accepts:

      - whitespace = AND (default conjunction)
      - ``|`` = OR
      - ``-token`` = NOT (must NOT contain)
      - ``"phrase"`` = exact phrase
      - ``(...)`` = grouping
      - ``token*`` = suffix wildcard

    The worker is instructed to write ``AND`` / ``OR`` / ``NOT`` as words
    plus quoted phrases and parens. We translate the words; quotes and
    parens pass through unchanged.
    """
    q = _KEYWORD_NOT.sub("-", query)
    q = _KEYWORD_OR.sub("|", q)
    q = _KEYWORD_AND.sub(" ", q)
    q = _REPEATED_WS.sub(" ", q).strip()
    return q


# ---------------------------------------------------------------------------
# JSON parsing (robust against fenced output / leading prose)
# ---------------------------------------------------------------------------


_CODE_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_CODE_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_QUERY_FIELD = re.compile(r'"query"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')


def parse_query_reply(raw: str) -> str:
    """Pull the ``query`` field out of a worker reply.

    Tries strict JSON first; falls back to a regex scan for the
    ``"query": "..."`` field so a leading-prose answer or a fenced block
    still gives us a usable query. Returns ``""`` on total failure — the
    caller logs and skips that iteration.
    """
    if not raw:
        return ""

    cleaned = raw.strip()
    cleaned = _CODE_FENCE_OPEN.sub("", cleaned)
    cleaned = _CODE_FENCE_CLOSE.sub("", cleaned)
    cleaned = cleaned.strip()

    # Path 1: parse the whole reply as JSON.
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        q = obj.get("query")
        if isinstance(q, str) and q.strip():
            return q.strip()

    # Path 2: find the first ``{...}`` block and try again.
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            obj = json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            q = obj.get("query")
            if isinstance(q, str) and q.strip():
                return q.strip()

    # Path 3: regex-scrape the field directly.
    m = _QUERY_FIELD.search(cleaned)
    if m:
        return m.group(1).strip()

    return ""


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


def run_iterative_search(
    *,
    topic: str,
    ctx: Any,
    llm: LLMClient,
    config: AgentConfig,
) -> tuple[list[str], list[AgentTurn]]:
    """Run the search agent and return (deduped paperIds, per-turn records).

    Currently capped at ``max_iterations`` turns; for the shipped 1-shot
    mode the cap is 1. Each turn:

      - Asks the LLM for a natural-language Boolean query.
      - Translates to S2 syntax and calls ``ctx.s2.search_bulk``.
      - Adds every novel ``paperId`` to the aggregate.

    The aggregate is returned in iteration order; the caller is
    responsible for screening it through the standard ExpandBy*
    pipeline (``screen_expand_candidates``).
    """
    aggregate: list[str] = []
    seen_pids: set[str] = set()
    turns: list[AgentTurn] = []

    if not topic.strip():
        log.warning(
            "ExpandBySearch agent: empty topic_description — refusing to query",
        )
        return aggregate, turns

    system_msg = WORKER_SYSTEM.format(description=topic)

    for i in range(max(1, config.max_iterations)):
        user_msg = WORKER_PROPOSE_FIRST.format(description=topic)

        # response_schema is intentionally NOT passed: xAI's grok-4.20
        # reasoning model truncates to 7 visible tokens (~``{"query": "((``)
        # when ``response_format=json_schema`` is combined with a long
        # system prompt, regardless of ``max_tokens``. The verbatim
        # prompt already asks for ``{"query": "..."}`` and parse_query_reply
        # is robust to fenced / prose / extra-text replies, so the schema
        # buys us nothing on the providers that work and breaks the one
        # provider that doesn't.
        try:
            response = llm.call(
                system_msg,
                user_msg,
                category="search_agent",
            )
        except Exception as exc:  # noqa: BLE001 — keep the pipeline alive
            log.warning(
                "ExpandBySearch agent: LLM call failed on iter %d: %s",
                i + 1, exc,
            )
            break

        raw = (response.text or "").strip()
        query_natural = parse_query_reply(raw)
        if not query_natural:
            log.warning(
                "ExpandBySearch agent: could not parse query from LLM reply "
                "on iter %d (raw=%r)",
                i + 1, raw[:200],
            )
            break

        query_s2 = translate_query(query_natural)
        if not query_s2:
            log.warning(
                "ExpandBySearch agent: empty query after translation on iter %d "
                "(natural=%r)",
                i + 1, query_natural[:200],
            )
            break

        log.info(
            "ExpandBySearch agent iter %d: %r -> %r",
            i + 1, query_natural[:120], query_s2[:120],
        )

        # Paginate through S2 ``/paper/search/bulk`` (1000 results per
        # page; ``token`` continues to the next page). We stop when we
        # hit ``max_papers_per_iteration``, exhaust the query's matches,
        # or get an error.
        cap = max(1, int(config.max_papers_per_iteration))
        per_page = 1000  # S2 server-side max per call
        next_token: str | None = None
        page_num = 0
        new_ids: list[str] = []
        total = 0
        broke_on_error = False

        while True:
            try:
                result = ctx.s2.search_bulk(
                    query_s2,
                    limit=per_page,
                    token=next_token,
                    sort=config.sort,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "ExpandBySearch agent: search_bulk failed on iter %d page %d: %s",
                    i + 1, page_num + 1, exc,
                )
                broke_on_error = True
                break

            if not isinstance(result, dict):
                break

            page_data = result.get("data") or []
            if page_num == 0 and isinstance(result.get("total"), int):
                total = int(result["total"])

            page_added = 0
            for entry in page_data:
                if not isinstance(entry, dict):
                    continue
                pid = entry.get("paperId")
                if isinstance(pid, str) and pid and pid not in seen_pids:
                    seen_pids.add(pid)
                    aggregate.append(pid)
                    new_ids.append(pid)
                    page_added += 1
                if len(new_ids) >= cap:
                    break

            page_num += 1
            log.info(
                "ExpandBySearch agent iter %d page %d: +%d ids (running total %d / cap %d, S2 total %d)",
                i + 1, page_num, page_added, len(new_ids), cap, total,
            )

            if len(new_ids) >= cap:
                log.info(
                    "ExpandBySearch agent iter %d: hit cap %d, stopping pagination",
                    i + 1, cap,
                )
                break

            next_token = result.get("token") if isinstance(result, dict) else None
            if not next_token:
                break  # No more pages.
            if not page_data:
                break  # Defensive: empty page with a token = end.

        turns.append(AgentTurn(
            iteration=i + 1, raw_response=raw,
            query_natural=query_natural, query_s2=query_s2,
            total=total, new_ids=new_ids,
        ))
        log.info(
            "ExpandBySearch agent iter %d: matched=%d on S2; fetched=%d across %d pages; aggregate=%d",
            i + 1, total, len(new_ids), page_num, len(aggregate),
        )

        if broke_on_error:
            break

    return aggregate, turns
