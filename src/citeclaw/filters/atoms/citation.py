"""CitationFilter — accept papers whose citation count outpaces ``beta × age``."""

from __future__ import annotations

from datetime import datetime

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


def _current_year() -> int:
    """Wall-clock year — extracted so tests can monkeypatch it deterministically."""
    return datetime.now().year


class CitationFilter:
    """Reject papers whose citation count is below ``beta × years_since_publication``.

    The age-aware threshold (``beta × age``) penalises young papers far less
    than old ones, so a 2-year-old preprint with 60 citations passes a
    ``beta=30`` bar that a 10-year-old paper would need 300 citations to
    clear. ``years_since`` is computed against ``reference_year`` (defaulting
    to the current calendar year) and floored at 1 so brand-new papers still
    have a non-zero bar.

    Two knobs cover the recency-skip case:

    * ``reference_year`` — anchor year for both the age math and the
      exemption window. Defaults to the current calendar year. Pin it
      to keep results reproducible across runs that span a year
      boundary.
    * ``exemption_years`` — papers from the last ``N+1`` years skip
      the citation check. The semantics work as follows:

        * ``N = 0`` (the default): only papers published in
          ``reference_year`` itself skip — current-calendar-year work
          almost always has zero citations on S2, so without this the
          SPECTER2 / search-agent expansion has nothing to feed into
          the citation gate.
        * ``N = 1``: ``reference_year`` AND ``reference_year - 1`` skip
          (i.e. last-two-years window).
        * ``N = -1``: disable the exemption entirely — every paper
          must clear ``years_since * beta``, matching the original
          pre-2026-05 strict behaviour. Use this when you genuinely
          want to filter out brand-new work that hasn't earned its
          citations yet.

      Note that the implementation accepts any integer; values < -1 are
      treated the same as -1 (no exemption).
    """

    def __init__(
        self,
        name: str = "citation",
        *,
        beta: float = 5.0,
        exemption_years: int = 0,
        reference_year: int | None = None,
    ) -> None:
        self.name = name
        self._beta = beta
        self._exemption_years = exemption_years
        self._reference_year = reference_year

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        """Evaluate ``paper`` in three stages.

        1. Recency exemption — if configured and the paper falls inside
           the window, pass without consulting the citation count.
        2. Missing-data reject — papers with no ``citation_count`` reject
           in the ``missing_data`` category, distinct from a real
           citation-floor failure.
        3. Threshold — ``citation_count >= max(anchor - paper.year, 1) * beta``.
           Papers with ``year is None`` are effectively treated as
           ancient (``years_since = anchor``), so only extreme citation
           counts can pass.
        """
        anchor = self._reference_year if self._reference_year is not None else _current_year()
        if (
            self._exemption_years >= 0
            and paper.year is not None
            and paper.year >= anchor - self._exemption_years
        ):
            return PASS
        if paper.citation_count is None:
            return FilterOutcome(False, "missing citation count", "missing_data")
        years_since = max(anchor - (paper.year or 0), 1)
        threshold = years_since * self._beta
        if paper.citation_count < threshold:
            return FilterOutcome(
                False,
                f"cit {paper.citation_count} < {threshold:.0f} (β={self._beta})",
                "citation",
            )
        return PASS
