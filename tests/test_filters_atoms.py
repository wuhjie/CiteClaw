"""Tests for atom filters: YearFilter, CitationFilter, LLMFilter, predicates."""

from __future__ import annotations

from datetime import datetime

import pytest

from citeclaw.filters.atoms.citation import CitationFilter
from citeclaw.filters.atoms.keyword import (
    AbstractKeywordFilter,
    TitleKeywordFilter,
    VenueKeywordFilter,
)
from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.filters.atoms.predicates import CitAtLeast, VenueIn, VenuePreset, YearAtLeast
from citeclaw.filters.atoms.year import YearFilter
from citeclaw.filters.base import FilterContext
from citeclaw.models import PaperRecord


def _fctx(ctx):
    return FilterContext(ctx=ctx)


# ---------------------------------------------------------------------------
# YearFilter
# ---------------------------------------------------------------------------


class TestYearFilter:
    def test_in_range(self, ctx):
        f = YearFilter(min=2020, max=2025)
        assert f.check(PaperRecord(paper_id="p", year=2022), _fctx(ctx)).passed

    def test_below_min(self, ctx):
        f = YearFilter(min=2020)
        out = f.check(PaperRecord(paper_id="p", year=2010), _fctx(ctx))
        assert not out.passed
        assert out.category == "year"

    def test_above_max(self, ctx):
        f = YearFilter(max=2020)
        out = f.check(PaperRecord(paper_id="p", year=2023), _fctx(ctx))
        assert not out.passed

    def test_missing_year_rejected(self, ctx):
        f = YearFilter(min=2020)
        out = f.check(PaperRecord(paper_id="p", year=None), _fctx(ctx))
        assert not out.passed
        assert "None" in out.reason

    def test_no_bounds_is_permissive(self, ctx):
        f = YearFilter()
        assert f.check(PaperRecord(paper_id="p", year=1900), _fctx(ctx)).passed
        # …but year=None still fails (it's a hard-data check).
        assert not f.check(PaperRecord(paper_id="p", year=None), _fctx(ctx)).passed


# ---------------------------------------------------------------------------
# CitationFilter
# ---------------------------------------------------------------------------


class TestCitationFilter:
    def test_missing_citation_count(self, ctx):
        f = CitationFilter(beta=5)
        out = f.check(PaperRecord(paper_id="p", year=2020), _fctx(ctx))
        assert not out.passed
        assert out.category == "missing_data"

    def test_very_high_citation_always_passes(self, ctx):
        f = CitationFilter(beta=5)
        # Old paper with massive citation count
        p = PaperRecord(paper_id="p", year=1990, citation_count=100_000)
        assert f.check(p, _fctx(ctx)).passed

    def test_low_citation_rejected(self, ctx):
        """A paper from N years ago needs ``N * beta`` citations."""
        f = CitationFilter(beta=50)
        # 5-year-old paper with only a few citations — 5 * 50 = 250 threshold.
        # Current-year papers would be exempt under the new default, so the
        # test pins to an older year to actually hit the citation gate.
        p = PaperRecord(paper_id="p", year=datetime.now().year - 5, citation_count=1)
        out = f.check(p, _fctx(ctx))
        assert not out.passed
        assert out.category == "citation"

    def test_years_floor_at_one(self, ctx):
        """A paper from the current year uses ``max(diff, 1)`` so a single
        citation isn't held to a 0-threshold bar. Also pin the exemption off
        so the citation threshold actually runs."""
        f = CitationFilter(beta=1, exemption_years=-1)
        p = PaperRecord(paper_id="p", year=datetime.now().year, citation_count=1)
        assert f.check(p, _fctx(ctx)).passed

    def test_year_none_uses_full_age(self, ctx):
        """If the paper has no year, it's treated as ancient — huge threshold.
        (Real-world: a PaperRecord with no year probably has no citation count
        either, but the filter is defensive.)"""
        f = CitationFilter(beta=5)
        p = PaperRecord(paper_id="p", year=None, citation_count=10)
        out = f.check(p, _fctx(ctx))
        assert not out.passed

    def test_exemption_zero_passes_anchor_year(self, ctx):
        """exemption_years=0 lets papers from reference_year skip the check."""
        f = CitationFilter(beta=50, exemption_years=0, reference_year=2026)
        p = PaperRecord(paper_id="p", year=2026, citation_count=0)
        assert f.check(p, _fctx(ctx)).passed

    def test_exemption_zero_does_not_pass_prior_year(self, ctx):
        f = CitationFilter(beta=50, exemption_years=0, reference_year=2026)
        p = PaperRecord(paper_id="p", year=2025, citation_count=1)
        out = f.check(p, _fctx(ctx))
        assert not out.passed
        assert out.category == "citation"

    def test_exemption_one_passes_prior_year(self, ctx):
        """exemption_years=1, reference_year=2026 → 2025+ are exempt."""
        f = CitationFilter(beta=50, exemption_years=1, reference_year=2026)
        p = PaperRecord(paper_id="p", year=2025, citation_count=0)
        assert f.check(p, _fctx(ctx)).passed

    def test_exemption_one_rejects_two_years_back(self, ctx):
        f = CitationFilter(beta=50, exemption_years=1, reference_year=2026)
        p = PaperRecord(paper_id="p", year=2024, citation_count=1)
        out = f.check(p, _fctx(ctx))
        assert not out.passed

    def test_exemption_default_zero_exempts_current_year(self, ctx):
        """Default exemption_years=0 exempts current-year papers.

        Default flipped 2026-05 from None (strict) to 0 because SPECTER2
        recommendations and broad search agents almost always return
        current-year papers with 0 citations — under the old default the
        citation gate killed 78% of legitimate on-topic candidates."""
        f = CitationFilter(beta=50, reference_year=2026)
        p = PaperRecord(paper_id="p", year=2026, citation_count=1)
        out = f.check(p, _fctx(ctx))
        assert out.passed

    def test_exemption_negative_one_means_strict(self, ctx):
        """exemption_years=-1 opts BACK IN to the pre-2026-05 strict
        behaviour: even current-year papers must clear beta."""
        f = CitationFilter(beta=50, exemption_years=-1, reference_year=2026)
        p = PaperRecord(paper_id="p", year=2026, citation_count=1)
        out = f.check(p, _fctx(ctx))
        assert not out.passed
        assert out.category == "citation"

    def test_reference_year_used_for_age_math(self, ctx):
        """Pinning reference_year affects threshold = (ref - year) * beta."""
        f = CitationFilter(beta=10, reference_year=2026)
        # 2020 paper, anchor 2026 → 6-year-old → threshold = 60.
        p_pass = PaperRecord(paper_id="p", year=2020, citation_count=60)
        assert f.check(p_pass, _fctx(ctx)).passed
        p_fail = PaperRecord(paper_id="p", year=2020, citation_count=59)
        assert not f.check(p_fail, _fctx(ctx)).passed


# ---------------------------------------------------------------------------
# LLMFilter (stub dispatch path)
# ---------------------------------------------------------------------------


class TestLLMFilter:
    def test_scope_validation(self):
        with pytest.raises(ValueError):
            LLMFilter(scope="abstract")

    def test_model_and_reasoning_defaults(self):
        f = LLMFilter(scope="title", prompt="x")
        assert f.model is None
        assert f.reasoning_effort is None

    def test_model_and_reasoning_stored(self):
        f = LLMFilter(
            scope="title", prompt="x",
            model="gpt-4o", reasoning_effort="medium",
        )
        assert f.model == "gpt-4o"
        assert f.reasoning_effort == "medium"

    def test_votes_defaults(self):
        f = LLMFilter(scope="title", prompt="x")
        assert f.votes == 1
        assert f.min_accepts == 1

    def test_votes_stored(self):
        f = LLMFilter(scope="title", prompt="x", votes=5, min_accepts=3)
        assert f.votes == 5
        assert f.min_accepts == 3

    def test_votes_validation_zero(self):
        with pytest.raises(ValueError, match="votes must be >= 1"):
            LLMFilter(scope="title", prompt="x", votes=0)

    def test_min_accepts_validation_zero(self):
        with pytest.raises(ValueError, match="min_accepts must be >= 1"):
            LLMFilter(scope="title", prompt="x", min_accepts=0)

    def test_min_accepts_validation_too_high(self):
        with pytest.raises(ValueError, match="cannot exceed votes"):
            LLMFilter(scope="title", prompt="x", votes=3, min_accepts=4)


class TestLLMFilterFormulaMode:
    def test_formula_parsed_eagerly(self):
        f = LLMFilter(
            scope="title",
            formula="(q1 | q2) & !q3",
            queries={"q1": "is ml", "q2": "is stats", "q3": "is survey"},
        )
        assert f.formula_expr == "(q1 | q2) & !q3"
        assert f._formula is not None
        assert f.queries == {"q1": "is ml", "q2": "is stats", "q3": "is survey"}

    def test_formula_without_queries_raises(self):
        with pytest.raises(ValueError, match="requires a non-empty 'queries'"):
            LLMFilter(scope="title", formula="q1")

    def test_queries_without_formula_raises(self):
        with pytest.raises(ValueError, match="only valid together with 'formula'"):
            LLMFilter(scope="title", queries={"q1": "is ml"})

    def test_formula_and_prompt_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            LLMFilter(
                scope="title", prompt="x",
                formula="q1", queries={"q1": "is ml"},
            )

    def test_formula_undefined_query_name_raises(self):
        with pytest.raises(ValueError, match="undefined queries"):
            LLMFilter(
                scope="title",
                formula="q1 & q_missing",
                queries={"q1": "is ml"},
            )

    def test_formula_with_malformed_expression_raises(self):
        with pytest.raises(ValueError, match="bad formula"):
            LLMFilter(
                scope="title",
                formula="q1 &",
                queries={"q1": "is ml"},
            )

    def test_formula_defaults_single_prompt_mode(self):
        """A filter without formula/queries is in single-prompt mode."""
        f = LLMFilter(scope="title", prompt="is ml")
        assert f.formula_expr is None
        assert f._formula is None
        assert f.queries == {}

    def test_formula_tolerates_extra_unused_queries(self, caplog):
        """Extra queries that the formula doesn't reference log a warning
        but don't raise — users may iterate on the formula without
        cleaning up the queries dict every time."""
        with caplog.at_level("WARNING", logger="citeclaw.filters.atoms.llm_query"):
            LLMFilter(
                scope="title",
                formula="q1",
                queries={"q1": "is ml", "q_unused": "is stats"},
            )
        assert any("unused sub-queries" in r.message for r in caplog.records)

    def test_content_for_title(self):
        f = LLMFilter(scope="title", prompt="x")
        p = PaperRecord(paper_id="p", title="Hello", abstract="World")
        assert f.content_for(p) == "Hello"

    def test_content_for_venue(self):
        f = LLMFilter(scope="venue", prompt="x")
        p = PaperRecord(paper_id="p", venue="Nature", title="T")
        assert f.content_for(p) == "Nature"

    def test_content_for_title_abstract(self):
        f = LLMFilter(scope="title_abstract", prompt="x")
        p = PaperRecord(paper_id="p", title="T", abstract="A")
        content = f.content_for(p)
        assert "Title: T" in content
        assert "Abstract: A" in content

    def test_content_for_title_abstract_handles_missing(self):
        f = LLMFilter(scope="title_abstract", prompt="x")
        p = PaperRecord(paper_id="p", title="T")
        assert "(no abstract)" in f.content_for(p)

    def test_check_dispatches_via_stub(self, ctx):
        """The stub LLM client says ``match: true`` for everything, so
        check() should return passed=True."""
        f = LLMFilter(scope="title", prompt="relevant")
        p = PaperRecord(paper_id="p", title="Anything")
        out = f.check(p, _fctx(ctx))
        assert out.passed


# ---------------------------------------------------------------------------
# Predicates (VenueIn, CitAtLeast, YearAtLeast)
# ---------------------------------------------------------------------------


class TestVenueIn:
    def test_match_substring(self, ctx):
        pred = VenueIn(values=["arXiv", "bioRxiv"])
        p = PaperRecord(paper_id="p", venue="arXiv preprint")
        assert pred.check(p, _fctx(ctx)).passed

    def test_case_insensitive(self, ctx):
        pred = VenueIn(values=["ARXIV"])
        p = PaperRecord(paper_id="p", venue="arxiv.org")
        assert pred.check(p, _fctx(ctx)).passed

    def test_no_match(self, ctx):
        pred = VenueIn(values=["Nature"])
        p = PaperRecord(paper_id="p", venue="Cell")
        assert not pred.check(p, _fctx(ctx)).passed

    def test_empty_venue(self, ctx):
        pred = VenueIn(values=["arXiv"])
        p = PaperRecord(paper_id="p", venue=None)
        assert not pred.check(p, _fctx(ctx)).passed


class TestVenuePreset:
    def test_nature_family_match(self, ctx):
        pred = VenuePreset(presets=["nature"])
        assert pred.check(
            PaperRecord(paper_id="p", venue="Nature Chemistry"), _fctx(ctx)
        ).passed
        assert pred.check(
            PaperRecord(paper_id="p", venue="Nature"), _fctx(ctx)
        ).passed

    def test_case_insensitive(self, ctx):
        pred = VenuePreset(presets=["science"])
        assert pred.check(
            PaperRecord(paper_id="p", venue="SCIENCE ADVANCES"), _fctx(ctx)
        ).passed

    def test_whitespace_normalized(self, ctx):
        pred = VenuePreset(presets=["cell"])
        assert pred.check(
            PaperRecord(paper_id="p", venue="  Cell   Reports  "), _fctx(ctx)
        ).passed

    def test_exact_match_rejects_false_positive(self, ctx):
        # Substring match on "Nature" would false-positive here; exact
        # match must reject.
        pred = VenuePreset(presets=["nature"])
        out = pred.check(
            PaperRecord(paper_id="p", venue="Nature-Inspired Computing"),
            _fctx(ctx),
        )
        assert not out.passed
        assert out.category == "venue_preset"

    def test_no_match(self, ctx):
        pred = VenuePreset(presets=["nature", "science"])
        assert not pred.check(
            PaperRecord(paper_id="p", venue="Journal of Chemical Physics"),
            _fctx(ctx),
        ).passed

    def test_empty_venue(self, ctx):
        pred = VenuePreset(presets=["nature"])
        assert not pred.check(
            PaperRecord(paper_id="p", venue=None), _fctx(ctx)
        ).passed

    def test_preprint_preset(self, ctx):
        pred = VenuePreset(presets=["preprint"])
        for v in ("arXiv", "bioRxiv", "medRxiv", "ChemRxiv"):
            assert pred.check(
                PaperRecord(paper_id="p", venue=v), _fctx(ctx)
            ).passed, v

    def test_multiple_presets_union(self, ctx):
        pred = VenuePreset(presets=["nature", "cell"])
        assert pred.check(
            PaperRecord(paper_id="p", venue="Nature Methods"), _fctx(ctx)
        ).passed
        assert pred.check(
            PaperRecord(paper_id="p", venue="Cell Systems"), _fctx(ctx)
        ).passed

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown venue preset"):
            VenuePreset(presets=["nonexistent_preset"])


class TestCitAtLeast:
    def test_threshold(self, ctx):
        pred = CitAtLeast(n=100)
        assert pred.check(PaperRecord(paper_id="p", citation_count=150), _fctx(ctx)).passed
        assert not pred.check(PaperRecord(paper_id="p", citation_count=50), _fctx(ctx)).passed
        assert not pred.check(PaperRecord(paper_id="p", citation_count=None), _fctx(ctx)).passed


class TestYearAtLeast:
    def test_threshold(self, ctx):
        pred = YearAtLeast(n=2020)
        assert pred.check(PaperRecord(paper_id="p", year=2021), _fctx(ctx)).passed
        assert not pred.check(PaperRecord(paper_id="p", year=2019), _fctx(ctx)).passed
        assert not pred.check(PaperRecord(paper_id="p", year=None), _fctx(ctx)).passed


# ---------------------------------------------------------------------------
# TitleKeywordFilter
# ---------------------------------------------------------------------------


class TestTitleKeywordFilter:
    def test_simple_match(self, ctx):
        f = TitleKeywordFilter(keyword="deep learning")
        p = PaperRecord(paper_id="p", title="Deep Learning for Biology")
        assert f.check(p, _fctx(ctx)).passed

    def test_simple_no_match(self, ctx):
        f = TitleKeywordFilter(keyword="quantum")
        p = PaperRecord(paper_id="p", title="Deep Learning for Biology")
        out = f.check(p, _fctx(ctx))
        assert not out.passed
        assert out.category == "title_keyword"
        assert "quantum" in out.reason

    def test_case_insensitive_default(self, ctx):
        f = TitleKeywordFilter(keyword="DEEP LEARNING")
        p = PaperRecord(paper_id="p", title="deep learning works")
        assert f.check(p, _fctx(ctx)).passed

    def test_case_sensitive_opt(self, ctx):
        f = TitleKeywordFilter(keyword="DEEP", case_sensitive=True)
        p = PaperRecord(paper_id="p", title="deep learning")
        assert not f.check(p, _fctx(ctx)).passed
        p2 = PaperRecord(paper_id="p", title="DEEP learning")
        assert f.check(p2, _fctx(ctx)).passed

    def test_whole_word(self, ctx):
        f = TitleKeywordFilter(keyword="learn", match="whole_word")
        # 'learn' is a substring of 'learning' but not a standalone word
        assert not f.check(
            PaperRecord(paper_id="p", title="deep learning"), _fctx(ctx)
        ).passed
        assert f.check(
            PaperRecord(paper_id="p", title="we learn fast"), _fctx(ctx)
        ).passed

    def test_starts_with(self, ctx):
        f = TitleKeywordFilter(keyword="Survey", match="starts_with")
        assert f.check(
            PaperRecord(paper_id="p", title="Survey of deep learning"), _fctx(ctx)
        ).passed
        assert f.check(
            PaperRecord(paper_id="p", title="survey of methods"), _fctx(ctx)
        ).passed  # case-insensitive by default
        assert not f.check(
            PaperRecord(paper_id="p", title="A Survey of deep learning"), _fctx(ctx)
        ).passed
        # Word boundary defends against prefix-substring traps
        assert not f.check(
            PaperRecord(paper_id="p", title="Surveying methods"), _fctx(ctx)
        ).passed

    def test_invalid_match_mode_raises(self):
        with pytest.raises(ValueError, match="'match' must be one of"):
            TitleKeywordFilter(keyword="x", match="bogus")

    def test_empty_title_rejects(self, ctx):
        f = TitleKeywordFilter(keyword="ml")
        assert not f.check(PaperRecord(paper_id="p", title=""), _fctx(ctx)).passed

    def test_formula_and(self, ctx):
        f = TitleKeywordFilter(
            formula="dl & bio",
            keywords={"dl": "deep learning", "bio": "biology"},
        )
        assert f.check(
            PaperRecord(paper_id="p", title="Deep Learning for Biology"),
            _fctx(ctx),
        ).passed
        assert not f.check(
            PaperRecord(paper_id="p", title="Deep Learning Survey"), _fctx(ctx)
        ).passed

    def test_formula_or(self, ctx):
        f = TitleKeywordFilter(
            formula="ml | rl",
            keywords={"ml": "machine learning", "rl": "reinforcement learning"},
        )
        assert f.check(
            PaperRecord(paper_id="p", title="Machine Learning Methods"),
            _fctx(ctx),
        ).passed
        assert f.check(
            PaperRecord(paper_id="p", title="Reinforcement Learning Agents"),
            _fctx(ctx),
        ).passed
        assert not f.check(
            PaperRecord(paper_id="p", title="Quantum Computing"), _fctx(ctx)
        ).passed

    def test_formula_not(self, ctx):
        f = TitleKeywordFilter(
            formula="ml & !survey",
            keywords={"ml": "machine learning", "survey": "survey"},
        )
        assert f.check(
            PaperRecord(paper_id="p", title="Machine Learning Methods"),
            _fctx(ctx),
        ).passed
        assert not f.check(
            PaperRecord(paper_id="p", title="A Survey of Machine Learning"),
            _fctx(ctx),
        ).passed

    def test_formula_complex(self, ctx):
        f = TitleKeywordFilter(
            formula="(dl | transformer) & !survey",
            keywords={
                "dl": "deep learning",
                "transformer": "transformer",
                "survey": "survey",
            },
        )
        assert f.check(
            PaperRecord(paper_id="p", title="Transformer Networks"), _fctx(ctx)
        ).passed
        assert not f.check(
            PaperRecord(paper_id="p", title="Survey of Transformers"), _fctx(ctx)
        ).passed
        assert not f.check(
            PaperRecord(paper_id="p", title="Quantum Computing"), _fctx(ctx)
        ).passed

    def test_no_keyword_or_formula_raises(self):
        with pytest.raises(ValueError, match="provide either"):
            TitleKeywordFilter()

    def test_both_keyword_and_formula_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            TitleKeywordFilter(keyword="x", formula="a", keywords={"a": "a"})

    def test_keywords_without_formula_raises(self):
        with pytest.raises(ValueError, match="only valid together with"):
            TitleKeywordFilter(keyword="x", keywords={"a": "a"})

    def test_formula_missing_keywords_raises(self):
        with pytest.raises(ValueError, match="undefined keywords"):
            TitleKeywordFilter(formula="a & b", keywords={"a": "x"})

    def test_empty_keyword_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            TitleKeywordFilter(keyword="")
        with pytest.raises(ValueError, match="non-empty string"):
            TitleKeywordFilter(keyword="   ")

    def test_empty_keyword_value_in_formula_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            TitleKeywordFilter(formula="a", keywords={"a": "  "})

    def test_bad_formula_raises(self):
        with pytest.raises(ValueError, match="bad formula"):
            TitleKeywordFilter(formula="a &", keywords={"a": "x"})


# ---------------------------------------------------------------------------
# AbstractKeywordFilter
# ---------------------------------------------------------------------------


class TestAbstractKeywordFilter:
    def test_simple_match(self, ctx):
        f = AbstractKeywordFilter(keyword="machine learning")
        p = PaperRecord(
            paper_id="p", title="T", abstract="We propose a new machine learning method."
        )
        assert f.check(p, _fctx(ctx)).passed

    def test_simple_no_match(self, ctx):
        f = AbstractKeywordFilter(keyword="quantum")
        p = PaperRecord(paper_id="p", title="T", abstract="Just classical methods here.")
        out = f.check(p, _fctx(ctx))
        assert not out.passed
        assert out.category == "abstract_keyword"

    def test_missing_abstract_required_keyword_rejects(self, ctx):
        f = AbstractKeywordFilter(keyword="ml")
        p = PaperRecord(paper_id="p", title="T", abstract=None)
        assert not f.check(p, _fctx(ctx)).passed

    def test_missing_abstract_negation_passes(self, ctx):
        # !survey is True when 'survey' isn't found — empty abstract has no
        # 'survey', so the formula evaluates True.
        f = AbstractKeywordFilter(formula="!survey", keywords={"survey": "survey"})
        p = PaperRecord(paper_id="p", title="T", abstract=None)
        assert f.check(p, _fctx(ctx)).passed

    def test_formula_in_abstract(self, ctx):
        f = AbstractKeywordFilter(
            formula="dl & bio",
            keywords={"dl": "deep learning", "bio": "biology"},
        )
        p = PaperRecord(
            paper_id="p", title="T",
            abstract="We apply deep learning to biology problems at scale.",
        )
        assert f.check(p, _fctx(ctx)).passed

    def test_case_insensitive_default(self, ctx):
        f = AbstractKeywordFilter(keyword="MACHINE")
        p = PaperRecord(paper_id="p", title="T", abstract="machine learning works")
        assert f.check(p, _fctx(ctx)).passed

    def test_whole_word_in_abstract(self, ctx):
        f = AbstractKeywordFilter(keyword="bio", match="whole_word")
        # 'bio' is part of 'biology' but not a standalone word
        assert not f.check(
            PaperRecord(paper_id="p", title="T", abstract="biology paper"),
            _fctx(ctx),
        ).passed
        assert f.check(
            PaperRecord(paper_id="p", title="T", abstract="bio research"),
            _fctx(ctx),
        ).passed


# ---------------------------------------------------------------------------
# VenueKeywordFilter
# ---------------------------------------------------------------------------


class TestVenueKeywordFilter:
    def test_simple_match(self, ctx):
        f = VenueKeywordFilter(keyword="Nature")
        p = PaperRecord(paper_id="p", title="T", venue="Nature")
        assert f.check(p, _fctx(ctx)).passed

    def test_substring_default_admits_substring_journals(self, ctx):
        # Default substring mode: 'Cell' matches 'Cellulose' — that's why
        # the richer match modes exist. Pin the loose default explicitly.
        f = VenueKeywordFilter(keyword="Cell")
        assert f.check(
            PaperRecord(paper_id="p", title="T", venue="Cellulose"), _fctx(ctx)
        ).passed

    def test_whole_word_excludes_substring_journals(self, ctx):
        f = VenueKeywordFilter(keyword="Cell", match="whole_word")
        assert not f.check(
            PaperRecord(paper_id="p", title="T", venue="Cellulose"), _fctx(ctx)
        ).passed
        assert f.check(
            PaperRecord(paper_id="p", title="T", venue="Cell Reports"), _fctx(ctx)
        ).passed
        assert f.check(
            PaperRecord(paper_id="p", title="T", venue="Cell"), _fctx(ctx)
        ).passed
        # whole_word still accepts mid-string matches
        assert f.check(
            PaperRecord(paper_id="p", title="T", venue="Stem Cell Reports"), _fctx(ctx)
        ).passed

    def test_starts_with_for_strict_journal_allowlist(self, ctx):
        f = VenueKeywordFilter(
            formula="nature | science | cell",
            keywords={"nature": "Nature", "science": "Science", "cell": "Cell"},
            match="starts_with",
        )
        # Nature / Science / Cell families — venue starts with the keyword.
        for venue in [
            "Nature", "Nature Methods", "Nature Catalysis",
            "Science", "Science Advances", "Science Robotics",
            "Cell", "Cell Reports", "Cell Stem Cell",
        ]:
            assert f.check(
                PaperRecord(paper_id="p", title="T", venue=venue), _fctx(ctx)
            ).passed, venue

        # The exact false positives the user reported on a previous run —
        # all contain Nature / Science / Cell *somewhere*, but none of
        # them BEGIN with it. starts_with rejects all of them.
        for venue in [
            "Royal Society Open Science",
            "Chemical Science",
            "Energy & Environmental Science",
            "Journal of Materials Science",
            "Science of the Total Environment",  # starts with "Science" but
            # is NOT a Nature/Science/Cell journal — caveat: this slips
            # through too. starts_with is necessary but not sufficient
            # for perfect filtering; rely on downstream LLM to catch it.
            "Stem Cell Reports",                  # Cell Press but doesn't begin with Cell
            "Cellulose",
            "PLoS ONE",
            "arXiv",
            "Advancement of science",
            "Photon Science",
            "Machine Learning: Science and Technology",
        ]:
            out = f.check(
                PaperRecord(paper_id="p", title="T", venue=venue), _fctx(ctx)
            )
            # Pin only those that should really be rejected by starts_with.
            should_reject = venue not in ("Science of the Total Environment",)
            if should_reject:
                assert not out.passed, f"expected reject: {venue}"
                assert out.category == "venue_keyword"

    def test_missing_venue_rejects(self, ctx):
        f = VenueKeywordFilter(keyword="Nature")
        p = PaperRecord(paper_id="p", title="T", venue=None)
        assert not f.check(p, _fctx(ctx)).passed
