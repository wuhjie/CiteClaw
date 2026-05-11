"""Cache-aware PDF fetch + parse pipeline used by ExpandByPDF.

Three fallback layers, tried in order, all driven by a single
:meth:`PdfClawBridge.fetch_text` call:

  1. **Cache** — ``paper_full_text`` in the SQLite cache (instant).
  2. **HTTP** — :func:`download_pdf_bytes` against S2's
     ``openAccessPdf.url`` (fast, no browser).
  3. **pdfclaw recipes** — the full publisher-aware fallback chain
     from :mod:`pdfclaw.publishers` (HTTP + Elsevier/Wiley TDM,
     browser SSO, LLM finder, etc.).

PDF parsing — the bytes-to-text step shared by Layers 2 and 3 — is
delegated to :func:`pdfclaw.parsers.parse`.  The engine
(``"pymupdf"`` / ``"docling"`` / ``"grobid"``) is fixed at bridge
construction time so the on-disk cache stays consistent for the whole
run.  See :mod:`pdfclaw.parsers` for the engine matrix.

The browser is opened **lazily** the first time a Layer-3 recipe
needs it and **reused** across every subsequent paper in the
bridge's lifetime.  Call :meth:`close` (or use the bridge as a
context manager) to shut down the browser cleanly.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from citeclaw.clients.pdf import download_pdf_bytes, make_pdf_http_client
from pdfclaw.parsers import ParserError, parse as parse_pdf

if TYPE_CHECKING:
    from citeclaw.cache import Cache
    from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.clients.pdfclaw_bridge")

_DEFAULT_MAX_TEXT_CHARS = 80_000
_DEFAULT_PROFILE_PATH = Path.home() / ".pdfclaw-chrome-profile"


class PdfClawBridge:
    """Cache-aware PDF fetcher that falls through to pdfclaw browser recipes.

    Designed to be instantiated **once per ExpandByPDF run** and reused
    across all papers in the signal.  The browser is opened lazily
    (only when a browser recipe is actually needed) and closed by
    :meth:`close`.
    """

    def __init__(
        self,
        cache: "Cache",
        *,
        max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS,
        profile_path: Path | None = None,
        headless: bool = True,
        sleep_between: float = 3.0,
        parser: str = "pymupdf",
        parser_kwargs: dict | None = None,
    ) -> None:
        self._cache = cache
        self._max_text_chars = max_text_chars
        self._profile_path = (profile_path or _DEFAULT_PROFILE_PATH).expanduser()
        self._headless = headless
        self._sleep_between = sleep_between
        # Engine selection from :mod:`pdfclaw.parsers`. Fixed for the
        # life of the bridge so cache entries remain comparable across
        # the run.  Heavy-quality engines (``"docling"``, ``"grobid"``)
        # are opt-in via the ExpandByPDF YAML config.
        self._parser = parser
        self._parser_kwargs = parser_kwargs or {}

        # Lazy-init state
        self._http = make_pdf_http_client()
        self._pdfclaw_available: bool | None = None
        self._registry: list | None = None
        self._browser_ctx_manager = None
        self._browser_page = None
        # Per-run recipe suppression — mirrors :class:`pdfclaw.fetcher.Fetcher`
        # so a failing publisher doesn't get hammered for every paper
        # in a batch.
        self._auth_failed: set[str] = set()
        self._consecutive_failures: dict[str, int] = {}
        # Serialises every entry into ``_try_pdfclaw`` (browser, recipe
        # registry init, auth-failed bookkeeping).  Playwright's Chrome
        # context is single-threaded and the recipe registry is built
        # lazily on first use, so concurrent ExpandByPDF workers would
        # race without this.  The fast paths (cache hit, HTTP fetch via
        # the shared thread-safe httpx Client) stay lock-free.
        self._pdfclaw_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_text(self, paper: "PaperRecord") -> str | None:
        """Return parsed PDF body text for *paper*, or ``None`` on failure.

        Tries cache → HTTP → pdfclaw browser in order.  Successes and
        categorised failures are cached so repeated calls for the same
        paper never re-fetch.
        """
        if not paper.paper_id:
            return None

        # 1. Cache hit (text or known failure)
        cached = self._cache.get_full_text(paper.paper_id)
        if cached is not None:
            text = cached.get("text")
            if text:
                return text[: self._max_text_chars] if len(text) > self._max_text_chars else text
            # Cached failure from HTTP — still try pdfclaw browser below.
            # But if the error is "parse_failed" or "too_large", pdfclaw
            # won't help either.
            if cached.get("error") in ("parse_failed", "too_large"):
                return None

        # 2. HTTP fetch (uses S2's openAccessPdf.url)
        if cached is None and paper.pdf_url:
            text = self._try_http(paper)
            if text:
                self._cache.put_full_text(paper.paper_id, text=text)
                return text

        # 3. PDFClaw browser-based recipes
        text = self._try_pdfclaw(paper)
        if text:
            self._cache.put_full_text(paper.paper_id, text=text)
            return text

        # All attempts failed — cache the failure.
        if cached is None:
            self._cache.put_full_text(paper.paper_id, error="download_failed")
        return None

    def close(self) -> None:
        """Release browser and HTTP resources.

        Both the browser context exit and the http-client close can
        legitimately raise during interpreter shutdown (playwright and
        httpx can each fail to talk to their event loops at that
        point). The bridge's contract is "must not propagate" —
        consumers rely on ``with PdfClawBridge(...) as b: ...`` or a
        bare ``b.close()`` in a finally block, neither of which tolerate
        a close failure. DEBUG logs give a diagnostic trail without
        breaking shutdown.
        """
        if self._browser_ctx_manager is not None:
            try:
                self._browser_ctx_manager.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                log.debug("pdfclaw browser context exit failed: %s", exc)
            self._browser_ctx_manager = None
            self._browser_page = None
        if self._http is not None:
            try:
                self._http.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("pdfclaw bridge http client close failed: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def fetch_pdf_bytes(self, paper: "PaperRecord") -> bytes | None:
        """Like :meth:`fetch_text`, but stops at the raw PDF bytes.

        Used by code paths that need *structured* parser output (e.g.
        GROBID's TEI ``<biblStruct>`` references) which the body-text
        cache can't carry.  Skips the ``paper_full_text`` cache for
        successes — that cache only stores ``text``, so reading from
        it gives us nothing useful for callers that want bytes — but
        still respects the cached error rows (``no_pdf``,
        ``download_failed``, ``parse_failed``, ``too_large``) so we
        don't re-download a known-broken paper for every reader.

        Returns ``None`` for paths that only ever produce body text
        (publisher recipes like Elsevier TDM / EuropePMC BioC that
        return XML directly), or when every fallback layer fails.
        """
        if not paper.paper_id:
            return None

        cached = self._cache.get_full_text(paper.paper_id)
        if cached is not None and cached.get("error") in (
            "no_pdf", "download_failed", "parse_failed", "too_large",
        ):
            return None

        # HTTP path — S2's openAccessPdf URL
        if paper.pdf_url:
            try:
                body, err = download_pdf_bytes(self._http, paper.pdf_url)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "fetch_pdf_bytes: http failed for %s: %s",
                    paper.paper_id[:20], exc,
                )
                body, err = None, exc
            if err is None and body:
                return body

        # PDFClaw browser-recipe path
        return self._try_pdfclaw_bytes(paper)

    # ------------------------------------------------------------------
    # Layer 2: HTTP
    # ------------------------------------------------------------------

    def _try_http(self, paper: "PaperRecord") -> str | None:
        url = paper.pdf_url
        if not url:
            return None
        body, err = download_pdf_bytes(self._http, url)
        if err is not None:
            return None
        return self._parse_bytes(body)

    # ------------------------------------------------------------------
    # Layer 3: PDFClaw browser recipes
    # ------------------------------------------------------------------

    def _try_pdfclaw(self, paper: "PaperRecord") -> str | None:
        # Serialise the whole pdfclaw path. The browser is the only
        # genuinely non-reentrant resource, but the recipe-registry
        # lazy-init and the per-run suppression dicts are also written
        # here; a single coarse lock keeps the implementation honest at
        # the cost of running publisher recipes one paper at a time.
        # The expensive concurrent step (LLM extraction) runs outside
        # this method, so this serialization barely affects throughput.
        with self._pdfclaw_lock:
            return self._try_pdfclaw_locked(paper)

    def _try_pdfclaw_locked(self, paper: "PaperRecord") -> str | None:
        if not self._ensure_pdfclaw():
            return None

        doi = self._extract_doi(paper)
        if not doi:
            return None

        from pdfclaw.publishers import find_recipes
        from pdfclaw.publishers.base import STATUS_AUTH

        recipes = find_recipes(doi, self._registry)
        if not recipes:
            return None

        for recipe in recipes:
            if recipe.name in self._auth_failed:
                continue

            # Lazy browser open
            if recipe.needs_browser:
                page = self._ensure_browser()
                if page is None:
                    continue
            else:
                page = None

            try:
                result = recipe.fetch(
                    paper.paper_id,
                    doi,
                    browser_page=page if recipe.needs_browser else None,
                    http=self._http if not recipe.needs_browser else None,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("pdfclaw recipe %s raised: %s", recipe.name, exc)
                if recipe.needs_browser:
                    self._bump_failures(recipe.name)
                continue

            if result.ok:
                self._consecutive_failures[recipe.name] = 0
                return self._extract_text(result)

            if result.status == STATUS_AUTH:
                self._auth_failed.add(recipe.name)
                log.info(
                    "pdfclaw: recipe %s needs auth; suppressed for this run",
                    recipe.name,
                )
                continue

            if recipe.needs_browser and result.status in ("error", "blocked"):
                self._bump_failures(recipe.name)

        return None

    def _try_pdfclaw_bytes(self, paper: "PaperRecord") -> bytes | None:
        """Same fallback chain as :meth:`_try_pdfclaw` but returns the
        recipe's raw ``pdf_bytes``.  Text-only recipes (Elsevier TDM
        XML, EuropePMC BioC) come back as ``None`` here because they
        never produce a PDF — the caller falls back to whatever
        text-based path makes sense for that recipe.

        Locked the same way as :meth:`_try_pdfclaw` so concurrent
        ExpandBackward / ExpandByPDF workers don't race on the browser.
        """
        with self._pdfclaw_lock:
            return self._try_pdfclaw_bytes_locked(paper)

    def _try_pdfclaw_bytes_locked(self, paper: "PaperRecord") -> bytes | None:
        if not self._ensure_pdfclaw():
            return None
        doi = self._extract_doi(paper)
        if not doi:
            return None

        from pdfclaw.publishers import find_recipes
        from pdfclaw.publishers.base import STATUS_AUTH

        recipes = find_recipes(doi, self._registry)
        if not recipes:
            return None

        for recipe in recipes:
            if recipe.name in self._auth_failed:
                continue

            if recipe.needs_browser:
                page = self._ensure_browser()
                if page is None:
                    continue
            else:
                page = None

            try:
                result = recipe.fetch(
                    paper.paper_id,
                    doi,
                    browser_page=page if recipe.needs_browser else None,
                    http=self._http if not recipe.needs_browser else None,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("pdfclaw recipe %s raised: %s", recipe.name, exc)
                if recipe.needs_browser:
                    self._bump_failures(recipe.name)
                continue

            if result.ok:
                self._consecutive_failures[recipe.name] = 0
                if result.pdf_bytes:
                    return result.pdf_bytes
                # body_text-only recipes (Elsevier TDM, EuropePMC, …)
                # produce no PDF — keep walking other recipes in case a
                # later one does.
                continue

            if result.status == STATUS_AUTH:
                self._auth_failed.add(recipe.name)
                continue

            if recipe.needs_browser and result.status in ("error", "blocked"):
                self._bump_failures(recipe.name)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_doi(self, paper: "PaperRecord") -> str | None:
        """Get DOI from paper metadata.  Try external_ids first, then ArXiv fallback."""
        doi = paper.external_ids.get("DOI")
        if doi:
            return doi
        arxiv = paper.external_ids.get("ArXiv")
        if arxiv:
            return f"10.48550/arXiv.{arxiv}"
        return None

    def _extract_text(self, result) -> str | None:
        """Parse a successful FetchResult into body text.

        Some recipes return already-extracted text directly
        (Elsevier TDM XML, EuropePMC BioC) — those bypass the
        parser engine because the publisher's API gave us text in
        the first place.  Recipes that return PDF bytes go through
        the configured :mod:`pdfclaw.parsers` engine.
        """
        if result.body_text:
            text = result.body_text
        elif result.pdf_bytes:
            text = self._parse_bytes(result.pdf_bytes)
        else:
            return None
        if text and len(text) > self._max_text_chars:
            text = text[: self._max_text_chars]
        return text or None

    def _parse_bytes(self, body: bytes) -> str | None:
        """Run the configured parser engine; return body text or None.

        Engine failures are caught and surfaced as ``None`` so the
        fallback chain in :meth:`fetch_text` keeps walking — the
        caller sees a categorised "parse_failed" cache entry rather
        than an exception bubbling out of the bridge.
        """
        try:
            result = parse_pdf(
                body,
                parser=self._parser,
                max_chars=self._max_text_chars,
                **self._parser_kwargs,
            )
        except ParserError as exc:
            log.info(
                "pdf bridge: parser %s failed: %s",
                self._parser, exc,
            )
            return None
        return result.body_text or None

    def _ensure_pdfclaw(self) -> bool:
        """Check whether pdfclaw is importable; cache the result."""
        if self._pdfclaw_available is not None:
            return self._pdfclaw_available
        try:
            from pdfclaw.publishers import build_default_registry

            self._registry = build_default_registry()
            self._pdfclaw_available = True
        except ImportError:
            log.info("pdfclaw not installed — browser-based PDF fetching disabled")
            self._pdfclaw_available = False
        return self._pdfclaw_available

    def _ensure_browser(self):
        """Lazily open a persistent browser context; return the Page or None."""
        if self._browser_page is not None:
            return self._browser_page
        try:
            from pdfclaw.browser import open_browser_context

            self._browser_ctx_manager = open_browser_context(
                self._profile_path,
                headless=self._headless,
            )
            _ctx, self._browser_page = self._browser_ctx_manager.__enter__()
            log.info("pdfclaw browser opened (headless=%s)", self._headless)
            return self._browser_page
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to open pdfclaw browser: %s", exc)
            self._pdfclaw_available = False  # Don't retry browser recipes
            return None

    def _bump_failures(self, recipe_name: str) -> None:
        count = self._consecutive_failures.get(recipe_name, 0) + 1
        self._consecutive_failures[recipe_name] = count
        if count >= 3 and recipe_name not in self._auth_failed:
            self._auth_failed.add(recipe_name)
            log.warning(
                "pdfclaw: recipe %s hit %d consecutive failures; suppressed",
                recipe_name,
                count,
            )

    def sleep(self) -> None:
        """Polite delay between fetches."""
        if self._sleep_between > 0:
            time.sleep(self._sleep_between)
