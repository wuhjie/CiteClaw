"""Live dashboard + Rich helpers for the CiteClaw pipeline.

The dashboard renders a live mission-control panel during a pipeline run:

  - one-time step header card per step
  - a Live-updating ``rich.Panel`` containing
      * a 4-row metric grid (LLM / S2 / Accept / Budget)
      * a top-N rejection breakdown with proportional mini-bars
      * a dual-progress block (outer "source papers" + inner "now: <phase>")
  - an above-the-bar acceptance stream (`✓ Title  cit · year · venue`)
  - a transient retry banner driven by the S2 client when a request retries
  - an end-of-run summary block

Steps reach the active dashboard via ``ctx.dashboard``. The S2 HTTP retry
callback reaches it via the ``_active_dashboard`` ContextVar so retry
messages can be surfaced live without threading the dashboard through
every client constructor.

A no-op ``NullDashboard`` is the default so steps can call
``ctx.dashboard.<anything>`` unconditionally.
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from citeclaw.budget import BudgetTracker
    from citeclaw.context import Context
    from citeclaw.models import PaperRecord


# ── custom progress columns (tolerate total=None for indeterminate phases) ─


class _CountsColumn(ProgressColumn):
    def render(self, task: "Task"):
        completed = int(task.completed or 0)
        if task.total is None:
            return Text.from_markup(f"[bold]{completed:>4}[/][dim]/ ?  [/]")
        return Text.from_markup(
            f"[bold]{completed:>4}[/][dim]/{int(task.total):<4}[/]"
        )


class _PercentColumn(ProgressColumn):
    def render(self, task: "Task"):
        if task.total is None:
            return Text.from_markup("[dim]·     [/]")
        return Text.from_markup(
            f"[dim]·[/] [bold]{int(task.percentage):>3.0f}%[/]"
        )


# ── theme + shared console (preserved for backward compatibility) ──────────

_THEME = Theme(
    {
        "phase":         "bold bright_blue",
        "step.idx":      "bold bright_cyan",
        "step.name":     "bold white",
        "status":        "dim white",
        "ok":            "bright_green",
        "warn":          "bright_yellow",
        "retry":         "yellow",
        "metric.label":  "dim cyan",
        "metric.value":  "bold white",
        "reject.name":   "red",
        "reject.count":  "bold red",
        "reject.bar":    "red",
        "paper.cit":     "bright_yellow",
        "paper.year":    "dim white",
        "paper.venue":   "magenta",
        "paper.pdf_yes": "bright_green",
        "paper.pdf_no":  "grey50",
        "panel.border":  "bright_cyan",
        "bar.back":      "grey23",
        "bar.complete":  "bright_cyan",
        "bar.finished":  "bright_green",
        "metric":        "bold white",
    }
)

console = Console(stderr=True, theme=_THEME)


# ── legacy helpers (still imported by annotate.py + checkpoint.py) ─────────


def create_progress(*, transient: bool = True) -> Progress:
    """Create a consistently styled rich progress bar (legacy helper)."""
    return Progress(
        SpinnerColumn("dots", style="bright_cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style="bright_cyan", finished_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("│", style="dim"),
        TextColumn("{task.fields[status]}", style="status"),
        TextColumn("│", style="dim"),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
    )


def phase_header(phase: str, message: str) -> None:
    """Print the ``━━━ <phase> <message> ━━━`` header banner for a pipeline phase."""
    console.print()
    console.print(f"[phase]{'━' * 65}[/]")
    console.print(f"[phase]  {phase}[/]  [white]{message}[/]")
    console.print(f"[phase]{'━' * 65}[/]")


def phase_done(message: str) -> None:
    """Print the green-check ``✓`` success line below a phase header."""
    console.print(f"  [bright_green]✓[/] {message}")


def phase_warn(message: str) -> None:
    """Print the yellow-bang ``!`` warning line below a phase header."""
    console.print(f"  [bright_yellow]![/] {message}")


def stat_line(label: str, value: str) -> None:
    """Print a two-column ``<label>: <value>`` stat row (dim label, metric value)."""
    console.print(f"    [dim]{label}:[/] [metric]{value}[/]")


# ── active-dashboard ContextVar (used by S2 retry callback) ────────────────

_active_dashboard: contextvars.ContextVar[Optional["DashboardLike"]] = contextvars.ContextVar(
    "citeclaw_active_dashboard", default=None
)


def get_active_dashboard() -> Optional["DashboardLike"]:
    """Return the dashboard bound to the current execution context, or None.

    Read by the S2 HTTP retry callback (and anyone else who can't be
    threaded a dashboard through their constructor) to surface live
    status messages without introducing a hard dependency on a
    specific run's Dashboard instance.
    """
    return _active_dashboard.get()


def set_active_dashboard(dash: Optional["DashboardLike"]) -> contextvars.Token:
    """Bind ``dash`` as the active dashboard for this context; return the token.

    The returned token must be passed back to
    :func:`reset_active_dashboard` to restore the prior binding — typical
    usage wraps the binding in a try/finally around the pipeline run.
    """
    return _active_dashboard.set(dash)


def reset_active_dashboard(token: contextvars.Token) -> None:
    """Restore the prior dashboard binding from the token returned by set_*."""
    _active_dashboard.reset(token)


# ── NullDashboard (default; everything is a no-op) ─────────────────────────


class NullDashboard:
    """Default dashboard that does nothing.

    Lets every step call ``ctx.dashboard.<anything>`` unconditionally
    without paying for terminal updates when the dashboard is disabled
    (e.g. in tests, or when stdout is not a TTY).
    """

    is_null: bool = True

    def attach(self, ctx: "Context") -> None: ...
    def begin_run(self) -> None: ...
    def begin_step(self, idx: int, name: str, desc: str = "") -> None: ...
    def end_step(self, *, candidates: int | None = None) -> None: ...
    def enable_outer_bar(self, total: int, *, description: str = "source papers") -> None: ...
    def begin_phase(self, description: str, total: int) -> None: ...
    def retotal_phase(self, total: int) -> None: ...
    def tick_inner(self, n: int = 1) -> None: ...
    def complete_phase(self) -> None: ...
    def advance_outer(self, n: int = 1) -> None: ...
    def note_candidates_seen(self, n: int = 1) -> None: ...
    def paper_accepted(self, paper: "PaperRecord", *, saturation: float | None = None) -> None: ...
    def set_retry_status(self, msg: str) -> None: ...
    def clear_retry_status(self) -> None: ...
    def warn(self, msg: str) -> None:
        console.print(f"  [warn]![/] {msg}")
    def note(self, msg: str) -> None:
        """Print an informational line above the live region."""
        console.print(f"  [dim]·[/] {msg}")
    def finalize(self) -> None: ...


# Type alias used in type hints — both NullDashboard and Dashboard satisfy it.
DashboardLike = NullDashboard


# ── Dashboard (the real thing) ─────────────────────────────────────────────


class Dashboard(NullDashboard):
    """Live mission-control dashboard rendered via Rich's ``Live`` API.

    Held on ``ctx.dashboard`` and on the ``_active_dashboard`` ContextVar.
    Lifecycle is driven by ``pipeline.run_pipeline``: it calls
    :meth:`begin_run` once, then :meth:`begin_step`/:meth:`end_step` per
    step, then :meth:`finalize` at the end.

    Steps drive the inner-bar phase cascade by calling :meth:`begin_phase`,
    :meth:`tick_inner`, and :meth:`advance_outer`.
    """

    is_null: bool = False

    def __init__(
        self,
        *,
        model: str = "stub",
        data_dir: str = "runs/data",
        pipeline_length: int = 1,
        budget_cap_usd: float = 0.70,
        max_reject_rows: int = 6,
        reject_bar_width: int = 26,
    ) -> None:
        self._console = console
        self._model = model
        self._data_dir = data_dir
        self._pipeline_length = pipeline_length
        self._budget_cap_usd = budget_cap_usd
        self._max_reject_rows = max_reject_rows
        self._reject_bar_width = reject_bar_width

        # Step state (set by begin_step / cleared by end_step)
        self._step_idx: int = 0
        self._step_name: str = ""
        self._step_desc: str = ""
        self._step_started: float = 0.0
        self._step_accepted: int = 0
        self._step_seen: int = 0
        self._step_start_llm_calls: int = 0
        self._step_start_seen_count: int = 0

        # Live + Progress (per-step instances)
        self._progress: Progress | None = None
        self._outer_task: TaskID | None = None
        self._inner_task: TaskID | None = None
        self._inner_total: int = 1
        self._live: Live | None = None
        self._live_token: Any = None  # contextvar token

        # Pulled from ctx after attach()
        self._budget: "BudgetTracker | None" = None
        self._rejection_counts: Counter | None = None
        self._collection: dict | None = None
        self._n_accepted: int = 0
        self._n_seen: int = 0
        # Running saturation = mean of per-paper saturation across accepts.
        self._sat_sum: float = 0.0
        self._sat_count: int = 0

        # Transient banner shown by S2 retry callbacks
        self._retry_status: str = ""

        # For installed log handler swap
        self._installed_handler: logging.Handler | None = None
        self._removed_handlers: list[logging.Handler] = []

        # Run-wide totals
        self._run_started: float = 0.0
        self._steps_completed: int = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def attach(self, ctx: "Context") -> None:
        """Bind the dashboard to a context's budget + rejection counters."""
        self._budget = ctx.budget
        self._rejection_counts = ctx.rejection_counts
        self._collection = ctx.collection

    def begin_run(self) -> None:
        """Print the run header and install the log handler."""
        self._run_started = time.time()
        self._console.print()
        self._console.print(f"[phase]{'═' * 81}[/]")
        self._console.print(
            f"  [bold bright_white]CiteClaw[/]  [dim]·[/]  "
            f"[bright_cyan]{self._model}[/]  [dim]·[/]  "
            f"[white]{self._data_dir}[/]  [dim]·[/]  "
            f"[dim]{self._pipeline_length} steps[/]"
        )
        self._console.print(f"[phase]{'═' * 81}[/]")
        self._install_log_handler()

    def begin_step(
        self,
        idx: int,
        name: str,
        desc: str = "",
    ) -> None:
        """Open a new step's Live region.

        The pipeline runner calls this with the step's index, name, and a
        short description. Both progress bars are pre-created (the outer is
        hidden by default); the step itself activates the outer bar later
        via :meth:`enable_outer_bar` if it has a per-source loop.
        """
        self._step_idx = idx
        self._step_name = name
        self._step_desc = desc
        self._step_started = time.time()
        self._step_accepted = 0
        self._step_seen = 0
        self._retry_status = ""

        # Snapshot pre-step counters so end_step can compute deltas.
        self._step_start_llm_calls = self._budget.llm_calls if self._budget else 0
        self._step_start_seen_count = 0  # filled in below if budget present

        self._print_step_header()

        self._progress = self._make_progress()
        # Outer task added first so it renders ABOVE the inner task.
        # Hidden by default — steps that want it call enable_outer_bar.
        self._outer_task = self._progress.add_task(
            "source papers", total=1, visible=False
        )
        self._inner_task = self._progress.add_task(
            "now: starting", total=1
        )

        self._live = Live(
            self,                              # __rich__ returns the panel
            console=self._console,
            refresh_per_second=12,
            transient=True,                    # clears the panel on exit
        )
        self._live.__enter__()

    def enable_outer_bar(self, total: int, *, description: str = "source papers") -> None:
        """Activate the outer 'source papers' bar with the given total.

        Called by ExpandForward / ExpandBackward inside their ``run()``;
        no-op for steps that don't have a per-source outer loop.

        Resets ``completed`` to 0 so multiple sub-steps within a single
        pipeline step (e.g. two Expand* steps in different ``Parallel``
        branches) don't carry over the previous sub-step's count and
        produce ``N/M`` displays where N > M.
        """
        if self._progress is None or self._outer_task is None:
            return
        self._progress.update(
            self._outer_task,
            total=max(1, total),
            completed=0,
            description=description,
            visible=True,
        )

    def end_step(self, *, candidates: int | None = None) -> None:
        """Close the current step's Live region and print the done line.

        ``candidates`` is optional; if omitted, uses the step's running
        ``_step_seen`` counter (incremented by ``note_candidates_seen``).
        ``llm_calls`` is computed from the budget delta vs. begin_step.
        """
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            finally:
                self._live = None
        self._progress = None
        self._outer_task = None
        self._inner_task = None
        self._steps_completed += 1

        cur_llm = self._budget.llm_calls if self._budget else 0
        llm_calls = cur_llm - self._step_start_llm_calls
        accepted = self._step_accepted
        seen = candidates if candidates is not None else self._step_seen

        elapsed = time.time() - self._step_started
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        self._console.print()
        self._console.print(
            f"  [ok]✓[/]  [step.name]{self._step_name}[/]  "
            f"[dim]─[/]  "
            f"[bold bright_green]{accepted:>4}[/] [ok]accepted[/] "
            f"[dim]/ {seen} screened[/]"
            f"   [dim]·  {llm_calls} LLM  ·  {mins:02d}:{secs:02d}[/]"
        )

    def note_candidates_seen(self, n: int = 1) -> None:
        """Increment the step's 'seen' counter (used by end_step + metrics)."""
        self._step_seen += n
        self._n_seen += n

    def finalize(self) -> None:
        """Print the run-end summary block."""
        self._remove_log_handler()
        elapsed = time.time() - self._run_started if self._run_started else 0.0
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        self._console.print()
        self._console.print(f"[phase]{'═' * 81}[/]")
        self._console.print(f"  [ok]✓[/]  [bold bright_white]pipeline complete[/]   [dim]· {mins:02d}:{secs:02d}[/]")
        n_collection = len(self._collection) if self._collection else 0
        self._console.print(
            f"     [ok]★[/]  [bold bright_green]{n_collection}[/] [bold]papers accepted[/] "
            f"[dim]/ {self._n_seen} screened  ·  {self._steps_completed} steps[/]"
        )

        if self._budget is not None:
            tot_in = self._budget.llm_input_tokens
            tot_out = self._budget.llm_output_tokens
            tot_reason = self._budget.llm_reasoning_tokens
            n_calls = self._budget.llm_calls
            s2_api = self._budget.s2_requests
            s2_cache = self._budget.s2_cache_hits
            hit_pct = (s2_cache / max(1, s2_api + s2_cache)) * 100
            cost = self._budget.cost_estimate(self._model)

            llm_cache_hits = self._budget.llm_cache_hits
            cache_suffix = (
                f"  [metric.value]{llm_cache_hits}[/] cached"
                if llm_cache_hits else ""
            )
            self._console.print(
                f"     llm  [metric.value]{tot_in / 1000:.1f}k[/] in  "
                f"[metric.value]{tot_out / 1000:.1f}k[/] out  "
                f"[metric.value]{tot_reason / 1000:.1f}k[/] reason  "
                f"([metric.value]{n_calls}[/] calls{cache_suffix})"
            )
            self._console.print(
                f"     s2   [metric.value]{s2_api}[/] api  "
                f"[metric.value]{s2_cache}[/] cached  "
                f"([metric.value]{hit_pct:.0f}%[/] hit)"
            )
            self._console.print(
                f"     cost [metric.value]${cost:.4f}[/]"
                f"[dim] / ${self._budget_cap_usd:.2f} budget   "
                f"(local estimate from MODEL_PRICING table)[/]"
            )
            # Per-category cost breakdown — surfaces which filter ate the
            # budget. Skipped if there's nothing to show or the stub model
            # was used (all costs would be $0).
            breakdown = self._budget.cost_breakdown(self._model)
            if breakdown and cost > 0:
                self._console.print(
                    f"     [dim]cost by category:[/]"
                )
                for cat, info in breakdown.items():
                    if info["cost_usd"] <= 0:
                        continue
                    self._console.print(
                        f"       [dim]{cat:<26}[/] "
                        f"[metric.value]${info['cost_usd']:>8.4f}[/] "
                        f"[dim]({info['input']:>7,} ↑  {info['output']:>5,} ↓"
                        + (f"  {info['reason']:>4,} reason" if info['reason'] else "")
                        + f"  {info['calls']:>3} calls)[/]"
                    )
        if self._sat_count:
            mean = self._sat_sum / self._sat_count
            self._console.print(
                f"     sat  [metric.value]{mean:.3f}[/]"
                f"[dim]  (mean across {self._sat_count} accepted papers)[/]"
            )

        # Open-access PDF coverage. Computed from ``ctx.collection`` rather
        # than a running counter so checkpoint-loaded and ReScreen-filtered
        # collections report accurately. Skipped when no papers landed.
        if self._collection:
            n_total = len(self._collection)
            n_pdf = sum(
                1 for p in self._collection.values()
                if getattr(p, "pdf_url", None)
            )
            pct = (n_pdf / n_total * 100) if n_total else 0.0
            self._console.print(
                f"     pdf  [metric.value]{n_pdf} / {n_total}[/] open-access "
                f"[dim]({pct:.0f}%; run `citeclaw fetch-pdfs <data_dir>` to download)[/]"
            )

        if self._rejection_counts:
            self._console.print()
            self._console.print(
                f"  [bold red]Rejection breakdown[/] [dim](final)[/]"
            )
            total = sum(self._rejection_counts.values())
            for name, count in sorted(self._rejection_counts.items(), key=lambda kv: -kv[1]):
                pct = count / total * 100 if total else 0.0
                self._console.print(
                    f"    [reject.name]{name:<24}[/] "
                    f"[reject.count]{count:>5}[/] [dim]({pct:>5.1f}%)[/]"
                )

        self._console.print(f"[phase]{'═' * 81}[/]")
        self._console.print()

    # ── inner / outer bar driving ────────────────────────────────────────

    def _inner_task_obj(self):
        """Look up the Rich ``Task`` object for the current inner task.

        Direct mutation of ``task.total`` is the only way to switch a
        task BACK to indeterminate — both ``Progress.reset(total=None)``
        and ``Progress.update(total=None)`` treat ``None`` as "leave
        total alone" (see Rich 14 ``progress.py``). Returns ``None`` if
        the inner task no longer exists.
        """
        if self._progress is None or self._inner_task is None:
            return None
        return next(
            (t for t in self._progress.tasks if t.id == self._inner_task),
            None,
        )

    def begin_phase(self, description: str, total: int | None) -> None:
        """Reset the inner bar to a new phase.

        ``total=None`` renders an indeterminate (pulsing) bar — use it
        when the caller knows work is happening but can't predict the
        unit count (e.g. paginating refs with no known page count, or
        an atomic batched call that returns all results at once).
        Any callable that ticks the inner bar thereafter bumps the
        displayed "completed" counter but the bar stays indeterminate
        until ``complete_phase`` snaps total to completed.
        """
        if self._progress is None or self._inner_task is None:
            return
        self._inner_total = None if total is None else max(1, total)
        # Rich's reset() drops the progress samples + zeroes completed,
        # but won't accept ``total=None`` (it's interpreted as "keep
        # current total"). Pass a placeholder 1, then mutate the Task
        # directly when we actually want indeterminate.
        self._progress.reset(
            self._inner_task,
            total=self._inner_total if self._inner_total is not None else 1,
            description=f"now: {description}",
        )
        if self._inner_total is None:
            task = self._inner_task_obj()
            if task is not None:
                task.total = None

    def retotal_phase(self, total: int | None) -> None:
        """Adjust the inner bar's total without resetting completed.

        Like :meth:`begin_phase`, switching back to indeterminate
        (``total=None``) requires direct Task mutation since Rich's
        ``update(total=None)`` is a no-op.
        """
        if self._progress is None or self._inner_task is None:
            return
        self._inner_total = None if total is None else max(1, total)
        if self._inner_total is None:
            task = self._inner_task_obj()
            if task is not None:
                task.total = None
        else:
            self._progress.update(self._inner_task, total=self._inner_total)

    def tick_inner(self, n: int = 1) -> None:
        if self._progress is None or self._inner_task is None:
            return
        self._progress.update(self._inner_task, advance=n)

    def complete_phase(self) -> None:
        # Clamp inner bar to its total — prevents A>B overshoot when a
        # nested dispatcher has already driven the bar independently.
        # For indeterminate phases (total=None), snap total to the
        # current completed count so the bar shows "N/N" at the end.
        if self._progress is None or self._inner_task is None:
            return
        if self._inner_total is None:
            task = self._inner_task_obj()
            completed = int(task.completed) if task else 0
            self._inner_total = max(1, completed)
            self._progress.update(
                self._inner_task, total=self._inner_total, completed=self._inner_total,
            )
        else:
            self._progress.update(self._inner_task, completed=self._inner_total)

    def advance_outer(self, n: int = 1) -> None:
        if self._progress is None or self._outer_task is None:
            return
        self._progress.update(self._outer_task, advance=n)

    # ── streaming acceptance feed ────────────────────────────────────────

    def paper_accepted(self, paper: "PaperRecord", *, saturation: float | None = None) -> None:
        """Print a ✓ paper line above the live region.

        ``saturation`` (the fraction of the paper's refs that are already
        in the collection) is computed by the calling step. It is no
        longer printed per-paper (visual noise) but is still aggregated
        as a running mean shown in the metric grid + end-of-run summary.
        """
        title = (getattr(paper, "title", None) or "")[:55].ljust(55)
        cit = getattr(paper, "citation_count", None) or 0
        year = getattr(paper, "year", None) or "—"
        venue = (getattr(paper, "venue", None) or "")[:24] or "—"
        has_pdf = bool(getattr(paper, "pdf_url", None))
        pdf_mark = "[paper.pdf_yes]●[/]" if has_pdf else "[paper.pdf_no]○[/]"
        self._console.print(
            f"  [ok]✓[/]  {title}  "
            f"[paper.cit]{cit:>8,}[/] cit · "
            f"[paper.year]{year}[/] · "
            f"{pdf_mark} · "
            f"[paper.venue]{venue}[/]"
        )
        self._step_accepted += 1
        self._n_accepted += 1
        if saturation is not None:
            self._sat_sum += saturation
            self._sat_count += 1

    # ── retry banner ─────────────────────────────────────────────────────

    def set_retry_status(self, msg: str) -> None:
        self._retry_status = msg

    def clear_retry_status(self) -> None:
        self._retry_status = ""

    # ── warnings + info (route through console) ──────────────────────────

    def warn(self, msg: str) -> None:
        self._console.print(f"  [warn]![/] {msg}")

    def note(self, msg: str) -> None:
        """Print an informational line above the live region.

        Useful for steps whose work isn't a clean "n of m" progress
        signal — e.g. the iterative search agent, which surfaces one
        line per turn ("iter 2/4 · 30 results · 5 new · refine").
        """
        self._console.print(f"  [dim]·[/] {msg}")

    # ── helpers ──────────────────────────────────────────────────────────

    def _print_step_header(self) -> None:
        self._console.print()
        self._console.print(f"[phase]{'─' * 81}[/]")
        self._console.print(
            f"  [step.idx]●  Step {self._step_idx} / {self._pipeline_length}[/]   "
            f"[step.name]{self._step_name}[/]"
        )
        if self._step_desc:
            self._console.print(f"     [dim]{self._step_desc}[/]")
        self._console.print(f"[phase]{'─' * 81}[/]")
        self._console.print()

    def _make_progress(self) -> Progress:
        return Progress(
            SpinnerColumn("dots", style="bright_cyan"),
            TextColumn("[bold bright_cyan]{task.description:<24}[/]"),
            BarColumn(
                bar_width=32,
                complete_style="bright_cyan",
                finished_style="bright_green",
            ),
            _CountsColumn(),
            _PercentColumn(),
            TextColumn("[dim]· ETA[/]"),
            TimeRemainingColumn(),
            console=self._console,
            transient=True,
            expand=False,
        )

    def _metric_grid(self) -> Table:
        b = self._budget
        in_tok = b.llm_input_tokens if b else 0
        out_tok = b.llm_output_tokens if b else 0
        reason_tok = b.llm_reasoning_tokens if b else 0
        n_calls = b.llm_calls if b else 0
        s2_api = b.s2_requests if b else 0
        s2_cache = b.s2_cache_hits if b else 0
        hit_pct = (s2_cache / max(1, s2_api + s2_cache)) * 100
        cost = b.cost_estimate(self._model) if b else 0.0
        budget_pct = (cost / self._budget_cap_usd) * 100 if self._budget_cap_usd else 0.0

        accept_pct = (self._n_accepted / max(1, self._n_seen)) * 100 if self._n_seen else 0.0
        step_pct = (
            (self._step_accepted / max(1, self._step_seen)) * 100
            if self._step_seen
            else 0.0
        )
        sat_mean = (self._sat_sum / self._sat_count) if self._sat_count else None
        sat_str = f"sat {sat_mean:.2f} (n={self._sat_count})" if sat_mean is not None else "sat —"

        g = Table.grid(padding=(0, 2), expand=False)
        g.add_column(justify="right", style="metric.label", no_wrap=True)
        g.add_column(style="metric.value", no_wrap=True)
        g.add_row(
            "LLM",
            f"{in_tok:>7,} ↑   "
            f"{out_tok:>6,} ↓   "
            f"{reason_tok:>5,} reason   "
            f"{n_calls:>4} calls",
        )
        g.add_row(
            "S2",
            f"{s2_api:>7,} api   "
            f"{s2_cache:>6,} cached   "
            f"{hit_pct:>4.0f}% hit",
        )
        g.add_row(
            "Accept",
            f"{self._n_accepted:>7,} / {self._n_seen:<6,}  "
            f"{accept_pct:>4.0f}% all   "
            f"step {self._step_accepted}/{self._step_seen} ({step_pct:.0f}%)",
        )
        g.add_row(
            "Quality",
            f"{sat_str}",
        )
        g.add_row(
            "Budget",
            f"${cost:>6.4f} / ${self._budget_cap_usd:.2f}   "
            f"{budget_pct:>4.0f}% used",
        )
        return g

    def _rejection_block(self) -> Group:
        rc = self._rejection_counts
        if not rc:
            return Group(Text.from_markup("  [dim]Rejected  none yet[/]"))
        total = sum(rc.values())
        sorted_counts = sorted(rc.items(), key=lambda kv: -kv[1])[: self._max_reject_rows]
        max_count = max((c for _, c in sorted_counts), default=1)

        header = Text.from_markup(
            f"  [bold red]Rejected[/]  [dim]{total} total[/]"
        )
        g = Table.grid(padding=(0, 2), expand=False)
        g.add_column(justify="right", style="reject.name", no_wrap=True)
        g.add_column(justify="right", style="reject.count", no_wrap=True, width=5)
        g.add_column(style="reject.bar", no_wrap=True)
        for name, count in sorted_counts:
            bar_len = max(1, int(round(count / max(1, max_count) * self._reject_bar_width)))
            g.add_row(name, str(count), "▰" * bar_len)
        return Group(header, g)

    def __rich__(self) -> RenderableType:
        elapsed = time.time() - self._step_started if self._step_started else 0.0
        mins, secs = int(elapsed // 60), int(elapsed % 60)

        parts: list[RenderableType] = [
            self._metric_grid(),
            Text(""),
            self._rejection_block(),
            Text(""),
        ]
        if self._progress is not None:
            parts.append(self._progress)
        if self._retry_status:
            parts.append(Text.from_markup(f"  [retry]↻ {self._retry_status}[/]"))

        title = (
            f"[step.idx] Step {self._step_idx} / {self._pipeline_length} [/]"
            f"[step.name]· {self._step_name}[/]"
        )
        if self._step_desc:
            title += f" [dim]· {self._step_desc}[/]"

        return Panel(
            Group(*parts),
            title=title,
            title_align="left",
            subtitle=f"[dim]elapsed[/] [bold]{mins:02d}:{secs:02d}[/]",
            subtitle_align="right",
            border_style="panel.border",
            padding=(1, 2),
            expand=False,
        )

    # ── log handler swap ─────────────────────────────────────────────────

    def _install_log_handler(self) -> None:
        """Replace existing console handlers on the citeclaw logger with one
        that routes warnings/errors through the dashboard's Rich Console.

        Without this, ``log.warning`` calls would print on their own line
        and collide with the live region. RichHandler is Live-aware and
        prints above the live block.
        """
        try:
            from rich.logging import RichHandler
        except ImportError:
            return
        citeclaw_logger = logging.getLogger("citeclaw")
        # Remove pure StreamHandlers (keep FileHandlers in place).
        for h in list(citeclaw_logger.handlers):
            if isinstance(h, logging.FileHandler):
                continue
            if isinstance(h, logging.StreamHandler):
                citeclaw_logger.removeHandler(h)
                self._removed_handlers.append(h)
        # Install our Rich-aware handler.
        rh = RichHandler(
            console=self._console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
            log_time_format="[%H:%M:%S]",
        )
        rh.setLevel(logging.WARNING)
        citeclaw_logger.addHandler(rh)
        self._installed_handler = rh

    def _remove_log_handler(self) -> None:
        citeclaw_logger = logging.getLogger("citeclaw")
        if self._installed_handler is not None:
            citeclaw_logger.removeHandler(self._installed_handler)
            self._installed_handler = None
        for h in self._removed_handlers:
            citeclaw_logger.addHandler(h)
        self._removed_handlers = []
