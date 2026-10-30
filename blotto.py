"""Fictitious play for Colonel Blotto.

Five castles are worth 1, 2, 3, 4, and 5. An allocation is five non-negative
integers that sum to 100.

Game 1 is head-to-head. On each castle the larger allocation takes that
castle's value, and a tie splits the value. ``game1_score`` averages this
over every row of the pool.

Game 2 draws three distinct pool rows. On each castle the value is split
among the players who allocated the most soldiers. Those four totals are
ranked and paid 9, 4, 2, 0. Players with the same total split the payouts
for the places they share. ``game2_score`` averages that payout.

Each game starts from 200 uniform random allocations and adds a best
response for 300 rounds. After ``burn_in`` rounds the initial random rows
are dropped so the pool holds only best responses. Optionally score against
only the most recent ``recent_window`` entries. The search scores a random
sample, then hill-climbs by moving 1 to 5 soldiers from one castle to
another. Game 2 freezes one draw of opponent groups for that search so
candidates are compared on the same matches. ``SEED`` fixes both runs.
Mean-soldier and exploitability-gap plots are written under ``figures/``.

    py -3 blotto.py
    py -3 blotto.py --both
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 0
N_CASTLES = 5
TOTAL_TROOPS = 100
INITIAL_POOL = 200
N_ROUNDS = 300
BURN_IN = 50
RECENT_WINDOW: int | None = None
EXPLOIT_EVERY = 50
N_GROUPS = 20_000
N_SHOW = 5
MOVE_MAX = 5
MAX_HILL_STEPS = 40
SCORE_CHUNK = 64
GAME1_VALUE = 7.5
GAME2_VALUE = 3.75

# Twelfths keep splits by 2, 3, and 4 exact (12 is divisible by each).
POINT_SCALE = 12
CASTLE_VALUES = np.array([1, 2, 3, 4, 5], dtype=np.float64)
SCALED_VALUES = np.array([12, 24, 36, 48, 60], dtype=np.int32)
RANK_PAYOUTS = np.array([9.0, 4.0, 2.0, 0.0])
PLOT_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
FIG_DIR = Path(__file__).resolve().parent / "figures"
COMMON_HUMAN = np.array(
    [
        [20, 20, 20, 20, 20],
        [0, 0, 0, 50, 50],
        [10, 10, 20, 30, 30],
        [0, 0, 25, 35, 40],
    ],
    dtype=np.int16,
)


def _payout_table() -> np.ndarray:
    """Payout for a player with ``higher`` scores above them and ``tied`` players in the group (including them)."""
    table = np.full((4, 5), np.nan, dtype=np.float64)
    for higher in range(4):
        for tied in range(1, 5 - higher):
            table[higher, tied] = RANK_PAYOUTS[higher : higher + tied].sum() / tied
    return table


PAYOUT_TABLE = _payout_table()


def _log(message: str) -> None:
    print(message, flush=True)


def _as_allocations(rows, name: str) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    bad_shape = arr.ndim != 2 or arr.shape[1] != N_CASTLES or arr.shape[0] == 0
    bad_values = (
        bad_shape
        or not np.all(np.isfinite(arr))
        or np.any(np.abs(arr - np.rint(arr)) > 1e-6)
    )
    if bad_values:
        raise ValueError(f"{name} must be 5 non-negative integers summing to 100")
    out = np.rint(arr).astype(np.int16)
    if np.any(out < 0) or not np.all(out.sum(axis=1) == TOTAL_TROOPS):
        raise ValueError(f"{name} must be 5 non-negative integers summing to 100")
    return out


def _as_alloc(a) -> np.ndarray:
    rows = _as_allocations(a, "allocation")
    if rows.shape[0] != 1:
        raise ValueError("allocation must be 5 non-negative integers summing to 100")
    return rows[0]


def _as_pool(pool) -> np.ndarray:
    arr = np.asarray(pool)
    if arr.ndim != 2:
        raise ValueError("pool must have shape (n, 5)")
    return _as_allocations(arr, "each pool allocation")


def random_allocations(
    rng: np.random.Generator,
    n: int,
    total: int = TOTAL_TROOPS,
    k: int = N_CASTLES,
) -> np.ndarray:
    """Draw ``n`` allocations uniformly from the compositions of ``total`` into ``k`` parts."""
    if n <= 0:
        return np.empty((0, k), dtype=np.int16)
    # Stars and bars: k-1 bar positions chosen from total + k - 1 slots.
    universe = total + k - 1
    keys = rng.random((n, universe))
    bars = np.argpartition(keys, k - 2, axis=1)[:, : k - 1]
    bars.sort(axis=1)
    padded = np.empty((n, k + 1), dtype=np.int32)
    padded[:, 0] = -1
    padded[:, 1:-1] = bars
    padded[:, -1] = universe
    alloc = np.diff(padded, axis=1) - 1
    return alloc.astype(np.int16)


def human_like_allocations(rng: np.random.Generator, n: int = INITIAL_POOL) -> np.ndarray:
    """Pool of multiples-of-5 allocations plus a few common human patterns.

    Draws compositions of 20 units (each unit = 5 soldiers), then appends
    ``COMMON_HUMAN`` so those rows always stay in the seed pool.
    """
    n_common = int(COMMON_HUMAN.shape[0])
    n_rand = max(0, n - n_common)
    units = random_allocations(rng, n_rand, total=TOTAL_TROOPS // 5, k=N_CASTLES)
    rows = (units * np.int16(5)).astype(np.int16)
    if n_rand == 0:
        return COMMON_HUMAN.copy()
    return np.vstack([rows, COMMON_HUMAN])


def _sorted_columns(pool: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.sort(pool[:, castle]) for castle in range(N_CASTLES))


def _game1_from_columns(
    allocs: np.ndarray,
    columns: tuple[np.ndarray, ...],
    n_pool: int,
) -> np.ndarray:
    allocs = np.asarray(allocs, dtype=np.int16)
    if allocs.ndim == 1:
        allocs = allocs.reshape(1, -1)
    less = np.empty(allocs.shape, dtype=np.int32)
    tied = np.empty(allocs.shape, dtype=np.int32)
    for castle, column in enumerate(columns):
        left = np.searchsorted(column, allocs[:, castle], side="left")
        right = np.searchsorted(column, allocs[:, castle], side="right")
        less[:, castle] = left
        tied[:, castle] = right - left
    points = less * CASTLE_VALUES + tied * (CASTLE_VALUES * 0.5)
    return points.sum(axis=1) / n_pool


def game1_scores(allocs: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Game 1 score of each row of ``allocs`` against every row of ``pool``."""
    pool = np.asarray(pool, dtype=np.int16)
    return _game1_from_columns(allocs, _sorted_columns(pool), pool.shape[0])


def game1_score(a, pool) -> float:
    """Average castle points of ``a`` against every allocation in ``pool``.

    On each castle the larger side takes the castle's value. A tie splits it.
    """
    alloc = _as_alloc(a)
    rows = _as_pool(pool)
    return float(game1_scores(alloc, rows)[0])


def _sample_indices(n_pool: int, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    if n_pool < 3:
        raise ValueError("pool must contain at least 3 allocations")
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    idx = rng.integers(0, n_pool, size=(n_groups, 3), dtype=np.int32)
    for _ in range(16):
        bad = (idx[:, 0] == idx[:, 1]) | (idx[:, 0] == idx[:, 2]) | (idx[:, 1] == idx[:, 2])
        n_bad = int(np.count_nonzero(bad))
        if n_bad == 0:
            return idx
        idx[bad] = rng.integers(0, n_pool, size=(n_bad, 3), dtype=np.int32)
    bad = (idx[:, 0] == idx[:, 1]) | (idx[:, 0] == idx[:, 2]) | (idx[:, 1] == idx[:, 2])
    for row in np.flatnonzero(bad):
        idx[row] = rng.choice(n_pool, size=3, replace=False)
    return idx


def _summarize(opponents: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute, per castle, the opponent max and the shares for a loss or a tie."""
    max_opp = opponents.max(axis=1)
    at_max = (opponents == max_opp[:, None, :]).astype(np.int16)
    count = at_max.sum(axis=1, dtype=np.int16)
    if int(count.min()) < 1:
        raise RuntimeError("opponent summary has an empty castle")
    # int16 on purpose: a castle share is at most 60, and int64 temps dominated the runtime.
    lose_share = (SCALED_VALUES // count).astype(np.int16)
    tie_share = (SCALED_VALUES // (count + 1)).astype(np.int16)
    return max_opp, at_max, lose_share, tie_share


def _prepare_groups(
    pool: np.ndarray,
    n_groups: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = _sample_indices(pool.shape[0], n_groups, rng)
    return _summarize(pool[idx])


def _mean_payouts(
    allocs: np.ndarray,
    max_opp: np.ndarray,
    at_max: np.ndarray,
    lose_share: np.ndarray,
    tie_share: np.ndarray,
) -> np.ndarray:
    """Mean rank payout of each allocation against a frozen set of opponent groups."""
    allocs = np.asarray(allocs, dtype=np.int16)
    if allocs.ndim == 1:
        allocs = allocs.reshape(1, -1)
    scaled = SCALED_VALUES.astype(np.int16)
    zero = np.int16(0)
    out = np.empty(allocs.shape[0], dtype=np.float64)
    for start in range(0, allocs.shape[0], SCORE_CHUNK):
        block = allocs[start : start + SCORE_CHUNK]
        troops = block[:, None, :]
        ahead = troops > max_opp
        tied = troops == max_opp
        own = np.where(ahead, scaled, np.where(tied, tie_share, zero))
        own_pts = own.sum(axis=2, dtype=np.int16)
        share = np.where(ahead, zero, np.where(tied, tie_share, lose_share))
        opp_pts = np.einsum("gpc,bgc->bgp", at_max, share, optimize=True)
        higher = (opp_pts > own_pts[:, :, None]).sum(axis=2)
        n_tied = 1 + (opp_pts == own_pts[:, :, None]).sum(axis=2)
        out[start : start + SCORE_CHUNK] = PAYOUT_TABLE[higher, n_tied].mean(axis=1)
    return out


def game2_score(a, pool, n_groups: int = N_GROUPS, rng: np.random.Generator | None = None) -> float:
    """Average Game 2 payout of ``a`` over ``n_groups`` random groups of three opponents.

    Each castle's value is split among the players who put the most soldiers
    there. The four totals are ranked and paid 9, 4, 2, 0. Tied players split
    the payouts for the ranks they share. Pass ``rng`` for a reproducible draw.
    """
    alloc = _as_alloc(a)
    rows = _as_pool(pool)
    if rng is None:
        rng = np.random.default_rng()
    prepared = _prepare_groups(rows, n_groups, rng)
    return float(_mean_payouts(alloc, *prepared)[0])


def neighbors(alloc: np.ndarray) -> np.ndarray:
    """Allocations reached by moving 1..5 soldiers from one castle to another."""
    alloc = np.asarray(alloc, dtype=np.int16)
    rows: list[np.ndarray] = []
    for src in range(N_CASTLES):
        k_max = min(MOVE_MAX, int(alloc[src]))
        for k in range(1, k_max + 1):
            step = np.int16(k)
            for dst in range(N_CASTLES):
                if dst == src:
                    continue
                nxt = alloc.copy()
                nxt[src] -= step
                nxt[dst] += step
                rows.append(nxt)
    if not rows:
        return np.empty((0, N_CASTLES), dtype=np.int16)
    return np.stack(rows)


def best_response(
    evaluate,
    rng: np.random.Generator,
    warm: np.ndarray | None = None,
    n_random: int = 96,
    n_starts: int = 1,
    max_steps: int = MAX_HILL_STEPS,
) -> tuple[np.ndarray, float, int, bool]:
    """Best of a random sample, polished by strict hill-climbing.

    ``evaluate(allocs) -> scores`` must be fixed for this call. The previous
    best response can be passed as ``warm`` and is included in the sample.
    """
    starts = random_allocations(rng, n_random)
    if warm is not None:
        warm_row = np.asarray(warm, dtype=np.int16).reshape(1, -1)
        starts = np.vstack([warm_row, starts])
    scores = np.asarray(evaluate(starts), dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    best = starts[int(order[0])].copy()
    best_score = float(scores[int(order[0])])
    steps = 0
    capped = False
    climbed: set[tuple[int, ...]] = set()
    for pos in order[:n_starts]:
        key = tuple(int(x) for x in starts[pos])
        if key in climbed:
            continue
        climbed.add(key)
        current = starts[pos].copy()
        current_score = float(scores[pos])
        for _ in range(max_steps):
            nbrs = neighbors(current)
            if len(nbrs) == 0:
                break
            nbr_scores = np.asarray(evaluate(nbrs), dtype=np.float64)
            pick = int(np.argmax(nbr_scores))
            improved = float(nbr_scores[pick])
            if improved <= current_score:
                break
            current = nbrs[pick].copy()
            current_score = improved
            steps += 1
        else:
            capped = True
        if current_score > best_score:
            best = current.copy()
            best_score = current_score
    return best, best_score, steps, capped


def _scoring_pool(pool: np.ndarray, recent_window: int | None) -> np.ndarray:
    """Rows used as opponents: whole pool, or only the most recent ``recent_window``."""
    if recent_window is None or recent_window >= pool.shape[0]:
        return pool
    if recent_window < 1:
        raise ValueError("recent_window must be positive or None")
    return pool[-recent_window:]


def _evaluator(
    kind: str,
    pool: np.ndarray,
    rng: np.random.Generator,
) -> Callable[[np.ndarray], np.ndarray]:
    if kind == "game1":
        columns = _sorted_columns(pool)
        n_pool = pool.shape[0]

        def evaluate(allocs: np.ndarray, columns=columns, n_pool=n_pool) -> np.ndarray:
            return _game1_from_columns(allocs, columns, n_pool)

        return evaluate

    # Freeze one draw so the random sample and the hill-climb share an objective.
    prepared = _prepare_groups(pool, N_GROUPS, rng)

    def evaluate(allocs: np.ndarray, prepared=prepared) -> np.ndarray:
        return _mean_payouts(allocs, *prepared)

    return evaluate


def _symmetric_value(kind: str) -> float:
    return GAME1_VALUE if kind == "game1" else GAME2_VALUE


def run_fictitious_play(
    kind: str,
    rng: np.random.Generator,
    n_rounds: int = N_ROUNDS,
    burn_in: int = BURN_IN,
    recent_window: int | None = RECENT_WINDOW,
    human: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Grow a pool by best responses, optionally dropping the initial random seed.

    After ``burn_in`` rounds the starting random allocations are removed so the
    pool keeps only best responses. When ``recent_window`` is set, each search
    scores against only the most recent that many pool rows. Human mode seeds
    multiples-of-5 allocations and forces ``burn_in=0`` so they stay.

    Every ``EXPLOIT_EVERY`` rounds a fresh best response is searched against the
    current scoring pool; its score minus the symmetric value is recorded as the
    exploitability gap.
    """
    if human:
        burn_in = 0
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if burn_in > n_rounds:
        raise ValueError("burn_in cannot exceed n_rounds")
    pool = human_like_allocations(rng, INITIAL_POOL) if human else random_allocations(rng, INITIAL_POOL)
    initial_n = pool.shape[0]
    history = np.empty((n_rounds + 1, N_CASTLES), dtype=np.float64)
    history[0] = pool.mean(axis=0)
    exploit_rounds: list[int] = []
    exploit_gaps: list[float] = []
    warm: np.ndarray | None = None
    capped_rounds = 0
    n_random = 256 if kind == "game1" else 48
    n_starts = 3 if kind == "game1" else 1
    sym = _symmetric_value(kind)
    started = time.perf_counter()
    window_note = "whole pool" if recent_window is None else f"last {recent_window}"
    seed_note = "human-like (multiples of 5 + commons)" if human else f"{INITIAL_POOL} random"
    _log(
        f"{kind}: {seed_note}, then {n_rounds} best responses "
        f"(burn_in={burn_in}, score vs {window_note})"
    )
    for rnd in range(1, n_rounds + 1):
        opponents = _scoring_pool(pool, recent_window)
        evaluate = _evaluator(kind, opponents, rng)
        starts = 3 if warm is None else n_starts
        chosen, score, steps, capped = best_response(
            evaluate,
            rng,
            warm=warm,
            n_random=n_random,
            n_starts=starts,
            max_steps=MAX_HILL_STEPS,
        )
        if capped:
            capped_rounds += 1
        pool = np.vstack([pool, chosen.reshape(1, -1)])
        if burn_in > 0 and rnd == burn_in:
            pool = pool[initial_n:].copy()
            _log(f"  burn-in complete: dropped {initial_n} random allocations, pool={len(pool)}")
        history[rnd] = pool.mean(axis=0)
        warm = chosen.copy()
        if rnd % EXPLOIT_EVERY == 0 or rnd == n_rounds:
            # Fresh search: no warm start, same sampling + hill-climb as the loop.
            check_pool = _scoring_pool(pool, recent_window)
            check_eval = _evaluator(kind, check_pool, rng)
            _br, br_score, _steps, _capped = best_response(
                check_eval,
                rng,
                warm=None,
                n_random=n_random,
                n_starts=n_starts,
                max_steps=MAX_HILL_STEPS,
            )
            gap = float(br_score - sym)
            exploit_rounds.append(rnd)
            exploit_gaps.append(gap)
            _log(f"  exploitability @ {rnd:3d}: gap {gap:+.4f}  (BR {br_score:.4f} vs {sym})")
        if rnd == 1 or rnd % 25 == 0 or rnd == n_rounds:
            elapsed = time.perf_counter() - started
            label = "score" if kind == "game1" else "sample"
            _log(
                f"  round {rnd:3d}/{n_rounds}  {label} {score:8.4f}  "
                f"steps {steps:3d}  {_format_alloc(chosen)}  ({elapsed:.0f}s)"
            )
    if not np.allclose(history.sum(axis=1), TOTAL_TROOPS, atol=1e-6):
        raise RuntimeError("average allocations do not sum to 100")
    return (
        pool,
        history,
        np.asarray(exploit_rounds, dtype=np.int32),
        np.asarray(exploit_gaps, dtype=np.float64),
        capped_rounds,
    )


def _format_alloc(alloc: np.ndarray) -> str:
    return "(" + ", ".join(f"{int(x):3d}" for x in alloc) + ")"


def round_mean_to_allocation(mean: np.ndarray, total: int = TOTAL_TROOPS) -> np.ndarray:
    """Round a mean allocation to non-negative integers that sum to ``total``.

    Largest-remainder method: take floors, then give leftover soldiers to the
    castles with the largest fractional parts. A rough guide, not a strategy.
    """
    mean = np.asarray(mean, dtype=np.float64)
    if mean.shape != (N_CASTLES,):
        raise ValueError("mean must have shape (5,)")
    if not np.isfinite(mean).all() or abs(float(mean.sum()) - total) > 1e-6:
        raise ValueError("mean must be finite and sum to the troop total")
    floors = np.floor(mean).astype(np.int64)
    rem = int(total - floors.sum())
    order = np.argsort(-(mean - floors), kind="mergesort")
    out = floors.copy()
    out[order[:rem]] += 1
    if int(out.sum()) != total or np.any(out < 0):
        raise RuntimeError("rounded mean is not a valid allocation")
    return out.astype(np.int16)


def plot_history(history: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    rounds = np.arange(history.shape[0])
    for castle in range(N_CASTLES):
        worth = int(CASTLE_VALUES[castle])
        ax.plot(
            rounds,
            history[:, castle],
            color=PLOT_COLORS[castle],
            linewidth=1.8,
            label=f"Worth {worth}",
        )
    ax.axhline(
        TOTAL_TROOPS / N_CASTLES,
        color="#9a9a9a",
        linestyle="--",
        linewidth=1,
        label="Equal split",
    )
    ax.set_title(title)
    ax.set_xlabel("Round (0 = initial random pool)")
    ax.set_ylabel("Average soldiers")
    ax.set_xlim(0, history.shape[0] - 1)
    ax.grid(True, axis="y", color="#e6e6e6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_exploitability(rounds: np.ndarray, gaps: np.ndarray, title: str, path: Path) -> None:
    """Plot best-response score minus the symmetric value at each checkpoint."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(rounds, gaps, color=PLOT_COLORS[0], marker="o", linewidth=1.8)
    ax.axhline(0.0, color="#9a9a9a", linestyle="--", linewidth=1, label="Symmetric value")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Exploitability gap")
    if len(rounds) > 1:
        ax.set_xlim(float(rounds[0]), float(rounds[-1]))
    ax.grid(True, axis="y", color="#e6e6e6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def report(
    kind: str,
    pool: np.ndarray,
    history: np.ndarray,
    exploit_rounds: np.ndarray,
    exploit_gaps: np.ndarray,
    rng: np.random.Generator,
    capped_rounds: int,
    tag: str = "standard",
    seed: int = SEED,
) -> None:
    uniq, counts = np.unique(pool, axis=0, return_counts=True)
    if kind == "game1":
        scores = game1_scores(uniq, pool)
        note = "head-to-head average against the final pool (symmetric value 7.5)"
    else:
        scores = _mean_payouts(uniq, *_prepare_groups(pool, N_GROUPS, rng))
        note = (
            f"mean payout over {N_GROUPS:,} shared opponent groups "
            "(a symmetric seat pays 3.75)"
        )
    order = np.argsort(-scores, kind="mergesort")
    order = order[np.argsort(-counts[order], kind="mergesort")]
    _log("")
    _log(f"{kind} [{tag}]: pool {len(pool)}, distinct {len(uniq)}")
    _log(f"most common allocations, {note}")
    _log("allocation columns are soldiers on castles worth 1, 2, 3, 4, 5")
    _log(f"{'count':>5}  {'score':>8}  allocation")
    for pos in order[:N_SHOW]:
        _log(f"{int(counts[pos]):5d}  {scores[pos]:8.4f}  {_format_alloc(uniq[pos])}")
    mean_vec = history[-1]
    means = "  ".join(
        f"worth {int(CASTLE_VALUES[c])} {mean_vec[c]:6.2f}" for c in range(N_CASTLES)
    )
    _log(f"final mean  {means}")
    guide = round_mean_to_allocation(mean_vec)
    _log(f"rounded mean (rough guide)  {_format_alloc(guide)}")
    if len(exploit_rounds):
        gaps = "  ".join(f"r{int(r)} {g:+.3f}" for r, g in zip(exploit_rounds, exploit_gaps))
        _log(f"exploitability gaps  {gaps}")
    if capped_rounds:
        _log(f"hill-climb step cap hit on {capped_rounds} rounds")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    title = "Game 1" if kind == "game1" else "Game 2"
    mean_path = FIG_DIR / f"blotto_{kind}_{tag}.png"
    plot_history(history, f"{title} [{tag}]: average soldiers per castle (seed {seed})", mean_path)
    _log(f"wrote {mean_path}")
    if len(exploit_rounds):
        gap_path = FIG_DIR / f"blotto_{kind}_{tag}_gap.png"
        plot_exploitability(
            exploit_rounds,
            exploit_gaps,
            f"{title} [{tag}]: exploitability gap (seed {seed})",
            gap_path,
        )
        _log(f"wrote {gap_path}")


def _self_check() -> None:
    from fractions import Fraction

    def check(cond: bool, message: str) -> None:
        if not cond:
            raise AssertionError(message)

    def compositions(total: int, k: int):
        if k == 1:
            yield (total,)
            return
        for troops in range(total + 1):
            for rest in compositions(total - troops, k - 1):
                yield (troops,) + rest

    rng = np.random.default_rng(0)
    small = random_allocations(rng, 3000, total=4, k=3)
    found = set(map(tuple, small.tolist()))
    expect = set(compositions(4, 3))
    check(found == expect, f"composition support {found ^ expect}")
    _uniq, comp_counts = np.unique(small, axis=0, return_counts=True)
    check(int(comp_counts.min()) > 100 and int(comp_counts.max()) < 350, "composition sampler looks biased")

    sample = random_allocations(rng, 4000)
    check(np.all(sample >= 0) and np.all(sample.sum(axis=1) == TOTAL_TROOPS), "invalid allocation")
    check(np.all(np.abs(sample.mean(axis=0) - 20) < 1.0), f"biased marginals {sample.mean(axis=0)}")
    human_pool = human_like_allocations(rng, INITIAL_POOL)
    check(human_pool.shape == (INITIAL_POOL, N_CASTLES), "human pool size")
    check(np.all(human_pool.sum(axis=1) == TOTAL_TROOPS), "human pool sum")
    check(np.all(human_pool % 5 == 0), "human pool not multiples of 5")
    for row in COMMON_HUMAN:
        check(np.any(np.all(human_pool == row, axis=1)), f"missing common human {row}")

    base = np.array([10, 10, 10, 10, 60], dtype=np.int16)
    nbrs = neighbors(base)
    check(len(nbrs) == 100 and len(np.unique(nbrs, axis=0)) == 100, "expected 100 neighbors")
    check(np.all(nbrs.sum(axis=1) == TOTAL_TROOPS) and np.all(nbrs >= 0), "neighbor broke the budget")
    distance = np.abs(nbrs.astype(np.int32) - base).sum(axis=1)
    check(np.all((distance >= 2) & (distance <= 10) & (distance % 2 == 0)), "neighbor is not a 1..5 transfer")
    corner = neighbors(np.array([0, 0, 0, 0, 100], dtype=np.int16))
    check(len(corner) == 20, f"corner neighborhood has {len(corner)} moves")

    def oracle_game1(left, right) -> float:
        score = 0.0
        for castle, value in enumerate(CASTLE_VALUES):
            if left[castle] > right[castle]:
                score += float(value)
            elif left[castle] == right[castle]:
                score += float(value) / 2
        return score

    guide = round_mean_to_allocation(np.array([19.6, 20.2, 20.0, 20.1, 20.1]))
    check(int(guide.sum()) == TOTAL_TROOPS and np.all(guide >= 0), "rounded mean invalid")
    check(tuple(int(x) for x in guide) == (20, 20, 20, 20, 20), f"unexpected rounded mean {guide}")

    identical = np.array([20, 20, 20, 20, 20], dtype=np.int16)
    stacked = np.array([0, 0, 0, 0, 100], dtype=np.int16)
    opposite = np.array([100, 0, 0, 0, 0], dtype=np.int16)
    check(abs(game1_score(identical, identical.reshape(1, -1)) - 7.5) < 1e-12, "tie score")
    check(abs(game1_score(stacked, opposite.reshape(1, -1)) - 9.5) < 1e-12, "known game 1 score")
    pool = random_allocations(rng, 20)
    allocs = random_allocations(rng, 8)
    got = game1_scores(allocs, pool)
    for row in range(len(allocs)):
        ref = np.mean([oracle_game1(allocs[row], pool[col]) for col in range(len(pool))])
        check(abs(got[row] - ref) < 1e-9, "game 1 batch mismatch")
    left, right = allocs[0], pool[0]
    paired = game1_score(left, right.reshape(1, -1)) + game1_score(right, left.reshape(1, -1))
    check(abs(paired - 15) < 1e-9, "game 1 not constant-sum")

    def walk_payout(scores: list[int]) -> list[float]:
        order = sorted(range(4), key=lambda i: (-scores[i], i))
        payouts = [0.0] * 4
        start = 0
        while start < 4:
            end = start + 1
            while end < 4 and scores[order[end]] == scores[order[start]]:
                end += 1
            share = float(RANK_PAYOUTS[start:end].sum() / (end - start))
            for pos in range(start, end):
                payouts[order[pos]] = share
            start = end
        return payouts

    tie_rng = np.random.default_rng(1)
    for _ in range(300):
        scores = [int(x) for x in tie_rng.integers(0, 6, size=4)]
        higher = sum(score > scores[0] for score in scores[1:])
        tied = 1 + sum(score == scores[0] for score in scores[1:])
        walked = walk_payout(scores)
        check(abs(PAYOUT_TABLE[higher, tied] - walked[0]) < 1e-12, "payout table mismatch")
        check(abs(sum(walked) - 15) < 1e-12, "rank payouts do not sum to 15")

    def oracle_game2(players) -> float:
        points = [Fraction(0) for _ in range(4)]
        for castle, value in enumerate((1, 2, 3, 4, 5)):
            column = [int(players[player, castle]) for player in range(4)]
            best = max(column)
            winners = [player for player, troops in enumerate(column) if troops == best]
            share = Fraction(value, len(winners))
            for player in winners:
                points[player] += share
        higher = sum(points[player] > points[0] for player in range(1, 4))
        tied = sum(point == points[0] for point in points)
        prizes = [Fraction(9), Fraction(4), Fraction(2), Fraction(0)]
        return float(sum(prizes[higher : higher + tied]) / tied)

    many = random_allocations(rng, 40)
    groups = random_allocations(rng, 15).reshape(5, 3, 5)
    got = _mean_payouts(many, *_summarize(groups))
    for row in range(len(many)):
        ref = np.mean(
            [oracle_game2(np.vstack([many[row], groups[group]])) for group in range(groups.shape[0])]
        )
        check(abs(got[row] - ref) < 1e-9, "game 2 batch mismatch")

    alone = np.array([0, 0, 0, 0, 100], dtype=np.int16)
    field = np.array(
        [[100, 0, 0, 0, 0], [0, 100, 0, 0, 0], [0, 0, 100, 0, 0]],
        dtype=np.int16,
    )
    check(abs(game2_score(alone, field, n_groups=40, rng=rng) - 9) < 1e-12, "unique first payout")
    flat = np.tile(identical, (5, 1))
    check(abs(game2_score(identical, flat, n_groups=30, rng=rng) - 3.75) < 1e-12, "four-way tie payout")
    tied_pool = np.array(
        [[0, 0, 0, 50, 50], [50, 50, 0, 0, 0], [0, 0, 100, 0, 0]],
        dtype=np.int16,
    )
    check(
        abs(game2_score(tied_pool[0], tied_pool, n_groups=20, rng=rng) - 6.5) < 1e-12,
        "split first-place payout",
    )

    weak = np.tile(np.array([100, 0, 0, 0, 0], dtype=np.int16), (25, 1))

    def evaluate_weak(allocs, pool=weak):
        return game1_scores(allocs, pool)

    found_br, found_score, _steps, _capped = best_response(
        evaluate_weak,
        np.random.default_rng(123),
        n_random=16,
        n_starts=2,
        max_steps=30,
    )
    check(found_score >= 13.99, f"search missed the obvious best response ({found_score}, {found_br})")
    check(int(found_br.sum()) == TOTAL_TROOPS and np.all(found_br >= 0), "search returned an invalid allocation")

    try:
        game1_score([1, 1, 1, 1, 1], identical.reshape(1, -1))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a short allocation")
    try:
        game2_score(identical, np.vstack([identical, identical]), n_groups=1, rng=rng)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a pool smaller than 3")


def _run_mode(
    *,
    human: bool,
    n_rounds: int,
    burn_in: int,
    seed: int,
    recent_window: int | None,
) -> dict[str, dict]:
    label = "human" if human else "standard"
    _log("")
    _log(f"=== mode: {label} ===")
    results: dict[str, dict] = {}
    for kind in ("game1", "game2"):
        rng = np.random.default_rng(seed)
        started = time.perf_counter()
        effective_burn = 0 if human else burn_in
        pool, history, exploit_rounds, exploit_gaps, capped_rounds = run_fictitious_play(
            kind,
            rng,
            n_rounds=n_rounds,
            burn_in=effective_burn,
            recent_window=recent_window,
            human=human,
        )
        report(
            kind,
            pool,
            history,
            exploit_rounds,
            exploit_gaps,
            rng,
            capped_rounds,
            tag=label,
            seed=seed,
        )
        elapsed = time.perf_counter() - started
        _log(f"{kind} ({label}) finished in {elapsed:.0f}s")
        results[kind] = {
            "mean": history[-1].copy(),
            "guide": round_mean_to_allocation(history[-1]),
            "exploit_rounds": exploit_rounds.copy(),
            "exploit_gaps": exploit_gaps.copy(),
            "pool_size": int(len(pool)),
        }
    return results


def _print_side_by_side(standard: dict[str, dict], human: dict[str, dict]) -> None:
    _log("")
    _log("=== side by side ===")
    for kind in ("game1", "game2"):
        s, h = standard[kind], human[kind]
        _log(f"{kind}")
        _log(f"  standard mean   {_format_alloc(np.rint(s['mean']).astype(int))}  "
             f"guide {_format_alloc(s['guide'])}  pool {s['pool_size']}")
        _log(f"  human mean      {_format_alloc(np.rint(h['mean']).astype(int))}  "
             f"guide {_format_alloc(h['guide'])}  pool {h['pool_size']}")
        if len(s["exploit_gaps"]) and len(h["exploit_gaps"]):
            s_gap = float(s["exploit_gaps"][-1])
            h_gap = float(h["exploit_gaps"][-1])
            _log(f"  final exploitability  standard {s_gap:+.3f}  human {h_gap:+.3f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colonel Blotto fictitious play")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument(
        "--human",
        action="store_true",
        help="seed multiples-of-5 human-like allocations and keep burn-in off",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="run standard and human modes, then print a side-by-side summary",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _log(f"Colonel Blotto fictitious play, seed {args.seed}")
    _log("castles worth 1, 2, 3, 4, 5; allocations are 5 non-negative integers summing to 100")
    _self_check()
    # Default: standard. --human: human only. --both: standard then human with a summary.
    if args.both:
        modes = (False, True)
    elif args.human:
        modes = (True,)
    else:
        modes = (False,)
    collected: dict[str, dict[str, dict]] = {}
    for human in modes:
        collected["human" if human else "standard"] = _run_mode(
            human=human,
            n_rounds=args.rounds,
            burn_in=args.burn_in,
            seed=args.seed,
            recent_window=RECENT_WINDOW,
        )
    if "standard" in collected and "human" in collected:
        _print_side_by_side(collected["standard"], collected["human"])


if __name__ == "__main__":
    main()
