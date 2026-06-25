"""
pam_engine.py — Persistence Amplitude Measure Engine
=====================================================

Theory
------
**Persistent Homology (review)**

Given a time series r: {1,...,T} → R, the sub-level set filtration builds:

    R_epsilon = {t : r(t) <= epsilon}

as epsilon increases from min(r) to max(r). Connected components (H0 features)
are born when new local minima appear and die when they merge at local maxima.
This produces a **persistence diagram** D = {(b_i, d_i)} where:

    b_i = birth value  (return level at which component i appeared)
    d_i = death value  (return level at which component i merged)

**Standard persistence (TDA-HOMOLOGY engine):**

    lifetime_i = d_i - b_i    (how long the feature persists in filtration scale)

**Persistence Amplitude (this engine):**

    amplitude_i = |b_i + d_i| / 2    (absolute return level of the cycle)

These are orthogonal summary statistics of the same persistence diagram:
- Lifetime measures *duration* in filtration scale
- Amplitude measures *magnitude* in return space

**Amplitude-Weighted Persistence Diagram**

Instead of the standard barcode (ordered by lifetime), we weight each
persistence pair by its amplitude:

    w_i = amplitude_i / Sigma(amplitude_j)

This reweights the diagram so that large-return events dominate the summary
statistics, regardless of how long they persisted.

**Sub-level set H0 persistence via Union-Find (Elder's lemma)**

For 1D time series, the H0 persistence diagram under sub-level set filtration
can be computed exactly in O(N log N):

1. Sort the N return values by level (ascending)
2. Process each point in order; use Union-Find to track connected components
3. When two components merge, record (birth of younger, current level) as a
   persistence pair
4. The oldest component (global minimum) gets paired with +infinity

This is essentially the same as finding all local minima/maxima pairs.

**Distinction from TDA-HOMOLOGY (in suite)**

TDA-HOMOLOGY uses the standard persistence summary (Betti numbers, total
persistence, persistence entropy) — all derived from lifetimes.

This engine uses amplitude-weighted summaries:
    - amplitude-weighted mean  (what return magnitude do features live at?)
    - coefficient of variation of amplitudes (scale-free or homogeneous?)
    - skewness of amplitudes (asymmetry of up vs. down moves)
    - amplitude/lifetime ratio (burst vs. slow-trend discrimination)

Two ETFs can have identical persistence diagrams (same lifetimes) but
completely different amplitude profiles — and vice versa.

**Distinction from PERSISTENT-EXCITATION (in suite)**

PERSISTENT-EXCITATION is a control theory concept measuring whether the
input signal is rich enough for system identification. No topological content.

References
----------
- Cohen-Steiner, D., Edelsbrunner, H. & Harer, J. (2007). Stability of
  persistence diagrams. Discrete & Computational Geometry, 37(1), 103–120.
- Chazal, F., de Silva, V., Glisse, M. & Oudot, S. (2016). The Structure and
  Stability of Persistence Modules. Springer.
- Edelsbrunner, H. & Harer, J. (2010). Computational Topology: An Introduction.
  AMS.
- Carlsson, G. (2009). Topology and data. Bulletin of the AMS, 46(2), 255–308.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── Union-Find for H0 persistence ────────────────────────────────────────────

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n
        # Birth level: when was this component created?
        self.birth  = [None] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> Tuple[int, int]:
        """
        Merge components of x and y.
        Returns (younger_root, older_root) — younger dies, older survives.
        The component with the higher birth level is younger.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx, rx

        # Older component (lower birth level) survives
        if self.birth[rx] > self.birth[ry]:
            rx, ry = ry, rx   # ry is younger (higher birth)

        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx

        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

        return ry, rx   # younger, older


# ── H0 persistence diagram via sub-level set filtration ──────────────────────

def _compute_h0_persistence(returns: np.ndarray) -> np.ndarray:
    """
    Compute H0 persistence diagram for a 1D return series via
    sub-level set filtration (Elder's lemma).

    Returns array of shape (N_pairs, 2) where each row is (birth, death).
    The last pair has death = max(returns) (oldest component, paired to sup).

    Algorithm:
    1. Sort indices by return value (ascending)
    2. Process each point; check left/right neighbours
    3. Use Union-Find to track components
    4. Record (birth, death) when components merge
    """
    T  = len(returns)
    if T < 3:
        return np.empty((0, 2))

    # Sort indices by return value
    order = np.argsort(returns)

    uf       = _UnionFind(T)
    added    = np.zeros(T, dtype=bool)
    pairs    = []

    for idx in order:
        val = returns[idx]
        uf.birth[idx] = val
        added[idx]    = True

        # Check left neighbour
        if idx > 0 and added[idx-1]:
            younger, older = uf.union(idx, idx-1)
            if younger != older:
                pairs.append((uf.birth[younger], val))
                uf.birth[older] = min(uf.birth[younger], uf.birth[older])

        # Check right neighbour
        if idx < T-1 and added[idx+1]:
            younger, older = uf.union(idx, idx+1)
            if younger != older:
                pairs.append((uf.birth[younger], val))
                uf.birth[older] = min(uf.birth[younger], uf.birth[older])

    # The oldest component is paired to the supremum
    oldest_root  = uf.find(order[0])
    oldest_birth = uf.birth[oldest_root]
    pairs.append((oldest_birth, returns.max()))

    return np.array(pairs)   # (N_pairs, 2): each row = (birth, death)


# ── Amplitude summary statistics ──────────────────────────────────────────────

def _amplitude_stats(
    diagram: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute amplitude-based summary statistics from a persistence diagram.

    Parameters
    ----------
    diagram : (N, 2) array of (birth, death) pairs

    Returns
    -------
    amp_mean       : amplitude-weighted mean
    amp_cv         : coefficient of variation of amplitudes
    amp_skew       : skewness of amplitude distribution
    lifetime_ratio : mean_amplitude / mean_lifetime
    """
    if len(diagram) == 0:
        return 0.0, 0.0, 0.0, 0.0

    births  = diagram[:, 0]
    deaths  = diagram[:, 1]

    # Amplitude = midpoint of (birth, death) in absolute return space
    amplitudes = np.abs(births + deaths) / 2.0

    # Lifetime = death - birth (standard persistence)
    lifetimes  = deaths - births

    # Filter below minimum amplitude
    mask = amplitudes > config.MIN_AMPLITUDE
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0

    amps = amplitudes[mask]
    lts  = lifetimes[mask]

    # 1. Amplitude-weighted mean
    amp_mean = float(amps.mean())

    # 2. Coefficient of variation (std/mean)
    amp_cv = float(amps.std() / (amp_mean + 1e-10))

    # 3. Skewness of amplitude distribution
    n    = len(amps)
    if n < 3:
        amp_skew = 0.0
    else:
        mu   = amps.mean()
        std  = amps.std() + 1e-10
        amp_skew = float(np.mean(((amps - mu) / std) ** 3))
        amp_skew = float(np.clip(amp_skew, -5, 5))

    # 4. Amplitude-to-lifetime ratio
    lt_mean        = float(lts.mean()) + 1e-10
    lifetime_ratio = amp_mean / lt_mean

    return amp_mean, amp_cv, amp_skew, lifetime_ratio


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_pam_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Compute Persistence Amplitude Measure scores for all ETFs.

    For each ETF over the rolling window:
      1. Compute H0 persistence diagram via sub-level set filtration
      2. Extract amplitude-based summary statistics
      3. Score = weighted combination, cross-sectionally z-scored

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite PAM z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    if len(prices) < window + 5:
        return pd.Series(dtype=float)

    # Align macro (for macro alignment signal)
    common    = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a  = prices.loc[common]
    macro_a   = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    # Latest macro direction for sign adjustment
    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0 and len(macro_vals) >= 21:
        recent    = macro_vals[-21:]
        mac_chg   = recent[-1] - recent[0]
        # VIX direction: falling = risk-on = positive
        vix_dir   = -np.sign(mac_chg[0]) if mac_chg.shape[0] > 0 else 0.0
    else:
        vix_dir = 0.0

    raw_scores = {}

    for ticker in avail:
        price_series = prices_a[ticker].dropna()
        if len(price_series) < window + 2:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values
        ret_win = log_ret[-window:]

        if len(ret_win) < 10:
            continue

        # ── Compute H0 persistence diagram ───────────────────────────────────
        diagram = _compute_h0_persistence(ret_win)

        if len(diagram) < 2:
            continue

        # ── Amplitude statistics ──────────────────────────────────────────────
        amp_mean, amp_cv, amp_skew, lt_ratio = _amplitude_stats(diagram)

        print(f"    {ticker}: n_pairs={len(diagram)}  "
              f"amp_mean={amp_mean:.5f}  amp_cv={amp_cv:.3f}  "
              f"amp_skew={amp_skew:.3f}  lt_ratio={lt_ratio:.3f}")

        # ── Score construction ────────────────────────────────────────────────
        # Low amp_mean → small moves → calm regime → neutral/positive
        # High amp_cv → scale-free → near-critical → positive (cf. SOC)
        # Positive amp_skew → large upside bursts → momentum positive
        # Low lt_ratio → slow trends (large duration per unit amplitude) → positive

        # Negate amp_mean: we want calm (small amplitude) as positive signal
        # But combine with macro direction: calm + risk-on = strongest positive
        s_amp_mean   = -(amp_mean * 100)          # scale to comparable magnitude
        s_amp_cv     = amp_cv                      # high CV = near-critical = positive
        s_amp_skew   = amp_skew * vix_dir if vix_dir != 0 else amp_skew
        s_lt_ratio   = -lt_ratio                   # low ratio = slow trend = positive

        composite = (
            config.WEIGHT_AMP_MEAN       * s_amp_mean
            + config.WEIGHT_AMP_CV       * s_amp_cv
            + config.WEIGHT_AMP_SKEW     * s_amp_skew
            + config.WEIGHT_LIFETIME_RATIO * s_lt_ratio
        )
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
