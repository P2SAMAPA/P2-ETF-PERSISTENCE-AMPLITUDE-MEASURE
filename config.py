import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-persistence-amplitude-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Persistence diagram construction ─────────────────────────────────────────
# We use the sub-level set filtration on the return time series.
# At threshold epsilon, the sub-level set is {t : r(t) <= epsilon}.
# As epsilon increases from min(r) to max(r), connected components (H0)
# are born and merge — this is the 0-dimensional persistence diagram.
#
# For 1-dimensional features (loops / H1), we use the Rips complex
# on a delay-embedding of the return path.
#
# We focus on H0 (connected components) for efficiency — this is
# computable in O(N log N) via a simple union-find on sorted returns.

# Delay embedding dimension for H1 Rips complex (set to 0 to skip H1)
# H0 only is fast and captures vol bursts and trend cycles
H0_ONLY = True

# Rips complex parameters (only used if H0_ONLY = False)
RIPS_DELAY  = 5    # delay in days for embedding
RIPS_DIM    = 2    # embedding dimension

# ── Amplitude vs lifetime distinction ────────────────────────────────────────
# For each persistence pair (birth, death):
#   lifetime  = death - birth                (standard persistence)
#   amplitude = (birth + death) / 2          (midpoint = absolute return level)
#   amplitude_range = death - birth in return space (= same as lifetime for H0)
#
# For H0 sub-level set filtration on returns:
#   birth  = return value at which component appears (local minimum)
#   death  = return value at which it merges (local maximum reached)
#   amplitude = |death + birth| / 2  → absolute return magnitude of the cycle
#   lifetime  = death - birth         → return range of the cycle

# Minimum amplitude to include in summary statistics
MIN_AMPLITUDE = 1e-4

# ── Score construction ────────────────────────────────────────────────────────
# From the amplitude-weighted persistence diagram:
#
#   amp_mean   : amplitude-weighted mean of persistence diagram
#                Large = big moves dominate → high vol regime
#                Small = small moves dominate → calm regime
#
#   amp_cv     : coefficient of variation of amplitudes (std/mean)
#                High CV → scale-free / near-critical (cf. SOC engine)
#                Low CV  → homogeneous move sizes → trending
#
#   amp_skew   : skewness of amplitude distribution
#                Positive skew → occasional large positive bursts → momentum
#                Negative skew → occasional large negative bursts → mean-rev
#
#   lifetime_ratio : mean_amplitude / mean_lifetime
#                Large → features are large relative to their duration → bursts
#                Small → features are small relative to duration → slow trends

WEIGHT_AMP_MEAN      = 0.35
WEIGHT_AMP_CV        = 0.30
WEIGHT_AMP_SKEW      = 0.20
WEIGHT_LIFETIME_RATIO = 0.15

TOP_N = 3
