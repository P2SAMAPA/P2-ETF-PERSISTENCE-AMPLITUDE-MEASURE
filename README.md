# 📊 P2-ETF-PERSISTENCE-AMPLITUDE-MEASURE

**Persistence Amplitude Measure Engine — Amplitude-Weighted Persistence Diagrams**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine extracts **amplitude-based** summary statistics from the persistence
diagram of ETF return series — measuring *how large* topological features are
in return space, rather than *how long* they persist (which TDA-HOMOLOGY measures).

---

## How It Differs from the Other Two "Persistent" Engines

| Engine | Field | Measures |
|--------|-------|---------|
| PERSISTENT-EXCITATION | Control theory | Signal richness for parameter identification — no topology |
| TDA-HOMOLOGY | Algebraic topology | Feature **lifetime** (death − birth in filtration scale) |
| **PAM (this engine)** | Algebraic topology | Feature **amplitude** ((birth + death)/2 in return space) |

TDA-HOMOLOGY and PAM use the same persistence diagram but extract
**orthogonal** information — one reads the spread (lifetime), the other
reads the absolute position (amplitude). They are complementary signals.

---

## Theory

### Persistence Diagram (H0 Sub-Level Set Filtration)

Given returns r: {1,...,T} → R, the sub-level set at threshold ε is:

```
R_epsilon = {t : r(t) <= epsilon}
```

As ε increases from min(r) to max(r), connected components appear (born at
local minima) and merge (die at local maxima). Each birth-death pair (b_i, d_i)
is one entry in the persistence diagram.

Computed in O(N log N) via Union-Find (Elder's lemma).

### Amplitude vs Lifetime

```
lifetime_i  = d_i - b_i          (TDA-HOMOLOGY: spread in filtration scale)
amplitude_i = |b_i + d_i| / 2    (PAM: midpoint = absolute return magnitude)
```

Two features with identical lifetimes can have very different amplitudes:
- Short-lived, large-amplitude feature → **volatility burst**
- Long-lived, small-amplitude feature → **structural trend**

Lifetime alone cannot distinguish these. Amplitude captures it directly.

### Score Components

From the amplitude distribution {amplitude_i}:

| Component | Formula | Weight | Meaning |
|-----------|---------|--------|---------|
| Amp mean | mean(amplitudes) | 35% | Average move magnitude — lower = calmer |
| Amp CV | std/mean | 30% | Scale-free = near-critical (cf. SOC engine) |
| Amp skew | skewness | 20% | Positive = upside bursts = momentum |
| Lifetime ratio | amp/lifetime | 15% | Low = slow trends |

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-PERSISTENCE-AMPLITUDE-MEASURE/
├── config.py          # Universes, amplitude thresholds, score weights
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── pam_engine.py      # Core: Union-Find H0 persistence, amplitude stats
├── trainer.py         # Orchestrator: load → score → JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Single job (O(N log N) — very fast)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-PERSISTENCE-AMPLITUDE-MEASURE
cd P2-ETF-PERSISTENCE-AMPLITUDE-MEASURE
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-persistence-amplitude-results`

---

## References

- Cohen-Steiner, D., Edelsbrunner, H. & Harer, J. (2007). Stability of
  persistence diagrams. *Discrete & Computational Geometry*, 37(1), 103–120.
- Chazal, F., de Silva, V., Glisse, M. & Oudot, S. (2016). *The Structure and
  Stability of Persistence Modules*. Springer.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.
- Carlsson, G. (2009). Topology and data. *Bulletin of the AMS*, 46(2), 255–308.
