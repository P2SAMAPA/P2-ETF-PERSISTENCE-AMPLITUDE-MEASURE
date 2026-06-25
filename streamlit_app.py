import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Persistence Amplitude Measure Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1a1a2e; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #e94560; }
.etf-card    { background:linear-gradient(135deg,#1a1a2e 0%,#e94560 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#1a1a2e 0%,#c0392b 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Persistence Amplitude Measure Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Cohen-Steiner et al. (2007) · '
    'H0 sub-level set filtration · '
    'Amplitude-weighted persistence diagram · '
    'Distinct from TDA-Homology (lifetime) and Persistent Excitation (control theory)</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## PAM Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Min amplitude:** {config.MIN_AMPLITUDE}")
st.sidebar.markdown(
    f"**Weights:** Amp mean {config.WEIGHT_AMP_MEAN:.0%} | "
    f"CV {config.WEIGHT_AMP_CV:.0%} | "
    f"Skew {config.WEIGHT_AMP_SKEW:.0%} | "
    f"LT ratio {config.WEIGHT_LIFETIME_RATIO:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "pam_engine_2")
tab2_path = find_latest(files, "pam_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Persistence Amplitude Signal")

    with st.expander("Persistence Amplitude Methodology", expanded=True):
        st.markdown("""
**Persistent Homology** builds a persistence diagram D = {(birth_i, death_i)}
from the sub-level set filtration of the return series. Each pair captures
a topological feature (connected component) that appears at level birth_i
and disappears at level death_i.

**Standard persistence (TDA-HOMOLOGY engine):**
```
lifetime_i = death_i - birth_i
```
Measures how long the feature persists across the filtration scale.

**Persistence Amplitude (this engine):**
```
amplitude_i = |birth_i + death_i| / 2
```
Measures the absolute return magnitude at which the feature lives.

Two ETFs can have identical persistence diagrams but completely different
amplitude profiles — and this engine extracts what lifetime ignores.

**Score components:**

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| Amplitude mean | mean(amplitudes) | Average return magnitude — lower = calmer |
| Amplitude CV | std/mean | Scale-free = near-critical (cf. SOC engine) |
| Amplitude skew | skewness | Positive = large upside bursts = momentum |
| Lifetime ratio | amp/lifetime | Low = slow trends = positive signal |

**Three-engine comparison:**

| Engine | Field | Measures |
|--------|-------|---------|
| PERSISTENT-EXCITATION | Control theory | Signal richness for ID |
| TDA-HOMOLOGY | Topology | Feature lifetime (how long) |
| PAM (this engine) | Topology | Feature amplitude (how large) |
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">PAM score = {etf['pam_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    score = info.get("score", info) if isinstance(info, dict) else info
                    win   = info.get("best_window", "N/A") if isinstance(info, dict) else "N/A"
                    rows.append({"ETF": t, "PAM Score": score, "Best Window (d)": win})
                df = pd.DataFrame(rows).sort_values("PAM Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Cohen-Steiner et al. (2007) · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore PAM Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **63d** — short-term amplitude structure; captures recent vol bursts
- **126d** — 6-month amplitude profile; more stable diagram statistics
- **252d** — 1-year amplitude; full annual return cycle represented
- **504d** — 2-year amplitude; structural amplitude patterns
        """)

    st.markdown(f"### PAM Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">PAM score = {etf['pam_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(rows, columns=["ETF", "PAM Score"])
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
