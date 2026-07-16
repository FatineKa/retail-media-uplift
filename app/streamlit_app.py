"""Budget-allocation dashboard. Run after training:
python -m src.train && streamlit run app/streamlit_app.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.policy import policy_profit, profit_curve

REPORTS_DIR = Path("reports")

COLOR_UPLIFT = "#2a78d6"       # categorical slot 1 (blue) -- hero policy
COLOR_PROPENSITY = "#eb6834"   # categorical slot 6 (orange) -- naive competitor
COLOR_EVERYONE = "#898781"     # muted ink -- reference line
MODEL_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]  # slots 1-4, all-pairs safe
SEGMENT_COLORS = {
    "persuadable": "#2a78d6",
    "sure thing": "#eda100",
    "lost cause": "#898781",
    "do-not-disturb": "#e34948",
}

st.set_page_config(page_title="Retail Media Budget Allocator", layout="wide")
st.title("Retail Media Budget Allocator")
st.caption(
    "Ads shown ≠ sales caused. This dashboard scores users by *predicted "
    "uplift* -- the causal effect of showing them an ad -- using models trained "
    "on Criteo's randomized ad experiment, and compares that targeting policy "
    "against propensity-to-convert (what most teams target with today)."
)


@st.cache_data
def load_artifacts():
    scores = pd.read_parquet(REPORTS_DIR / "test_scores.parquet")
    qini = pd.read_parquet(REPORTS_DIR / "qini_curves.parquet")
    summary = json.loads((REPORTS_DIR / "summary.json").read_text())
    return scores, qini, summary


if not (REPORTS_DIR / "test_scores.parquet").exists():
    st.warning(
        "No trained model artifacts found. Run `python -m src.train` first -- "
        "it saves the held-out scores this dashboard reads from `reports/`."
    )
    st.stop()

scores, qini, summary = load_artifacts()
best_model = summary["best_model"]
y, t = scores["conversion"].to_numpy(), scores["treatment"].to_numpy()
uplift_scores = scores[best_model].to_numpy()
propensity_scores = scores["propensity"].to_numpy()

st.sidebar.header("Assumptions")
cost = st.sidebar.slider("Cost per exposure ($)", 0.02, 0.50, 0.10, 0.01)
value = st.sidebar.slider("Value per incremental conversion ($)", 5.0, 100.0, 40.0, 5.0)
budget = st.sidebar.slider("% of audience to target", 2, 100, 20)
st.sidebar.caption(
    f"Best model: **{best_model}**  \n"
    f"Qini coefficient: {summary['results'][best_model]['qini_coefficient']:.4f}  \n"
    f"Held out on {summary['n_test']:,} users"
)

# --- Money chart: incremental profit vs. budget, propensity vs. uplift ----
st.subheader("Incremental profit vs. budget")
budgets = np.linspace(0.02, 1.0, 25)
uplift_curve = profit_curve(y, t, uplift_scores, budgets, cost, value)
prop_curve = profit_curve(y, t, propensity_scores, budgets, cost, value)
everyone_profit = uplift_curve.iloc[-1]["profit"]  # ranking doesn't matter at 100%

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axhline(0, color="#c3c2b7", linewidth=1)
ax.fill_between(uplift_curve["budget_frac"], uplift_curve["ci_low"], uplift_curve["ci_high"],
                 color=COLOR_UPLIFT, alpha=0.15, linewidth=0)
ax.fill_between(prop_curve["budget_frac"], prop_curve["ci_low"], prop_curve["ci_high"],
                 color=COLOR_PROPENSITY, alpha=0.15, linewidth=0)
ax.plot(uplift_curve["budget_frac"], uplift_curve["profit"], color=COLOR_UPLIFT,
        linewidth=2, label=f"Uplift targeting ({best_model})")
ax.plot(prop_curve["budget_frac"], prop_curve["profit"], color=COLOR_PROPENSITY,
        linewidth=2, label="Propensity targeting (naive)")
ax.axhline(everyone_profit, color=COLOR_EVERYONE, linewidth=1.5, linestyle="--",
           label="Target everyone")
ax.axvline(budget / 100, color="#0b0b0b", linewidth=1, linestyle=":", alpha=0.5)
ax.set_xlabel("% of audience targeted")
ax.set_ylabel("Expected incremental profit ($)")
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.legend(frameon=False)
st.pyplot(fig)

with st.expander("View profit-curve data"):
    table = pd.concat([
        uplift_curve.assign(policy=f"uplift ({best_model})"),
        prop_curve.assign(policy="propensity"),
    ], ignore_index=True)
    st.dataframe(table[["policy", "budget_frac", "n_targeted", "uplift", "profit",
                         "ci_low", "ci_high"]], width="stretch")

# --- Budget readout at the selected slider position ------------------------
sel_uplift = policy_profit(y, t, uplift_scores, budget / 100, cost, value)
sel_prop = policy_profit(y, t, propensity_scores, budget / 100, cost, value)
gap = sel_uplift["profit"] - sel_prop["profit"]

col1, col2, col3 = st.columns(3)
col1.metric(f"Uplift targeting profit @ {budget}%", f"${sel_uplift['profit']:,.0f}")
col2.metric(f"Propensity targeting profit @ {budget}%", f"${sel_prop['profit']:,.0f}")
col3.metric("Uplift advantage", f"${gap:,.0f}")
st.caption(
    f"At {budget}% budget ({sel_uplift['n_targeted']:,} users targeted), uplift "
    f"targeting nets ${gap:,.0f} more expected profit than propensity targeting, "
    f"under ${cost:.2f}/exposure and ${value:.0f}/incremental conversion."
)

# --- Qini curves: model comparison -----------------------------------------
st.subheader("Model comparison — Qini curves")
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
model_names = list(qini["model"].unique())
diagonal_end = qini[qini.model == best_model]["qini"].iloc[-1]
ax2.plot([0, 1], [0, diagonal_end], color="#c3c2b7", linewidth=1, linestyle="--",
         label="Random targeting")
for i, name in enumerate(model_names):
    sub = qini[qini.model == name]
    ax2.plot(sub["frac"], sub["qini"], color=MODEL_COLORS[i % len(MODEL_COLORS)],
              linewidth=2, label=name)
ax2.set_xlabel("% of audience targeted")
ax2.set_ylabel("Cumulative incremental conversions")
ax2.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax2.legend(frameon=False, fontsize=8)
st.pyplot(fig2)

# --- Segments: who to target ------------------------------------------------
st.subheader("Who to target: four business segments")
seg_df = pd.DataFrame({
    "propensity": propensity_scores,
    "uplift": uplift_scores,
    "segment": scores["segment"],
})
sample = seg_df.sample(min(5000, len(seg_df)), random_state=0)
fig3, ax3 = plt.subplots(figsize=(9, 4.5))
for name, color in SEGMENT_COLORS.items():
    sub = sample[sample.segment == name]
    ax3.scatter(sub["propensity"], sub["uplift"], s=8, alpha=0.5, color=color,
                label=name, linewidths=0)
ax3.axhline(0, color="#c3c2b7", linewidth=1)
ax3.set_xlabel("Predicted conversion propensity (would they convert anyway?)")
ax3.set_ylabel("Predicted uplift (does the ad change their behavior?)")
ax3.spines[["top", "right"]].set_visible(False)
ax3.legend(frameon=False, markerscale=2)
st.pyplot(fig3)

counts = seg_df["segment"].value_counts()
seg_cols = st.columns(4)
for col, name in zip(seg_cols, ["persuadable", "sure thing", "lost cause", "do-not-disturb"]):
    n = int(counts.get(name, 0))
    col.metric(name, f"{n:,}", f"{n / len(seg_df):.0%} of users")
st.caption(
    "**Persuadables** are worth paying to reach -- ads change their behavior. "
    "**Sure things** convert regardless; **lost causes** won't convert either way "
    "-- don't pay for either. **Do-not-disturbs** show negative predicted uplift "
    "-- showing them ads may actively suppress conversion."
)
