"""
Loan Default Risk Dashboard
========================================
Phase 3: Interactive Streamlit dashboard showing:
  - Portfolio overview (KPIs)
  - Top risk drivers (SHAP feature importance)
  - Geographic concentration heatmap
  - LTV and FICO distributions by outcome
  - Delinquency forecasts by vintage
  - Individual loan risk scorer

Run from project root:
  streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
PROC_DIR    = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

import sys
sys.path.insert(0, str(Path(__file__).parent))

FEATURE_COLS = [
    "fico_score", "dti_ratio", "num_borrowers", "first_time_homebuyer_flag",
    "orig_ltv", "orig_cltv", "cltv_ltv_diff", "mortgage_insurance_flag",
    "orig_interest_rate", "loan_term_years", "loan_purpose_encoded",
    "product_type_encoded", "property_type_encoded", "num_units",
    "occupancy_encoded", "property_state", "orig_year",
    "mortgage_rate_at_orig", "unemp_at_orig", "hpi_growth_at_orig",
]

FEATURE_LABELS = {
    "fico_score":               "FICO Score",
    "dti_ratio":                "DTI Ratio (%)",
    "num_borrowers":            "# Borrowers",
    "first_time_homebuyer_flag":"First-Time Homebuyer",
    "orig_ltv":                 "Original LTV (%)",
    "orig_cltv":                "Original CLTV (%)",
    "cltv_ltv_diff":            "CLTV - LTV Gap",
    "mortgage_insurance_flag":  "MI Flag",
    "orig_interest_rate":       "Interest Rate (%)",
    "loan_term_years":          "Loan Term (yrs)",
    "loan_purpose_encoded":     "Loan Purpose",
    "product_type_encoded":     "Product Type",
    "property_type_encoded":    "Property Type",
    "num_units":                "# Units",
    "occupancy_encoded":        "Occupancy Type",
    "property_state":           "State Code",
    "orig_year":                "Origination Year",
    "mortgage_rate_at_orig":    "Mortgage Rate at Orig",
    "unemp_at_orig":            "Unemployment at Orig",
    "hpi_growth_at_orig":       "HPI Growth at Orig",
}

STATE_CODES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train    = pd.read_parquet(PROC_DIR / "train.parquet")
    validate = pd.read_parquet(PROC_DIR / "validate.parquet")
    test     = pd.read_parquet(PROC_DIR / "test.parquet")
    return pd.concat([train, validate, test], ignore_index=True)

@st.cache_resource
def load_models():
    path = RESULTS_DIR / "models.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_importance(target):
    path = RESULTS_DIR / f"importance_{target}.csv"
    return pd.read_csv(path)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fannie Mae Default Risk Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #457b9d;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e0e0e0; }
    .metric-label { font-size: 0.85rem; color: #9aa0b4; margin-top: 4px; }
    .risk-high   { color: #e63946; font-weight: 700; }
    .risk-med    { color: #f4a261; font-weight: 700; }
    .risk-low    { color: #2a9d8f; font-weight: 700; }
    h1 { color: #e0e0e0 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Fannie_Mae.svg/200px-Fannie_Mae.svg.png",
             width=140)
    st.markdown("### Dashboard Controls")

    target = st.selectbox(
        "Prediction Target",
        ["ever_90dpd_24mo", "ever_default"],
        format_func=lambda x: "90-Day Delinquency" if "90dpd" in x else "Default / Foreclosure"
    )

    vintage_filter = st.multiselect(
        "Filter by Vintage Year",
        options=list(range(2010, 2020)),
        default=list(range(2010, 2020)),
    )

    st.markdown("---")
    st.markdown("**Model Performance**")
    perf = {
        "ever_90dpd_24mo": {"auc": 0.791, "ks": 0.440},
        "ever_default":    {"auc": 0.765, "ks": 0.407},
    }
    st.metric("Test AUC",  f"{perf[target]['auc']:.3f}")
    st.metric("Test KS",   f"{perf[target]['ks']:.3f}")
    st.markdown("*Trained 2010–2017 | Tested 2019*")
    st.markdown("---")
    st.caption("Data: Fannie Mae SFLL (synthetic)\nModel: XGBoost + SHAP")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df     = load_data()
    models = load_models()
    df_filtered = df[df["orig_year"].isin(vintage_filter)].copy()
except Exception as e:
    st.error(f"Could not load data: {e}\nRun synthetic_data.py → preprocessing.py → model.py first.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 Fannie Mae Single-Family Default Risk Dashboard")
target_label = "90-Day Delinquency (24mo)" if "90dpd" in target else "Default / Foreclosure"
st.markdown(f"**Active model:** {target_label} &nbsp;|&nbsp; "
            f"**Vintages:** {min(vintage_filter)}–{max(vintage_filter)} &nbsp;|&nbsp; "
            f"**Loans:** {len(df_filtered):,}")

st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_loans    = len(df_filtered)
event_rate     = df_filtered[target].mean() * 100
avg_fico       = df_filtered["fico_score"].mean()
avg_ltv        = df_filtered["orig_ltv"].mean()
avg_dti        = df_filtered["dti_ratio"].mean()

with col1:
    st.metric("Total Loans",    f"{total_loans:,}")
with col2:
    st.metric(f"{target_label} Rate", f"{event_rate:.2f}%",
              delta=f"{event_rate - 1.8:.2f}% vs baseline",
              delta_color="inverse")
with col3:
    st.metric("Avg FICO",       f"{avg_fico:.0f}")
with col4:
    st.metric("Avg LTV",        f"{avg_ltv:.1f}%")
with col5:
    st.metric("Avg DTI",        f"{avg_dti:.1f}%")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Risk Drivers",
    "🗺️ Geographic",
    "📈 Distributions",
    "📅 Vintage Analysis",
    "🔍 Loan Scorer",
    "🤖 AI Narrative",
])

# ── Tab 1: Risk Drivers ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Top Risk Drivers — SHAP Feature Importance")
    st.caption("Mean absolute SHAP value across training set (2010–2017). "
               "Higher = stronger predictor of outcome.")

    try:
        imp = load_importance(target).head(15)
        imp = imp.sort_values("mean_abs_shap", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        highlight = ["FICO Score", "Original LTV (%)", "DTI Ratio (%)"]
        colors = ["#e63946" if f in highlight else "#457b9d"
                  for f in imp["feature"]]
        ax.barh(imp["feature"], imp["mean_abs_shap"],
                color=colors, edgecolor="none", height=0.6)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(f"Feature Importance — {target_label}", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top risk-increasing features:**")
            st.markdown("- High LTV / CLTV → less equity cushion")
            st.markdown("- High DTI → income stretched thin")
            st.markdown("- First-time homebuyer → less experience")
            st.markdown("- Investor occupancy → less motivated to pay")
        with col_b:
            st.markdown("**Top risk-reducing features:**")
            st.markdown("- High FICO → strong payment history")
            st.markdown("- Low unemployment at origination → stable job market")
            st.markdown("- Strong HPI growth → equity building")
            st.markdown("- Lower interest rate → more affordable payment")
    except Exception as e:
        st.warning(f"Could not load importance file: {e}")
        st.image(str(RESULTS_DIR / f"shap_importance_{target}.png"))

# ── Tab 2: Geographic ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Geographic Risk Concentration")

    if "property_state" in df_filtered.columns:
        # Decode state codes back (property_state is encoded as int)
        state_risk = df_filtered.groupby("property_state").agg(
            n=(target, "count"),
            rate=(target, "mean"),
            avg_fico=("fico_score", "mean"),
            avg_ltv=("orig_ltv", "mean"),
        ).reset_index()
        state_risk["rate_pct"] = state_risk["rate"] * 100
        state_risk = state_risk.sort_values("rate_pct", ascending=False)

        col_a, col_b = st.columns([2, 1])

        with col_a:
            fig, ax = plt.subplots(figsize=(12, 5))
            top20 = state_risk.head(20)
            colors = ["#e63946" if r > state_risk["rate_pct"].quantile(0.75)
                      else "#457b9d" for r in top20["rate_pct"]]
            ax.bar(range(len(top20)), top20["rate_pct"],
                   color=colors, edgecolor="none")
            ax.set_xticks(range(len(top20)))
            ax.set_xticklabels([f"State {int(s)}" for s in top20["property_state"]],
                               rotation=45, ha="right", fontsize=9)
            ax.set_ylabel(f"{target_label} Rate (%)")
            ax.set_title(f"Top 20 States by {target_label} Rate")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("**Highest Risk States**")
            display = state_risk.head(10)[["property_state","n","rate_pct","avg_fico","avg_ltv"]].copy()
            display.columns = ["State", "Loans", "Rate%", "Avg FICO", "Avg LTV"]
            display["Rate%"] = display["Rate%"].round(2)
            display["Avg FICO"] = display["Avg FICO"].round(0)
            display["Avg LTV"] = display["Avg LTV"].round(1)
            st.dataframe(display, use_container_width=True, hide_index=True)

# ── Tab 3: Distributions ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Feature Distributions by Outcome")

    col_a, col_b = st.columns(2)

    with col_a:
        # FICO distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = range(500, 860, 20)
        ax.hist(df_filtered.loc[df_filtered[target]==0, "fico_score"],
                bins=bins, alpha=0.6, color="#457b9d", label="No Event",
                density=True)
        ax.hist(df_filtered.loc[df_filtered[target]==1, "fico_score"],
                bins=bins, alpha=0.7, color="#e63946", label=target_label,
                density=True)
        ax.set_xlabel("FICO Score")
        ax.set_ylabel("Density")
        ax.set_title("FICO Distribution by Outcome")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        # LTV distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        bins_ltv = range(20, 106, 5)
        ax.hist(df_filtered.loc[df_filtered[target]==0, "orig_ltv"],
                bins=bins_ltv, alpha=0.6, color="#457b9d", label="No Event",
                density=True)
        ax.hist(df_filtered.loc[df_filtered[target]==1, "orig_ltv"],
                bins=bins_ltv, alpha=0.7, color="#e63946", label=target_label,
                density=True)
        ax.set_xlabel("Original LTV (%)")
        ax.set_ylabel("Density")
        ax.set_title("LTV Distribution by Outcome")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        # DTI distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df_filtered.loc[df_filtered[target]==0, "dti_ratio"],
                bins=30, alpha=0.6, color="#457b9d", label="No Event", density=True)
        ax.hist(df_filtered.loc[df_filtered[target]==1, "dti_ratio"],
                bins=30, alpha=0.7, color="#e63946", label=target_label, density=True)
        ax.set_xlabel("DTI Ratio (%)")
        ax.set_ylabel("Density")
        ax.set_title("DTI Distribution by Outcome")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_d:
        # FICO vs LTV scatter (sampled)
        sample = df_filtered.sample(min(2000, len(df_filtered)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(sample.loc[sample[target]==0, "fico_score"],
                   sample.loc[sample[target]==0, "orig_ltv"],
                   alpha=0.3, s=8, color="#457b9d", label="No Event")
        ax.scatter(sample.loc[sample[target]==1, "fico_score"],
                   sample.loc[sample[target]==1, "orig_ltv"],
                   alpha=0.7, s=15, color="#e63946", label=target_label, zorder=3)
        ax.set_xlabel("FICO Score")
        ax.set_ylabel("Original LTV (%)")
        ax.set_title("FICO vs LTV — Risk Boundary")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Tab 4: Vintage Analysis ───────────────────────────────────────────────────
with tab4:
    st.subheader("Vintage Analysis — Predicted PD vs Actual Rate")
    st.caption("How well does the model's predicted default probability "
               "track actual delinquency by origination year?")

    vintage = df_filtered.groupby("orig_year").agg(
        n=(target, "count"),
        actual_rate=(target, "mean"),
    ).reset_index()

    # Get model predictions for full dataset
    model_obj = models[target]["model"]
    imputer   = models[target]["imputer"]
    X = df_filtered[[c for c in FEATURE_COLS if c in df_filtered.columns]]
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)
    probs = model_obj.predict_proba(X_imp)[:, 1]
    df_filtered = df_filtered.copy()
    df_filtered["pred_pd"] = probs

    vintage = df_filtered.groupby("orig_year").agg(
        n=(target, "count"),
        actual_rate=(target, "mean"),
        predicted_pd=("pred_pd", "mean"),
    ).reset_index()

    col_a, col_b = st.columns([3, 1])

    with col_a:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(vintage))
        ax.bar(x - 0.18, vintage["actual_rate"] * 100, width=0.35,
               label="Actual Rate", color="#e63946", alpha=0.85)
        ax.bar(x + 0.18, vintage["predicted_pd"] * 100, width=0.35,
               label="Predicted PD (scaled)", color="#457b9d", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(vintage["orig_year"].astype(int))
        ax.set_xlabel("Origination Year (Vintage)")
        ax.set_ylabel("Rate (%)")
        ax.set_title(f"Vintage Analysis — {target_label}", fontsize=13)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**By Vintage**")
        vt = vintage.copy()
        vt["actual_rate"] = (vt["actual_rate"] * 100).round(2)
        vt["predicted_pd"] = (vt["predicted_pd"] * 100).round(2)
        vt.columns = ["Year", "N", "Actual %", "Pred PD %"]
        st.dataframe(vt, use_container_width=True, hide_index=True)

    st.info("📌 Note: Predicted PD absolute values are intentionally higher than "
            "actual rates due to class imbalance scaling. The model is calibrated "
            "for ranking (AUC/KS), not absolute probability. "
            "Platt scaling can be applied for calibrated probabilities.")

# ── Tab 5: Loan Scorer ────────────────────────────────────────────────────────
with tab5:
    st.subheader("🔍 Individual Loan Risk Scorer")
    st.caption("Enter loan characteristics to get a real-time default risk score "
               "and top risk drivers.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Borrower**")
        fico      = st.slider("FICO Score",        300, 850, 720)
        dti       = st.slider("DTI Ratio (%)",     10,  65,  36)
        n_borrow  = st.radio("# Borrowers",        [1, 2], horizontal=True)
        fthb      = st.checkbox("First-Time Homebuyer", value=False)

    with col_b:
        st.markdown("**Collateral**")
        ltv       = st.slider("Original LTV (%)",  20,  105, 80)
        cltv      = st.slider("Original CLTV (%)", 20,  105, 80)
        rate      = st.slider("Interest Rate (%)",  2.0, 9.0, 4.5, step=0.1)
        mi        = st.checkbox("Mortgage Insurance", value=(ltv > 80))

    with col_c:
        st.markdown("**Loan & Property**")
        purpose   = st.selectbox("Loan Purpose",
                                 ["Purchase (0)", "Rate/Term Refi (1)", "Cash-Out Refi (2)"])
        occ       = st.selectbox("Occupancy",
                                 ["Primary (0)", "Second Home (1)", "Investor (2)"])
        prop_type = st.selectbox("Property Type",
                                 ["Single Family (0)", "Condo (1)", "PUD (2)", "MH (3)"])
        num_units = st.radio("# Units", [1, 2, 3, 4], horizontal=True)
        term      = st.radio("Term (yrs)", [15, 20, 30], horizontal=True, index=2)
        orig_yr   = st.slider("Origination Year", 2010, 2019, 2018)

    # Score button
    if st.button("⚡ Score This Loan", type="primary", use_container_width=True):
        loan_input = {
            "fico_score":               fico,
            "dti_ratio":                dti,
            "num_borrowers":            n_borrow,
            "first_time_homebuyer_flag":int(fthb),
            "orig_ltv":                 ltv,
            "orig_cltv":                cltv,
            "cltv_ltv_diff":            max(cltv - ltv, 0),
            "mortgage_insurance_flag":  int(mi),
            "orig_interest_rate":       rate,
            "loan_term_years":          term,
            "loan_purpose_encoded":     int(purpose.split("(")[1].split(")")[0]) if "(" in purpose else 0,
            "product_type_encoded":     0,
            "property_type_encoded":    int(prop_type.split("(")[1].split(")")[0]),
            "num_units":                num_units,
            "occupancy_encoded":        int(occ.split("(")[1].split(")")[0]),
            "property_state":           25,
            "orig_year":                orig_yr,
            "mortgage_rate_at_orig":    rate,
            "unemp_at_orig":            5.0,
            "hpi_growth_at_orig":       5.0,
        }

        X_loan = pd.DataFrame([loan_input])[[c for c in FEATURE_COLS
                                              if c in loan_input]]
        X_imp  = pd.DataFrame(
            models[target]["imputer"].transform(X_loan),
            columns=X_loan.columns
        )
        prob = models[target]["model"].predict_proba(X_imp)[0, 1]

        # Relative risk vs population mean
        pop_mean = df[target].mean()
        rel_risk = prob / max(pop_mean, 0.001)

        # Risk tier
        if rel_risk < 0.8:
            tier, color = "LOW RISK", "risk-low"
        elif rel_risk < 1.5:
            tier, color = "MODERATE RISK", "risk-med"
        else:
            tier, color = "HIGH RISK", "risk-high"

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Model Score", f"{prob*100:.1f}%")
        with r2:
            st.metric("Relative Risk", f"{rel_risk:.2f}x population",
                      delta=f"{(rel_risk-1)*100:.0f}% vs avg",
                      delta_color="inverse")
        with r3:
            st.markdown(f"<div class='metric-card'>"
                        f"<div class='metric-value {color}'>{tier}</div>"
                        f"<div class='metric-label'>{target_label}</div>"
                        f"</div>", unsafe_allow_html=True)

        # SHAP for this loan
        import shap
        explainer  = shap.TreeExplainer(models[target]["model"])
        shap_vals  = explainer.shap_values(X_imp)[0]
        contribs   = pd.Series(shap_vals, index=X_imp.columns)
        top5       = contribs.abs().nlargest(5).index

        st.markdown("**Top Risk Drivers for This Loan:**")
        driver_data = []
        for feat in top5:
            val  = loan_input.get(feat, "N/A")
            sv   = contribs[feat]
            pct  = abs(sv) / max(abs(shap_vals).sum(), 1e-6) * 100
            direction = "⬆ Increases" if sv > 0 else "⬇ Decreases"
            driver_data.append({
                "Feature":    FEATURE_LABELS.get(feat, feat),
                "Value":      round(float(val), 1) if isinstance(val, (int, float)) else val,
                "Direction":  direction,
                "Impact":     f"{pct:.1f}%",
                "SHAP":       round(float(sv), 4),
            })
        st.dataframe(pd.DataFrame(driver_data), use_container_width=True, hide_index=True)

        st.info("💡 Go to the **🤖 AI Narrative** tab to generate a plain-English "
                "explanation of this loan's risk profile using Claude.")

# ── Tab 6: AI Narrative ──────────────────────────────────────────────────────
with tab6:
    st.subheader("🤖 AI-Generated Risk Narrative")
    st.caption("Claude generates a plain-English explanation of loan risk "
               "based on calibrated probability and SHAP drivers.")

    st.markdown("**Configure Loan for Narrative**")
    col_n1, col_n2, col_n3 = st.columns(3)
    with col_n1:
        n_fico    = st.slider("FICO Score",       300, 850, 650, key="n_fico")
        n_dti     = st.slider("DTI Ratio (%)",    10,  65,  45,  key="n_dti")
        n_fthb    = st.checkbox("First-Time Homebuyer", value=True, key="n_fthb")
    with col_n2:
        n_ltv     = st.slider("Original LTV (%)", 20,  105, 95,  key="n_ltv")
        n_cltv    = st.slider("Original CLTV (%)",20,  105, 95,  key="n_cltv")
        n_rate    = st.slider("Interest Rate (%)", 2.0, 9.0, 5.5, step=0.1, key="n_rate")
    with col_n3:
        n_purpose = st.selectbox("Loan Purpose",
                                 ["Purchase (0)", "Rate/Term Refi (1)", "Cash-Out Refi (2)"],
                                 key="n_purpose")
        n_occ     = st.selectbox("Occupancy",
                                 ["Primary (0)", "Second Home (1)", "Investor (2)"],
                                 key="n_occ")
        n_yr      = st.slider("Origination Year", 2010, 2019, 2018, key="n_yr")
        n_target  = st.selectbox("Model Target",
                                 ["ever_90dpd_24mo", "ever_default"],
                                 format_func=lambda x: "90-Day Delinquency" if "90dpd" in x else "Default/Foreclosure",
                                 key="n_target")

    api_key = st.text_input("Anthropic API Key",
                             type="password",
                             placeholder="sk-ant-...",
                             help="Get your key at console.anthropic.com")

    if st.button("🤖 Generate Risk Narrative", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your Anthropic API key.")
        else:
            loan_input = {
                "fico_score":               n_fico,
                "dti_ratio":                n_dti,
                "num_borrowers":            1,
                "first_time_homebuyer_flag":int(n_fthb),
                "orig_ltv":                 n_ltv,
                "orig_cltv":                n_cltv,
                "cltv_ltv_diff":            max(n_cltv - n_ltv, 0),
                "mortgage_insurance_flag":  int(n_ltv > 80),
                "orig_interest_rate":       n_rate,
                "loan_term_years":          30,
                "loan_purpose_encoded":     int(n_purpose.split("(")[1].split(")")[0]),
                "product_type_encoded":     0,
                "property_type_encoded":    0,
                "num_units":                1,
                "occupancy_encoded":        int(n_occ.split("(")[1].split(")")[0]),
                "property_state":           25,
                "orig_year":                n_yr,
                "mortgage_rate_at_orig":    n_rate,
                "unemp_at_orig":            5.0,
                "hpi_growth_at_orig":       5.0,
            }
            with st.spinner("Scoring loan and generating narrative..."):
                try:
                    from llm_explain import explain_loan
                    result = explain_loan(loan_input, target=n_target, api_key=api_key)

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("Calibrated Probability", f"{result['cal_prob']*100:.1f}%",
                                  help="Platt-scaled probability (not raw model output)")
                    with r2:
                        st.metric("Relative Risk", f"{result['rel_risk']:.1f}x population")
                    with r3:
                        colors = {"LOW":"🟢","MODERATE":"🟡","ELEVATED":"🟠","HIGH":"🔴"}
                        st.metric("Risk Tier",
                                  f"{colors.get(result['risk_tier'],'')} {result['risk_tier']}")

                    st.markdown("---")
                    st.markdown("**📝 Risk Narrative**")
                    st.markdown(f"> {result['narrative']}")

                    st.markdown("**Top SHAP Drivers**")
                    driver_df = pd.DataFrame(result["top_drivers"])[
                        ["label","value","direction","pct_impact"]
                    ]
                    driver_df.columns = ["Feature","Value","Direction","Impact %"]
                    st.dataframe(driver_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Loan Default Risk Dashboard | "
           "Model: XGBoost + SHAP | "
           "Data: SFLL Public Data (synthetic for dev) | "
           "Not for production use")
