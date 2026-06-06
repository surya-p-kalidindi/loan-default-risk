"""
Fannie Mae Loan Default — Phase 1 & 2: Model + SHAP
=====================================================
XGBoost classifier with time-based walk-forward validation.
Trains two models:
  Model A: ever_90dpd_24mo  (early delinquency signal)
  Model B: ever_default     (foreclosure / loss event)

Evaluation:
  - Validate on 2018 (tune), Test on 2019 (final holdout)
  - Metrics: AUC-ROC, Average Precision, KS statistic
  - SHAP: global feature importance + per-loan explanations
  - Vintage analysis: predicted PD vs actual delinquency by year
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")
import pickle

from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap

PROC_DIR    = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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
    "product_type_encoded":     "Product Type (ARM/FRM)",
    "property_type_encoded":    "Property Type",
    "num_units":                "# Units",
    "occupancy_encoded":        "Occupancy Type",
    "property_state":           "Property State",
    "orig_year":                "Origination Year",
    "mortgage_rate_at_orig":    "Mortgage Rate at Orig",
    "unemp_at_orig":            "Unemployment at Orig",
    "hpi_growth_at_orig":       "HPI Growth at Orig",
}


def load_splits():
    train    = pd.read_parquet(PROC_DIR / "train.parquet")
    validate = pd.read_parquet(PROC_DIR / "validate.parquet")
    test     = pd.read_parquet(PROC_DIR / "test.parquet")
    return train, validate, test


def prep(df, target, imputer=None, fit_imputer=False):
    """Extract features and target, impute missing values."""
    X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
    y = df[target].astype(int).copy()
    if fit_imputer:
        imputer = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        return X_imp, y, imputer
    else:
        X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)
        return X_imp, y


def make_model(scale_pos_weight, n_estimators=500):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )


def ks_statistic(y_true, y_prob):
    """KS statistic — standard metric in credit risk modeling."""
    from scipy.stats import ks_2samp
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    ks, _ = ks_2samp(pos, neg)
    return ks


def evaluate(model, X, y, label):
    probs = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, probs)
    ap    = average_precision_score(y, probs)
    ks    = ks_statistic(y.values, probs)
    print(f"  {label:12s} | AUC={auc:.3f}  AP={ap:.3f}  KS={ks:.3f}  "
          f"Events={y.sum():,}/{len(y):,} ({100*y.mean():.1f}%)")
    return probs, auc, ap, ks


def train_model(train, validate, target):
    """Train with early stopping on validate set."""
    print(f"\n{'='*55}")
    print(f"Training: {target}")
    print(f"{'='*55}")

    X_tr, y_tr, imputer = prep(train, target, fit_imputer=True)
    X_val, y_val        = prep(validate, target, imputer=imputer)

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"  Class balance — train: {y_tr.sum():,} events ({100*y_tr.mean():.1f}%), "
          f"scale_pos_weight={spw:.1f}")

    model = make_model(spw)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    print("\nModel performance:")
    _, auc_tr, _, _  = evaluate(model, X_tr,  y_tr,  "Train")
    probs_val, auc_val, ap_val, ks_val = evaluate(model, X_val, y_val, "Validate")

    return model, imputer, probs_val


def test_model(model, imputer, test, target):
    X_te, y_te = prep(test, target, imputer=imputer)
    probs_te, auc_te, ap_te, ks_te = evaluate(model, X_te, y_te, "Test")
    return probs_te, auc_te, ap_te, ks_te


def plot_roc_pr(models_data, out_dir):
    """ROC and PR curves for both targets on test set."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"ever_90dpd_24mo": "#e63946", "ever_default": "#457b9d"}
    names  = {"ever_90dpd_24mo": "90-Day Delinquency", "ever_default": "Default/Foreclosure"}

    for target, (y_te, probs_te) in models_data.items():
        RocCurveDisplay.from_predictions(
            y_te, probs_te, ax=axes[0],
            name=names[target], color=colors[target]
        )
        PrecisionRecallDisplay.from_predictions(
            y_te, probs_te, ax=axes[1],
            name=names[target], color=colors[target]
        )

    axes[0].plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
    axes[0].set_title("ROC Curve — Test Set (2019)", fontsize=13)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Precision-Recall — Test Set (2019)", fontsize=13)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = out_dir / "roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {path}")


def plot_shap(model, imputer, train, target, out_dir):
    """SHAP importance bar + beeswarm for a given target."""
    X_tr, y_tr = prep(train, target, imputer=imputer)

    # Sample for speed if large
    if len(X_tr) > 10000:
        idx = np.random.choice(len(X_tr), 10000, replace=False)
        X_sample = X_tr.iloc[idx]
    else:
        X_sample = X_tr

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    feat_labels = [FEATURE_LABELS.get(c, c) for c in X_tr.columns]

    # ── Bar plot ──────────────────────────────────────────────────────────
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_tr.columns
    ).sort_values(ascending=True)
    mean_shap.index = [FEATURE_LABELS.get(c,c) for c in mean_shap.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    top_features = ["FICO Score", "Original LTV (%)", "DTI Ratio (%)"]
    colors = ["#e63946" if label in top_features else "#457b9d"
              for label in mean_shap.index]
    mean_shap.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    label = "90-Day Delinquency" if "90dpd" in target else "Default/Foreclosure"
    ax.set_title(f"Feature Importance — {label}\n(trained on 2010–2017, evaluated on 2018)",
                 fontsize=12)
    plt.tight_layout()
    path = out_dir / f"shap_importance_{target}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    # ── Beeswarm ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feat_labels,
                      show=False, max_display=15)
    plt.title(f"SHAP Summary — {label}", fontsize=12, pad=12)
    plt.tight_layout()
    path = out_dir / f"shap_beeswarm_{target}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    return mean_shap


def vintage_analysis(model, imputer, full_df, target, out_dir):
    """
    Predicted PD vs actual delinquency rate by origination year.
    This is how mortgage investors think about portfolio risk.
    """
    X_all, y_all = prep(full_df, target, imputer=imputer)
    probs = model.predict_proba(X_all)[:, 1]

    full_df = full_df.copy()
    full_df["pred_pd"] = probs
    full_df["actual"]  = y_all.values

    vintage = full_df.groupby("orig_year").agg(
        n=("loan_id", "count"),
        actual_rate=("actual", "mean"),
        predicted_pd=("pred_pd", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(vintage["orig_year"] - 0.2, vintage["actual_rate"] * 100,
           width=0.35, label="Actual Rate", color="#e63946", alpha=0.8)
    ax.bar(vintage["orig_year"] + 0.2, vintage["predicted_pd"] * 100,
           width=0.35, label="Predicted PD", color="#457b9d", alpha=0.8)

    ax.set_xlabel("Origination Year (Vintage)", fontsize=11)
    ax.set_ylabel("Rate (%)", fontsize=11)
    label = "90-Day Delinquency" if "90dpd" in target else "Default/Foreclosure"
    ax.set_title(f"Vintage Analysis — Predicted vs Actual {label}", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    ax.set_xticks(vintage["orig_year"])
    plt.tight_layout()

    path = out_dir / f"vintage_analysis_{target}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    print(f"\nVintage Analysis — {label}:")
    print(vintage.to_string(index=False, float_format="{:.3f}".format))
    return vintage


def explain_loan(model, imputer, loan_row: pd.Series, target: str) -> dict:
    """
    Compute per-loan SHAP explanation.
    Returns dict of {feature: shap_value} for the top drivers.
    Used by Phase 4 LLM narrative generator.
    """
    X = pd.DataFrame([loan_row[[c for c in FEATURE_COLS if c in loan_row.index]]])
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_imp)[0]
    base_value  = explainer.expected_value

    prob = model.predict_proba(X_imp)[0, 1]

    contributions = pd.Series(shap_vals, index=X_imp.columns)
    top = contributions.abs().nlargest(5)

    return {
        "loan_id":     loan_row.get("loan_id", "unknown"),
        "target":      target,
        "probability": round(float(prob), 4),
        "base_rate":   round(float(1 / (1 + np.exp(-base_value))), 4),
        "top_drivers": [
            {
                "feature":     FEATURE_LABELS.get(f, f),
                "value":       round(float(loan_row.get(f, np.nan)), 2),
                "shap_value":  round(float(contributions[f]), 4),
                "direction":   "increases" if contributions[f] > 0 else "decreases",
                "pct_impact":  round(abs(float(contributions[f])) /
                               max(abs(shap_vals).sum(), 1e-6) * 100, 1),
            }
            for f in top.index
        ],
    }


def save_models(models: dict, out_dir: Path):
    path = out_dir / "models.pkl"
    with open(path, "wb") as f:
        pickle.dump(models, f)
    print(f"Saved → {path}")


def run():
    print("=" * 55)
    print("Phase 1 & 2: Model Training + SHAP Analysis")
    print("=" * 55)

    train, validate, test = load_splits()
    full_df = pd.concat([train, validate, test], ignore_index=True)

    saved_models = {}
    roc_data     = {}
    results      = []

    for target in ["ever_90dpd_24mo", "ever_default"]:
        # Train
        model, imputer, probs_val = train_model(train, validate, target)

        # Test (final holdout)
        print("\nFinal holdout (2019 test set):")
        probs_te, auc_te, ap_te, ks_te = test_model(model, imputer, test, target)

        X_te, y_te = prep(test, target, imputer=imputer)
        roc_data[target] = (y_te, probs_te)

        results.append({
            "target": target,
            "test_auc": auc_te,
            "test_ap":  ap_te,
            "test_ks":  ks_te,
        })

        # SHAP
        print(f"\nComputing SHAP for {target}...")
        mean_shap = plot_shap(model, imputer, train, target, RESULTS_DIR)

        # Vintage analysis
        vintage_analysis(model, imputer, full_df, target, RESULTS_DIR)

        # Save importance
        imp_df = mean_shap.sort_values(ascending=False).reset_index()
        imp_df.columns = ["feature", "mean_abs_shap"]
        imp_df.to_csv(RESULTS_DIR / f"importance_{target}.csv", index=False)

        saved_models[target] = {"model": model, "imputer": imputer}

    # ROC/PR plot for both targets
    plot_roc_pr(roc_data, RESULTS_DIR)

    # Save models
    save_models(saved_models, RESULTS_DIR)

    # Summary
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    for r in results:
        print(f"\n  {r['target']}:")
        print(f"    Test AUC : {r['test_auc']:.3f}")
        print(f"    Test AP  : {r['test_ap']:.3f}")
        print(f"    Test KS  : {r['test_ks']:.3f}")

    # Demo: explain one loan
    print("\n--- Sample loan explanation (Phase 4 preview) ---")
    sample_loan = test.iloc[0]
    for target in ["ever_90dpd_24mo", "ever_default"]:
        explanation = explain_loan(
            saved_models[target]["model"],
            saved_models[target]["imputer"],
            sample_loan, target
        )
        print(f"\n{target}:")
        print(f"  Probability: {explanation['probability']*100:.1f}%")
        print(f"  Top drivers:")
        for d in explanation["top_drivers"][:3]:
            print(f"    {d['feature']:30s} = {d['value']:.1f}  "
                  f"→ {d['direction']} risk by {d['pct_impact']:.1f}%")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
