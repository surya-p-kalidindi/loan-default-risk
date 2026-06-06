"""
Fannie Mae Loan Default — Phase 4: LLM Narrative Generator
============================================================
Uses Claude to generate plain-English risk explanations from
SHAP values and loan characteristics.

Example output:
  "This loan carries elevated 90-day delinquency risk, driven
   primarily by a high CLTV of 95% which limits the borrower's
   equity cushion. The DTI of 48% suggests income is stretched,
   leaving little buffer for payment shocks. These risk factors
   are partially mitigated by a solid FICO score of 720, indicating
   a strong payment history. Overall assessment: HIGH RISK.
   Recommend enhanced servicing monitoring in months 6-18."

Platt scaling applied before narrative to convert raw model
scores to calibrated probabilities.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import anthropic
import shap

ROOT        = Path(__file__).parent.parent
PROC_DIR    = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

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
    "property_state":           "State",
    "orig_year":                "Origination Year",
    "mortgage_rate_at_orig":    "Mortgage Rate at Orig (%)",
    "unemp_at_orig":            "Unemployment Rate at Orig (%)",
    "hpi_growth_at_orig":       "HPI Growth at Orig (%)",
}

PURPOSE_MAP  = {0: "Purchase", 1: "Rate/Term Refinance", 2: "Cash-Out Refinance"}
OCC_MAP      = {0: "Primary Residence", 1: "Second Home", 2: "Investment Property"}
PROPTYPE_MAP = {0: "Single Family", 1: "Condo", 2: "PUD/Multi-Unit", 3: "Manufactured Housing"}


def load_models():
    with open(RESULTS_DIR / "models.pkl", "rb") as f:
        return pickle.load(f)


def platt_scale(model, imputer, validate_df, target):
    """
    Fit Platt scaling (logistic regression on model outputs)
    to convert raw XGBoost scores to calibrated probabilities.
    Uses validation set (2018) for fitting — no data leakage.
    """
    X_val = validate_df[[c for c in FEATURE_COLS if c in validate_df.columns]]
    y_val = validate_df[target].astype(int)
    X_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    raw_probs = model.predict_proba(X_imp)[:, 1].reshape(-1, 1)

    calibrator = LogisticRegression()
    calibrator.fit(raw_probs, y_val)

    return calibrator


def get_calibrated_prob(model, imputer, calibrator, X_loan):
    """Return calibrated probability for a single loan."""
    X_imp = pd.DataFrame(imputer.transform(X_loan), columns=X_loan.columns)
    raw   = model.predict_proba(X_imp)[0, 1]
    cal   = calibrator.predict_proba([[raw]])[0, 1]
    return raw, cal


def get_shap_drivers(model, imputer, X_loan, top_n=5):
    """Return top SHAP drivers for a loan as a list of dicts."""
    X_imp = pd.DataFrame(imputer.transform(X_loan), columns=X_loan.columns)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_imp)[0]
    contribs   = pd.Series(shap_vals, index=X_imp.columns)
    total_abs  = abs(shap_vals).sum()

    top = contribs.abs().nlargest(top_n).index
    drivers = []
    for feat in top:
        val = float(X_loan[feat].iloc[0]) if feat in X_loan.columns else None
        sv  = float(contribs[feat])
        drivers.append({
            "feature":      feat,
            "label":        FEATURE_LABELS.get(feat, feat),
            "value":        round(val, 2) if val is not None else "N/A",
            "shap_value":   round(sv, 4),
            "direction":    "increases" if sv > 0 else "decreases",
            "pct_impact":   round(abs(sv) / max(total_abs, 1e-6) * 100, 1),
        })
    return drivers


def build_prompt(loan_data: dict, drivers: list, cal_prob: float,
                 target: str, pop_mean: float) -> str:
    """
    Build a structured prompt for Claude.
    Gives Claude the loan facts, SHAP drivers, and calibrated probability.
    Instructs it to produce a concise credit risk narrative.
    """
    target_label = ("90-day delinquency within 24 months"
                    if "90dpd" in target else "mortgage default or foreclosure")

    rel_risk = cal_prob / max(pop_mean, 1e-4)
    if rel_risk < 0.7:
        risk_tier = "LOW"
    elif rel_risk < 1.3:
        risk_tier = "MODERATE"
    elif rel_risk < 2.0:
        risk_tier = "ELEVATED"
    else:
        risk_tier = "HIGH"

    # Human-readable loan summary
    purpose  = PURPOSE_MAP.get(int(loan_data.get("loan_purpose_encoded", 0)), "Purchase")
    occ      = OCC_MAP.get(int(loan_data.get("occupancy_encoded", 0)), "Primary")
    proptype = PROPTYPE_MAP.get(int(loan_data.get("property_type_encoded", 0)), "Single Family")
    fthb     = "Yes" if loan_data.get("first_time_homebuyer_flag", 0) == 1 else "No"
    mi       = "Yes" if loan_data.get("mortgage_insurance_flag", 0) == 1 else "No"

    # Format SHAP drivers for prompt
    driver_lines = []
    for i, d in enumerate(drivers, 1):
        direction_word = "risk-increasing" if d["direction"] == "increases" else "risk-reducing"
        driver_lines.append(
            f"  {i}. {d['label']} = {d['value']} "
            f"({direction_word}, {d['pct_impact']:.1f}% of total risk attribution)"
        )
    drivers_text = "\n".join(driver_lines)

    prompt = f"""You are a senior credit risk analyst at a mortgage company reviewing a loan for risk assessment.

LOAN CHARACTERISTICS:
- FICO Score: {loan_data.get('fico_score', 'N/A')}
- DTI Ratio: {loan_data.get('dti_ratio', 'N/A')}%
- Original LTV: {loan_data.get('orig_ltv', 'N/A')}%
- Original CLTV: {loan_data.get('orig_cltv', 'N/A')}%
- Loan Purpose: {purpose}
- Occupancy: {occ}
- Property Type: {proptype}
- First-Time Homebuyer: {fthb}
- Mortgage Insurance: {mi}
- Interest Rate: {loan_data.get('orig_interest_rate', 'N/A')}%
- Loan Term: {loan_data.get('loan_term_years', 30)} years
- Origination Year: {int(loan_data.get('orig_year', 2018))}
- Unemployment Rate at Origination: {loan_data.get('unemp_at_orig', 'N/A')}%

MODEL OUTPUT:
- Prediction target: {target_label}
- Calibrated probability: {cal_prob*100:.1f}%
- Population average rate: {pop_mean*100:.1f}%
- Relative risk: {rel_risk:.1f}x population average
- Risk tier: {risk_tier}

TOP SHAP RISK DRIVERS (from XGBoost model):
{drivers_text}

Write a concise credit risk narrative (3-4 sentences) for a portfolio analyst. Follow this structure exactly:
1. State that this loan falls within the [RISK TIER]-risk segment based on historical loan performance patterns — do not say "this loan is classified as" or imply a credit decision. Frame it as a pattern observation, e.g. "This loan falls within the HIGH-risk segment based on historical performance patterns observed in similar loans."
2. Explain the top 2-3 risk-increasing factors in plain English, using the actual values.
3. Note any risk-mitigating factors.

Use professional but plain English. No bullet points. No headers. No recommendations, approvals, denials, or credit decisions. This is a descriptive analytical summary only — it describes what historical patterns suggest, not what should be done."""

    return prompt


def generate_narrative(prompt: str, api_key: str = None) -> str:
    """Call Claude API and return the narrative."""
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "Export it: export ANTHROPIC_API_KEY=your_key_here"
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text.strip()


def explain_loan(loan_data: dict, target: str = "ever_90dpd_24mo",
                 api_key: str = None, validate_df=None) -> dict:
    """
    Full pipeline: loan dict → calibrated risk score → SHAP drivers → narrative.

    Args:
        loan_data:    dict of feature values (raw, pre-encoding)
        target:       'ever_90dpd_24mo' or 'ever_default'
        api_key:      Anthropic API key (or set ANTHROPIC_API_KEY env var)
        validate_df:  validation DataFrame for Platt scaling (2018 data)

    Returns:
        dict with probability, risk_tier, drivers, narrative
    """
    models    = load_models()
    model     = models[target]["model"]
    imputer   = models[target]["imputer"]

    # Load validate set for Platt scaling
    if validate_df is None:
        validate_df = pd.read_parquet(PROC_DIR / "validate.parquet")

    # Fit calibrator on validate set
    calibrator = platt_scale(model, imputer, validate_df, target)

    # Prep loan
    X_loan = pd.DataFrame([loan_data])[[c for c in FEATURE_COLS
                                         if c in loan_data]]

    # Get calibrated probability
    raw_prob, cal_prob = get_calibrated_prob(model, imputer, calibrator, X_loan)

    # Get SHAP drivers
    drivers = get_shap_drivers(model, imputer, X_loan, top_n=5)

    # Population mean from validate set
    pop_mean = validate_df[target].mean()

    # Build prompt
    prompt = build_prompt(loan_data, drivers, cal_prob, target, pop_mean)

    # Generate narrative
    narrative = generate_narrative(prompt, api_key=api_key)

    # Risk tier
    rel_risk = cal_prob / max(pop_mean, 1e-4)
    if rel_risk < 0.7:
        risk_tier = "LOW"
    elif rel_risk < 1.3:
        risk_tier = "MODERATE"
    elif rel_risk < 2.0:
        risk_tier = "ELEVATED"
    else:
        risk_tier = "HIGH"

    return {
        "target":        target,
        "raw_prob":      round(raw_prob, 4),
        "cal_prob":      round(cal_prob, 4),
        "risk_tier":     risk_tier,
        "rel_risk":      round(rel_risk, 2),
        "pop_mean":      round(pop_mean, 4),
        "top_drivers":   drivers,
        "narrative":     narrative,
        "prompt":        prompt,
    }


def batch_explain(loans: list, target: str = "ever_90dpd_24mo",
                  api_key: str = None) -> list:
    """Explain multiple loans. Returns list of result dicts."""
    validate_df = pd.read_parquet(PROC_DIR / "validate.parquet")
    results = []
    for i, loan in enumerate(loans):
        print(f"  Explaining loan {i+1}/{len(loans)}...")
        result = explain_loan(loan, target=target, api_key=api_key,
                              validate_df=validate_df)
        results.append(result)
    return results


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LLM risk narrative for a loan.")
    parser.add_argument("--api-key",  type=str, default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--target",   type=str, default="ever_90dpd_24mo",
                        choices=["ever_90dpd_24mo", "ever_default"])
    parser.add_argument("--demo",     action="store_true",
                        help="Run with a demo loan (no real API call)")
    args = parser.parse_args()

    # Example high-risk loan
    high_risk_loan = {
        "fico_score":               620,
        "dti_ratio":                48,
        "num_borrowers":            1,
        "first_time_homebuyer_flag":1,
        "orig_ltv":                 97,
        "orig_cltv":                97,
        "cltv_ltv_diff":            0,
        "mortgage_insurance_flag":  1,
        "orig_interest_rate":       5.5,
        "loan_term_years":          30,
        "loan_purpose_encoded":     0,
        "product_type_encoded":     0,
        "property_type_encoded":    0,
        "num_units":                1,
        "occupancy_encoded":        0,
        "property_state":           10,
        "orig_year":                2018,
        "mortgage_rate_at_orig":    4.54,
        "unemp_at_orig":            3.9,
        "hpi_growth_at_orig":       5.1,
    }

    # Example low-risk loan
    low_risk_loan = {
        "fico_score":               780,
        "dti_ratio":                28,
        "num_borrowers":            2,
        "first_time_homebuyer_flag":0,
        "orig_ltv":                 65,
        "orig_cltv":                65,
        "cltv_ltv_diff":            0,
        "mortgage_insurance_flag":  0,
        "orig_interest_rate":       3.8,
        "loan_term_years":          30,
        "loan_purpose_encoded":     0,
        "product_type_encoded":     0,
        "property_type_encoded":    0,
        "num_units":                1,
        "occupancy_encoded":        0,
        "property_state":           5,
        "orig_year":                2016,
        "mortgage_rate_at_orig":    3.65,
        "unemp_at_orig":            4.9,
        "hpi_growth_at_orig":       5.6,
    }

    if args.demo:
        # Show prompt without calling API
        validate_df = pd.read_parquet(PROC_DIR / "validate.parquet")
        models      = load_models()
        model       = models[args.target]["model"]
        imputer     = models[args.target]["imputer"]
        calibrator  = platt_scale(model, imputer, validate_df, args.target)
        X_loan      = pd.DataFrame([high_risk_loan])[[c for c in FEATURE_COLS
                                                       if c in high_risk_loan]]
        _, cal_prob = get_calibrated_prob(model, imputer, calibrator, X_loan)
        drivers     = get_shap_drivers(model, imputer, X_loan)
        pop_mean    = validate_df[args.target].mean()
        prompt      = build_prompt(high_risk_loan, drivers, cal_prob,
                                   args.target, pop_mean)
        print("=" * 60)
        print("DEMO MODE — Prompt that would be sent to Claude:")
        print("=" * 60)
        print(prompt)
        print("\n" + "=" * 60)
        print(f"Calibrated probability: {cal_prob*100:.1f}%")
        print(f"Population mean:        {pop_mean*100:.1f}%")
        print(f"Relative risk:          {cal_prob/pop_mean:.1f}x")
    else:
        print("=" * 60)
        print("Testing HIGH-RISK loan explanation...")
        print("=" * 60)
        result = explain_loan(high_risk_loan, target=args.target,
                              api_key=args.api_key)
        print(f"\nCalibrated probability: {result['cal_prob']*100:.1f}%")
        print(f"Risk tier:              {result['risk_tier']}")
        print(f"Relative risk:          {result['rel_risk']}x population")
        print(f"\nTop drivers:")
        for d in result["top_drivers"][:3]:
            print(f"  {d['label']:30s} = {d['value']:6}  "
                  f"→ {d['direction']} risk ({d['pct_impact']:.1f}%)")
        print(f"\n{'='*60}")
        print("NARRATIVE:")
        print("=" * 60)
        print(result["narrative"])

        print("\n\nTesting LOW-RISK loan explanation...")
        result2 = explain_loan(low_risk_loan, target=args.target,
                               api_key=args.api_key)
        print(f"\nCalibrated probability: {result2['cal_prob']*100:.1f}%")
        print(f"Risk tier:              {result2['risk_tier']}")
        print(f"\n{'='*60}")
        print("NARRATIVE:")
        print("=" * 60)
        print(result2["narrative"])
