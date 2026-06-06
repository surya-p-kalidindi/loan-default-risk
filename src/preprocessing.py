"""
Fannie Mae Loan Default — Preprocessing Pipeline
==================================================
Loads acquisition data (synthetic or real Fannie Mae files),
engineers features, and produces time-based train/validate/test splits.

Real Fannie Mae file format:
  Acquisition: Fnma_sf_acq_YYYY_QQ.txt (pipe-delimited, no header)
  Performance: Fnma_sf_per_YYYY_QQ.txt (pipe-delimited, no header)

Targets:
  ever_90dpd_24mo  — 90+ DPD within first 24 months of origination
  ever_default     — foreclosure / charge-off / REO at any point

Time-based split (no random CV — mortgage data is time-series-like):
  Train:    2010–2017
  Validate: 2018
  Test:     2019
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Real Fannie Mae acquisition file column names (official schema) ───────────
# Source: Fannie Mae SF Loan Performance Data File Layout
ACQ_COLS = [
    "loan_id", "orig_channel", "seller_name", "orig_interest_rate",
    "orig_upb", "orig_loan_term", "orig_date", "first_pay_date",
    "orig_ltv", "orig_cltv", "num_borrowers", "dti_ratio",
    "borrower_credit_score", "first_time_homebuyer", "loan_purpose",
    "property_type", "num_units", "occupancy_status", "property_state",
    "zip_3", "msa", "mortgage_insurance_pct", "product_type",
    "co_borrower_credit_score", "mortgage_insurance_type",
    "relocation_mortgage_indicator",
]

# Performance file columns
PERF_COLS = [
    "loan_id", "monthly_period_str", "servicer_name", "current_interest_rate",
    "current_upb", "loan_age", "remaining_months", "adjusted_months",
    "maturity_date", "msa", "current_dlq", "modification_flag",
    "zero_bal_code", "zero_bal_date", "last_paid_install_date",
    "foreclosure_date", "disposition_date", "foreclosure_costs",
    "prop_preservation_repair_costs", "asset_recovery_costs",
    "misc_holding_expenses", "taxes_insurance_holding",
    "net_sale_proceeds", "credit_enhancement_proceeds",
    "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
    "non_mi_recoveries", "expenses", "legal_costs", "maintenance_costs",
    "taxes_holding", "misc_holding", "actual_loss", "modification_cost",
    "step_modification_flag", "deferred_payment_plan",
    "estimated_ltv", "zero_bal_removal_upb", "delinquent_accrued_interest",
    "delinquency_due_disaster", "borrower_assistance_status",
    "current_month_modification_cost", "interest_bearing_upb",
]

# ── Feature columns used in model ─────────────────────────────────────────────
FEATURE_COLS = [
    # Borrower credit
    "fico_score", "dti_ratio", "num_borrowers",
    "first_time_homebuyer_flag",
    # Collateral
    "orig_ltv", "orig_cltv", "cltv_ltv_diff",
    "mortgage_insurance_flag",
    # Loan structure
    "orig_interest_rate", "loan_term_years",
    "loan_purpose_encoded", "product_type_encoded",
    # Property
    "property_type_encoded", "num_units", "occupancy_encoded",
    # Geography
    "property_state",
    # Vintage / macro
    "orig_year", "mortgage_rate_at_orig", "unemp_at_orig",
    "hpi_growth_at_orig",
]

TARGET_COLS = ["ever_90dpd_24mo", "ever_default"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_synthetic(path=None):
    """Load pre-generated synthetic dataset."""
    if path is None:
        path = PROC_DIR / "fannie_mae_synthetic.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded synthetic data: {df.shape[0]:,} loans")
    return df


def load_real_acquisition(data_dir: Path, years: list) -> pd.DataFrame:
    """
    Load real Fannie Mae acquisition files.
    Files named: Acquisition_YYYYQQ.txt
    Pipe-delimited, no header row.
    """
    frames = []
    for year in years:
        for q in range(1, 5):
            fname = f"Acquisition_{year}Q{q}.txt"
            path  = data_dir / fname
            if not path.exists():
                continue
            df = pd.read_csv(path, sep="|", header=None,
                             names=ACQ_COLS, low_memory=False)
            df["orig_year"] = year
            frames.append(df)
            print(f"  Loaded {fname}: {len(df):,} rows")

    if not frames:
        raise FileNotFoundError(
            f"No acquisition files found in {data_dir}\n"
            f"Expected: Acquisition_YYYYQQ.txt"
        )
    return pd.concat(frames, ignore_index=True)


def load_real_performance(data_dir: Path, years: list) -> pd.DataFrame:
    """Load real Fannie Mae performance files."""
    frames = []
    for year in years:
        for q in range(1, 5):
            fname = f"Performance_{year}Q{q}.txt"
            path  = data_dir / fname
            if not path.exists():
                continue
            # Only load columns we need to save memory
            df = pd.read_csv(path, sep="|", header=None,
                             names=PERF_COLS[:14],  # first 14 cols sufficient
                             usecols=[0, 1, 5, 10, 12, 13],
                             low_memory=False)
            df.columns = ["loan_id", "monthly_period_str", "loan_age",
                          "current_dlq", "zero_bal_code", "zero_bal_date"]
            frames.append(df)
            print(f"  Loaded {fname}: {len(df):,} rows")

    return pd.concat(frames, ignore_index=True)


def derive_targets_from_real(acq_df: pd.DataFrame,
                              perf_df: pd.DataFrame) -> pd.DataFrame:
    """Derive targets from real performance data."""
    # Clean loan_age
    perf_df["loan_age"] = pd.to_numeric(perf_df["loan_age"], errors="coerce")

    # ever_90dpd_24mo
    early = perf_df[perf_df["loan_age"] <= 24].copy()
    # dlq codes: "0"=current, "1"=30dpd, "2"=60dpd, "3"=90dpd+
    early["dlq_num"] = pd.to_numeric(early["current_dlq"], errors="coerce").fillna(0)
    dlq_24 = early.groupby("loan_id")["dlq_num"].max().reset_index()
    dlq_24["ever_90dpd_24mo"] = (dlq_24["dlq_num"] >= 3).astype(int)

    # ever_default: zero_bal_code 03=short sale, 06=repurchase, 09=REO,
    #               02=third party sale, 15=note sale
    default_codes = {"02","03","06","09","15"}
    defaults = perf_df[perf_df["zero_bal_code"].astype(str).isin(default_codes)]
    default_ids = set(defaults["loan_id"].unique())
    dlq_24["ever_default"] = dlq_24["loan_id"].isin(default_ids).astype(int)

    return acq_df.merge(
        dlq_24[["loan_id","ever_90dpd_24mo","ever_default"]],
        on="loan_id", how="left"
    )


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features from raw acquisition columns.
    All features use origination-time data only — no leakage.
    """
    out = df.copy()

    # ── Credit ────────────────────────────────────────────────────────────
    # Use borrower FICO; fall back to co-borrower if missing
    out["fico_score"] = pd.to_numeric(
        out.get("borrower_credit_score", out.get("fico_score")),
        errors="coerce"
    )
    # If co-borrower exists, use minimum (conservative)
    if "co_borrower_credit_score" in out.columns:
        co = pd.to_numeric(out["co_borrower_credit_score"], errors="coerce")
        out["fico_score"] = np.where(
            co.notna(),
            np.minimum(out["fico_score"], co),
            out["fico_score"]
        )
    out["fico_score"] = out["fico_score"].clip(300, 850)

    # DTI
    out["dti_ratio"] = pd.to_numeric(out["dti_ratio"], errors="coerce").clip(0, 65)

    # ── Collateral ─────────────────────────────────────────────────────────
    out["orig_ltv"]  = pd.to_numeric(out["orig_ltv"],  errors="coerce").clip(0, 105)
    out["orig_cltv"] = pd.to_numeric(out["orig_cltv"], errors="coerce").clip(0, 105)

    # CLTV - LTV gap (captures second lien presence)
    out["cltv_ltv_diff"] = (out["orig_cltv"] - out["orig_ltv"]).clip(0, 30)

    # MI flag
    mi_pct = pd.to_numeric(out.get("mortgage_insurance_pct", 0), errors="coerce").fillna(0)
    out["mortgage_insurance_flag"] = (mi_pct > 0).astype(int)

    # ── Loan structure ──────────────────────────────────────────────────────
    out["orig_interest_rate"] = pd.to_numeric(
        out["orig_interest_rate"], errors="coerce"
    ).clip(1.0, 12.0)

    out["loan_term_years"] = pd.to_numeric(
        out.get("orig_loan_term", 360), errors="coerce"
    ).fillna(360) / 12

    # ── Categorical encodings ───────────────────────────────────────────────
    out["first_time_homebuyer_flag"] = (
        out["first_time_homebuyer"].astype(str).str.upper() == "Y"
    ).astype(int)

    purpose_map = {"P": 0, "R": 1, "C": 2, "U": 0}
    out["loan_purpose_encoded"] = out["loan_purpose"].map(purpose_map).fillna(0)

    prop_map = {"SF": 0, "CO": 1, "PU": 2, "MH": 3, "CP": 2}
    out["property_type_encoded"] = out["property_type"].map(prop_map).fillna(0)

    occ_map = {"P": 0, "S": 1, "I": 2, "U": 0}
    out["occupancy_encoded"] = out["occupancy_status"].map(occ_map).fillna(0)

    product_map = {"FRM": 0, "ARM": 1}
    out["product_type_encoded"] = out.get("product_type", pd.Series("FRM", index=out.index)
                                          ).map(product_map).fillna(0)

    out["num_units"] = pd.to_numeric(out["num_units"], errors="coerce").fillna(1).clip(1, 4)
    out["num_borrowers"] = pd.to_numeric(out["num_borrowers"], errors="coerce").fillna(1).clip(1, 4)

    # State as categorical code
    out["property_state"] = out["property_state"].astype("category").cat.codes

    # ── Macro / vintage ─────────────────────────────────────────────────────
    out["orig_year"] = pd.to_numeric(out["orig_year"], errors="coerce").fillna(2015)

    # Use pre-computed macro cols if present (synthetic), else fill with year avg
    for col, default_val in [("mortgage_rate_at_orig", 4.0),
                              ("unemp_at_orig", 5.0),
                              ("hpi_growth_at_orig", 5.0)]:
        if col not in out.columns:
            out[col] = default_val
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default_val)

    return out


# ── Time-based split ───────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame):
    """
    Walk-forward time split — mirrors how models are evaluated in production.

    Train:    2010–2017  (learn patterns from historical book)
    Validate: 2018       (tune hyperparameters on recent unseen year)
    Test:     2019       (final holdout — touch once)

    Rationale: Random k-fold on mortgage data lets future vintages
    predict past ones, inflating AUC. Time split is honest.
    """
    train    = df[df["orig_year"] <= 2017].copy()
    validate = df[df["orig_year"] == 2018].copy()
    test     = df[df["orig_year"] == 2019].copy()

    print(f"\nTime-based split:")
    print(f"  Train    (2010-2017): {len(train):,} loans")
    print(f"  Validate (2018):      {len(validate):,} loans")
    print(f"  Test     (2019):      {len(test):,} loans")

    for label, split in [("Train", train), ("Validate", validate), ("Test", test)]:
        if "ever_90dpd_24mo" in split.columns:
            dpd_rate = split["ever_90dpd_24mo"].mean() * 100
            def_rate = split["ever_default"].mean() * 100
            print(f"    {label}: 90DPD={dpd_rate:.1f}%  Default={def_rate:.1f}%")

    return train, validate, test


# ── Missing value report ───────────────────────────────────────────────────────

def report_missingness(df: pd.DataFrame, label: str = ""):
    existing = [c for c in FEATURE_COLS if c in df.columns]
    missing = {c: df[c].isna().sum() for c in existing if df[c].isna().sum() > 0}
    if missing:
        print(f"\nMissing values {label}:")
        for col, n in sorted(missing.items(), key=lambda x: -x[1]):
            print(f"  {col:30s}: {n:,} ({100*n/len(df):.1f}%)")
    else:
        print(f"  No missing values in features {label}")


# ── Main ───────────────────────────────────────────────────────────────────────

def preprocess(use_synthetic: bool = True,
               real_data_dir: Path = None,
               save: bool = True):
    """
    Full preprocessing pipeline.

    Args:
        use_synthetic: If True, use generated synthetic data.
                       Set False when real Fannie Mae files are available.
        real_data_dir: Path to folder containing real Acquisition/Performance files.
        save: Write processed parquet files to data/processed/.
    """
    print("=" * 55)
    print("Fannie Mae Preprocessing Pipeline")
    print("=" * 55)

    if use_synthetic:
        print("\nMode: SYNTHETIC (swap use_synthetic=False for real data)")
        df = load_synthetic()
    else:
        print(f"\nMode: REAL DATA from {real_data_dir}")
        years = list(range(2010, 2020))
        acq  = load_real_acquisition(real_data_dir, years)
        perf = load_real_performance(real_data_dir, years)
        df   = derive_targets_from_real(acq, perf)

    # Feature engineering
    print("\nEngineering features...")
    df = engineer_features(df)

    # Validate all feature cols present
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"  WARNING: Missing feature columns: {missing_cols}")

    report_missingness(df, "(full dataset)")

    # Time split
    train, validate, test = time_split(df)

    # Feature + target arrays
    existing_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\nFeature set ({len(existing_features)} features):")
    for f in existing_features:
        print(f"  {f}")

    if save:
        train.to_parquet(PROC_DIR / "train.parquet",    index=False)
        validate.to_parquet(PROC_DIR / "validate.parquet", index=False)
        test.to_parquet(PROC_DIR / "test.parquet",     index=False)
        print(f"\nSaved → {PROC_DIR}/train.parquet")
        print(f"Saved → {PROC_DIR}/validate.parquet")
        print(f"Saved → {PROC_DIR}/test.parquet")

    return train, validate, test, existing_features


if __name__ == "__main__":
    train, validate, test, features = preprocess(use_synthetic=True)
    print(f"\nReady for modeling.")
    print(f"Features: {len(features)}")
    print(f"Targets:  ever_90dpd_24mo, ever_default")
