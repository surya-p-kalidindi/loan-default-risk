"""
Synthetic Fannie Mae Loan Performance Data Generator
======================================================
Generates realistic synthetic data matching the exact Fannie Mae
Single Family Loan Performance dataset schema.

Schema reference:
  https://capitalmarkets.fanniemae.com/media/9391/display
  Fannie Mae Single Family Loan Performance Data — File Layout

Two files per acquisition cohort (matching real Fannie Mae structure):
  1. Acquisition file  — loan characteristics at origination
  2. Performance file  — monthly performance history per loan

Targets derived:
  ever_90dpd_24mo  — 90+ DPD within first 24 months (delinquency model)
  ever_default     — foreclosure / charge-off / REO ever (default model)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import truncnorm

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── State list (for geographic distribution) ──────────────────────────────────
STATES = [
    "CA","TX","FL","NY","IL","PA","OH","GA","NC","MI",
    "NJ","VA","WA","AZ","MA","TN","IN","MO","MD","WI",
    "CO","MN","SC","AL","LA","KY","OR","OK","CT","UT",
    "IA","NV","AR","MS","KS","NM","NE","ID","WV","HI",
    "NH","ME","RI","MT","DE","SD","ND","AK","VT","WY"
]

# State-level baseline default rate adjustments (relative risk)
STATE_RISK = {
    "FL": 1.4, "NV": 1.3, "AZ": 1.2, "CA": 1.1, "IL": 1.1,
    "MI": 1.2, "OH": 1.1, "GA": 1.1, "TX": 0.9, "CO": 0.85,
    "WA": 0.85, "MA": 0.85, "NY": 1.0, "VA": 0.9, "MD": 0.95,
}

# Macro environment by year (30yr mortgage rate, unemployment proxy)
YEAR_MACRO = {
    2010: {"rate": 4.69, "unemp": 9.6,  "hpi_growth": -3.1},
    2011: {"rate": 4.45, "unemp": 8.9,  "hpi_growth": -4.0},
    2012: {"rate": 3.66, "unemp": 8.1,  "hpi_growth":  4.3},
    2013: {"rate": 3.98, "unemp": 7.4,  "hpi_growth":  9.3},
    2014: {"rate": 4.17, "unemp": 6.2,  "hpi_growth":  5.7},
    2015: {"rate": 3.85, "unemp": 5.3,  "hpi_growth":  5.4},
    2016: {"rate": 3.65, "unemp": 4.9,  "hpi_growth":  5.6},
    2017: {"rate": 3.99, "unemp": 4.4,  "hpi_growth":  6.2},
    2018: {"rate": 4.54, "unemp": 3.9,  "hpi_growth":  5.1},
    2019: {"rate": 3.94, "unemp": 3.7,  "hpi_growth":  4.8},
    2020: {"rate": 3.11, "unemp": 8.1,  "hpi_growth":  9.1},
    2021: {"rate": 2.96, "unemp": 5.4,  "hpi_growth": 17.4},
    2022: {"rate": 5.34, "unemp": 3.6,  "hpi_growth":  8.2},
}

# ── Origination feature distributions ────────────────────────────────────────
def sample_fico(n, rng):
    """FICO scores — roughly normal, clipped 300-850, mean ~740."""
    mu, sigma = 740, 60
    lo, hi = (300 - mu) / sigma, (850 - mu) / sigma
    return truncnorm.rvs(lo, hi, loc=mu, scale=sigma, size=n,
                         random_state=rng).astype(int)

def sample_ltv(n, rng):
    """LTV — bimodal: cluster at 80 (conventional) and 95-97 (low down)."""
    mix = rng.uniform(0, 1, n)
    ltv = np.where(
        mix < 0.45,
        rng.normal(80, 5, n),       # conventional 20% down
        np.where(
            mix < 0.75,
            rng.normal(95, 3, n),   # low down payment
            rng.normal(70, 10, n),  # equity-rich refi
        )
    )
    return np.clip(ltv, 20, 105).round(1)

def sample_dti(n, rng):
    """DTI — right-skewed, mean ~35, hard cap at 50 for conforming."""
    dti = rng.normal(35, 8, n)
    return np.clip(dti, 10, 50).round(1)


# ── Default probability model ────────────────────────────────────────────────
def compute_default_prob(df):
    """
    Logistic model calibrated to approximate real Fannie Mae default rates:
    - Post-crisis (2010-2013): 3-5% 90DPD rates
    - Recovery (2014-2018): 1-2%
    - Pre-COVID (2019): ~1%
    - COVID vintages (2020-2021): TBD

    Features weighted by published GSE credit risk research.
    """
    # FICO effect: strong negative (higher FICO = lower default)
    fico_effect = -0.015 * (df["fico_score"] - 700)

    # LTV effect: positive (higher LTV = higher default)
    ltv_effect = 0.04 * (df["orig_ltv"] - 80)

    # CLTV additional effect
    cltv_effect = 0.02 * (df["orig_cltv"] - df["orig_ltv"]).clip(0)

    # DTI effect
    dti_effect = 0.05 * (df["dti_ratio"] - 36)

    # First-time homebuyer: higher risk
    fthb_effect = 0.3 * df["first_time_homebuyer"].map({"Y": 1, "N": 0, "U": 0.3})

    # Loan purpose: cash-out refi slightly higher risk
    purpose_map = {"P": 0.0, "R": -0.1, "C": 0.2, "U": 0.0}
    purpose_effect = df["loan_purpose"].map(purpose_map).fillna(0)

    # Property type: multi-unit and condo slightly higher
    prop_map = {"SF": 0.0, "CO": 0.15, "PU": 0.1, "MH": 0.3, "CP": 0.1}
    prop_effect = df["property_type"].map(prop_map).fillna(0)

    # Units: 2-4 unit higher risk
    unit_effect = (df["num_units"] - 1) * 0.15

    # Occupancy: investor highest risk
    occ_map = {"P": 0.0, "S": 0.2, "I": 0.5, "U": 0.1}
    occ_effect = df["occupancy_status"].map(occ_map).fillna(0)

    # State risk
    state_risk = df["property_state"].map(STATE_RISK).fillna(1.0)
    state_effect = np.log(state_risk)

    # Macro: high unemployment + low HPI = higher default
    unemp_effect  =  0.08 * (df["unemp_at_orig"] - 5.0)
    hpi_effect    = -0.03 * (df["hpi_growth_at_orig"] - 5.0)
    rate_effect   =  0.05 * (df["mortgage_rate_at_orig"] - 4.0)

    # Intercept calibrated for ~2% base rate
    intercept = -4.2

    log_odds = (intercept + fico_effect + ltv_effect + cltv_effect +
                dti_effect + fthb_effect + purpose_effect + prop_effect +
                unit_effect + occ_effect + state_effect +
                unemp_effect + hpi_effect + rate_effect)

    return 1 / (1 + np.exp(-log_odds))


# ── Main generator ────────────────────────────────────────────────────────────
def generate_acquisition_file(n_loans, orig_year, rng):
    """Generate one year's worth of originations."""
    macro = YEAR_MACRO.get(orig_year, YEAR_MACRO[2018])

    loan_ids = [f"{orig_year}Q{q:01d}{i:07d}"
                for i, q in zip(range(n_loans),
                                rng.integers(1, 5, n_loans))]

    states = rng.choice(STATES, size=n_loans)

    fico   = sample_fico(n_loans, rng)
    ltv    = sample_ltv(n_loans, rng)
    cltv   = np.minimum(ltv + rng.choice([0, 0, 0, 5, 10, 15], size=n_loans), 105)
    dti    = sample_dti(n_loans, rng)

    df = pd.DataFrame({
        "loan_id":               loan_ids,
        "orig_year":             orig_year,
        "orig_channel":          rng.choice(["R","B","C"], n_loans, p=[0.45,0.35,0.20]),
        "seller_name":           rng.choice(["WELLS FARGO","JPMORGAN","BOFA","CHASE","OTHER"],
                                            n_loans, p=[0.18,0.16,0.15,0.14,0.37]),
        "orig_interest_rate":    (macro["rate"] + rng.normal(0, 0.4, n_loans)).round(3).clip(2.0, 8.0),
        "orig_upb":              rng.integers(50000, 750000, n_loans),
        "orig_loan_term":        rng.choice([180, 240, 360], n_loans, p=[0.15,0.05,0.80]),
        "orig_date":             [f"{orig_year}{rng.integers(1,13):02d}" for _ in range(n_loans)],
        "first_pay_date":        [f"{orig_year}{rng.integers(2,13):02d}" for _ in range(n_loans)],
        "orig_ltv":              ltv,
        "orig_cltv":             cltv,
        "num_borrowers":         rng.choice([1, 2], n_loans, p=[0.35, 0.65]),
        "dti_ratio":             dti,
        "borrower_credit_score": fico,
        "co_borrower_credit_score": np.where(
                                    rng.random(n_loans) < 0.65,
                                    (fico + rng.normal(0, 30, n_loans)).clip(300, 850).astype(int),
                                    np.nan),
        "first_time_homebuyer":  rng.choice(["Y","N","U"], n_loans, p=[0.35,0.60,0.05]),
        "loan_purpose":          rng.choice(["P","R","C","U"], n_loans, p=[0.55,0.25,0.18,0.02]),
        "property_type":         rng.choice(["SF","CO","PU","MH","CP"], n_loans,
                                            p=[0.72,0.12,0.08,0.04,0.04]),
        "num_units":             rng.choice([1,2,3,4], n_loans, p=[0.86,0.07,0.04,0.03]),
        "occupancy_status":      rng.choice(["P","S","I","U"], n_loans, p=[0.80,0.08,0.10,0.02]),
        "property_state":        states,
        "zip_3":                 [str(z).zfill(3) for z in rng.integers(100, 999, n_loans)],
        "msa":                   rng.integers(10000, 49999, n_loans),
        "mortgage_insurance_pct":np.where(ltv > 80,
                                          rng.choice([6,12,18,25,30], n_loans),
                                          0),
        "product_type":          rng.choice(["FRM","ARM"], n_loans, p=[0.92,0.08]),
        "fico_score":            fico,  # alias for model use
        "unemp_at_orig":         macro["unemp"]   + rng.normal(0, 0.5, n_loans),
        "hpi_growth_at_orig":    macro["hpi_growth"] + rng.normal(0, 1.0, n_loans),
        "mortgage_rate_at_orig": macro["rate"],
    })

    return df


def generate_performance_file(acq_df, n_months=60, rng=None):
    """
    Generate monthly performance records for each loan.
    Returns long-format DataFrame with one row per loan-month.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    default_probs = compute_default_prob(acq_df)
    records = []

    for idx, (_, loan) in enumerate(acq_df.iterrows()):
        pd_base = default_probs.iloc[idx]

        current_upb = loan["orig_upb"]
        dlq_status  = 0   # current delinquency counter
        terminated  = False
        disposition = None

        for month in range(1, n_months + 1):
            if terminated:
                break

            # Monthly default probability (annualize → monthly)
            monthly_pd = 1 - (1 - pd_base) ** (1/12)

            # Slightly higher risk in months 12-36 (seasoning peak)
            if 12 <= month <= 36:
                monthly_pd *= 1.3

            # Transition: current → 30 DPD
            if dlq_status == 0:
                if rng.random() < monthly_pd:
                    dlq_status = 1
            elif dlq_status == 1:
                if rng.random() < 0.6:
                    dlq_status = 2   # 60 DPD
                else:
                    dlq_status = 0   # cured
            elif dlq_status == 2:
                if rng.random() < 0.55:
                    dlq_status = 3   # 90 DPD
                else:
                    dlq_status = 0
            elif dlq_status >= 3:
                if rng.random() < 0.15:
                    # Default event
                    disposition = rng.choice(["F","S","T","N"],
                                             p=[0.45, 0.25, 0.20, 0.10])
                    terminated = True
                elif rng.random() < 0.20:
                    dlq_status = 0   # reinstatement
                else:
                    dlq_status = min(dlq_status + 1, 9)

            # UPB amortization (simplified)
            current_upb *= 0.9985

            records.append({
                "loan_id":          loan["loan_id"],
                "monthly_period":   month,
                "current_upb":      round(current_upb, 2),
                "dlq_status":       dlq_status * 30,  # 0,30,60,90,120...
                "loan_age":         month,
                "remaining_months": loan["orig_loan_term"] - month,
                "repurchase_flag":  "N",
                "modification_flag":"N",
                "zero_bal_code":    disposition if terminated else None,
                "zero_bal_date":    f"2020{month:02d}" if terminated else None,
            })

    return pd.DataFrame(records)


def derive_targets(acq_df, perf_df):
    """Derive model targets from performance history."""
    # ever_90dpd_24mo: any 90+ DPD in first 24 months
    early = perf_df[perf_df["monthly_period"] <= 24]
    dlq_24 = early.groupby("loan_id")["dlq_status"].max().reset_index()
    dlq_24["ever_90dpd_24mo"] = (dlq_24["dlq_status"] >= 90).astype(int)

    # ever_default: foreclosure (F), short sale (S), or deed-in-lieu (T)
    defaults = perf_df[perf_df["zero_bal_code"].isin(["F","S","T"])]
    default_ids = set(defaults["loan_id"].unique())
    dlq_24["ever_default"] = dlq_24["loan_id"].isin(default_ids).astype(int)

    return acq_df.merge(dlq_24[["loan_id","ever_90dpd_24mo","ever_default"]],
                        on="loan_id", how="left")


def generate_dataset(years=range(2010, 2020), n_per_year=5000, seed=42):
    """Generate full multi-year dataset."""
    rng = np.random.default_rng(seed)

    print("Generating synthetic Fannie Mae loan performance data...")
    print(f"Years: {list(years)}, ~{n_per_year:,} loans/year")

    all_acq  = []
    all_perf = []

    for year in years:
        print(f"  {year}... ", end="", flush=True)
        acq  = generate_acquisition_file(n_per_year, year, rng)
        perf = generate_performance_file(acq, n_months=48, rng=rng)
        all_acq.append(acq)
        all_perf.append(perf)
        print(f"{len(acq):,} loans, {len(perf):,} performance records")

    acq_full  = pd.concat(all_acq,  ignore_index=True)
    perf_full = pd.concat(all_perf, ignore_index=True)

    # Derive targets
    print("\nDeriving targets...")
    final = derive_targets(acq_full, perf_full)

    # Report
    print(f"\nDataset summary:")
    print(f"  Total loans:        {len(final):,}")
    print(f"  Ever 90DPD (24mo):  {final['ever_90dpd_24mo'].sum():,} "
          f"({100*final['ever_90dpd_24mo'].mean():.1f}%)")
    print(f"  Ever default:       {final['ever_default'].sum():,} "
          f"({100*final['ever_default'].mean():.1f}%)")
    print(f"\nBy vintage:")
    vintage = final.groupby("orig_year").agg(
        n=("loan_id","count"),
        dpd_rate=("ever_90dpd_24mo","mean"),
        default_rate=("ever_default","mean")
    )
    vintage["dpd_rate"]     = (vintage["dpd_rate"]     * 100).round(2)
    vintage["default_rate"] = (vintage["default_rate"] * 100).round(2)
    print(vintage.to_string())

    # Save
    out = PROC_DIR / "fannie_mae_synthetic.parquet"
    final.to_parquet(out, index=False)
    print(f"\nSaved → {out}")

    # Save performance file separately (needed for survival analysis later)
    perf_out = PROC_DIR / "fannie_mae_performance.parquet"
    perf_full.to_parquet(perf_out, index=False)
    print(f"Saved → {perf_out}")

    return final, perf_full


if __name__ == "__main__":
    df, perf = generate_dataset(
        years=range(2010, 2020),
        n_per_year=5000,
        seed=42
    )
