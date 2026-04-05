import numpy as np
import pandas as pd

from investment_lab.constants import DAYS_PER_YEAR, TENOR_TO_PERIOD
from investment_lab.util import check_is_true


def compute_forward(df_options: pd.DataFrame, df_rates: pd.DataFrame) -> pd.DataFrame:
    """Compute interpolated risk-free rates and forward prices for option data.

    Args:
        df_options (pd.DataFrame): Daily option dataset containing at least date, spot,
        day_to_expiration and option_id.
        df_rates (pd.DataFrame): Daily interest rate curve with tenor columns defined
        in TENOR_TO_PERIOD.

    Returns:
        pd.DataFrame: Option dataset enriched with interpolated risk-free rates and
        forward prices.
    """
    
    missing_cols = set(TENOR_TO_PERIOD.keys()).difference(df_rates.columns)
    check_is_true(
        len(missing_cols) == 0, f"df_rates is missing columns: {missing_cols}"
    )
    missing_options_cols = {
        "date",
        "spot",
        "day_to_expiration",
        "option_id",
    }.difference(df_options.columns)
    check_is_true(
        len(missing_options_cols) == 0,
        f"df_options is missing columns: {missing_options_cols}",
    )

    tenor_cols = list(TENOR_TO_PERIOD.keys())
    tenor_values = np.asarray([TENOR_TO_PERIOD[col] for col in tenor_cols], dtype=float)

    df = df_options.merge(df_rates, on="date", how="left").copy()
    df = df.sort_values("date")
    df[tenor_cols] = df[tenor_cols].ffill().bfill()

    def _compute_values(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        dte = float(group["day_to_expiration"].iloc[0]) / DAYS_PER_YEAR
        rate_curve = group[tenor_cols].iloc[0].to_numpy(dtype=float)

        valid_mask = np.isfinite(rate_curve)
        if valid_mask.sum() == 0:
            group["risk_free_rate"] = 0.0
            return group

        interpolated_rate = interpolate_rates(
            eval_tenor=dte,
            tenors=tenor_values[valid_mask],
            rate_curve=rate_curve[valid_mask],
        )
        group["risk_free_rate"] = interpolated_rate
        return group

    df = (
        df.groupby(["date", "expiration"], group_keys=False)
        .apply(_compute_values)
        .reset_index(drop=True)
    )

    df["forward"] = df["spot"] * np.exp(
        df["risk_free_rate"] * df["day_to_expiration"] / DAYS_PER_YEAR
    )

    df_forward = (
        df.groupby(["ticker", "date", "expiration"])[["forward"]]
        .first()
        .ffill()
        .reset_index()
    )

    return df.drop(columns=tenor_cols + ["forward"]).merge(
        df_forward,
        how="left",
        on=["ticker", "date", "expiration"],
    )


def interpolate_rates(
    eval_tenor: float,
    tenors: pd.Series | np.ndarray,
    rate_curve: pd.Series | np.ndarray,
) -> float:
    """Linearly interpolate a risk-free rate for a given tenor.

    Args:
        eval_tenor (float): Target maturity (in years) at which the rate is evaluated.
        tenors (pd.Series | np.ndarray): Available maturities of the rate curve.
        rate_curve (pd.Series | np.ndarray): Corresponding interest rates.

    Returns:
        float: Interpolated risk-free rate.
    """
    tenors = np.asarray(tenors, dtype=float)
    rate_curve = np.asarray(rate_curve, dtype=float)

    check_is_true(
        len(tenors) == len(rate_curve),
        "Tenors and rate curve must have the same length.",
    )
    check_is_true(len(tenors) > 0, "Tenors and rate_curve must not be empty.")

    valid_mask = np.isfinite(tenors) & np.isfinite(rate_curve)
    tenors = tenors[valid_mask]
    rate_curve = rate_curve[valid_mask]

    check_is_true(len(tenors) > 0, "No valid tenor/rate pair available for interpolation.")

    sort_idx = np.argsort(tenors)
    tenors = tenors[sort_idx]
    rate_curve = rate_curve[sort_idx]

    unique_tenors, unique_indices = np.unique(tenors, return_index=True)
    tenors = unique_tenors
    rate_curve = rate_curve[unique_indices]

    if eval_tenor <= tenors[0]:
        return float(rate_curve[0])
    if eval_tenor >= tenors[-1]:
        return float(rate_curve[-1])

    idx_above = np.searchsorted(tenors, eval_tenor, side="left")
    idx_below = idx_above - 1

    tenor_below, tenor_above = tenors[idx_below], tenors[idx_above]
    rate_below, rate_above = rate_curve[idx_below], rate_curve[idx_above]

    if np.isclose(tenor_above, tenor_below):
        return float(rate_below)

    weight_above = (eval_tenor - tenor_below) / (tenor_above - tenor_below)
    weight_below = 1.0 - weight_above

    return float(weight_below * rate_below + weight_above * rate_above)