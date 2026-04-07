import numpy as np
import pandas as pd

from investment_lab.util import check_is_true


def build_zscore_allocation_signal(
    df_spread: pd.DataFrame,
    spread_col: str = "spread",
    rolling_window: int = 60,
    min_allocation: float = 0.0,
    max_allocation: float = 1.0,
    zscore_scaling: float = 0.25,
    center_at: float = 0.5,
) -> pd.DataFrame:
    """Build a dynamic allocation signal from the rolling z-score of an implied-realized spread.

    Args:
        df_spread (pd.DataFrame): Daily spread series by date and ticker.
        spread_col (str): Name of the spread column used to build the signal.
        rolling_window (int): Window length used to compute the rolling mean and standard deviation.
        min_allocation (float): Lower bound of the allocation signal.
        max_allocation (float): Upper bound of the allocation signal.
        zscore_scaling (float): Scaling coefficient applied to the spread z-score.
        center_at (float): Central allocation level around which the signal fluctuates.

    Returns:
        pd.DataFrame: Daily allocation signal by date and ticker, including the spread,
        spread z-score, and allocation.
    """
    required_cols = {"date", "ticker", spread_col}
    missing_cols = required_cols.difference(df_spread.columns)
    check_is_true(len(missing_cols) == 0, f"df_spread is missing columns: {missing_cols}")
    check_is_true(rolling_window >= 5, "rolling_window must be >= 5.")

    df = df_spread[["date", "ticker", spread_col]].copy().sort_values(["ticker", "date"])
    df["spread_mean"] = df.groupby("ticker")[spread_col].transform(lambda x: x.rolling(rolling_window).mean())
    df["spread_std"] = df.groupby("ticker")[spread_col].transform(lambda x: x.rolling(rolling_window).std())
    df["spread_zscore"] = (df[spread_col] - df["spread_mean"]) / df["spread_std"].replace(0, np.nan)
    df["allocation"] = (center_at + zscore_scaling * df["spread_zscore"]).clip(lower=min_allocation, upper=max_allocation)
    return df[["date", "ticker", spread_col, "spread_zscore", "allocation"]].dropna(subset=["allocation"]).reset_index(drop=True)


def build_percentile_allocation_signal(
    df_spread: pd.DataFrame,
    spread_col: str = "spread",
    rolling_window: int = 60,
    high_threshold: float = 0.75,
    low_threshold: float = 0.25,
    min_allocation: float = -1.0,
    max_allocation: float = 1.0,
) -> pd.DataFrame:
    """Build a dynamic allocation signal from the percentile ranking of an implied-realized spread.

    Args:
        df_spread (pd.DataFrame): Daily spread series by date and ticker.
        spread_col (str): Name of the spread column used to build the signal.
        rolling_window (int): Window length for the rolling percentile.
        high_threshold (float): Percentile above which we go short vol (e.g. 0.75).
        low_threshold (float): Percentile below which we go long vol (e.g. 0.25).
        min_allocation (float): Lower bound of the allocation signal.
        max_allocation (float): Upper bound of the allocation signal.

    Returns:
        pd.DataFrame: Daily allocation signal with spread_percentile and allocation.
    """
    df = df_spread[["date", "ticker", spread_col]].copy().sort_values(["ticker", "date"])

    df["spread_percentile"] = df.groupby("ticker")[spread_col].transform(
        lambda x: x.rolling(rolling_window).rank(pct=True)
    )

    conditions = [
        df["spread_percentile"] >= high_threshold,
        df["spread_percentile"] <= low_threshold,
    ]

    choices = [
        max_allocation * (df["spread_percentile"] - high_threshold) / (1.0 - high_threshold),
        min_allocation * (low_threshold - df["spread_percentile"]) / low_threshold,
    ]
    
    df["allocation"] = np.select(conditions, choices, default=0.0)

    return df.dropna(subset=["allocation"])