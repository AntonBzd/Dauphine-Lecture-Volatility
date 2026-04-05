import numpy as np
import pandas as pd

from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.util import check_is_true


def build_naive_implied_realized_spread(
    df_iv_reference: pd.DataFrame,
    df_spot: pd.DataFrame,
    rolling_window: int = 30,
) -> pd.DataFrame:
    """Build a benchmark implied-realized spread using rolling realized volatility.

    Args:
        df_iv_reference (pd.DataFrame): Daily reference implied volatility series by date and ticker.
        df_spot (pd.DataFrame): Daily spot price series by date and ticker.
        rolling_window (int): Window length used to compute rolling realized volatility.

    Returns:
        pd.DataFrame: Daily benchmark implied-realized spread with reference implied volatility,
        rolling realized volatility, and spread value.
    """
    required_iv_cols = {"date", "ticker", "iv_reference"}
    required_spot_cols = {"date", "ticker", "spot"}

    check_is_true(
        len(required_iv_cols.difference(df_iv_reference.columns)) == 0,
        f"df_iv_reference missing columns: {required_iv_cols.difference(df_iv_reference.columns)}",
    )
    check_is_true(
        len(required_spot_cols.difference(df_spot.columns)) == 0,
        f"df_spot missing columns: {required_spot_cols.difference(df_spot.columns)}",
    )

    df_spot_sorted = df_spot[["date", "ticker", "spot"]].copy().sort_values(["ticker", "date"])
    df_spot_sorted["log_return"] = df_spot_sorted.groupby("ticker")["spot"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_spot_sorted["rolling_realized_vol"] = df_spot_sorted.groupby("ticker")["log_return"].transform(
        lambda x: x.rolling(rolling_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    df = (
        df_iv_reference[["date", "ticker", "iv_reference"]]
        .merge(
            df_spot_sorted[["date", "ticker", "rolling_realized_vol"]],
            on=["date", "ticker"],
            how="inner",
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    df["spread"] = df["iv_reference"] - df["rolling_realized_vol"]
    return df