import pandas as pd

from investment_lab.util import check_is_true


def build_implied_realized_spread(
    df_iv_reference: pd.DataFrame,
    df_heston_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Build the implied-realized spread from reference implied volatility and Heston forecast volatility.

    Args:
        df_iv_reference (pd.DataFrame): Daily reference implied volatility series by date and ticker.
        df_heston_forecast (pd.DataFrame): Daily Heston forecast results by date and ticker.

    Returns:
        pd.DataFrame: Daily implied-realized spread with reference implied volatility,
        forecast volatility, filtered variance, and spread value.
    """
    required_iv_cols = {"date", "ticker", "iv_reference"}
    required_forecast_cols = {"date", "ticker", "forecast_volatility"}

    missing_iv_cols = required_iv_cols.difference(df_iv_reference.columns)
    missing_forecast_cols = required_forecast_cols.difference(df_heston_forecast.columns)

    check_is_true(len(missing_iv_cols) == 0, f"df_iv_reference is missing columns: {missing_iv_cols}")
    check_is_true(len(missing_forecast_cols) == 0, f"df_heston_forecast is missing columns: {missing_forecast_cols}")

    df = (
        df_iv_reference[["date", "ticker", "iv_reference"]]
        .merge(
            df_heston_forecast[["date", "ticker", "forecast_volatility", "filtered_variance"]],
            on=["date", "ticker"],
            how="inner",
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    df["spread"] = df["iv_reference"] - df["forecast_volatility"]
    return df