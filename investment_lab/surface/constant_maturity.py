import numpy as np
import pandas as pd

from investment_lab.constants import DAYS_PER_YEAR
from investment_lab.util import check_is_true


def interpolate_total_variance(
    target_time_to_maturity: float,
    time_to_maturities: pd.Series | np.ndarray,
    implied_volatilities: pd.Series | np.ndarray,
) -> float:
    """Interpolate implied volatility at a target maturity through total variance interpolation.

    Args:
        target_time_to_maturity (float): Target maturity expressed in years.
        time_to_maturities (pd.Series | np.ndarray): Available maturities expressed in years.
        implied_volatilities (pd.Series | np.ndarray): Implied volatilities associated with the available maturities.

    Returns:
        float: Interpolated implied volatility at the target maturity.
    """

    time_to_maturities = np.asarray(time_to_maturities, dtype=float)
    implied_volatilities = np.asarray(implied_volatilities, dtype=float)

    check_is_true(
        len(time_to_maturities) == len(implied_volatilities),
        "time_to_maturities and implied_volatilities must have the same length.",
    )
    check_is_true(len(time_to_maturities) >= 1, "Need at least one maturity to interpolate.")
    check_is_true(target_time_to_maturity > 0, "target_time_to_maturity must be > 0.")

    total_variances = (implied_volatilities**2) * time_to_maturities
    sort_idx = np.argsort(time_to_maturities)
    time_to_maturities = time_to_maturities[sort_idx]
    total_variances = total_variances[sort_idx]

    if len(time_to_maturities) == 1:
        interpolated_total_variance = total_variances[0]
    elif target_time_to_maturity <= time_to_maturities.min():
        interpolated_total_variance = total_variances[0]
    elif target_time_to_maturity >= time_to_maturities.max():
        interpolated_total_variance = total_variances[-1]
    else:
        interpolated_total_variance = np.interp(
            x=target_time_to_maturity,
            xp=time_to_maturities,
            fp=total_variances,
        )

    return float(np.sqrt(max(interpolated_total_variance / target_time_to_maturity, 0.0)))


def interpolate_total_variance_from_days(
    target_day_to_maturity: int,
    day_to_maturities: pd.Series | np.ndarray,
    implied_volatilities: pd.Series | np.ndarray,
) -> float:
    """Interpolate implied volatility at a target maturity expressed in days through total variance interpolation.

    Args:
        target_day_to_maturity (int): Target maturity expressed in days.
        day_to_maturities (pd.Series | np.ndarray): Available maturities expressed in days.
        implied_volatilities (pd.Series | np.ndarray): Implied volatilities associated with the available maturities.

    Returns:
        float: Interpolated implied volatility at the target maturity.
    """
    check_is_true(target_day_to_maturity >= 1, "target_day_to_maturity must be >= 1.")
    return interpolate_total_variance(
        target_time_to_maturity=target_day_to_maturity / DAYS_PER_YEAR,
        time_to_maturities=np.asarray(day_to_maturities, dtype=float) / DAYS_PER_YEAR,
        implied_volatilities=implied_volatilities,
    )