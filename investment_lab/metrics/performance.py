import pandas as pd

from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.metrics.util import returns_to_levels
from investment_lab.metrics.volatility import realized_volatility


# Summary

def evaluate_backtest(df_nav: pd.DataFrame, strategy_name: str, df_alloc: pd.DataFrame | None = None):

    """Compute a set of performance metrics for a backtested strategy.

    Args:
        df_nav : DataFrame containing the NAV time series indexed by date.
        strategy_name : Name of the evaluated strategy.
        df_alloc : Allocation time series with an "allocation" column.

    Returns:
        pd.Series: Summary of performance metrics including returns, volatility, drawdowns and ratios.
    """

    df_nav = df_nav.copy().sort_index()
    returns = df_nav["NAV"].pct_change().dropna()

    metrics = {
        "strategy": strategy_name,
        "annual_return": realized_returns(returns),
        "annual_vol": realized_volatility(returns),
        "max_drawdown": max_drawdown(returns),
        "sharpe": sharpe_ratio(returns),
        "calmar": calmar_ratio(returns),
        "hit_ratio": hit_ratio(returns),
    }

    if df_alloc is not None and "allocation" in df_alloc.columns:
        metrics["avg_turnover"] = allocation_turnover(df_alloc["allocation"])

    return pd.Series(metrics)

def allocation_turnover(allocation: pd.Series) -> float:
    """Compute the average daily turnover of an allocation time series.

    Args:
        allocation (pd.Series): Time series of portfolio allocation weights.

    Returns:
        float: Mean absolute daily change in allocation.
    """
    """Average daily turnover of the allocation (absolute mean variation))."""
    changes = allocation.diff().abs().dropna()
    if len(changes) == 0:
        return 0.0
    return float(changes.mean())


# Ratio

def sharpe_ratio(returns: pd.Series, risk_free_rate: float | pd.Series = 0.0) -> float:
    """Compute the annualized Sharpe ratio of a daily returns series.

    Args:
        returns (pd.Series): Series of daily returns.
        risk_free_rate (float | pd.Series): Annualized scalar risk-free rate or daily series.

    Returns:
        float: Annualized Sharpe ratio.
    """
    return (
        realized_returns(excess_return(returns, risk_free_rate))
    ) / realized_volatility(returns)


def calmar_ratio(returns: pd.Series) -> float:
    """Compute the Calmar ratio of a strategy.

    Args:
        returns (pd.Series): Series of daily returns.

    Returns:
        float: Ratio of annualized return to maximum drawdown.
    """
    annualized_return = realized_returns(returns)
    maximum_drawdown = -max_drawdown(returns)
    if maximum_drawdown == 0:
        return float("inf")
    return annualized_return / maximum_drawdown


def hit_ratio(returns: pd.Series) -> float:
    """Compute the proportion of positive return observations.

    Args:
        returns (pd.Series): Series of daily returns.

    Returns:
        float: Fraction of days with positive returns.
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return 0.0
    return float((returns_clean > 0).sum() / len(returns_clean))


# Return

def realized_returns(returns: pd.Series) -> float:
    """Compute annualized realized return

    Args:
        returns (pd.Series): _description_

    Returns:
        float: Annualized realized returns.
    """
    return returns.mean() * TRADING_DAYS_PER_YEAR

def excess_return(
    returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.0,
    annualized_risk_free_rate: bool = True,
) -> pd.Series:
    """Compute excess returns over a risk-free rate.

    Args:
        returns (pd.Series): Series of daily returns.
        risk_free_rate (float | pd.Series): Risk-free rate, either annualized scalar or daily series.
        annualized_risk_free_rate (bool): Whether the risk-free rate is annualized.

    Returns:
        pd.Series: Series of excess returns.
    """
    if annualized_risk_free_rate:
        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    else:
        daily_rf = risk_free_rate
    return returns - daily_rf

# Drawdown

def drawdown(returns: pd.Series) -> pd.Series:
    """Compute the drawdown time series from daily returns.

    Args:
        returns (pd.Series): Series of daily returns.

    Returns:
        pd.Series: Drawdown series expressed as percentage decline from peak.
    """
    nav = returns_to_levels(returns)
    return (nav / nav.cummax()) - 1


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown of a return series.

    Args:
        returns (pd.Series): Series of daily returns.

    Returns:
        float: Maximum observed drawdown.
    """
    return drawdown(returns).min()




