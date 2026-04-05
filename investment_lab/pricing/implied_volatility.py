import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from investment_lab.pricing.black_scholes import black_scholes_price, vega_black_scholes


def implied_volatility_vectorized(
    market_price: pd.Series,
    S: pd.Series,
    K: pd.Series,
    T: pd.Series,
    r: pd.Series,
    option_type: pd.Series,
    initial_guess: float = 0.2,
    tol: float = 1e-7,
    max_iterations: int = 10000,
) -> pd.Series:
    """Compute implied volatility using a vectorized Newton-Raphson method.

    Args:
        market_price (pd.Series): Observed market prices of the options.
        S (pd.Series): Underlying asset prices.
        K (pd.Series): Strike prices.
        T (pd.Series): Time to maturity in years.
        r (pd.Series): Risk-free interest rates.
        option_type (pd.Series): Option types ("C" or "P").
        initial_guess (float): Initial guess for volatility.
        tol (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        pd.Series: Implied volatility estimates.
    """

    sigma = pd.Series(initial_guess, index=market_price.index, dtype=float)
    logging.info("Calculate implied volatility using Newton-Raphson method")
    logging.info(
        "Parameters: initial_guess=%s, tol=%s, max_iteration=%s",
        initial_guess,
        tol,
        max_iterations,
    )
    for i in tqdm(
        range(max_iterations), desc="Calculating Implied Volatility", leave=True
    ):
  
        current_price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = vega_black_scholes(S, K, T, r, sigma)

        price_diff = market_price - current_price

        sigma += (price_diff / vega.replace(0, np.nan)).fillna(0)

        sigma = sigma.clip(lower=1e-5, upper=5.0)

        if price_diff.abs().max() < tol:
            logging.info("Converged after %s iterations", i + 1)
            break

    return sigma
