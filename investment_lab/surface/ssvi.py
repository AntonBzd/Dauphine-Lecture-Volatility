import logging
from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_lab.metrics.distance import mse
from investment_lab.surface.base import VolSmoother
from investment_lab.util import check_is_true


class SSVISmoother(VolSmoother):
    def __init__(self, initial_params: tuple[float, float, float, float]) -> None:
        """Initialize an SSVI volatility smoother.

        Args:
            initial_params (tuple[float, float, float, float]): Initial parameter tuple
            ordered as (sigma, rho, eta, lamb).
        """
        check_is_true(
            len(initial_params) == 4,
            "Initial parameters must be a tuple of (sigma, rho, eta, lamb).",
        )
        super().__init__(initial_params)

    def _fit(
        self,
        forward: float | pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray | float,
        market_implied_vols: pd.Series | np.ndarray,
        **kwargs,
    ) -> Self:
        """Calibrate SSVI parameters to market implied volatilities.

        Args:
            forward (float | pd.Series | np.ndarray): Forward price.
            strike (pd.Series | np.ndarray): Strike prices.
            time_to_maturities (pd.Series | np.ndarray | float): Time to maturities in years.
            market_implied_vols (pd.Series | np.ndarray): Observed market implied volatilities.
            **kwargs: Additional keyword arguments.

        Returns:
            Self: Fitted SSVI smoother.
        """
        time_to_maturities = np.asarray(time_to_maturities, dtype=float)
        market_implied_vols = np.asarray(market_implied_vols, dtype=float)

        market_total_variance = (market_implied_vols ** 2) * time_to_maturities

        def objective(params):
            self._params = params
            model_iv = self._transform(
                forward=forward, strike=strike, time_to_maturities=time_to_maturities,
            )
            model_total_variance = (model_iv ** 2) * time_to_maturities
            return mse(market_total_variance, model_total_variance)

        bounds = [(1e-5, None), (-0.999, 0.999), (1e-6, 5.0), (1e-6, 0.5)]
        logging.info("Fitting SSVI model on %s records", len(market_implied_vols))
        result = minimize(
            objective,
            self._params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "disp": False},
        )
        self._params = result.x
        logging.info("Successfully fitted SSVI. Parameters: %s", self._params)
        return self

    def _transform(
        self,
        forward: pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray,
        **kwargs,
    ) -> pd.Series | np.ndarray:
        """Compute SSVI implied volatilities from model parameters.

        Args:
            forward (pd.Series | np.ndarray): Forward prices.
            strike (pd.Series | np.ndarray): Strike prices.
            time_to_maturities (pd.Series | np.ndarray): Time to maturities in years.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series | np.ndarray: Model implied volatilities.
        """
        sigma, rho, eta, lamb = self._params

        k = np.log(np.asarray(strike) / np.asarray(forward))
        time_to_maturities = np.asarray(time_to_maturities, dtype=float)
        theta = sigma * sigma * time_to_maturities

        theta_safe = np.maximum(theta, 1e-12)
        phi = eta / (theta_safe ** lamb)

        inner_sqrt = np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
        total_variance = 0.5 * theta * (1 + rho * phi * k + inner_sqrt)
        total_variance = np.maximum(total_variance, 0.0)

        return np.sqrt(total_variance / np.maximum(time_to_maturities, 1e-12))
