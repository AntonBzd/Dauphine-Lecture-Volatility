from dataclasses import dataclass

import numpy as np

from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.util import check_is_true


@dataclass
class HestonParams:
    """Container for Heston model parameters.

    Attributes:
        mu: Drift of the underlying asset.
        kappa: Mean reversion speed of the variance process.
        theta: Long-run variance level.
        xi: Volatility of volatility.
        rho: Instantaneous correlation between spot and variance shocks.
    """
    mu: float
    kappa: float
    theta: float
    xi: float
    rho: float


class HestonModel:
    def __init__(
        self,
        params: HestonParams,
        dt: float = 1 / TRADING_DAYS_PER_YEAR,
        minimum_variance: float = 1e-10,
        measurement_error: float = 1e-6,
        augmented: bool = False,
        variance_proxy_error: float = 0.01,
    ) -> None:
        """Initialize a Heston stochastic volatility model for UKF filtering.

        Args:
            params (HestonParams): Heston model parameters.
            dt (float): Time step expressed in years.
            minimum_variance (float): Lower bound used to enforce positivity of the variance.
            measurement_error (float): Observation noise variance on log-spot.
            augmented (bool): Whether to use an augmented observation vector including a variance proxy.
            variance_proxy_error (float): Observation noise variance on the realized variance proxy.
        """

        check_is_true(dt > 0, "dt must be > 0.")
        check_is_true(minimum_variance > 0, "minimum_variance must be > 0.")
        check_is_true(measurement_error > 0, "measurement_error must be > 0.")
        check_is_true(params.kappa > 0, "kappa must be > 0.")
        check_is_true(params.theta > 0, "theta must be > 0.")
        check_is_true(params.xi > 0, "xi must be > 0.")
        check_is_true(-1.0 < params.rho < 1.0, "rho must be in (-1, 1).")

        self._params = params
        self._dt = dt
        self._minimum_variance = minimum_variance
        self._measurement_error = measurement_error
        self._augmented = augmented
        self._variance_proxy_error = variance_proxy_error

    @property
    def params(self) -> HestonParams:
        """Return the current Heston model parameters.

        Returns:
            HestonParams: Model parameters.
        """
        return self._params

    @property
    def dt(self) -> float:
        """Return the discretization time step.

        Returns:
            float: Time step in years.
        """
        return self._dt

    @property
    def observation_dim(self) -> int:
        """Return the dimension of the observation vector.

        Returns:
            int: Observation dimension.
        """
        return 2 if self._augmented else 1

    def transition_function(self, state: np.ndarray) -> np.ndarray:
        """Propagate the latent state through the discretized Heston dynamics.

        Args:
            state (np.ndarray): Current state vector containing log-spot and variance.

        Returns:
            np.ndarray: Next-period predicted state vector.
        """

        log_spot, variance = state
        variance = max(float(variance), self._minimum_variance)

        next_log_spot = log_spot + (self._params.mu - 0.5 * variance) * self._dt
        next_variance = variance + self._params.kappa * (self._params.theta - variance) * self._dt
        next_variance = max(next_variance, self._minimum_variance)
        return np.asarray([next_log_spot, next_variance], dtype=float)

    def process_noise_covariance(self, state: np.ndarray) -> np.ndarray:
        """Compute the state-dependent process noise covariance matrix.

        Args:
            state (np.ndarray): Current state vector containing log-spot and variance.

        Returns:
            np.ndarray: Process noise covariance matrix.
        """

        _, variance = state
        variance = max(float(variance), self._minimum_variance)
        scaled_variance = variance * self._dt

        return np.asarray(
            [
                [scaled_variance, self._params.rho * self._params.xi * scaled_variance],
                [self._params.rho * self._params.xi * scaled_variance, (self._params.xi ** 2) * scaled_variance],
            ],
            dtype=float,
        )

    def observation_function(self, state: np.ndarray) -> np.ndarray:
        """Map the latent state to the observation space.

        Args:
            state (np.ndarray): Current state vector containing log-spot and variance.

        Returns:
            np.ndarray: Observation vector implied by the current state.
        """

        log_spot, variance = state
        if self._augmented:
            return np.asarray([log_spot, max(variance, self._minimum_variance)], dtype=float)
        else:
            return np.asarray([log_spot], dtype=float)


    def measurement_noise_covariance(self, state: np.ndarray) -> np.ndarray:
        """Return the measurement noise covariance matrix.

        Args:
            state (np.ndarray): Current state vector containing log-spot and variance.

        Returns:
            np.ndarray: Measurement noise covariance matrix.
        """

        if self._augmented:
            return np.asarray(
                [
                    [self._measurement_error, 0.0],
                    [0.0, self._variance_proxy_error],
                ],
                dtype=float,
            )
        else:
            return np.asarray([[self._measurement_error]], dtype=float)

    def forecast_average_variance(
        self,
        current_variance: float,
        horizon_in_years: float,
    ) -> float:
        """Forecast the average future variance over a fixed horizon.

        Args:
            current_variance (float): Current filtered instantaneous variance.
            horizon_in_years (float): Forecast horizon expressed in years.

        Returns:
            float: Forecast average variance over the target horizon.
        """

        check_is_true(horizon_in_years > 0, "horizon_in_years must be > 0.")
        current_variance = max(float(current_variance), self._minimum_variance)

        kappa = self._params.kappa
        theta = self._params.theta

        if abs(kappa) < 1e-12:
            return current_variance

        average_variance = theta + (current_variance - theta) * (1 - np.exp(-kappa * horizon_in_years)) / (kappa * horizon_in_years)
        return max(float(average_variance), self._minimum_variance)

    def forecast_volatility(
        self,
        current_variance: float,
        horizon_in_years: float,
    ) -> float:
        """Forecast the average future volatility over a fixed horizon.

        Args:
            current_variance (float): Current filtered instantaneous variance.
            horizon_in_years (float): Forecast horizon expressed in years.

        Returns:
            float: Forecast average volatility over the target horizon.
        """
        
        return float(np.sqrt(self.forecast_average_variance(current_variance, horizon_in_years)))