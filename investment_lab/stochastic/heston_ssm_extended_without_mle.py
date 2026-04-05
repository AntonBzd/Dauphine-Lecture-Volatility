import logging
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd

from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.stochastic.heston import HestonModel, HestonParams
from investment_lab.stochastic.ukf import UnscentedKalmanFilter
from investment_lab.util import check_is_true


class HestonStateSpaceModel:
    """Heston State Space Model with augmented UKF.

    The UKF observes both log_spot AND a realized variance proxy,
    giving it two "eyes" instead of one. This makes the latent
    variance much more identifiable.

    Pipeline:
        1. Calibrate parameters via method of moments (fast, robust)
        2. Filter latent variance via augmented UKF
        3. Forecast forward volatility via Heston analytical formula
    """

    _PARAMETER_ORDER = ["mu", "kappa", "theta", "xi", "rho"]

    def __init__(
        self,
        rolling_window: int,
        forecast_horizon_days: int = 30,
        dt: float = 1 / TRADING_DAYS_PER_YEAR,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa_ukf: float = 0.0,
        minimum_variance: float = 1e-10,
        measurement_error: float = 1e-6,
        variance_proxy_error: float = 0.01,
        rv_proxy_window: int = 5,
    ) -> None:
        """
        Args:
            rolling_window: number of days for each calibration window
            forecast_horizon_days: forward horizon for vol forecast (e.g. 30)
            measurement_error: noise on log_spot observation (small)
            variance_proxy_error: noise on RV proxy observation (larger)
                - 0.005 = trust proxy a lot  (reactive forecast)
                - 0.01  = balanced
                - 0.02  = trust proxy less   (smoother forecast)
            rv_proxy_window: days for rolling variance proxy (e.g. 5)
        """
        check_is_true(rolling_window >= 20, "rolling_window must be >= 20.")
        check_is_true(forecast_horizon_days >= 1, "forecast_horizon_days must be >= 1.")

        self._rolling_window = rolling_window
        self._forecast_horizon_days = forecast_horizon_days
        self._dt = dt
        self._minimum_variance = minimum_variance
        self._measurement_error = measurement_error
        self._variance_proxy_error = variance_proxy_error
        self._rv_proxy_window = rv_proxy_window

        # UKF with observation_dim=2 (log_spot + variance proxy)
        self._ukf = UnscentedKalmanFilter(
            state_dim=2,
            observation_dim=2,
            alpha=alpha,
            beta=beta,
            kappa=kappa_ukf,
        )

    @property
    def rolling_window(self) -> int:
        return self._rolling_window

    @property
    def forecast_horizon_days(self) -> int:
        return self._forecast_horizon_days

    def _build_model(self, params: HestonParams) -> HestonModel:
        return HestonModel(
            params=params,
            dt=self._dt,
            minimum_variance=self._minimum_variance,
            measurement_error=self._measurement_error,
            augmented=True,
            variance_proxy_error=self._variance_proxy_error,
        )

    @staticmethod
    def _to_parameter_array(params: HestonParams) -> np.ndarray:
        return np.asarray([params.mu, params.kappa, params.theta, params.xi, params.rho], dtype=float)

    @staticmethod
    def _from_parameter_array(params: np.ndarray) -> HestonParams:
        return HestonParams(
            mu=float(params[0]),
            kappa=float(params[1]),
            theta=float(params[2]),
            xi=float(params[3]),
            rho=float(params[4]),
        )

    def _compute_rv_proxy(self, log_spot: pd.Series) -> np.ndarray:
        """Compute a daily realized variance proxy.

        Uses rolling variance over rv_proxy_window days, annualized.
        First days without enough history use the global window variance.
        """
        returns = log_spot.diff()
        rolling_var = returns.rolling(self._rv_proxy_window).var() * TRADING_DAYS_PER_YEAR

        # Fill initial NaNs with global variance
        global_var = float(returns.dropna().var() * TRADING_DAYS_PER_YEAR)
        rv_proxy = rolling_var.fillna(global_var).values

        # Clip extremes
        rv_proxy = np.clip(rv_proxy, self._minimum_variance, 5.0)

        return rv_proxy

    def _estimate_params_from_returns(self, log_spot: pd.Series) -> HestonParams:
        """Estimate Heston parameters from return statistics.

        Uses rolling variance as proxy. Fast and robust — no optimization.
        """
        returns = log_spot.diff().dropna()

        # Variance proxy: rolling 10-day variance, annualized
        rolling_var = returns.rolling(10).var() * TRADING_DAYS_PER_YEAR
        variance_proxy = rolling_var.dropna()
        var_series = variance_proxy.values

        # theta: median (robust to COVID-type outliers)
        theta = float(np.median(var_series))
        theta = np.clip(theta, 0.005, 2.0)

        # kappa: from autocorrelation at lag 5 (weekly)
        var_centered = var_series - var_series.mean()
        lag = 5
        if len(var_centered) > lag + 10:
            autocorr = np.corrcoef(var_centered[:-lag], var_centered[lag:])[0, 1]
            autocorr = np.clip(autocorr, 0.01, 0.99)
            kappa = -np.log(autocorr) / (lag / TRADING_DAYS_PER_YEAR)
            kappa = np.clip(kappa, 0.1, 5.0)
        else:
            kappa = 2.0

        # xi: vol-of-vol
        mean_var = max(float(np.mean(var_series)), 1e-6)
        xi = float(np.std(var_series)) / float(np.sqrt(mean_var))
        xi = np.clip(xi, 0.1, 3.0)

        # rho: correlation between returns and variance changes
        delta_var = np.diff(var_series)
        returns_aligned = returns.iloc[10:].values
        min_len = min(len(returns_aligned), len(delta_var))
        if min_len > 10:
            rho = float(np.corrcoef(returns_aligned[:min_len], delta_var[:min_len])[0, 1])
            rho = np.clip(rho, -0.95, -0.01)
        else:
            rho = -0.70

        # mu
        mu = float(returns.mean() * TRADING_DAYS_PER_YEAR + 0.5 * theta)
        mu = np.clip(mu, -0.50, 0.50)

        logging.info(
            "Moments: mu=%.3f, kappa=%.2f, theta=%.4f (vol=%.1f%%), xi=%.3f, rho=%.3f",
            mu, kappa, theta, np.sqrt(theta) * 100, xi, rho,
        )
        return HestonParams(mu=mu, kappa=kappa, theta=theta, xi=xi, rho=rho)

    def _build_initial_state(
        self,
        log_spot: pd.Series,
        initial_variance: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        variance_guess = initial_variance or float(log_spot.diff().dropna().var() * TRADING_DAYS_PER_YEAR)
        variance_guess = float(np.clip(variance_guess, 1e-4, 2.0))

        initial_state_mean = np.asarray(
            [float(log_spot.iloc[0]), variance_guess],
            dtype=float,
        )
        initial_state_covariance = np.asarray(
            [[1e-4, 0.0], [0.0, 0.1]],
            dtype=float,
        )
        return initial_state_mean, initial_state_covariance

    def fit_filter(
        self,
        df_underlying: pd.DataFrame,
        params: HestonParams,
        initial_variance: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run augmented UKF filter on a window of spot data.

        Observations = [log_spot, rv_proxy] at each timestep.
        State        = [log_spot, variance]
        """
        required_cols = {"date", "spot"}
        missing_cols = required_cols.difference(df_underlying.columns)
        check_is_true(len(missing_cols) == 0, f"Missing columns: {missing_cols}")

        df = df_underlying[["date", "spot"]].copy().sort_values("date").reset_index(drop=True)
        df["log_spot"] = np.log(df["spot"])

        # Compute realized variance proxy
        rv_proxy = self._compute_rv_proxy(df["log_spot"])

        # Augmented observations: [log_spot, rv_proxy] per day
        observations = np.column_stack([
            df["log_spot"].values,
            rv_proxy,
        ])

        initial_state_mean, initial_state_covariance = self._build_initial_state(
            df["log_spot"],
            initial_variance=initial_variance,
        )

        model = self._build_model(params)
        df_filter = self._ukf.filter(
            observations=observations,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_function=model.transition_function,
            observation_function=model.observation_function,
            process_noise_cov_function=model.process_noise_covariance,
            measurement_noise_cov_function=model.measurement_noise_covariance,
        )

        df_out = pd.concat([df[["date", "spot", "log_spot"]], df_filter.drop(columns=["t"])], axis=1)
        df_out["filtered_variance"] = df_out["filtered_state_1"].clip(lower=self._minimum_variance)
        df_out["predicted_variance"] = df_out["predicted_state_1"].clip(lower=self._minimum_variance)
        df_out["rv_proxy"] = rv_proxy

        horizon_in_years = self._forecast_horizon_days / TRADING_DAYS_PER_YEAR
        df_out["forecast_average_variance"] = df_out["filtered_variance"].apply(
            lambda x: model.forecast_average_variance(current_variance=x, horizon_in_years=horizon_in_years)
        )
        df_out["forecast_volatility"] = np.sqrt(df_out["forecast_average_variance"])

        for key, value in asdict(params).items():
            df_out[key] = value

        return df_out

    def calibrate(
        self,
        df_underlying: pd.DataFrame,
        initial_params: Optional[HestonParams] = None,
        **kwargs,
    ) -> HestonParams:
        check_is_true(len(df_underlying) >= 20, "Need at least 20 observations.")
        df = df_underlying[["date", "spot"]].copy().sort_values("date").reset_index(drop=True)
        df["log_spot"] = np.log(df["spot"])
        return self._estimate_params_from_returns(df["log_spot"])

    @staticmethod
    def _is_valid_params(params: HestonParams) -> bool:
        return (
            params.kappa > 0.1
            and params.theta > 0.005
            and params.xi > 0.05
            and -0.95 < params.rho < -0.01
        )

    def rolling_fit_predict(
        self,
        df_underlying: pd.DataFrame,
        initial_params: Optional[HestonParams] = None,
        recalibration_frequency: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Rolling calibration + augmented UKF filtering.

        For each day:
            1. Calibrate Heston params via moments on the rolling window
            2. Run augmented UKF to filter latent variance
            3. Forecast 30-day forward volatility
        """
        required_cols = {"date", "spot"}
        missing_cols = required_cols.difference(df_underlying.columns)
        check_is_true(len(missing_cols) == 0, f"Missing columns: {missing_cols}")

        df = df_underlying[["date", "spot"]].copy().sort_values("date").reset_index(drop=True)
        check_is_true(len(df) > self._rolling_window, "Series too short for rolling window.")

        all_end_indices = list(range(self._rolling_window - 1, len(df)))

        # Phase 1: calibration by moments (instant)
        calibrated_params_map = {}
        last_calibrated = None
        for i, end_idx in enumerate(all_end_indices):
            if i % recalibration_frequency == 0:
                df_window = df.iloc[end_idx - self._rolling_window + 1: end_idx + 1].copy()
                last_calibrated = self.calibrate(df_underlying=df_window)
            calibrated_params_map[end_idx] = last_calibrated

        # Phase 2: augmented UKF filtering
        results = []
        for end_idx in all_end_indices:
            df_window = df.iloc[end_idx - self._rolling_window + 1: end_idx + 1].copy()
            params = calibrated_params_map[end_idx]
            df_filter = self.fit_filter(df_underlying=df_window, params=params)
            last_row = df_filter.iloc[-1]

            results.append({
                "date": last_row["date"],
                "spot": last_row["spot"],
                "filtered_variance": last_row["filtered_variance"],
                "predicted_variance": last_row["predicted_variance"],
                "forecast_average_variance": last_row["forecast_average_variance"],
                "forecast_volatility": last_row["forecast_volatility"],
                "loglikelihood": last_row["loglikelihood"],
                "rv_proxy": last_row["rv_proxy"],
                "mu": params.mu,
                "kappa": params.kappa,
                "theta": params.theta,
                "xi": params.xi,
                "rho": params.rho,
            })

        return pd.DataFrame(results)