import logging
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.stochastic.heston import HestonModel, HestonParams
from investment_lab.stochastic.ukf import UnscentedKalmanFilter
from investment_lab.util import check_is_true


class HestonStateSpaceModel:

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
        """Initialize a Heston state space model estimated with an augmented UKF.

        Args:
            rolling_window (int): Number of observations used in each rolling calibration window.
            forecast_horizon_days (int): Number of days used for the volatility forecast horizon.
            dt (float): Time step expressed in years.
            alpha (float): Spread parameter of the sigma points in the UKF.
            beta (float): Parameter incorporating prior distribution information in the UKF.
            kappa_ukf (float): Secondary scaling parameter for sigma point generation.
            minimum_variance (float): Lower bound used to enforce positivity of the variance.
            measurement_error (float): Observation noise variance on log-spot.
            variance_proxy_error (float): Observation noise variance on the realized variance proxy.
            rv_proxy_window (int): Window length used to compute the realized variance proxy.
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

        self._ukf = UnscentedKalmanFilter(
            state_dim=2,
            observation_dim=2,  # Augmented: log_spot + rv_proxy
            alpha=alpha,
            beta=beta,
            kappa=kappa_ukf,
        )

    @property
    def rolling_window(self) -> int:
        """Return the rolling calibration window length.

        Returns:
            int: Number of observations used in each rolling calibration.
        """
        return self._rolling_window

    @property
    def forecast_horizon_days(self) -> int:
        """Return the forecast horizon in days.

        Returns:
            int: Number of days for the volatility forecast horizon.
        """
        return self._forecast_horizon_days

    def _build_model(self, params: HestonParams) -> HestonModel:
        """Instantiate a Heston model from a parameter set.

        Args:
            params (HestonParams): Heston model parameters.

        Returns:
            HestonModel: Instantiated Heston model.
        """
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
        """Convert a Heston parameter object into a numeric array.

        Args:
            params (HestonParams): Heston model parameters.

        Returns:
            np.ndarray: Parameter array ordered as (mu, kappa, theta, xi, rho).
        """
        return np.asarray([params.mu, params.kappa, params.theta, params.xi, params.rho], dtype=float)

    @staticmethod
    def _from_parameter_array(params: np.ndarray) -> HestonParams:
        """Convert a numeric array into a Heston parameter object.

        Args:
            params (np.ndarray): Parameter array ordered as (mu, kappa, theta, xi, rho).

        Returns:
            HestonParams: Heston model parameters.
        """
        return HestonParams(
            mu=float(params[0]),
            kappa=float(params[1]),
            theta=float(params[2]),
            xi=float(params[3]),
            rho=float(params[4]),
        )

    # def _compute_rv_proxy(self, log_spot: pd.Series) -> np.ndarray:
    #     """Compute a rolling realized variance proxy from a log-spot series.

    #     Args:
    #         log_spot (pd.Series): Time series of log-spot prices.

    #     Returns:
    #         np.ndarray: Annualized realized variance proxy.
    #     """
    #     returns = log_spot.diff()
    #     rolling_var = returns.rolling(self._rv_proxy_window).var() * TRADING_DAYS_PER_YEAR

    #     global_var = float(returns.dropna().var() * TRADING_DAYS_PER_YEAR)
    #     rv_proxy = rolling_var.fillna(global_var).values
    #     rv_proxy = np.clip(rv_proxy, self._minimum_variance, 5.0)

    #     return rv_proxy

    def _compute_rv_proxy(self, log_spot: pd.Series) -> np.ndarray:
        """Compute a rolling realized variance proxy from a log-spot series.

        Args:
            log_spot (pd.Series): Time series of log-spot prices.

        Returns:
            np.ndarray: Annualized realized variance proxy.
        """
        returns = log_spot.diff()
        qv = returns.pow(2).rolling(self._rv_proxy_window).sum() * (TRADING_DAYS_PER_YEAR / self._rv_proxy_window)
        
        global_var = float(returns.dropna().pow(2).mean() * TRADING_DAYS_PER_YEAR)
        rv_proxy = qv.fillna(global_var).values
        rv_proxy = np.clip(rv_proxy, self._minimum_variance, 5.0)
        
        return rv_proxy


    def _estimate_diffusion_params(self, log_spot: pd.Series) -> dict:
        """Estimate Heston parameters from return moments.

        Args:
            log_spot (pd.Series): Time series of log-spot prices.

        Returns:
            HestonParams: Heston parameter estimates obtained from return statistics.
        """
        returns = log_spot.diff().dropna()

        rolling_var = returns.rolling(10).var() * TRADING_DAYS_PER_YEAR
        variance_proxy = rolling_var.dropna()
        var_series = variance_proxy.values

        mean_var = max(float(np.mean(var_series)), 1e-6)
        xi = float(np.std(var_series)) / float(np.sqrt(mean_var))
        xi = np.clip(xi, 0.1, 3.0)

        delta_var = np.diff(var_series)
        returns_aligned = returns.iloc[10:].values
        min_len = min(len(returns_aligned), len(delta_var))
        if min_len > 10:
            rho = float(np.corrcoef(returns_aligned[:min_len], delta_var[:min_len])[0, 1])
            rho = np.clip(rho, -0.95, -0.01)
        else:
            rho = -0.70

        theta_approx = float(np.mean(var_series)) # median
        mu = float(returns.mean() * TRADING_DAYS_PER_YEAR + 0.5 * theta_approx)
        mu = np.clip(mu, -0.50, 0.50)

        theta_guess = float(np.median(var_series))
        theta_guess = np.clip(theta_guess, 0.005, 2.0)

        var_centered = var_series - var_series.mean()
        lag = 5
        if len(var_centered) > lag + 10:
            autocorr = np.corrcoef(var_centered[:-lag], var_centered[lag:])[0, 1]
            autocorr = np.clip(autocorr, 0.01, 0.99)
            kappa_guess = -np.log(autocorr) / (lag / TRADING_DAYS_PER_YEAR)
            kappa_guess = np.clip(kappa_guess, 0.1, 5.0)
        else:
            kappa_guess = 2.0

        logging.info(
            "Diffusion params (moments): xi=%.3f, rho=%.3f, mu=%.3f | "
            "Initial kappa=%.2f, theta=%.4f (vol=%.1f%%)",
            xi, rho, mu, kappa_guess, theta_guess, np.sqrt(theta_guess) * 100,
        )

        return {
            "xi": xi,
            "rho": rho,
            "mu": mu,
            "kappa_guess": kappa_guess,
            "theta_guess": theta_guess,
        }


    def _build_initial_state(
        self,
        log_spot: pd.Series,
        initial_variance: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the initial latent state and covariance used by the UKF.

        Args:
            log_spot (pd.Series): Time series of log-spot prices.
            initial_variance (Optional[float]): Optional initial variance guess.

        Returns:
            tuple[np.ndarray, np.ndarray]: Initial state mean and initial state covariance.
        """

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
        """Run the augmented UKF on an underlying price series for a fixed Heston parameter set.

        Args:
            df_underlying (pd.DataFrame): Underlying price series with at least date and spot columns.
            params (HestonParams): Heston model parameters.
            initial_variance (Optional[float]): Optional initial variance guess.

        Returns:
            pd.DataFrame: Filtering results including filtered variance, predicted variance,
            realized variance proxy, forecast average variance, and forecast volatility.
        """

        required_cols = {"date", "spot"}
        missing_cols = required_cols.difference(df_underlying.columns)
        check_is_true(len(missing_cols) == 0, f"Missing columns: {missing_cols}")

        df = df_underlying[["date", "spot"]].copy().sort_values("date").reset_index(drop=True)
        df["log_spot"] = np.log(df["spot"])

        rv_proxy = self._compute_rv_proxy(df["log_spot"])

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
        
        # df_out["forecast_average_variance"] = df_out["filtered_variance"].apply(
        #     lambda x: model.forecast_average_variance(current_variance=x, horizon_in_years=horizon_in_years)
        # )
        # df_out["forecast_volatility"] = np.sqrt(df_out["forecast_average_variance"])

        df_out["filtered_volatility"] = np.sqrt(df_out["filtered_variance"])   # ← c'est σ̂_t de la consigne

        # On garde le forecast forward comme info additionnelle, sous un nom distinct
        df_out["forecast_average_variance"] = df_out["filtered_variance"].apply(
            lambda x: model.forecast_average_variance(current_variance=x, horizon_in_years=horizon_in_years)
        )
        df_out["forecast_volatility_forward"] = np.sqrt(df_out["forecast_average_variance"])

        # forecast_volatility pointe maintenant sur l'instantané (conforme consigne)
        df_out["forecast_volatility"] = df_out["filtered_volatility"]

        ###

        for key, value in asdict(params).items():
            df_out[key] = value

        return df_out


    def _negative_log_likelihood_kappa_theta(
        self,
        kappa_theta: np.ndarray,
        df_underlying: pd.DataFrame,
        fixed_mu: float,
        fixed_xi: float,
        fixed_rho: float,
    ) -> float:
        """NLL as a function of (kappa, theta) only.

        xi and rho are fixed from moments (diffusion params).
        This 2D optimization is well-conditioned because the augmented
        UKF makes variance identifiable.
        """
        try:
            kappa, theta = kappa_theta
            if not (np.isfinite(kappa) and np.isfinite(theta)):
                return 1e12

            params = HestonParams(
                mu=fixed_mu,
                kappa=kappa,
                theta=theta,
                xi=fixed_xi,
                rho=fixed_rho,
            )
            df_filter = self.fit_filter(df_underlying=df_underlying, params=params)

            nll = -float(df_filter["loglikelihood"].sum())

            feller_margin = fixed_xi ** 2 - 2 * kappa * theta
            if feller_margin > 0:
                nll += 1000.0 * feller_margin

            if not np.isfinite(nll):
                return 1e12
            return min(nll, 1e12)

        except Exception as exc:
            logging.debug("NLL failed for kappa=%.3f theta=%.4f: %s", kappa_theta[0], kappa_theta[1], exc)
            return 1e12


    def calibrate(
        self,
        df_underlying: pd.DataFrame,
        initial_params: Optional[HestonParams] = None,
        **kwargs,
    ) -> HestonParams:
        """Calibrate Heston parameters on a given time window using return moments.

        Args:
            df_underlying (pd.DataFrame): Underlying price series with at least date and spot columns.
            initial_params (Optional[HestonParams]): Optional initial parameter guess.
            **kwargs: Additional keyword arguments.

        Returns:
            HestonParams: Calibrated Heston parameters.
        """

        check_is_true(len(df_underlying) >= 20, "Need at least 20 observations.")
        df = df_underlying[["date", "spot"]].copy().sort_values("date").reset_index(drop=True)
        df["log_spot"] = np.log(df["spot"])

        diffusion = self._estimate_diffusion_params(df["log_spot"])
        fixed_xi = diffusion["xi"]
        fixed_rho = diffusion["rho"]
        fixed_mu = diffusion["mu"]
        kappa_guess = diffusion["kappa_guess"]
        theta_guess = diffusion["theta_guess"]

        bounds_kappa_theta = [
            (0.1, 5.0),     # kappa
            (0.005, 2.0),   # theta
        ]

        starting_points = [
            [kappa_guess, theta_guess],
            [kappa_guess * 0.5, theta_guess * 1.2],
            [kappa_guess * 2.0, theta_guess * 0.8],
        ]

        best_result = None
        best_nll = 1e12

        for x0 in starting_points:
            x0_clipped = [
                np.clip(x0[0], bounds_kappa_theta[0][0], bounds_kappa_theta[0][1]),
                np.clip(x0[1], bounds_kappa_theta[1][0], bounds_kappa_theta[1][1]),
            ]

            result = minimize(
                fun=lambda x: self._negative_log_likelihood_kappa_theta(
                    x, df, fixed_mu, fixed_xi, fixed_rho,
                ),
                x0=x0_clipped,
                method="L-BFGS-B",
                bounds=bounds_kappa_theta,
                options={"maxiter": 100, "disp": False},
            )

            if result.fun < best_nll and np.all(np.isfinite(result.x)):
                best_nll = result.fun
                best_result = result

        if best_result is not None and best_nll < 1e12:
            kappa_opt, theta_opt = best_result.x
            calibrated = HestonParams(
                mu=fixed_mu,
                kappa=kappa_opt,
                theta=theta_opt,
                xi=fixed_xi,
                rho=fixed_rho,
            )
            logging.info(
                "Hybrid calibrated: kappa=%.3f, theta=%.4f (vol=%.1f%%) | "
                "fixed xi=%.3f, rho=%.3f | NLL=%.2f",
                kappa_opt, theta_opt, np.sqrt(theta_opt) * 100,
                fixed_xi, fixed_rho, best_nll,
            )
            return calibrated


        logging.warning("MLE for kappa/theta failed, using full moment estimates.")
        return HestonParams(
            mu=fixed_mu,
            kappa=kappa_guess,
            theta=theta_guess,
            xi=fixed_xi,
            rho=fixed_rho,
        )

    @staticmethod
    def _is_valid_params(params: HestonParams) -> bool:
        """Check whether a Heston parameter set satisfies basic admissibility conditions.

        Args:
            params (HestonParams): Heston model parameters.

        Returns:
            bool: Whether the parameter set is considered valid.
        """
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
        """Run rolling calibration and augmented UKF filtering on an underlying price series.

        Args:
            df_underlying (pd.DataFrame): Underlying price series with at least date and spot columns.
            initial_params (Optional[HestonParams]): Optional initial parameter guess.
            recalibration_frequency (int): Number of days between two recalibrations.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: Rolling filtering and forecasting results evaluated at the end of each window.
        """
        required_cols = {"date", "spot"}
        missing_cols = required_cols.difference(df_underlying.columns)
        check_is_true(len(missing_cols) == 0, f"Missing columns: {missing_cols}")

        df = df_underlying[["date", "spot"]].copy().sort_values("date").reset_index(drop=True)
        check_is_true(len(df) > self._rolling_window, "Series too short for rolling window.")

        all_end_indices = list(range(self._rolling_window - 1, len(df)))


        calibrated_params_map = {}
        last_calibrated = None
        for i, end_idx in enumerate(all_end_indices):
            if i % recalibration_frequency == 0:
                df_window = df.iloc[end_idx - self._rolling_window + 1: end_idx + 1].copy()
                last_calibrated = self.calibrate(df_underlying=df_window)
            calibrated_params_map[end_idx] = last_calibrated


        results = []
        for end_idx in all_end_indices:
            df_window = df.iloc[end_idx - self._rolling_window + 1: end_idx + 1].copy()
            params = calibrated_params_map[end_idx]
            df_filter = self.fit_filter(df_underlying=df_window, params=params)
            last_row = df_filter.iloc[-1]

            results.append({
                "date": last_row["date"],
                "spot": last_row["spot"],
                #"filtered_variance": last_row["filtered_variance"],
                # "predicted_variance": last_row["predicted_variance"],
                # "forecast_average_variance": last_row["forecast_average_variance"],
                ###
                "filtered_variance": last_row["filtered_variance"],
                "filtered_volatility": last_row["filtered_volatility"],         
                "forecast_volatility": last_row["forecast_volatility"],         
                "forecast_volatility_forward": last_row["forecast_volatility_forward"],
                "forecast_average_variance": last_row["forecast_average_variance"],
                ###
                # "forecast_volatility": last_row["forecast_volatility"],
                "loglikelihood": last_row["loglikelihood"],
                "rv_proxy": last_row["rv_proxy"],
                "mu": params.mu,
                "kappa": params.kappa,
                "theta": params.theta,
                "xi": params.xi,
                "rho": params.rho,
            })

        return pd.DataFrame(results)