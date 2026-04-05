import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from investment_lab.util import check_is_true


class UnscentedKalmanFilter:
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        """Initialize an Unscented Kalman Filter for a non-linear state space model.

        Args:
            state_dim (int): Dimension of the latent state vector.
            observation_dim (int): Dimension of the observation vector.
            alpha (float): Spread parameter used to generate sigma points.
            beta (float): Parameter incorporating prior distribution information.
            kappa (float): Secondary scaling parameter for sigma point generation.
        """
        check_is_true(state_dim >= 1, "state_dim must be >= 1.")
        check_is_true(observation_dim >= 1, "observation_dim must be >= 1.")
        check_is_true(alpha > 0, "alpha must be > 0.")

        self._state_dim = state_dim
        self._observation_dim = observation_dim
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

        self._lambda = alpha**2 * (state_dim + kappa) - state_dim

        self._weights_mean = np.full(2 * state_dim + 1, 1 / (2 * (state_dim + self._lambda)))
        self._weights_cov = np.full(2 * state_dim + 1, 1 / (2 * (state_dim + self._lambda)))
        self._weights_mean[0] = self._lambda / (state_dim + self._lambda)
        self._weights_cov[0] = self._lambda / (state_dim + self._lambda) + (1 - alpha**2 + beta)

    @property
    def state_dim(self) -> int:
        """Return the dimension of the latent state vector.

        Returns:
            int: State dimension.
        """
        return self._state_dim

    @property
    def observation_dim(self) -> int:
        """Return the dimension of the observation vector.

        Returns:
            int: Observation dimension.
        """
        return self._observation_dim

    def _compute_sigma_points(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """Compute sigma points associated with a Gaussian state distribution.

        Args:
            mean (np.ndarray): Mean vector of the latent state distribution.
            covariance (np.ndarray): Covariance matrix of the latent state distribution.

        Returns:
            np.ndarray: Sigma points of shape (2 * state_dim + 1, state_dim).
        """
        mean = np.asarray(mean, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)

        check_is_true(
            mean.shape[0] == self._state_dim,
            f"mean must have shape ({self._state_dim},).",
        )
        check_is_true(
            covariance.shape == (self._state_dim, self._state_dim),
            f"covariance must have shape ({self._state_dim}, {self._state_dim}).",
        )

        covariance = 0.5 * (covariance + covariance.T)
        jitter = 1e-10 * np.eye(self._state_dim)

        try:
            chol = np.linalg.cholesky((self._state_dim + self._lambda) * covariance + jitter)
        except np.linalg.LinAlgError:
            logging.warning("Covariance not SPD, adding extra jitter in sigma-point generation.")
            chol = np.linalg.cholesky((self._state_dim + self._lambda) * (covariance + 1e-6 * np.eye(self._state_dim)))

        sigma_points = np.zeros((2 * self._state_dim + 1, self._state_dim))
        sigma_points[0] = mean
        for i in range(self._state_dim):
            sigma_points[i + 1] = mean + chol[:, i]
            sigma_points[self._state_dim + i + 1] = mean - chol[:, i]
        return sigma_points

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        transition_function: Callable[[np.ndarray], np.ndarray],
        process_noise_cov_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform the prediction step of the Unscented Kalman Filter.

        Args:
            mean (np.ndarray): Current filtered state mean.
            covariance (np.ndarray): Current filtered state covariance.
            transition_function (Callable[[np.ndarray], np.ndarray]): State transition function.
            process_noise_cov_function (Optional[Callable[[np.ndarray], np.ndarray]]): Function returning the process noise covariance.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Predicted state mean, predicted
            state covariance, and propagated sigma points.
        """
        
        sigma_points = self._compute_sigma_points(mean, covariance)
        propagated_sigma_points = np.asarray([transition_function(point) for point in sigma_points], dtype=float)

        predicted_mean = np.sum(
            self._weights_mean[:, None] * propagated_sigma_points,
            axis=0,
        )

        predicted_covariance = np.zeros((self._state_dim, self._state_dim), dtype=float)
        for i in range(propagated_sigma_points.shape[0]):
            diff = propagated_sigma_points[i] - predicted_mean
            predicted_covariance += self._weights_cov[i] * np.outer(diff, diff)

        if process_noise_cov_function is not None:
            process_noise_cov = np.zeros((self._state_dim, self._state_dim), dtype=float)
            for i in range(sigma_points.shape[0]):
                process_noise_cov += self._weights_cov[i] * process_noise_cov_function(sigma_points[i])
            predicted_covariance += process_noise_cov

        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        return predicted_mean, predicted_covariance, propagated_sigma_points

    def update(
        self,
        predicted_mean: np.ndarray,
        predicted_covariance: np.ndarray,
        propagated_sigma_points: np.ndarray,
        observation: np.ndarray,
        observation_function: Callable[[np.ndarray], np.ndarray],
        measurement_noise_cov_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Perform the update step of the Unscented Kalman Filter.

        Args:
            predicted_mean (np.ndarray): Predicted state mean from the prediction step.
            predicted_covariance (np.ndarray): Predicted state covariance from the prediction step.
            propagated_sigma_points (np.ndarray): Sigma points propagated through the transition function.
            observation (np.ndarray): Observation vector at the current step.
            observation_function (Callable[[np.ndarray], np.ndarray]): Observation function of the state space model.
            measurement_noise_cov_function (Optional[Callable[[np.ndarray], np.ndarray]]): Function returning the measurement noise covariance."""
        
        observation = np.asarray(observation, dtype=float).reshape(-1)
        check_is_true(
            observation.shape[0] == self._observation_dim,
            f"observation must have shape ({self._observation_dim},).",
        )

        predicted_observation_sigma_points = np.asarray(
            [observation_function(point) for point in propagated_sigma_points],
            dtype=float,
        )

        predicted_observation_mean = np.sum(
            self._weights_mean[:, None] * predicted_observation_sigma_points,
            axis=0,
        )

        innovation_covariance = np.zeros((self._observation_dim, self._observation_dim), dtype=float)
        cross_covariance = np.zeros((self._state_dim, self._observation_dim), dtype=float)

        for i in range(predicted_observation_sigma_points.shape[0]):
            y_diff = predicted_observation_sigma_points[i] - predicted_observation_mean
            x_diff = propagated_sigma_points[i] - predicted_mean
            innovation_covariance += self._weights_cov[i] * np.outer(y_diff, y_diff)
            cross_covariance += self._weights_cov[i] * np.outer(x_diff, y_diff)

        if measurement_noise_cov_function is not None:
            measurement_noise_cov = np.zeros((self._observation_dim, self._observation_dim), dtype=float)
            for i in range(propagated_sigma_points.shape[0]):
                measurement_noise_cov += self._weights_cov[i] * measurement_noise_cov_function(propagated_sigma_points[i])
            innovation_covariance += measurement_noise_cov

        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        innovation_covariance += 1e-10 * np.eye(self._observation_dim)

        kalman_gain = cross_covariance @ np.linalg.inv(innovation_covariance)
        innovation = observation - predicted_observation_mean

        updated_mean = predicted_mean + kalman_gain @ innovation
        updated_covariance = predicted_covariance - kalman_gain @ innovation_covariance @ kalman_gain.T
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)

        sign, logdet = np.linalg.slogdet(innovation_covariance)
        check_is_true(sign > 0, "Innovation covariance must be positive definite.")
        log_likelihood = float(
            -0.5
            * (
                self._observation_dim * np.log(2 * np.pi)
                + logdet
                + innovation.T @ np.linalg.inv(innovation_covariance) @ innovation
            )
        )

        return updated_mean, updated_covariance, innovation, innovation_covariance, log_likelihood

    def filter(
        self,
        observations: pd.Series | np.ndarray,
        initial_state_mean: np.ndarray,
        initial_state_covariance: np.ndarray,
        transition_function: Callable[[np.ndarray], np.ndarray],
        observation_function: Callable[[np.ndarray], np.ndarray],
        process_noise_cov_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        measurement_noise_cov_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Run the Unscented Kalman Filter on a sequence of observations.

        Args:
            observations (pd.Series | np.ndarray): Time series of observations.
            initial_state_mean (np.ndarray): Initial mean of the latent state.
            initial_state_covariance (np.ndarray): Initial covariance of the latent state.
            transition_function (Callable[[np.ndarray], np.ndarray]): State transition function.
            observation_function (Callable[[np.ndarray], np.ndarray]): Observation function of the state space model.
            process_noise_cov_function (Optional[Callable[[np.ndarray], np.ndarray]]): Function returning the process noise covariance.
            measurement_noise_cov_function (Optional[Callable[[np.ndarray], np.ndarray]]): Function returning the measurement noise covariance.

        Returns:
            pd.DataFrame: Filtering results including predicted states, filtered states,
            innovations, innovation variances, and log-likelihood contributions.
        """
        
        observations = np.asarray(observations, dtype=float).reshape(-1, self._observation_dim)
        filtered_rows = []

        current_mean = np.asarray(initial_state_mean, dtype=float).reshape(-1)
        current_covariance = np.asarray(initial_state_covariance, dtype=float)

        for t, observation in enumerate(observations):
            predicted_mean, predicted_covariance, propagated_sigma_points = self.predict(
                current_mean,
                current_covariance,
                transition_function=transition_function,
                process_noise_cov_function=process_noise_cov_function,
            )
            updated_mean, updated_covariance, innovation, innovation_covariance, log_likelihood = self.update(
                predicted_mean=predicted_mean,
                predicted_covariance=predicted_covariance,
                propagated_sigma_points=propagated_sigma_points,
                observation=observation,
                observation_function=observation_function,
                measurement_noise_cov_function=measurement_noise_cov_function,
            )

            row = {
                "t": t,
                "loglikelihood": log_likelihood,
            }
            for i in range(self._state_dim):
                row[f"predicted_state_{i}"] = predicted_mean[i]
                row[f"filtered_state_{i}"] = updated_mean[i]
                row[f"predicted_state_var_{i}"] = predicted_covariance[i, i]
                row[f"filtered_state_var_{i}"] = updated_covariance[i, i]
            for j in range(self._observation_dim):
                row[f"innovation_{j}"] = innovation[j]
                row[f"innovation_var_{j}"] = innovation_covariance[j, j]
                row[f"observation_{j}"] = observation[j]
            filtered_rows.append(row)

            current_mean = updated_mean
            current_covariance = updated_covariance

        return pd.DataFrame(filtered_rows)