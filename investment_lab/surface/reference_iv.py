import logging
from abc import ABC, abstractmethod
from typing import Optional, Self

import numpy as np
import pandas as pd

from investment_lab.constants import DAYS_PER_YEAR
from investment_lab.surface.constant_maturity import interpolate_total_variance_from_days
from investment_lab.surface.ssvi import SSVISmoother
from investment_lab.surface.svi import SVISmoother
from investment_lab.util import check_is_true


class ReferenceIVExtractor(ABC):
    _REQUIRED_COLUMNS = [
        "date",
        "ticker",
        "spot",
        "forward",
        "strike",
        "expiration",
        "day_to_expiration",
        "implied_volatility",
        "call_put",
        "option_id",
    ]

    def __init__(
        self,
        target_day_to_expiration: int,
        target_moneyness: float = 1.0,
        minimum_points_per_expiry: int = 5,
        volume_col: Optional[str] = "volume",
        bid_col: Optional[str] = "bid",
        ask_col: Optional[str] = "ask",
    ) -> None:
        """Initialize a reference implied volatility extractor.

        Args:
            target_day_to_expiration (int): Target maturity in days.
            target_moneyness (float): Target moneyness level used to extract the reference volatility.
            minimum_points_per_expiry (int): Minimum number of options required to fit one expiry slice.
            volume_col (Optional[str]): Name of the volume column used for filtering.
            bid_col (Optional[str]): Name of the bid column used for filtering.
            ask_col (Optional[str]): Name of the ask column used for filtering.
        """
        check_is_true(target_day_to_expiration >= 1, "target_day_to_expiration must be >= 1.")
        check_is_true(target_moneyness > 0, "target_moneyness must be > 0.")
        check_is_true(minimum_points_per_expiry >= 3, "minimum_points_per_expiry must be >= 3.")

        self._target_day_to_expiration = target_day_to_expiration
        self._target_moneyness = target_moneyness
        self._minimum_points_per_expiry = minimum_points_per_expiry
        self._volume_col = volume_col
        self._bid_col = bid_col
        self._ask_col = ask_col

    def _validate_input(self, df_options: pd.DataFrame) -> None:
        """Validate that the option dataset contains all required columns.

        Args:
            df_options (pd.DataFrame): Option dataset used for reference volatility extraction.
        """
        missing_cols = set(self._REQUIRED_COLUMNS).difference(df_options.columns)
        check_is_true(len(missing_cols) == 0, f"df_options is missing columns: {missing_cols}")

    def _filter_cross_section(self, df_options: pd.DataFrame) -> pd.DataFrame:
        """Filter an option cross-section before fitting a volatility surface.

        Args:
            df_options (pd.DataFrame): Raw option cross-section.

        Returns:
            pd.DataFrame: Filtered option cross-section.
        """
        df = df_options.copy()
        df = df[df["day_to_expiration"] > 0]
        df = df[df["implied_volatility"].notna()]
        df = df[df["implied_volatility"] > 0]
        df = df[df["spot"] > 0]
        df = df[df["strike"] > 0]
        df = df[df["forward"] > 0]

        if self._volume_col is not None and self._volume_col in df.columns:
            df = df[df[self._volume_col].fillna(0) >= 0]

        if self._bid_col is not None and self._ask_col is not None:
            if self._bid_col in df.columns and self._ask_col in df.columns:
                df = df[df[self._ask_col] >= df[self._bid_col]]

        return df

    @abstractmethod
    def _build_smoother(self):
        """Instantiate the volatility smoother used to fit the option surface.

        Returns:
            Any: Volatility smoother instance.
        """
        raise NotImplementedError

    def _fit_one_expiry_slice(self, df_expiry: pd.DataFrame) -> float:
        """Fit one expiry slice and extract the fitted volatility at the target moneyness.

        Args:
            df_expiry (pd.DataFrame): Option dataset associated with a single expiry.

        Returns:
            float: Fitted implied volatility at the target strike and expiry.
        """
        check_is_true(len(df_expiry) >= self._minimum_points_per_expiry, "Not enough points in expiry slice.")

        smoother = self._build_smoother()
        fitted_smoother = smoother.fit(
            forward=df_expiry["forward"].to_numpy(),
            strike=df_expiry["strike"].to_numpy(),
            time_to_maturities=(df_expiry["day_to_expiration"] / DAYS_PER_YEAR).to_numpy(),
            market_implied_vols=df_expiry["implied_volatility"].to_numpy(),
        )
        target_forward = float(df_expiry["forward"].iloc[0])
        target_strike = self._target_moneyness * target_forward

        fitted_vol = fitted_smoother.transform(
            forward=np.asarray([target_forward], dtype=float),
            strike=np.asarray([target_strike], dtype=float),
            time_to_maturities=np.asarray([df_expiry["day_to_expiration"].iloc[0] / DAYS_PER_YEAR], dtype=float),
        )
        return float(np.asarray(fitted_vol).reshape(-1)[0])
    
    def _extract_single_date_ticker(self, df_cross_section: pd.DataFrame) -> Optional[dict]:
        """Extract the reference implied volatility for one date and one ticker.

        Args:
            df_cross_section (pd.DataFrame): Option cross-section for a single date and ticker.

        Returns:
            Optional[dict]: Dictionary containing the extracted reference implied volatility
            and associated metadata.
        """
        df = self._filter_cross_section(df_cross_section)
        if len(df) == 0:
            return None

        fitted_points = []
        for expiration, df_expiry in df.groupby("expiration"):
            try:
                if len(df_expiry) < self._minimum_points_per_expiry:
                    continue
                fitted_iv = self._fit_one_expiry_slice(df_expiry)
                fitted_points.append(
                    {
                        "date": df_expiry["date"].iloc[0],
                        "ticker": df_expiry["ticker"].iloc[0],
                        "expiration": expiration,
                        "day_to_expiration": int(df_expiry["day_to_expiration"].iloc[0]),
                        "fitted_iv": fitted_iv,
                    }
                )
            except Exception as exc:
                logging.warning(
                    "Surface fitting failed for date=%s ticker=%s expiration=%s. Error=%s",
                    df_expiry["date"].iloc[0],
                    df_expiry["ticker"].iloc[0],
                    expiration,
                    exc,
                )

        if len(fitted_points) == 0:
            return None

        df_fitted = pd.DataFrame(fitted_points).sort_values("day_to_expiration")
        iv_reference = interpolate_total_variance_from_days(
            target_day_to_maturity=self._target_day_to_expiration,
            day_to_maturities=df_fitted["day_to_expiration"].to_numpy(),
            implied_volatilities=df_fitted["fitted_iv"].to_numpy(),
        )

        return {
            "date": df_fitted["date"].iloc[0],
            "ticker": df_fitted["ticker"].iloc[0],
            "iv_reference": iv_reference,
            "target_day_to_expiration": self._target_day_to_expiration,
            "target_moneyness": self._target_moneyness,
            "n_fitted_expiries": len(df_fitted),
        }

    def extract(
        self,
        df_options: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract a daily reference implied volatility series from option cross-sections.

        Args:
            df_options (pd.DataFrame): Option dataset containing multiple dates and tickers.

        Returns:
            pd.DataFrame: Daily reference implied volatility series with metadata.
        """
        self._validate_input(df_options)
        logging.info("Extracting reference IV on %s option rows.", len(df_options))

        # Pré-filtrage global une seule fois (au lieu de le faire par date/ticker)
        df_clean = self._filter_cross_section(df_options)
        logging.info("After filtering: %s rows remaining.", len(df_clean))

        rows = []
        for (date, ticker), df_cross_section in df_clean.groupby(["date", "ticker"]):
            extracted = self._extract_single_date_ticker_prefiltred(df_cross_section)
            if extracted is not None:
                rows.append(extracted)

        if len(rows) == 0:
            return pd.DataFrame(
                columns=[
                    "date", "ticker", "iv_reference",
                    "target_day_to_expiration", "target_moneyness", "n_fitted_expiries",
                ]
            )

        return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    def _extract_single_date_ticker_prefiltred(self, df_cross_section: pd.DataFrame) -> Optional[dict]:
        """Extract the reference implied volatility for one date and one ticker from a pre-filtered cross-section.

        Args:
            df_cross_section (pd.DataFrame): Pre-filtered option cross-section for a single date and ticker.

        Returns:
            Optional[dict]: Dictionary containing the extracted reference implied volatility
            and associated metadata.
        """
        df = df_cross_section
        if len(df) == 0:
            return None

        fitted_points = []
        for expiration, df_expiry in df.groupby("expiration"):
            try:
                if len(df_expiry) < self._minimum_points_per_expiry:
                    continue
                fitted_iv = self._fit_one_expiry_slice(df_expiry)
                if not np.isfinite(fitted_iv) or fitted_iv <= 0:
                    continue
                fitted_points.append({
                    "date": df_expiry["date"].iloc[0],
                    "ticker": df_expiry["ticker"].iloc[0],
                    "expiration": expiration,
                    "day_to_expiration": int(df_expiry["day_to_expiration"].iloc[0]),
                    "fitted_iv": fitted_iv,
                })
            except Exception as exc:
                logging.debug(
                    "Surface fitting failed date=%s ticker=%s exp=%s: %s",
                    df_expiry["date"].iloc[0], df_expiry["ticker"].iloc[0], expiration, exc,
                )

        if len(fitted_points) == 0:
            return None

        df_fitted = pd.DataFrame(fitted_points).sort_values("day_to_expiration")
        iv_reference = interpolate_total_variance_from_days(
            target_day_to_maturity=self._target_day_to_expiration,
            day_to_maturities=df_fitted["day_to_expiration"].to_numpy(),
            implied_volatilities=df_fitted["fitted_iv"].to_numpy(),
        )

        return {
            "date": df_fitted["date"].iloc[0],
            "ticker": df_fitted["ticker"].iloc[0],
            "iv_reference": iv_reference,
            "target_day_to_expiration": self._target_day_to_expiration,
            "target_moneyness": self._target_moneyness,
            "n_fitted_expiries": len(df_fitted),
        }


class SVIReferenceIVExtractor(ReferenceIVExtractor):
    def __init__(
        self,
        target_day_to_expiration: int,
        target_moneyness: float = 1.0,
        minimum_points_per_expiry: int = 5,
        initial_params: tuple[float, float, float, float, float] = (0.01, 0.2, -0.3, 0.0, 0.2),
        volume_col: Optional[str] = "volume",
        bid_col: Optional[str] = "bid",
        ask_col: Optional[str] = "ask",
    ) -> None:
        """Initialize an SVI-based reference implied volatility extractor.

        Args:
            target_day_to_expiration (int): Target maturity in days.
            target_moneyness (float): Target moneyness level used to extract the reference volatility.
            minimum_points_per_expiry (int): Minimum number of options required to fit one expiry slice.
            initial_params (tuple[float, float, float, float, float]): Initial SVI parameter tuple.
            volume_col (Optional[str]): Name of the volume column used for filtering.
            bid_col (Optional[str]): Name of the bid column used for filtering.
            ask_col (Optional[str]): Name of the ask column used for filtering.
        """
        super().__init__(
            target_day_to_expiration=target_day_to_expiration,
            target_moneyness=target_moneyness,
            minimum_points_per_expiry=minimum_points_per_expiry,
            volume_col=volume_col,
            bid_col=bid_col,
            ask_col=ask_col,
        )
        self._initial_params = initial_params

    def _build_smoother(self) -> SVISmoother:
        """Instantiate the SVI smoother used to fit the option surface.

        Returns:
            SVISmoother: SVI volatility smoother.
        """
        return SVISmoother(initial_params=self._initial_params)


class SSVIReferenceIVExtractor(ReferenceIVExtractor):
    def __init__(
        self,
        target_day_to_expiration: int,
        target_moneyness: float = 1.0,
        minimum_points_per_expiry: int = 5,
        initial_params: tuple[float, float, float, float] = (0.2, -0.3, 0.5, 0.2),
        volume_col: Optional[str] = "volume",
        bid_col: Optional[str] = "bid",
        ask_col: Optional[str] = "ask",
    ) -> None:
        """Initialize an SSVI-based reference implied volatility extractor.

        Args:
            target_day_to_expiration (int): Target maturity in days.
            target_moneyness (float): Target moneyness level used to extract the reference volatility.
            minimum_points_per_expiry (int): Minimum number of options required to fit one expiry slice.
            initial_params (tuple[float, float, float, float]): Initial SSVI parameter tuple.
            volume_col (Optional[str]): Name of the volume column used for filtering.
            bid_col (Optional[str]): Name of the bid column used for filtering.
            ask_col (Optional[str]): Name of the ask column used for filtering.
        """
        super().__init__(
            target_day_to_expiration=target_day_to_expiration,
            target_moneyness=target_moneyness,
            minimum_points_per_expiry=minimum_points_per_expiry,
            volume_col=volume_col,
            bid_col=bid_col,
            ask_col=ask_col,
        )
        self._initial_params = initial_params

    def _build_smoother(self) -> SSVISmoother:
        """Instantiate the SSVI smoother used to fit the option surface.

        Returns:
            SSVISmoother: SSVI volatility smoother.
        """
        return SSVISmoother(initial_params=self._initial_params)