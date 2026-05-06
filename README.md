# Realized Volatility Timing — Variance Risk Premium Harvesting

> **Course project** — Master 272, Paris Dauphine University  
> **Course**: Stratégie de la Volatilité (B. Zloch)  
> **Branch**: `Volatility_Strategies_Project`

This project builds on the course infrastructure ([main branch](https://github.com/AntonBzd/Dauphine-Lecture-Volatility)) to explore **realized volatility timing**: dynamically sizing a short variance swap based on the implied–realized volatility spread.

---

## Objective

The implied volatility priced by the options market is structurally higher than the subsequently realized volatility — this gap is the **Variance Risk Premium (VRP)**. Selling volatility harvests this premium, but doing so with constant sizing leads to large drawdowns during vol spikes (COVID, 2022 bear market).

The goal is to **time** the VRP: sell more when the premium is rich, reduce or reverse when it is compressed. We estimate the spread

$$s_t = \sigma_{IV,t} - \hat{\sigma}_t$$

where $\sigma_{IV,t}$ is the ATM implied vol at 30-day maturity (extracted via SSVI surface fitting) and $\hat{\sigma}_t$ is the forward-looking forecast from a Heston model filtered by an Unscented Kalman Filter. A z-score of this spread drives the dynamic allocation on a variance swap.

---

## Architecture

```
investment_lab/
├── data/                        # Data loading (options, rates)
│   ├── data_loader.py           # Abstract loader with date filtering
│   ├── option_db.py             # Option data loaders (AAPL, SPY)
│   └── rates_db.py              # US Treasury yield curve loader
│
├── surface/                     # Implied volatility surface
│   ├── base.py                  # Abstract VolSmoother (fit/transform)
│   ├── svi.py                   # SVI parametrization (raw)
│   ├── ssvi.py                  # SSVI parametrization (power-law kernel)
│   ├── sabr.py                  # SABR model
│   ├── constant_maturity.py     # Total variance interpolation across maturities
│   └── reference_iv.py          # Extract a single ATM IV reference per day
│
├── stochastic/                  # Stochastic volatility models
│   ├── base.py                  # Abstract StochasticProcess
│   ├── heston.py                # Heston model (transition, observation, forecast)
│   ├── heston_ssm.py            # Heston State Space Model (calibration + UKF)
│   └── ukf.py                   # Unscented Kalman Filter (generic)
│
├── signals/                     # Trading signals
│   ├── implied_realized_spread.py  # Spread: IV reference − Heston forecast
│   ├── naive_spread.py          # Benchmark: IV reference − rolling realized vol
│   └── allocation.py            # Z-score allocation signal
│
├── strategies/
│   └── dynamic_allocation.py    # Apply allocation overlay to trades
│
├── metrics/                     # Performance & risk metrics
│   ├── performance.py           # Sharpe, Calmar, hit ratio, drawdown
│   ├── volatility.py            # Realized vol, rolling vol
│   ├── distance.py              # MSE, SSE
│   └── util.py                  # Returns ↔ levels conversion
│
├── pricing/
│   ├── black_scholes.py         # BS price, greeks, IV (Newton-Raphson)
│   └── implied_volatility.py    # Vectorized IV computation
│
├── backtest.py                  # Strategy backtester with P&L attribution
├── option_trade.py              # Trade generation (selection, hedging)
├── option_selection.py          # Closest strike/maturity selection
├── option_strategies.py         # Pre-defined option strategies
├── rates.py                     # Rate interpolation and forward computation
├── constants.py                 # Trading days, tenor mappings
├── dataclass.py                 # TypedDicts for leg specs
└── util.py                      # Validation helpers

notebooks/
└── Realized_Volatility_Timing.ipynb   # Main notebook (full pipeline)
```

---

## Pipeline

### 1. IV Reference Extraction (`surface/`)

For each trading day, we fit an **SSVI** surface to the cross-section of option implied volatilities, then interpolate in total variance space to extract a constant-maturity ATM implied vol at 30 days:

$$w(k, \theta_t) = \frac{\theta_t}{2}\left(1 + \rho\,\varphi(\theta_t)\,k + \sqrt{(\varphi(\theta_t)\,k + \rho)^2 + 1 - \rho^2}\right)$$

where $k = \ln(K/F)$ is the log-forward-moneyness, $\theta_t = \sigma^2 t$ is the ATM total variance, and $\varphi(\theta) = \eta / \theta^\lambda$ is the power-law kernel. The reference IV is:

$$\sigma_{IV,t} = \sqrt{w(0,\, \theta_{30d}) \;/\; T_{30d}}$$

### 2. Heston State Space Model (`stochastic/`)

The latent variance follows the Heston dynamics:

$$dS_t = \mu\, S_t\, dt + \sqrt{v_t}\, S_t\, dW_{1,t}$$

$$dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_{2,t}, \quad dW_1 \cdot dW_2 = \rho\,dt$$

#### Hybrid Calibration

The change of measure from P (physical) to Q (risk-neutral) affects only the drift of the variance process — the diffusion parameters $\xi$ and $\rho$ are invariant. We exploit this:

| Parameter | Estimated by | Justification |
|-----------|-------------|---------------|
| $\xi$ (vol-of-vol) | Method of moments | Diffusion param — same under P and Q |
| $\rho$ (correlation) | Method of moments | Diffusion param — same under P and Q |
| $\kappa$ (mean-reversion speed) | MLE via UKF | Drift param — P-specific |
| $\theta$ (long-run variance) | MLE via UKF | Drift param — P-specific |
| $\mu$ (drift) | Method of moments | — |

The MLE optimizes only 2 parameters (κ, θ), making it well-conditioned and fast.

#### Augmented UKF

The standard UKF observes only the log-spot, making the latent variance weakly identifiable. We augment the observation vector with a **realized variance proxy** (5-day rolling variance, annualized):

$$\text{observation}_t = \begin{pmatrix} \ln S_t \\ \hat{v}_t^{RV} \end{pmatrix}$$

This gives the filter a direct (noisy) signal on the variance level, dramatically improving filtering quality. The measurement noise covariance is diagonal: small noise on log-spot ($\sim 10^{-6}$), larger noise on the RV proxy ($\sim 10^{-2}$) reflecting its estimation error.

#### Volatility Forecast

From the filtered variance $\hat{v}_t$, the expected average variance over horizon $h$ is:

$$\bar{V}(t, t+h) = \theta + (\hat{v}_t - \theta)\,\frac{1 - e^{-\kappa h}}{\kappa h}$$

The forecast volatility is $\hat{\sigma}_t = \sqrt{\bar{V}}$.

### 3. Signal Construction (`signals/`)

The **implied–realized spread** captures the instantaneous VRP:

$$s_t = \sigma_{IV,t} - \hat{\sigma}_t$$

We compute a rolling z-score to normalize for the structural positive bias (VRP):

$$z_t = \frac{s_t - \bar{s}_t^{(63)}}{\text{std}(s)_t^{(63)}}$$

The **allocation** is a linear function of the z-score, clipped to $[-1, +1]$:

$$a_t = 0.25 \times z_t$$

- $a_t > 0$: VRP is rich → **short vol** (sell the variance swap)
- $a_t < 0$: VRP is compressed → **long vol** (buy the variance swap)
- $a_t = 0$: neutral

### 4. Strategy & Backtest

The strategy is a **short variance swap** (30-day, weekly roll). The `DynamicAllocationOverlay` multiplies each trade's weight by the allocation signal from the previous day (lag = 1 business day to avoid look-ahead bias).

We compare three variants:
- **Static**: constant sizing, no timing
- **Dynamic Heston**: sizing driven by the Heston-based spread z-score
- **Dynamic Naive**: sizing driven by a simple IV − rolling realized vol spread (benchmark)

---

## Setup

```bash
pip install -r requirement.txt
pip install -e .
```

Then open `notebooks/Realized_Volatility_Timing.ipynb` and run all cells.

---

## Key Results

| Strategy | Annual Return | Sharpe | Max Drawdown |
|----------|--------------|--------|-------------|
| Var Swap Static | 0.09 | 0.78 | -0.11 |
| Var Swap Dynamic Z-Score (Heston) | 0.04 | 0.58 | -0.08 |
| Var Swap Dynamic Z-Score (Naive) | 0.04 | 0.78 | -0.06 |
| Var Swap Dynamic Percentile (Heston) | 0.02 | 0.35 | -0.06 |
| Var Swap Dynamic Percentile (Naive) | 0.03 | 0.92 | -0.03 |


The dynamic strategies do not surperform compared to the static approach. We should improve dynamically and leverage the exposure during vol spikes and increasing it when the VRP is rich.

---

## Potential Improvements

- **Hybrid calibration with surface-implied ξ, ρ**: extract diffusion parameters directly from the options surface (via Heston characteristic function) instead of moments on returns.
- **MLE regularization**: penalize deviations from moment estimates to prevent degenerate solutions.
- **Multi-asset**: extend to other assets or a basket for diversification of the carry.
- **Regime detection**: replace the z-score with a regime-switching model for asymmetric VRP dynamics.

---

## Acknowledgements

This project builds on the course infrastructure and codebase developed by **Baptiste Zloch** for the Master 272 Volatility Trading course at Paris Dauphine University.

## License

This project is licensed under the terms of the CC BY-NC-SA 4.0 license.

