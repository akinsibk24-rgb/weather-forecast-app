"""
forecaster.py
--------------
A lightweight, dependency-friendly forecasting engine.

Approach
--------
Nigerian weather is strongly seasonal (wet/dry seasons), so instead of a
heavy ML stack we fit a harmonic (Fourier) regression per city:

    temperature(t) ~ b0 + b1*sin(w*doy) + b2*cos(w*doy)
                        + b3*sin(2w*doy) + b4*cos(2w*doy) + b5*year_trend

This captures the annual cycle plus a slow trend very well with very
little data/compute, which keeps the packaged app small and fast.

Rainfall is modeled in two parts:
  1. Probability of rain on a given day-of-year (smoothed empirical rate)
  2. Expected rainfall amount on rainy days (harmonic regression in
     log-space, since rainfall is non-negative and right-skewed)

The final forecast blends the seasonal ("climatological") prediction with
the recent observed anomaly (last 14 days vs. what the model expects for
that period), decaying the anomaly's influence the further out we forecast.
This is a standard, explainable way to get short-range forecasts to track
current conditions while falling back to climatology further ahead.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _fourier_features(doy: np.ndarray, year_frac: np.ndarray) -> np.ndarray:
    w = 2 * np.pi / 365.25
    return np.column_stack([
        np.sin(w * doy), np.cos(w * doy),
        np.sin(2 * w * doy), np.cos(2 * w * doy),
        year_frac,
    ])


@dataclass
class CityForecastModel:
    city: str
    temp_model: LinearRegression
    rain_amount_model: LinearRegression
    rain_prob_by_doy: np.ndarray  # length 366, smoothed probability of rain
    last_date: pd.Timestamp
    recent_temp_anomaly: float
    recent_rain_anomaly_ratio: float  # multiplicative anomaly, 1.0 = normal


def _smoothed_rain_probability(df: pd.DataFrame) -> np.ndarray:
    doy = df["date"].dt.dayofyear.to_numpy()
    rained = (df["rainfall_mm"] > 0.1).to_numpy().astype(float)
    prob = np.zeros(367)
    counts = np.zeros(367)
    np.add.at(prob, doy, rained)
    np.add.at(counts, doy, 1)
    counts[counts == 0] = 1
    daily_prob = prob / counts

    # circular smoothing (15-day window) so the curve is not jagged
    window = 15
    padded = np.concatenate([daily_prob[-window:], daily_prob, daily_prob[:window]])
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="same")[window:window + 367]
    return np.clip(smoothed, 0.02, 0.97)


def fit_city_model(df_city: pd.DataFrame) -> CityForecastModel:
    df_city = df_city.sort_values("date").reset_index(drop=True)
    doy = df_city["date"].dt.dayofyear.to_numpy()
    first_year = df_city["date"].dt.year.min()
    year_frac = (df_city["date"].dt.year - first_year) + df_city["date"].dt.dayofyear / 365.25

    X = _fourier_features(doy, year_frac.to_numpy())

    temp_model = LinearRegression().fit(X, df_city["temp_mean_c"].to_numpy())

    rainy = df_city[df_city["rainfall_mm"] > 0.1]
    X_rainy = _fourier_features(
        rainy["date"].dt.dayofyear.to_numpy(),
        ((rainy["date"].dt.year - first_year) + rainy["date"].dt.dayofyear / 365.25).to_numpy(),
    )
    y_rainy = np.log1p(rainy["rainfall_mm"].to_numpy())
    rain_amount_model = LinearRegression().fit(X_rainy, y_rainy)

    rain_prob_by_doy = _smoothed_rain_probability(df_city)

    # recent anomaly: last 14 days actual vs. model-predicted
    last_date = df_city["date"].max()
    recent = df_city[df_city["date"] > last_date - pd.Timedelta(days=14)]
    if len(recent):
        recent_doy = recent["date"].dt.dayofyear.to_numpy()
        recent_yf = ((recent["date"].dt.year - first_year) + recent["date"].dt.dayofyear / 365.25).to_numpy()
        recent_pred_temp = temp_model.predict(_fourier_features(recent_doy, recent_yf))
        temp_anomaly = float((recent["temp_mean_c"].to_numpy() - recent_pred_temp).mean())

        recent_expected_rain = rain_prob_by_doy[recent_doy] * np.expm1(
            rain_amount_model.predict(_fourier_features(recent_doy, recent_yf))
        )
        actual_rain_mean = recent["rainfall_mm"].mean()
        expected_rain_mean = max(recent_expected_rain.mean(), 0.1)
        rain_ratio = float(np.clip(actual_rain_mean / expected_rain_mean, 0.2, 3.0))
    else:
        temp_anomaly, rain_ratio = 0.0, 1.0

    return CityForecastModel(
        city=df_city["city"].iloc[0],
        temp_model=temp_model,
        rain_amount_model=rain_amount_model,
        rain_prob_by_doy=rain_prob_by_doy,
        last_date=last_date,
        recent_temp_anomaly=temp_anomaly,
        recent_rain_anomaly_ratio=rain_ratio,
    )


def forecast(model: CityForecastModel, horizon_days: int = 7, first_year: int = None) -> pd.DataFrame:
    """Produce a day-by-day forecast starting the day after the last observed date."""
    future_dates = pd.date_range(model.last_date + timedelta(days=1), periods=horizon_days, freq="D")
    doy = future_dates.dayofyear.to_numpy()

    ref_year = model.last_date.year
    year_frac = (future_dates.year - ref_year) + future_dates.dayofyear / 365.25
    X = _fourier_features(doy, year_frac.to_numpy())

    base_temp = model.temp_model.predict(X)
    base_log_rain = model.rain_amount_model.predict(X)
    base_rain_amount = np.expm1(base_log_rain)
    rain_prob = model.rain_prob_by_doy[doy]

    # decay anomaly influence over the horizon (full weight day 1, ~0 by day 14+)
    decay = np.exp(-np.arange(1, horizon_days + 1) / 6.0)
    temp_forecast = base_temp + model.recent_temp_anomaly * decay
    rain_ratio_effective = 1 + (model.recent_rain_anomaly_ratio - 1) * decay
    expected_rain_amount = base_rain_amount * rain_ratio_effective

    expected_rainfall_mm = rain_prob * expected_rain_amount

    return pd.DataFrame({
        "date": future_dates,
        "city": model.city,
        "forecast_temp_c": np.round(temp_forecast, 1),
        "rain_probability_pct": np.round(rain_prob * 100, 0),
        "forecast_rainfall_mm": np.round(expected_rainfall_mm, 1),
    })
