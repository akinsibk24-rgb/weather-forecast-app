"""
generate_sample_data.py
------------------------
Builds a realistic multi-year DAILY rainfall + temperature sample dataset
for major Nigerian cities, based on published climate normals (wet/dry
season timing, average highs/lows, and rainfall totals per city).

This is NOT the real Kaggle dataset. It exists so the app has something
sensible to run on immediately. Replace/augment it with the real Kaggle
data using scripts/fetch_kaggle_data.py once you have Kaggle API access.

Run:
    python scripts/generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Approximate climate normals for each city:
#   temp_mean_c   -> annual mean temperature
#   temp_amp_c    -> seasonal swing (peak minus mean)
#   rain_annual_mm-> approx annual rainfall total
#   wet_peak_doy  -> day-of-year around which rainfall peaks
#   climate       -> coastal cities have smaller temp swings, Sahel cities bigger
CITY_PROFILES = {
    "Lagos":         dict(lat=6.52,  temp_mean=27.2, temp_amp=2.0, rain_annual=1700, wet_peak_doy=170, wet_width=110),
    "Abuja":         dict(lat=9.06,  temp_mean=26.0, temp_amp=3.5, rain_annual=1400, wet_peak_doy=220, wet_width=95),
    "Kano":          dict(lat=12.00, temp_mean=26.5, temp_amp=6.5, rain_annual=850,  wet_peak_doy=230, wet_width=70),
    "Port Harcourt": dict(lat=4.82,  temp_mean=26.8, temp_amp=1.5, rain_annual=2500, wet_peak_doy=200, wet_width=140),
    "Ibadan":        dict(lat=7.38,  temp_mean=26.3, temp_amp=2.5, rain_annual=1250, wet_peak_doy=180, wet_width=110),
    "Enugu":         dict(lat=6.44,  temp_mean=26.6, temp_amp=2.5, rain_annual=1800, wet_peak_doy=200, wet_width=120),
}

START_YEAR = 2015
END_YEAR = 2024  # inclusive, ~10 years of daily data
SEED = 42


def build_city_series(city, profile, rng):
    dates = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    year_idx = (dates.year - START_YEAR).to_numpy()

    # --- Temperature: seasonal sinusoid + mild long-term trend + noise ---
    seasonal = profile["temp_amp"] * np.sin(2 * np.pi * (doy - 60) / 365.0)
    trend = 0.02 * year_idx  # slight warming trend over the decade (~0.2C/decade)
    noise = rng.normal(0, 1.1, size=len(dates))
    temp_mean_c = profile["temp_mean"] + seasonal + trend + noise
    temp_max_c = temp_mean_c + rng.normal(4.0, 0.6, size=len(dates))
    temp_min_c = temp_mean_c - rng.normal(4.5, 0.6, size=len(dates))

    # --- Rainfall: wet-season bell curve controls probability & intensity ---
    wet_signal = np.exp(-0.5 * ((doy - profile["wet_peak_doy"]) / profile["wet_width"]) ** 2)
    # small second bump for cities with a "little dry season" break (common in south Nigeria)
    rain_prob = 0.08 + 0.75 * wet_signal
    rain_prob = np.clip(rain_prob, 0.02, 0.92)

    rain_occurs = rng.random(len(dates)) < rain_prob
    # Rain amount on rainy days: gamma distribution scaled by season strength & annual total target
    shape_k = 1.8
    scale = (profile["rain_annual"] / (365 * rain_prob.mean())) / shape_k
    rain_amount = rng.gamma(shape_k, scale, size=len(dates)) * (0.5 + wet_signal)
    rainfall_mm = np.where(rain_occurs, rain_amount, 0.0)
    rainfall_mm = np.round(rainfall_mm, 1)

    humidity = np.clip(55 + 35 * wet_signal + rng.normal(0, 4, size=len(dates)), 30, 99)

    df = pd.DataFrame({
        "date": dates,
        "city": city,
        "latitude": profile["lat"],
        "temp_min_c": np.round(temp_min_c, 1),
        "temp_max_c": np.round(temp_max_c, 1),
        "temp_mean_c": np.round(temp_mean_c, 1),
        "rainfall_mm": rainfall_mm,
        "humidity_pct": np.round(humidity, 0),
    })
    return df


def main():
    rng = np.random.default_rng(SEED)
    frames = [build_city_series(city, profile, rng) for city, profile in CITY_PROFILES.items()]
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["city", "date"]).reset_index(drop=True)

    out_path = Path(__file__).resolve().parent.parent / "data" / "nigeria_weather_sample.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    print(f"Wrote {len(full):,} rows for {len(CITY_PROFILES)} cities -> {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
