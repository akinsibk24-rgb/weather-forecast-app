"""
data_loader.py
---------------
Loads the Nigeria rain/temperature CSV (either the bundled sample data
or a real Kaggle dataset that has been reshaped by fetch_kaggle_data.py)
and exposes clean helpers for the GUI and forecasting layer.

Expected CSV schema (columns, order doesn't matter):
    date          (YYYY-MM-DD)
    city          (string)
    temp_min_c    (float, optional)
    temp_max_c    (float, optional)
    temp_mean_c   (float, required)
    rainfall_mm   (float, required)
    humidity_pct  (float, optional)
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FILE = DATA_DIR / "nigeria_weather_sample.csv"
REQUIRED_COLUMNS = {"date", "city", "temp_mean_c", "rainfall_mm"}


class DataLoadError(Exception):
    pass


def available_datasets():
    """Return every CSV in the data/ folder, most recently modified first."""
    if not DATA_DIR.exists():
        return []
    files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def load_weather_data(path: Path = None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_FILE
    if not path.exists():
        raise DataLoadError(
            f"Dataset not found at {path}.\n"
            "Run `python scripts/generate_sample_data.py` to create the sample "
            "dataset, or `python scripts/fetch_kaggle_data.py` to pull real "
            "Kaggle data."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataLoadError(
            f"Dataset {path.name} is missing required column(s): {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_COLUMNS)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "city", "temp_mean_c", "rainfall_mm"])
    df["city"] = df["city"].astype(str).str.strip()
    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df


def list_cities(df: pd.DataFrame):
    return sorted(df["city"].unique().tolist())


def city_slice(df: pd.DataFrame, city: str) -> pd.DataFrame:
    return df[df["city"] == city].sort_values("date").reset_index(drop=True)


def summary_stats(df: pd.DataFrame, city: str) -> dict:
    sub = city_slice(df, city)
    if sub.empty:
        return {}
    last_year = sub["date"].max().year
    ytd = sub[sub["date"].dt.year == last_year]
    return {
        "records": len(sub),
        "date_range": (sub["date"].min().date(), sub["date"].max().date()),
        "avg_temp_c": round(sub["temp_mean_c"].mean(), 1),
        "avg_annual_rainfall_mm": round(
            sub.groupby(sub["date"].dt.year)["rainfall_mm"].sum().mean(), 1
        ),
        "hottest_month": sub.groupby(sub["date"].dt.month)["temp_mean_c"].mean().idxmax(),
        "wettest_month": sub.groupby(sub["date"].dt.month)["rainfall_mm"].sum().idxmax(),
    }
