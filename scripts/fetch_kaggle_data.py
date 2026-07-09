"""
fetch_kaggle_data.py
----------------------
Downloads a real Nigeria rain/temperature dataset from Kaggle using the
official Kaggle API, then reshapes it into the schema this app expects:

    date, city, temp_min_c, temp_max_c, temp_mean_c, rainfall_mm, humidity_pct

SETUP (one-time)
-----------------
1. Create a Kaggle account (if you don't have one): https://www.kaggle.com
2. Go to Account settings -> "Create New API Token". This downloads
   kaggle.json.
3. Place it at:
      Linux/Mac: ~/.kaggle/kaggle.json   (chmod 600 ~/.kaggle/kaggle.json)
      Windows:   C:\\Users\\<you>\\.kaggle\\kaggle.json
4. Install the client:
      pip install kaggle

USAGE
-----
    python scripts/fetch_kaggle_data.py --dataset oyekanmiolamilekan/nigeria-cities-weather-forecast-data

Other Nigeria-relevant datasets worth trying:
    kalusamuel/lagos-weather-dataset
    (search "Nigeria weather" or "Nigeria rainfall" on kaggle.com/datasets
    for more, and pass --dataset <owner>/<slug>)

WHY THIS IS A SEPARATE STEP
----------------------------
Kaggle requires an authenticated API call per their Terms of Service, and
datasets are updated independently by their authors, so this repo does not
bundle the raw Kaggle CSV. The bundled data/nigeria_weather_sample.csv is
a synthetic stand-in so the app runs immediately; running this script
replaces it with real observed data.

NOTE ON COLUMN MAPPING
------------------------
Every Kaggle weather dataset names its columns slightly differently. This
script includes a best-effort auto-mapper (see COLUMN_ALIASES below) and
will print a warning + show you the raw columns if something can't be
matched automatically, so you can adjust COLUMN_ALIASES for your chosen
dataset.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw_kaggle"
OUTPUT_FILE = DATA_DIR / "nigeria_weather_kaggle.csv"

# Map common column-name variants (lowercased) to our canonical schema.
COLUMN_ALIASES = {
    "date": ["date", "datetime", "date_time", "obs_date", "day"],
    "city": ["city", "location", "station", "place", "town"],
    "temp_min_c": ["temp_min_c", "tmin", "min_temp", "mintemp", "temperature_min"],
    "temp_max_c": ["temp_max_c", "tmax", "max_temp", "maxtemp", "temperature_max"],
    "temp_mean_c": ["temp_mean_c", "temp", "temperature", "avg_temp", "mean_temp", "tavg"],
    "rainfall_mm": ["rainfall_mm", "rainfall", "precip", "precipitation", "rain_mm", "rain"],
    "humidity_pct": ["humidity_pct", "humidity", "rh", "relative_humidity"],
}


def download(dataset_slug: str):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "The 'kaggle' package is not installed.\n"
            "Install it with:  pip install kaggle\n"
            "Then make sure ~/.kaggle/kaggle.json exists (see script docstring)."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading '{dataset_slug}' from Kaggle...")
    api.dataset_download_files(dataset_slug, path=str(RAW_DIR), unzip=False)

    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        sys.exit("Download finished but no .zip file was found in data/raw_kaggle/.")
    with zipfile.ZipFile(zips[-1]) as zf:
        zf.extractall(RAW_DIR)
    print(f"Extracted to {RAW_DIR}")


def _find_column(columns_lower, aliases):
    for alias in aliases:
        if alias in columns_lower:
            return columns_lower[alias]
    return None


def reshape(default_city: str = None):
    csvs = list(RAW_DIR.glob("*.csv"))
    if not csvs:
        sys.exit(f"No CSV files found in {RAW_DIR}. Did the download step succeed?")

    frames = []
    for csv_path in csvs:
        raw = pd.read_csv(csv_path)
        columns_lower = {c.lower().strip(): c for c in raw.columns}

        mapped = {}
        for target, aliases in COLUMN_ALIASES.items():
            src = _find_column(columns_lower, aliases)
            if src:
                mapped[target] = raw[src]

        if "date" not in mapped or "temp_mean_c" not in mapped and "temp_max_c" not in mapped:
            print(f"⚠️  Could not confidently map columns in {csv_path.name}.")
            print(f"   Raw columns were: {list(raw.columns)}")
            print("   Edit COLUMN_ALIASES in this script to match, then rerun with --reshape-only.")
            continue

        df = pd.DataFrame(mapped)
        if "temp_mean_c" not in df.columns and {"temp_min_c", "temp_max_c"}.issubset(df.columns):
            df["temp_mean_c"] = (df["temp_min_c"] + df["temp_max_c"]) / 2
        if "city" not in df.columns:
            df["city"] = default_city or csv_path.stem
        if "rainfall_mm" not in df.columns:
            df["rainfall_mm"] = 0.0

        frames.append(df)

    if not frames:
        sys.exit("No files could be reshaped. See warnings above.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[[c for c in COLUMN_ALIASES if c in combined.columns]]
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(combined):,} rows -> {OUTPUT_FILE}")
    print("Update app/data_loader.py's DEFAULT_FILE (or use 'Load Different Dataset' "
          "in the app) to point at this file.")


def main():
    parser = argparse.ArgumentParser(description="Fetch & reshape a Kaggle Nigeria weather dataset.")
    parser.add_argument("--dataset", default="oyekanmiolamilekan/nigeria-cities-weather-forecast-data",
                         help="Kaggle dataset slug, e.g. owner/dataset-name")
    parser.add_argument("--city", default=None, help="Fallback city name if the dataset has none")
    parser.add_argument("--reshape-only", action="store_true",
                         help="Skip download; just reshape files already in data/raw_kaggle/")
    args = parser.parse_args()

    if not args.reshape_only:
        download(args.dataset)
    reshape(default_city=args.city)


if __name__ == "__main__":
    main()
