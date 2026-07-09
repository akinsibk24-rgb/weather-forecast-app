# Nigeria Weather Forecaster — Rain & Temperature

A desktop Python app (Tkinter + matplotlib) that visualizes historical rain
and temperature patterns for major Nigerian cities and produces a
short-range forecast (7 / 14 / 30 days) using a seasonal (harmonic
regression) model fit to the historical data.

Companion project style to the *Dave Pro Python Calc* GUI: Tkinter,
slate-blue/orange/red theme, single-screen desktop workflow.

## Features

- City picker: Lagos, Abuja, Kano, Port Harcourt, Ibadan, Enugu (easy to extend)
- Historical seasonal chart (avg. temperature + rainfall per month)
- Forecast chart + summary cards (tomorrow's outlook, expected rainfall
  total, rainy-day count)
- Swappable data source: bundled sample data out of the box, or your own
  CSV / real Kaggle dataset via "Load Different Dataset"

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_sample_data.py   # creates data/nigeria_weather_sample.csv
python main.py
```

`main.py` will auto-generate the sample dataset on first run if it's
missing, so `python main.py` alone is usually enough.

## Using real Kaggle data instead of the sample data

The app ships with a **synthetic sample dataset** (`data/nigeria_weather_sample.csv`),
built from published Nigerian climate normals (wet/dry season timing per
city, typical temperature ranges, annual rainfall totals). It's realistic
enough to demo the app immediately, but it is not observed data.

To use a real dataset from Kaggle:

1. Create a Kaggle account and API token:
   - Kaggle → your profile → **Account** → **Create New API Token**
   - This downloads `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json` (`chmod 600` on Linux/Mac) or
   `C:\Users\<you>\.kaggle\kaggle.json` on Windows.
3. Install the client: `pip install kaggle`
4. Run:
   ```bash
   python scripts/fetch_kaggle_data.py --dataset oyekanmiolamilekan/nigeria-cities-weather-forecast-data
   ```
   This downloads the dataset, extracts it, and reshapes it into
   `data/nigeria_weather_kaggle.csv` using the app's schema.
5. In the app, click **"Load Different Dataset"** and select that file
   (or set it as `DEFAULT_FILE` in `app/data_loader.py`).

Other Nigeria weather datasets worth trying on Kaggle:
- `kalusamuel/lagos-weather-dataset` (Lagos rainfall & air quality)
- Search ["Nigeria weather"](https://www.kaggle.com/search?q=nigeria+weather)
  or ["Nigeria rainfall"](https://www.kaggle.com/search?q=nigeria+rainfall)
  on kaggle.com/datasets for more, then pass `--dataset owner/slug`.

Every Kaggle dataset names its columns a bit differently. `fetch_kaggle_data.py`
includes a `COLUMN_ALIASES` auto-mapper for common variants (`tmin`/`tmax`,
`precip`/`rainfall`, etc.) — if it can't confidently map a dataset's columns
it will print the raw column names so you can extend the alias list.

### Expected CSV schema

Any CSV you point the app at needs at least these columns:

| column        | type   | notes                          |
|---------------|--------|---------------------------------|
| `date`        | date   | `YYYY-MM-DD`                    |
| `city`        | text   | e.g. `Lagos`                    |
| `temp_mean_c` | float  | daily mean temperature, °C      |
| `rainfall_mm` | float  | daily rainfall total, mm        |

Optional columns `temp_min_c`, `temp_max_c`, `humidity_pct` are used if
present.

## How the forecast works

Nigerian weather is strongly seasonal, so instead of a heavy ML stack the
app fits a compact **harmonic (Fourier) regression** per city:

```
temperature(t) ≈ b0 + b1·sin(ωt) + b2·cos(ωt) + b3·sin(2ωt) + b4·cos(2ωt) + b5·trend
```

Rainfall is modeled as *probability of rain on a given day-of-year*
(smoothed from history) × *expected rainfall amount on rainy days*
(log-space harmonic regression, since rainfall is non-negative/skewed).

The final forecast blends this seasonal ("climatology") baseline with the
last 14 days' observed anomaly, decaying that influence the further out
the forecast reaches — so short-range predictions track recent conditions,
and longer-range predictions fall back to the seasonal average. See
`app/forecaster.py` for the full implementation.

## Project structure

```
weather_forecast_nigeria/
├── main.py                        # entry point
├── requirements.txt
├── app/
│   ├── data_loader.py             # CSV loading & validation
│   ├── forecaster.py              # harmonic regression forecasting engine
│   └── gui.py                     # Tkinter GUI
├── data/
│   └── nigeria_weather_sample.csv # bundled sample dataset (~1MB)
├── scripts/
│   ├── generate_sample_data.py    # regenerates the sample dataset
│   └── fetch_kaggle_data.py       # pulls + reshapes real Kaggle data
```

## Notes on GitHub upload size

This repo (code + sample CSV) is well under GitHub's 25MB upload limit —
the sample dataset is roughly 1MB. If you fetch real Kaggle data and it's
large, keep it out of version control (already covered by `.gitignore` →
`data/raw_kaggle/`) and consider `.gitignore`-ing large reshaped CSVs too,
documenting the fetch command in the README instead so collaborators can
regenerate it locally.

## Limitations

This is a lightweight, explainable statistical forecaster for
learning/demo purposes — not a substitute for meteorological forecasts
(e.g. NiMet) for real-world planning or safety-critical decisions.
