"""
gui.py
-------
Desktop GUI for the Nigeria Rain & Temperature Forecaster.

Built with Tkinter (matching the "Dave Pro" project's toolkit choice) plus
matplotlib embedded via FigureCanvasTkAgg for the charts. Color palette
mirrors the slate-blue / orange / red token scheme used in the reference
calculator project.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.data_loader import (
    load_weather_data, list_cities, city_slice, summary_stats, DataLoadError, DEFAULT_FILE
)
from app.forecaster import fit_city_model, forecast

# ---- Theme (slate-blue / orange / red, matching the reference project) ----
BG_DARK = "#1f2937"       # slate-800
BG_PANEL = "#273449"      # slightly lighter panel
ACCENT_BLUE = "#3b82f6"   # slate-blue accent
ACCENT_ORANGE = "#f97316" # orange accent (rainfall)
ACCENT_RED = "#ef4444"    # red accent (alerts / hot temps)
TEXT_LIGHT = "#e5e7eb"
TEXT_MUTED = "#9ca3af"
FONT_HEADER = ("Segoe UI", 16, "bold")
FONT_SUB = ("Segoe UI", 10)
FONT_LABEL = ("Segoe UI", 10, "bold")


class WeatherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nigeria Weather Forecaster — Rain & Temperature")
        self.geometry("1120x700")
        self.configure(bg=BG_DARK)
        self.minsize(980, 620)

        self.df = None
        self.current_data_path = DEFAULT_FILE

        self._build_style()
        self._build_layout()
        self._load_data(self.current_data_path)

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("TLabel", background=BG_DARK, foreground=TEXT_LIGHT, font=FONT_SUB)
        style.configure("Panel.TLabel", background=BG_PANEL, foreground=TEXT_LIGHT, font=FONT_SUB)
        style.configure("Header.TLabel", background=BG_DARK, foreground=TEXT_LIGHT, font=FONT_HEADER)
        style.configure("Muted.TLabel", background=BG_PANEL, foreground=TEXT_MUTED, font=FONT_SUB)
        style.configure("Stat.TLabel", background=BG_PANEL, foreground=ACCENT_BLUE, font=("Segoe UI", 20, "bold"))
        style.configure(
            "Accent.TButton", font=FONT_LABEL, padding=8,
            background=ACCENT_BLUE, foreground="white"
        )
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        style.configure("TCombobox", fieldbackground=BG_PANEL, background=BG_PANEL, foreground="black")

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(14, 6))
        ttk.Label(header, text="🌦  Nigeria Weather Forecaster", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="Rain & Temperature — historical data + short-range forecast",
                  style="TLabel", foreground=TEXT_MUTED).pack(side="left", padx=(12, 0), pady=(6, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        self._build_control_panel(body)
        self._build_display_panel(body)

        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self, textvariable=self.status_var, style="TLabel", foreground=TEXT_MUTED)
        status.pack(fill="x", padx=16, pady=(0, 8), anchor="w")

    def _build_control_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_PANEL, width=280)
        panel.pack(side="left", fill="y", padx=(0, 12))
        panel.pack_propagate(False)

        pad = dict(padx=16, pady=(14, 4))

        ttk.Label(panel, text="City", style="Panel.TLabel").pack(anchor="w", **pad)
        self.city_var = tk.StringVar()
        self.city_combo = ttk.Combobox(panel, textvariable=self.city_var, state="readonly")
        self.city_combo.pack(fill="x", padx=16)
        self.city_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        ttk.Label(panel, text="Forecast horizon", style="Panel.TLabel").pack(anchor="w", **pad)
        self.horizon_var = tk.StringVar(value="7 days")
        horizon_combo = ttk.Combobox(
            panel, textvariable=self.horizon_var, state="readonly",
            values=["7 days", "14 days", "30 days"]
        )
        horizon_combo.pack(fill="x", padx=16)
        horizon_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        ttk.Button(panel, text="🔄  Refresh Forecast", style="Accent.TButton",
                   command=self.refresh).pack(fill="x", padx=16, pady=(18, 6))
        ttk.Button(panel, text="📂  Load Different Dataset",
                   command=self.load_dataset_dialog).pack(fill="x", padx=16, pady=4)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=16, pady=14)

        ttk.Label(panel, text="City Snapshot", style="Panel.TLabel").pack(anchor="w", padx=16)
        self.stats_frame = tk.Frame(panel, bg=BG_PANEL)
        self.stats_frame.pack(fill="x", padx=16, pady=8)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=16, pady=14)
        ttk.Label(
            panel,
            text="Data source:\nsample climate-normal data.\nSwap in real Kaggle data via\nscripts/fetch_kaggle_data.py",
            style="Muted.TLabel", justify="left"
        ).pack(anchor="w", padx=16)

    def _build_display_panel(self, parent):
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        # Forecast summary cards
        self.cards_frame = tk.Frame(right, bg=BG_DARK)
        self.cards_frame.pack(fill="x", pady=(0, 10))

        # Tabs: Historical trend / Forecast chart
        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        self.hist_tab = tk.Frame(notebook, bg=BG_DARK)
        self.forecast_tab = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(self.hist_tab, text="Historical Trend")
        notebook.add(self.forecast_tab, text="Forecast")

        self.hist_fig = Figure(figsize=(6.4, 4.2), dpi=100, facecolor=BG_DARK)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.hist_tab)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        self.fc_fig = Figure(figsize=(6.4, 4.2), dpi=100, facecolor=BG_DARK)
        self.fc_canvas = FigureCanvasTkAgg(self.fc_fig, master=self.forecast_tab)
        self.fc_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    # ------------------------------------------------------------------ #
    # Data handling
    # ------------------------------------------------------------------ #
    def _load_data(self, path):
        try:
            self.df = load_weather_data(path)
            self.current_data_path = path
        except DataLoadError as e:
            messagebox.showerror("Data Error", str(e))
            return

        cities = list_cities(self.df)
        self.city_combo["values"] = cities
        if cities:
            self.city_var.set(cities[0])
        self.status_var.set(f"Loaded {len(self.df):,} records from {path.name} — {len(cities)} cities.")
        self.refresh()

    def load_dataset_dialog(self):
        path = filedialog.askopenfilename(
            title="Select weather CSV", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            from pathlib import Path
            self._load_data(Path(path))

    def _horizon_days(self):
        return int(self.horizon_var.get().split()[0])

    # ------------------------------------------------------------------ #
    # Refresh: stats, cards, charts
    # ------------------------------------------------------------------ #
    def refresh(self):
        if self.df is None or not self.city_var.get():
            return
        city = self.city_var.get()
        sub = city_slice(self.df, city)
        if sub.empty:
            return

        stats = summary_stats(self.df, city)
        self._render_stats(stats)

        model = fit_city_model(sub)
        fc = forecast(model, horizon_days=self._horizon_days())

        self._render_cards(fc)
        self._render_history_chart(sub, city)
        self._render_forecast_chart(fc, city)

        self.status_var.set(
            f"{city}: {stats['records']:,} historical records "
            f"({stats['date_range'][0]} → {stats['date_range'][1]}). "
            f"Forecast generated for next {self._horizon_days()} days."
        )

    def _render_stats(self, stats):
        for w in self.stats_frame.winfo_children():
            w.destroy()
        if not stats:
            return
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        rows = [
            ("Avg. temperature", f"{stats['avg_temp_c']} °C"),
            ("Avg. annual rainfall", f"{stats['avg_annual_rainfall_mm']:,.0f} mm"),
            ("Hottest month", month_names[stats["hottest_month"]]),
            ("Wettest month", month_names[stats["wettest_month"]]),
        ]
        for label, value in rows:
            row = tk.Frame(self.stats_frame, bg=BG_PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_SUB).pack(side="left")
            tk.Label(row, text=value, bg=BG_PANEL, fg=TEXT_LIGHT, font=FONT_LABEL).pack(side="right")

    def _render_cards(self, fc):
        for w in self.cards_frame.winfo_children():
            w.destroy()

        avg_temp = fc["forecast_temp_c"].mean()
        total_rain = fc["forecast_rainfall_mm"].sum()
        rainy_days = (fc["rain_probability_pct"] >= 50).sum()
        tomorrow = fc.iloc[0]

        cards = [
            ("Tomorrow", f"{tomorrow['forecast_temp_c']:.1f} °C",
             f"{tomorrow['rain_probability_pct']:.0f}% chance of rain", ACCENT_BLUE),
            (f"Avg. temp ({len(fc)}d)", f"{avg_temp:.1f} °C", "forecast average", ACCENT_BLUE),
            (f"Total rainfall ({len(fc)}d)", f"{total_rain:.0f} mm", "expected accumulation", ACCENT_ORANGE),
            ("Rainy days", f"{rainy_days} / {len(fc)}", "≥50% chance of rain", ACCENT_RED),
        ]
        for title, value, sub, color in cards:
            card = tk.Frame(self.cards_frame, bg=BG_PANEL, highlightbackground=color,
                             highlightthickness=2, padx=14, pady=10)
            card.pack(side="left", fill="both", expand=True, padx=6)
            tk.Label(card, text=title, bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_SUB).pack(anchor="w")
            tk.Label(card, text=value, bg=BG_PANEL, fg=color, font=("Segoe UI", 18, "bold")).pack(anchor="w")
            tk.Label(card, text=sub, bg=BG_PANEL, fg=TEXT_MUTED, font=("Segoe UI", 8)).pack(anchor="w")

    def _render_history_chart(self, sub, city):
        self.hist_fig.clear()
        # Monthly aggregation over full history for a readable seasonal picture
        monthly = sub.copy()
        monthly["month"] = monthly["date"].dt.month
        temp_by_month = monthly.groupby("month")["temp_mean_c"].mean()
        rain_by_month = monthly.groupby("month")["rainfall_mm"].sum() / monthly["date"].dt.year.nunique()

        ax1 = self.hist_fig.add_subplot(111)
        ax1.set_facecolor(BG_DARK)
        months = list(range(1, 13))
        ax1.bar(months, rain_by_month.reindex(months, fill_value=0), color=ACCENT_ORANGE, alpha=0.75, label="Avg. rainfall (mm)")
        ax1.set_ylabel("Rainfall (mm)", color=ACCENT_ORANGE)
        ax1.tick_params(axis="y", colors=ACCENT_ORANGE)
        ax1.tick_params(axis="x", colors=TEXT_LIGHT)
        ax1.set_xticks(months)
        ax1.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

        ax2 = ax1.twinx()
        ax2.plot(months, temp_by_month.reindex(months), color=ACCENT_BLUE, marker="o", linewidth=2, label="Avg. temperature (°C)")
        ax2.set_ylabel("Temperature (°C)", color=ACCENT_BLUE)
        ax2.tick_params(axis="y", colors=ACCENT_BLUE)

        ax1.set_title(f"{city} — Average Seasonal Pattern", color=TEXT_LIGHT)
        for spine in list(ax1.spines.values()) + list(ax2.spines.values()):
            spine.set_color(TEXT_MUTED)
        self.hist_fig.tight_layout()
        self.hist_canvas.draw()

    def _render_forecast_chart(self, fc, city):
        self.fc_fig.clear()
        ax1 = self.fc_fig.add_subplot(111)
        ax1.set_facecolor(BG_DARK)
        x = fc["date"].dt.strftime("%b %d")
        ax1.bar(x, fc["forecast_rainfall_mm"], color=ACCENT_ORANGE, alpha=0.8, label="Rainfall (mm)")
        ax1.set_ylabel("Rainfall (mm)", color=ACCENT_ORANGE)
        ax1.tick_params(axis="y", colors=ACCENT_ORANGE)
        ax1.tick_params(axis="x", colors=TEXT_LIGHT, rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(x, fc["forecast_temp_c"], color=ACCENT_BLUE, marker="o", linewidth=2, label="Temp (°C)")
        ax2.set_ylabel("Temperature (°C)", color=ACCENT_BLUE)
        ax2.tick_params(axis="y", colors=ACCENT_BLUE)

        ax1.set_title(f"{city} — {len(fc)}-Day Forecast", color=TEXT_LIGHT)
        for spine in list(ax1.spines.values()) + list(ax2.spines.values()):
            spine.set_color(TEXT_MUTED)
        self.fc_fig.tight_layout()
        self.fc_canvas.draw()


def run():
    app = WeatherApp()
    app.mainloop()
