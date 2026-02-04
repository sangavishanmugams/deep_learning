"""
Inspect synthetic time series and produce diagnostic plots.
Saves plots and a short stats file to the output folder.

Usage:
    python scripts/inspect_data.py --in data/synthetic.csv --out figures --window 50

Outputs (in `--out` dir):
 - series.png         : raw series
 - rolling.png        : series with rolling mean/std
 - fft.png            : FFT magnitude (dominant periods shown)
 - stats.txt          : summary statistics and top-3 periods

"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal


def inspect_series(csv_path, out_dir="figures", window=50):
    df = pd.read_csv(csv_path)
    if "y" not in df.columns:
        raise ValueError("Input CSV must contain a 'y' column with the series values")

    n = len(df)
    if n < 1000:
        print(f"Warning: series length {n} < 1000 (recommended >=1000)")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t = df["t"].values if "t" in df.columns else np.arange(n)
    y = df["y"].values

    # 1) Raw series
    plt.figure(figsize=(12, 4))
    sns.lineplot(x=t, y=y)
    plt.title("Raw time series")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "series.png"))
    plt.close()

    # 2) Rolling mean and std
    roll_mean = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean()
    roll_std = pd.Series(y).rolling(window=window, center=True, min_periods=1).std()

    plt.figure(figsize=(12, 4))
    sns.lineplot(x=t, y=y, label="y", alpha=0.6)
    sns.lineplot(x=t, y=roll_mean, label=f"rolling_mean({window})", color="orange")
    plt.fill_between(t, roll_mean - roll_std, roll_mean + roll_std, color="orange", alpha=0.2, label="rolling_std")
    plt.title("Series with rolling mean and std")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling.png"))
    plt.close()

    # 3) FFT on detrended signal to find dominant periods
    # Use scipy.signal.detrend to remove linear trend before FFT
    y_detrended = signal.detrend(y)
    fft_vals = np.fft.rfft(y_detrended)
    fft_freqs = np.fft.rfftfreq(n, d=1.0)  # sampling interval = 1
    fft_power = np.abs(fft_vals)

    # Ignore the zero frequency (mean) when finding peaks
    fft_power[0] = 0.0

    # Find the top 3 peaks in the FFT magnitude
    top_k = 3
    peak_indices = np.argsort(fft_power)[-top_k:][::-1]
    top_periods = []
    for idx in peak_indices:
        freq = fft_freqs[idx]
        period = np.nan if freq == 0 else 1.0 / freq
        top_periods.append(period)

    plt.figure(figsize=(10, 4))
    sns.lineplot(x=fft_freqs[1:], y=fft_power[1:])
    plt.xlabel("Frequency (1 / time)")
    plt.ylabel("FFT magnitude")
    plt.title("FFT (detrended signal) — inspect dominant frequencies")
    for p in top_periods:
        if not np.isnan(p) and p < n:
            plt.axvline(x=1.0 / p, color="red", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fft.png"))
    plt.close()

    # 4) Basic stats
    s = pd.Series(y)
    stats = {
        "n": int(n),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurt()),
    }

    stats_path = os.path.join(out_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write("Summary statistics\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTop periods (time units):\n")
        for i, p in enumerate(top_periods, 1):
            f.write(f"{i}: {p}\n")

    print(f"Saved plots and stats to {out_dir}")
    print("Top periods (time units):", top_periods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect synthetic time series and save diagnostics")
    parser.add_argument("--in", dest="inpath", type=str, default="data/synthetic.csv", help="Input CSV path")
    parser.add_argument("--out", dest="outdir", type=str, default="figures", help="Output directory for plots")
    parser.add_argument("--window", dest="window", type=int, default=50, help="Rolling window size for mean/std")
    args = parser.parse_args()

    inspect_series(args.inpath, out_dir=args.outdir, window=args.window)
