"""
Synthetic time series generator for Neural ODE project.
Produces a non-linear trend + multi-period seasonality + heteroskedastic noise + occasional shocks.
Saves CSV with columns: `t`, `y`.
"""
import os
import argparse
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None


def generate_series(n=2000, seed=0, out_path=None):
    rng = np.random.RandomState(seed)
    t = np.arange(n).astype(float)

    # Nonlinear trend: quadratic growth plus a slow logistic-style saturation
    trend = 0.0005 * (t ** 2) / n + 2.0 / (1.0 + np.exp(-0.001 * (t - n / 2))) - 1.0

    # Seasonality: sum of two sinusoids with different periods and a slowly varying phase
    period1 = 50.0
    period2 = 200.0
    phase = 0.5 * np.sin(2 * np.pi * t / 800.0)
    season = 1.5 * np.sin(2 * np.pi * t / period1 + phase) + 0.7 * np.sin(2 * np.pi * t / period2)

    # Time-varying amplitude (makes the signal non-stationary)
    amp = 1.0 + 0.5 * np.sin(2 * np.pi * t / 500.0)

    # Heteroskedastic noise: variance changes over time
    base_sigma = 0.3
    sigma_t = base_sigma * (1.0 + 0.5 * np.sin(2 * np.pi * t / 300.0))
    noise = sigma_t * rng.randn(n)

    # Occasional shocks: Poisson arrivals with random magnitude
    shocks = np.zeros(n)
    jump_prob = 0.005
    num_jumps = rng.binomial(n, jump_prob)
    jump_indices = rng.choice(n, size=num_jumps, replace=False) if num_jumps > 0 else []
    for idx in jump_indices:
        shocks[idx:] += rng.normal(loc=0.0, scale=3.0)  # persistent step change

    # Compose the signal
    y = trend + amp * season + noise + shocks

    if pd is not None:
        df = pd.DataFrame({"t": t, "y": y})
        if out_path is not None:
            out_dir = os.path.dirname(out_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            df.to_csv(out_path, index=False)
        return df
    else:
        # Fallback: write CSV using numpy to avoid heavy dependency on pandas during initial steps
        if out_path is not None:
            out_dir = os.path.dirname(out_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            data = np.vstack([t, y]).T
            header = "t,y"
            np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.8f")
        # Return a lightweight dict-like object
        return {"t": t, "y": y}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic time series data")
    parser.add_argument("--n", type=int, default=2000, help="Number of time points (>=1000 recommended)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", type=str, default="data/synthetic.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_series(n=args.n, seed=args.seed, out_path=args.out)
    print(f"Wrote synthetic series with {len(df)} points to {args.out}")
