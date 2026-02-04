"""
Monte Carlo Dropout inference for uncertainty quantification.

Loads a trained Neural ODE model and runs MC Dropout by keeping dropout active
at inference time. Multiple forward passes sample dropout masks, producing a
distribution of predictions from which we compute mean and prediction intervals.

Usage:
    python scripts/mc_dropout_inference.py --model models/neural_ode_forecaster.pt --in data/synthetic.csv --seq_len 50 --mc_samples 100
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, Dataset

from models.neural_ode import NeuralODEForecaster


class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        self.series = series.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.series) - self.seq_len)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return x.reshape(-1), np.array([y], dtype=np.float32)


def mc_dropout_inference(model, device, loader, mc_samples=50, scaler=None):
    """
    Run MC Dropout: multiple forward passes with dropout active to sample
    the posterior distribution of predictions.
    
    Returns:
        mean_preds: [N] mean prediction per sample
        std_preds: [N] standard deviation (epistemic uncertainty)
        all_preds: [N, mc_samples] all sampled predictions
        trues: [N] true values
    """
    model.to(device)
    model.train()  # keep dropout active!
    
    all_preds_list = []
    trues_list = []
    
    with torch.no_grad():
        for sample_idx in range(mc_samples):
            preds = []
            trues = []
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                xb = xb.view(xb.size(0), xb.size(1))
                pred = model(xb)
                pred_next = pred[:, -1, :].cpu().numpy().ravel()
                true = yb.cpu().numpy().ravel()
                preds.append(pred_next)
                trues.append(true)
            
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            all_preds_list.append(preds)
            trues_list.append(trues)
    
    # Stack: [mc_samples, N] -> transpose to [N, mc_samples]
    all_preds = np.array(all_preds_list).T  # [N, mc_samples]
    trues = trues_list[0]  # all the same
    
    # Compute statistics
    mean_preds = all_preds.mean(axis=1)
    std_preds = all_preds.std(axis=1)
    
    # Inverse transform if scaler provided
    if scaler is not None:
        mean_preds = scaler.inverse_transform(mean_preds.reshape(-1, 1)).ravel()
        std_preds_scaled = scaler.scale_[0]  # scaling factor
        std_preds = std_preds * std_preds_scaled  # scale the std as well
        trues = scaler.inverse_transform(trues.reshape(-1, 1)).ravel()
        # also inverse-transform all samples for visualization
        all_preds_orig = scaler.inverse_transform(all_preds)
    else:
        all_preds_orig = all_preds
    
    return mean_preds, std_preds, all_preds_orig, trues


def compute_prediction_intervals(mean_preds, std_preds, z_score=1.96):
    """
    Compute prediction intervals: [mean - z*std, mean + z*std]
    z_score=1.96 corresponds to ~95% confidence interval for Gaussian.
    """
    lower = mean_preds - z_score * std_preds
    upper = mean_preds + z_score * std_preds
    return lower, upper


def compute_coverage(trues, lower, upper):
    """
    Compute empirical coverage: fraction of true values falling inside intervals.
    """
    inside = (trues >= lower) & (trues <= upper)
    coverage = inside.mean()
    return coverage


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/neural_ode_forecaster.pt")
    parser.add_argument("--in", dest="inpath", type=str, default="data/synthetic.csv")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--mc_samples", type=int, default=100, help="Number of MC Dropout samples")
    parser.add_argument("--z_score", type=float, default=1.96, help="Z-score for confidence interval (1.96 ~= 95%)")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Load data
    df = pd.read_csv(args.inpath)
    y = df["y"].values.reshape(-1, 1)

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y).ravel()

    # Prepare test dataloader (same split as training)
    n = len(y_scaled)
    test_frac = 0.15
    test_n = int(n * test_frac)
    train_n = int(n * (1 - test_frac - 0.15))
    test_series = y_scaled[train_n + int(n * 0.15) - args.seq_len:]
    test_ds = TimeSeriesDataset(test_series, args.seq_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralODEForecaster(seq_len=args.seq_len, encoder_hidden=64, latent_dim=16,
                                odefunc_hidden=64, decoder_hidden=32, dropout=0.1)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # Run MC Dropout
    print(f"Running MC Dropout with {args.mc_samples} samples...")
    mean_preds, std_preds, all_preds, trues = mc_dropout_inference(model, device, test_loader, 
                                                                      mc_samples=args.mc_samples, 
                                                                      scaler=scaler)
    
    # Compute prediction intervals
    lower, upper = compute_prediction_intervals(mean_preds, std_preds, z_score=args.z_score)
    coverage = compute_coverage(trues, lower, upper)

    print(f"Prediction interval coverage: {coverage:.2%} (target ~95%)")
    print(f"Mean std (epistemic uncertainty): {std_preds.mean():.4f}")
    print(f"Max std: {std_preds.max():.4f}, Min std: {std_preds.min():.4f}")

    # Plot: first 300 points with confidence bands
    plt.figure(figsize=(14, 5))
    nplot = min(300, len(trues))
    t_idx = np.arange(nplot)
    
    plt.plot(t_idx, trues[:nplot], 'ko-', label='true', markersize=3, linewidth=1)
    plt.plot(t_idx, mean_preds[:nplot], 'b-', label='mean prediction', linewidth=2)
    plt.fill_between(t_idx, lower[:nplot], upper[:nplot], alpha=0.3, color='blue', label=f'{int(args.z_score*100/1.96)}% PI')
    
    plt.legend()
    plt.title(f"Neural ODE with MC Dropout Uncertainty (coverage={coverage:.2%})")
    plt.xlabel("time step")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "mc_dropout_uncertainty.png"), dpi=150)
    plt.close()

    # Save uncertainty metrics
    with open(os.path.join(args.out_dir, "mc_dropout_metrics.txt"), "w") as f:
        f.write(f"MC Dropout Uncertainty Metrics\n")
        f.write(f"MC Samples: {args.mc_samples}\n")
        f.write(f"Z-score: {args.z_score}\n")
        f.write(f"Prediction Interval Coverage: {coverage:.4f}\n")
        f.write(f"Mean epistemic std: {std_preds.mean():.4f}\n")
        f.write(f"Max epistemic std: {std_preds.max():.4f}\n")
        f.write(f"Min epistemic std: {std_preds.min():.4f}\n")

    print("Saved MC Dropout plot and metrics.")


if __name__ == "__main__":
    main()
