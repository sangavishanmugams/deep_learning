"""
Phase 8 — Comprehensive Evaluation & Comparison.

Loads both baseline (LSTM) and Neural ODE models, runs inference on the test set,
and produces a detailed comparison report with visualizations.

Usage:
    python scripts/evaluate_and_compare.py --lstm_model models/lstm_baseline.pt --ode_model models/neural_ode_forecaster.pt --in data/synthetic.csv --seq_len 50
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
from torch.utils.data import DataLoader, Dataset

from models.neural_ode import NeuralODEForecaster
from scripts.baseline_lstm import LSTMForecaster


class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        self.series = series.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.series) - self.seq_len)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return x.reshape(-1, 1), np.array([y], dtype=np.float32)


def evaluate_model(model, device, loader, model_name, scaler=None):
    """Evaluate a model and return metrics."""
    model.to(device)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if model_name == "LSTM":
                pred = model(xb).cpu().numpy().ravel()
            else:  # Neural ODE
                pred = model(xb.view(xb.size(0), xb.size(1)))[:, -1, :].cpu().numpy().ravel()
            true = yb.cpu().numpy().ravel()
            preds.append(pred)
            trues.append(true)
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
        trues = scaler.inverse_transform(trues.reshape(-1, 1)).ravel()
    
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return preds, trues, rmse, mae


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm_model", type=str, default="models/lstm_baseline.pt")
    parser.add_argument("--ode_model", type=str, default="models/neural_ode_forecaster.pt")
    parser.add_argument("--in", dest="inpath", type=str, default="data/synthetic.csv")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Load data
    df = pd.read_csv(args.inpath)
    y = df["y"].values.reshape(-1, 1)

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y).ravel()

    # Prepare test dataloader
    n = len(y_scaled)
    test_frac = 0.15
    val_frac = 0.15
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    train_n = n - val_n - test_n
    test_series = y_scaled[train_n + val_n - args.seq_len:]
    test_ds = TimeSeriesDataset(test_series, args.seq_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate LSTM baseline
    print("Evaluating LSTM baseline...")
    lstm_model = LSTMForecaster(input_size=1, hidden_size=64)
    lstm_model.load_state_dict(torch.load(args.lstm_model, map_location=device))
    lstm_preds, trues, lstm_rmse, lstm_mae = evaluate_model(lstm_model, device, test_loader, "LSTM", scaler=scaler)

    # Evaluate Neural ODE
    print("Evaluating Neural ODE...")
    ode_model = NeuralODEForecaster(seq_len=args.seq_len, encoder_hidden=64, latent_dim=16,
                                     odefunc_hidden=64, decoder_hidden=32, dropout=0.1)
    ode_model.load_state_dict(torch.load(args.ode_model, map_location=device))
    ode_preds, _, ode_rmse, ode_mae = evaluate_model(ode_model, device, test_loader, "Neural ODE", scaler=scaler)

    # Comparison report
    print("\n" + "="*60)
    print("COMPARISON REPORT: LSTM Baseline vs Neural ODE")
    print("="*60)
    print(f"\nLSTM Baseline:")
    print(f"  RMSE: {lstm_rmse:.6f}")
    print(f"  MAE:  {lstm_mae:.6f}")
    print(f"\nNeural ODE:")
    print(f"  RMSE: {ode_rmse:.6f}")
    print(f"  MAE:  {ode_mae:.6f}")
    print(f"\nImprovement (Neural ODE vs LSTM):")
    rmse_improvement = (lstm_rmse - ode_rmse) / lstm_rmse * 100
    mae_improvement = (lstm_mae - ode_mae) / lstm_mae * 100
    print(f"  RMSE: {rmse_improvement:.2f}% better")
    print(f"  MAE:  {mae_improvement:.2f}% better")
    print("="*60 + "\n")

    # Save comparison metrics
    with open(os.path.join(args.out_dir, "comparison_metrics.txt"), "w") as f:
        f.write("Baseline vs Neural ODE Comparison\n")
        f.write("="*60 + "\n")
        f.write(f"LSTM Baseline:\n")
        f.write(f"  RMSE: {lstm_rmse}\n")
        f.write(f"  MAE: {lstm_mae}\n")
        f.write(f"\nNeural ODE:\n")
        f.write(f"  RMSE: {ode_rmse}\n")
        f.write(f"  MAE: {ode_mae}\n")
        f.write(f"\nImprovement:\n")
        f.write(f"  RMSE improvement: {rmse_improvement:.2f}%\n")
        f.write(f"  MAE improvement: {mae_improvement:.2f}%\n")

    # Comparison plot: first 300 points
    plt.figure(figsize=(14, 5))
    nplot = min(300, len(trues))
    t_idx = np.arange(nplot)
    
    plt.plot(t_idx, trues[:nplot], 'k-', label='true', linewidth=2)
    plt.plot(t_idx, lstm_preds[:nplot], 'r--', label=f'LSTM (RMSE={lstm_rmse:.4f})', linewidth=1.5, alpha=0.7)
    plt.plot(t_idx, ode_preds[:nplot], 'b--', label=f'Neural ODE (RMSE={ode_rmse:.4f})', linewidth=1.5, alpha=0.7)
    
    plt.legend(fontsize=11)
    plt.title("LSTM Baseline vs Neural ODE Forecasts")
    plt.xlabel("time step")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_vs_neuralode.png"), dpi=150)
    plt.close()

    # Error distribution plot
    lstm_errors = np.abs(lstm_preds - trues)
    ode_errors = np.abs(ode_preds - trues)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(lstm_errors, bins=40, alpha=0.7, label='LSTM', color='red')
    axes[0].hist(ode_errors, bins=40, alpha=0.7, label='Neural ODE', color='blue')
    axes[0].set_xlabel("Absolute Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Prediction Error Distributions")
    axes[0].legend()
    
    axes[1].boxplot([lstm_errors, ode_errors], labels=['LSTM', 'Neural ODE'])
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("Error Box Plot")
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "error_comparison.png"), dpi=150)
    plt.close()

    print("Saved comparison plots and metrics.")


if __name__ == "__main__":
    main()
