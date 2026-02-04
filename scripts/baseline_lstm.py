"""
Baseline LSTM forecasting script.
- Loads `data/synthetic.csv` (generated earlier)
- Prepares sliding windows for supervised learning
- Trains a small PyTorch LSTM for one-step-ahead forecasting
- Evaluates RMSE and MAE on a held-out test set
- Saves model to `models/lstm_baseline.pt` and forecast plot to `figures/baseline_forecast.png`

Usage:
    python scripts/baseline_lstm.py --in data/synthetic.csv --seq_len 50 --epochs 50
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def make_dataloaders(series, seq_len, batch_size=64, val_frac=0.15, test_frac=0.15):
    n = len(series)
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    train_n = n - val_n - test_n

    train_series = series[:train_n]
    val_series = series[train_n - seq_len: train_n + val_n]  # overlap so windows form correctly
    test_series = series[train_n + val_n - seq_len:]

    train_ds = TimeSeriesDataset(train_series, seq_len)
    val_ds = TimeSeriesDataset(val_series, seq_len)
    test_ds = TimeSeriesDataset(test_series, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, (train_n, val_n, test_n)


def train(model, device, train_loader, val_loader, epochs=50, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"Epoch {ep}/{epochs} — train MSE: {avg_train:.6f}, val MSE: {avg_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model, device, loader, scaler=None):
    model.to(device)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).cpu().numpy().ravel()
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
    parser.add_argument("--in", dest="inpath", type=str, default="data/synthetic.csv")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args(argv)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("figures"):
        os.makedirs("figures", exist_ok=True)

    df = pd.read_csv(args.inpath)
    y = df["y"].values.reshape(-1, 1)

    # scale
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y).ravel()

    train_loader, val_loader, test_loader, splits = make_dataloaders(y_scaled, args.seq_len, batch_size=args.batch)
    print("Data splits (train,val,test):", splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_size=1, hidden_size=args.hidden)

    model = train(model, device, train_loader, val_loader, epochs=args.epochs)

    preds, trues, rmse, mae = evaluate(model, device, test_loader, scaler=scaler)
    print(f"Test RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # Save model
    torch.save(model.state_dict(), "models/lstm_baseline.pt")

    # Plot first 300 points of test predictions vs true
    plt.figure(figsize=(12, 4))
    nplot = min(300, len(preds))
    plt.plot(trues[:nplot], label="true")
    plt.plot(preds[:nplot], label="pred")
    plt.legend()
    plt.title(f"LSTM baseline — test RMSE={rmse:.4f}, MAE={mae:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_forecast.png"))
    plt.close()

    # write metrics
    with open(os.path.join(args.out_dir, "baseline_metrics.txt"), "w") as f:
        f.write(f"RMSE: {rmse}\nMAE: {mae}\n")

    print("Saved model and figures.")


if __name__ == "__main__":
    main()
