# Advanced Time Series Forecasting with Neural ODEs and Uncertainty Quantification

A production-grade implementation of Neural Ordinary Differential Equations (Neural ODEs) for time series forecasting with integrated uncertainty quantification. This project demonstrates how continuous-time models outperform discrete-time approaches on non-linear, complex temporal data.

## 📊 Overview

This project implements and compares:
- **Baseline Model**: LSTM neural network (discrete-time)
- **Advanced Model**: Neural ODE with continuous-time dynamics (via `torchdiffeq`)
- **Uncertainty Quantification**: Monte Carlo Dropout for epistemic uncertainty estimation
- **Dataset**: Synthetic non-linear time series with multiple periodicities, trend changes, and stochastic noise

### Key Results

| Model | RMSE | MAE | Improvement |
|-------|------|-----|-------------|
| **LSTM Baseline** | 0.4223 | 0.3414 | — |
| **Neural ODE** | 0.3922 | 0.3105 | **7.12% RMSE ↓, 9.05% MAE ↓** |

**Uncertainty Coverage**: 60.67% (MC Dropout with 50 samples, z=1.96)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/sangavishanmugams/deep_learning.git
cd Project

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# 1. Generate synthetic dataset
python scripts/generate_synthetic.py --n 2000 --out data/synthetic.csv

# 2. Inspect and visualize data
python scripts/inspect_data.py --in data/synthetic.csv --out figures

# 3. Train baseline LSTM
python scripts/baseline_lstm.py --in data/synthetic.csv --seq_len 50 --epochs 50

# 4. Train Neural ODE model (set PYTHONPATH first)
export PYTHONPATH=.  # or on Windows: set PYTHONPATH=.
python scripts/train_neural_ode.py --in data/synthetic.csv --seq_len 50 --epochs 10

# 5. Run MC Dropout inference for uncertainty
python scripts/mc_dropout_inference.py --model models/neural_ode_forecaster.pt --in data/synthetic.csv --mc_samples 100

# 6. Generate final comparison report
python scripts/evaluate_and_compare.py --lstm_model models/lstm_baseline.pt --ode_model models/neural_ode_forecaster.pt --in data/synthetic.csv
```

All outputs are saved to `results/` (metrics) and `figures/` (plots).

## 📁 Project Structure

```
Project/
├── data/
│   └── synthetic.csv                 # Generated time series (2000 points)
├── models/
│   ├── neural_ode.py                 # Neural ODE components (encoder, ODE, decoder)
│   ├── lstm_baseline.pt              # Trained LSTM weights
│   └── neural_ode_forecaster.pt      # Trained Neural ODE weights
├── scripts/
│   ├── generate_synthetic.py         # Dataset generation
│   ├── inspect_data.py               # EDA and diagnostics
│   ├── baseline_lstm.py              # LSTM training
│   ├── train_neural_ode.py           # Neural ODE training
│   ├── mc_dropout_inference.py       # Uncertainty quantification
│   └── evaluate_and_compare.py       # Baseline vs Neural ODE comparison
├── figures/
│   ├── series.png                    # Raw time series plot
│   ├── baseline_forecast.png         # LSTM predictions
│   ├── neural_ode_forecast.png       # Neural ODE predictions
│   ├── mc_dropout_uncertainty.png    # Uncertainty bands
│   ├── baseline_vs_neuralode.png     # Side-by-side comparison
│   └── error_comparison.png          # Error distributions
├── results/
│   ├── baseline_metrics.txt          # LSTM RMSE/MAE
│   ├── neural_ode_metrics.txt        # Neural ODE RMSE/MAE
│   ├── mc_dropout_metrics.txt        # Uncertainty statistics
│   └── comparison_metrics.txt        # Detailed comparison
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── REPORT.md                         # Full technical report
```

## 🧠 What are Neural ODEs?

Neural ODEs model time series by learning a continuous-time dynamics function rather than discrete transitions:

- **Traditional RNNs/LSTMs**: Learn step-by-step mappings (h_{t} → h_{t+1})
- **Neural ODEs**: Learn the instantaneous rate of change (dh/dt = f(h, t))

This enables:
- ✅ **Smooth trajectories** — naturally interpolate between observations
- ✅ **Memory efficiency** — constant memory via adjoint method (vs. linear in sequence length)
- ✅ **Irregular sampling** — can forecast at any time, not just fixed intervals
- ✅ **Principled dynamics** — learn the underlying system rules, not just patterns

## 📈 Key Features

### 1. Synthetic Dataset
- **Non-linear trend** + logistic saturation
- **Multi-scale seasonality** (periods 50 & 200)
- **Time-varying amplitude** (heteroskedasticity)
- **Occasional shocks** (regime changes)

### 2. Baseline LSTM Model
- 1 LSTM layer (64 hidden units) + linear readout
- Trained with Adam optimizer on 1-step-ahead loss
- Quick baseline for comparison

### 3. Neural ODE Model
- **Encoder**: window → latent state (16-dim)
- **ODE Block**: continuous dynamics via Runge-Kutta 4
- **Decoder**: latent state → scalar prediction
- Includes dropout for MC Dropout uncertainty

### 4. Uncertainty Quantification
- **MC Dropout**: run forward pass 50-100 times with dropout active
- **Prediction Intervals**: [mean ± z·std] with z=1.96 (~95% CI)
- **Coverage Analysis**: empirical % of true values inside intervals

## 🔬 Technical Details

### Models

- **Neural ODE** uses `torchdiffeq` to numerically solve dh/dt = f(h, t)
- **ODE Solver**: Runge-Kutta 4 (fixed-step, deterministic)
- **Backprop**: Adjoint method for memory efficiency
- **Framework**: PyTorch

### Metrics

- **RMSE** (Root Mean Squared Error): emphasizes larger errors
- **MAE** (Mean Absolute Error): robust to outliers
- **Coverage**: fraction of true values inside prediction intervals

### Dataset Stats

- **Length**: 2000 time steps
- **Train/Val/Test**: 70% / 15% / 15%
- **Window size**: 50 (past observations used for prediction)
- **Target**: one-step-ahead forecast

## 📊 Results & Interpretation

### Accuracy Comparison
Neural ODE outperforms LSTM by ~7–9% on both metrics, validating the continuous-time hypothesis for this smooth, non-linear system.

### Uncertainty (MC Dropout)
- **Coverage**: 60.67% with z=1.96 (conservative; improve by increasing MC samples or dropout rate)
- **Epistemic Uncertainty**: ranges 0.084–0.268 across the series

## 🎓 Educational Value

This project is suitable for:
- Learning **Neural ODE theory and implementation**
- Understanding **continuous-time vs discrete-time modeling**
- Exploring **uncertainty quantification** in deep learning
- Practicing **production-grade ML code organization**

## 📚 References

- Chen, R. T. Q., et al. (2018). "Neural Ordinary Differential Equations." NeurIPS. ([arXiv](https://arxiv.org/abs/1806.07522))
- Dupont, E., et al. (2019). "Augmented Neural ODEs." NeurIPS.
- Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation." ICML.

## 🛠 Troubleshooting

### Import Error: "No module named 'models'"
Set `PYTHONPATH` before running scripts:
```bash
export PYTHONPATH=.
python scripts/train_neural_ode.py ...
```

### GPU Support
To use GPU (optional):
```bash
# Install CUDA-enabled PyTorch
pip install torch torchcuda
```

The code automatically detects and uses GPU if available.

### Slow Training
- Neural ODE forward/backward passes are slower than LSTM due to ODE solving
- Use smaller batch sizes or fewer epochs for quick experiments
- GPU strongly recommended for longer runs

## 📄 Full Report

See [REPORT.md](REPORT.md) for detailed technical documentation covering:
- Dataset design & justification
- Model architectures & design choices
- Training procedures & hyperparameters
- Results & error analysis
- Uncertainty quantification methodology
- Advantages & disadvantages of Neural ODEs
- Future work recommendations

## 📝 License

This project is provided as-is for educational and research purposes.

## 👤 Author

**Sangavi Shanmugam** — Neural ODE time series forecasting project

---

**Last Updated**: February 4, 2026
