Project: Advanced Time Series Forecasting with Neural ODEs and Uncertainty Quantification

Quick start

1. Create a Python 3.8+ virtual environment and install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

2. Generate synthetic dataset (default 2000 points):

```bash
python scripts/generate_synthetic.py --n 2000 --out data/synthetic.csv
```

Files created in this phase
- `requirements.txt` — Python dependencies
- `scripts/generate_synthetic.py` — generator for reproducible synthetic data

Next steps
- Inspect the generated series and plots (Phase 3)
- Implement baseline model (Phase 4)
- Implement Neural ODE model (Phase 6)

