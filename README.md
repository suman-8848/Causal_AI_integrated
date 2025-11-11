# Causal AI Integrated: Fair Risk Minimization for Treatment Effect Estimation

This repository implements the integration of Fair Risk Minimization under Causal Path-Specific Effect Constraints into deep learning models for treatment effect estimation.

## Overview

This project combines:
- **CFRNet** (Counterfactual Regression Network) for treatment effect estimation
- **Fair Risk Minimization** framework with path-specific effect constraints
- **DCE Module** (Differentiable Causal Estimation) for fairness-aware causal inference

## Architecture

The `FairCFRNet` model integrates:
- Representation network that takes `(X, A)` as input (covariates and sensitive attribute)
- Separate treatment and control heads for predicting `Y(1)` and `Y(0)`
- Propensity network for estimating `P(T|X)`
- DCE module for estimating path-specific effects `PE(A -> Y)` and canonical gradients

## Key Features

1. **Path-Specific Fairness**: Explicitly estimates and constrains the causal effect of sensitive attributes on outcomes through mediators
2. **Canonical Gradient**: Implements `d_theta` for fair risk minimization
3. **Closed-Form Adjustment**: Includes theoretical adjustment mechanism for fairness
4. **Modular Design**: Separate mediator model to avoid catastrophic interference

## Installation

```bash
pip install torch numpy pandas scikit-learn
```

## Usage

### Training FairCFRNet

```python
from scripts.train_fair_cfrnet import train_fair_cfrnet
from utils.data_loader import load_ihdp_data, preprocess_ihdp_data

# Load data
data = load_ihdp_data(fold=1)
T, Y, Y_cf, mu0, mu1, A, X = preprocess_ihdp_data(data)

# Train model
model, results = train_fair_cfrnet(
    X, T, Y, A,
    Y_cf=Y_cf,
    mu0=mu0,
    mu1=mu1,
    lambda_fairness=1.0,
    epochs=100,
    batch_size=128
)
```

### Model Architecture

```python
from models.fair_cfr_net import FairCFRNet

model = FairCFRNet(
    input_dim=24,          # Dimension of covariates X
    hidden_dim=200,        # Hidden dimension for representation
    alpha=1.0,             # Weight for IPM loss
    lambda_fairness=1.0,   # Weight for fairness constraint
    beta_fairness=0.5,     # Weight for fairness adjustment
    m_dim=1                # Dimension of mediator (1 for binary)
)
```

## Loss Function

The model optimizes:
```
L_total = L_pred + α·IPM + λ·L_fair
```

Where:
- `L_pred`: Prediction loss (MSE)
- `IPM`: Integral Probability Metric for treatment group balance
- `L_fair = |PE(A -> Y)| + β·||TE - TE*||²`: Fairness loss with path-specific constraints

## Project Structure

```
causal_ai_integrated/
├── models/
│   ├── cfr_net.py          # Base CFRNet implementation
│   ├── dce_module.py        # Differentiable Causal Estimation module
│   └── fair_cfr_net.py      # FairCFRNet with path-specific constraints
├── scripts/
│   ├── train_baseline.py    # Training script for baseline CFRNet
│   ├── train_fair_cfrnet.py # Training script for FairCFRNet
│   ├── evaluate_and_visualize.py
│   └── validate_implementation.py
└── utils/
    ├── data_loader.py       # Data loading utilities
    ├── metrics.py           # Evaluation metrics (PEHE, ATE, etc.)
    └── causal_graph.py      # Causal graph utilities
```

## Theoretical Foundation

This implementation is based on the framework from "Fair Risk Minimization under Causal Path-Specific Effect Constraints", integrating:
- Closed-form solutions for fair risk minimization
- AIPW (Augmented Inverse Probability Weighting) estimators
- Semiparametric estimation for robustness

## Evaluation Metrics

- **PEHE**: Precision in Estimation of Heterogeneous Effect
- **ATE Error**: Average Treatment Effect error
- **Path-Specific Effect**: `PE(A -> Y)` through mediators
- **Demographic Parity Gap**: Fairness metric

## Citation

If you use this code, please cite the relevant papers on fair risk minimization and CFRNet.

## License

[Add your license here]

## Contact

[Add contact information here]

