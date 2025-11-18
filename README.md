# Causal AI Integrated: Fair Risk Minimization for Treatment Effect Estimation

This repository implements the integration of Fair Risk Minimization under Causal Path-Specific Effect Constraints into deep learning models for treatment effect estimation.

## Overview

This project combines:
- **CFRNet** (Counterfactual Regression Network) for treatment effect estimation
- **Fair Risk Minimization** framework with path-specific effect constraints
- **DCE Module** (Differentiable Causal Estimation) for fairness-aware causal inference

## Project Structure

```
causal_ai_integrated/
├── models/
│   ├── cfr_net.py          # Base CFRNet implementation
│   ├── dce_module.py        # Differentiable Causal Estimation module
│   ├── fair_cfr_net.py      # FairCFRNet with path-specific constraints
│   ├── tar_net.py           # TARNet implementation
│   ├── naive_fair_cfr_net.py # Naive Fair CFRNet (A removed from input)
│   └── adversarial_cfr_net.py # Adversarial CFRNet with adversarial debiasing
├── scripts/
│   ├── train_baseline.py    # Training script for baseline CFRNet
│   ├── train_fair_cfrnet.py # Training script for FairCFRNet
│   ├── train_baselines.py   # Training script for all baseline models
│   ├── evaluate_and_visualize.py
│   └── validate_implementation.py
└── utils/
    ├── data_loader.py       # Data loading utilities
    ├── metrics.py           # Evaluation metrics (PEHE, ATE, fairness metrics)
    └── causal_graph.py      # Causal graph utilities
```
