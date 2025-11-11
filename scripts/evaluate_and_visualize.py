"""
Evaluation and visualization script for comparing baseline and fair CFRNet.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_ihdp_data, preprocess_ihdp_data
from utils.metrics import pehe, compute_ite, demographic_parity_gap, compute_ate_error
from models.cfr_net import CFRNet
from models.fair_cfr_net import FairCFRNet
from sklearn.preprocessing import StandardScaler


def load_model_results(model_path, model_type='baseline'):
    """
    Load a trained model and return it along with scaler.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    model_type : str
        'baseline' or 'fair'
    
    Returns:
    --------
    model : torch.nn.Module
        Loaded model
    scaler : StandardScaler
        Feature scaler
    config : dict
        Model configuration
    """
    # Load checkpoint (weights_only=False to allow sklearn StandardScaler)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    scaler = checkpoint['scaler']
    input_dim = checkpoint['input_dim']
    alpha = checkpoint.get('alpha', 1.0)
    lambda_fairness = checkpoint.get('lambda_fairness', 0.0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'baseline':
        model = CFRNet(input_dim=input_dim, alpha=alpha)
    else:
        model = FairCFRNet(input_dim=input_dim, alpha=alpha, lambda_fairness=lambda_fairness)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    config = {
        'alpha': alpha,
        'lambda_fairness': lambda_fairness,
        'input_dim': input_dim
    }
    
    return model, scaler, config


def evaluate_model(model, X, T, Y, Y_cf, mu0, mu1, A, scaler, model_type='baseline'):
    """
    Evaluate a model and compute all metrics.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    X : np.ndarray or pd.DataFrame
        Covariates
    T : np.ndarray
        Treatment indicator
    Y : np.ndarray
        Observed outcomes
    Y_cf : np.ndarray
        Counterfactual outcomes
    A : np.ndarray
        Sensitive attribute
    scaler : StandardScaler
        Feature scaler
    model_type : str
        'baseline' or 'fair'
    
    Returns:
    --------
    results : dict
        Evaluation metrics
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X)
    T = np.asarray(T)
    Y = np.asarray(Y)
    Y_cf = np.asarray(Y_cf)
    A = np.asarray(A)
    
    # Normalize features
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        T_tensor = torch.FloatTensor(T).to(device)
        A_tensor = torch.FloatTensor(A).to(device)
        
        if model_type == 'baseline':
            ite_pred = model.compute_ite(X_tensor).cpu().numpy().flatten()
        else:
            ite_pred = model.compute_ite(X_tensor, A_tensor).cpu().numpy().flatten()
    
    # Calculate true ITE using ground truth mu1 - mu0
    if mu0 is not None and mu1 is not None:
        ite_true = mu1 - mu0
    else:
        # Fallback: Use Y_cf to calculate ITE
        ite_true = np.zeros_like(Y)
        treated_mask = T == 1
        control_mask = T == 0
        ite_true[treated_mask] = Y[treated_mask] - Y_cf[treated_mask]
        ite_true[control_mask] = Y_cf[control_mask] - Y[control_mask]
    
    # Compute metrics
    pehe_val = pehe(ite_true, ite_pred)
    
    ate_true = ite_true.mean()
    ate_pred = ite_pred.mean()
    ate_error = compute_ate_error(ate_true, ate_pred)
    
    # For fairness, get predictions
    with torch.no_grad():
        if model_type == 'baseline':
            y_pred, _ = model(X_tensor, T_tensor)
        else:
            y_pred, _, _ = model(X_tensor, A_tensor, T_tensor)
        y_pred = y_pred.cpu().numpy().flatten()
    
    dp_gap = demographic_parity_gap(y_pred, A)
    
    # Compute PE(A -> Y) from DCE module if available
    fairness_penalty = None
    if model_type == 'fair' and hasattr(model, 'dce_module'):
        with torch.no_grad():
            X_tensor_cpu = X_tensor.to(device)
            A_tensor_cpu = A_tensor.to(device)
            T_tensor_cpu = T_tensor.to(device)
            fairness_penalty = model.dce_module(X_tensor_cpu, A_tensor_cpu, T_tensor_cpu).item()
    
    results = {
        'pehe': pehe_val,
        'ate_error': ate_error,
        'ate_true': ate_true,
        'ate_pred': ate_pred,
        'demographic_parity_gap': dp_gap,
        'fairness_penalty': fairness_penalty,
        'ite_pred': ite_pred,
        'ite_true': ite_true,
    }
    
    return results


def create_tradeoff_plot(results_dict, save_path='results/fairness_accuracy_tradeoff.png'):
    """
    Create fairness-accuracy trade-off plot.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping lambda values to results
    save_path : str
        Path to save the plot
    """
    lambdas = []
    pehe_values = []
    fairness_penalties = []
    dp_gaps = []
    
    for lambda_val in sorted(results_dict.keys()):
        res = results_dict[lambda_val]
        if 'pehe' in res and res['pehe'] is not None:
            lambdas.append(lambda_val)
            pehe_values.append(res['pehe'])
            fairness_penalties.append(res.get('fairness_penalty', 0))
            dp_gaps.append(res.get('demographic_parity_gap', 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: PEHE vs Fairness Penalty (PE(A -> Y))
    ax1 = axes[0]
    ax1.plot(fairness_penalties, pehe_values, 'o-', linewidth=2, markersize=8)
    for i, lambda_val in enumerate(lambdas):
        ax1.annotate(f'λ={lambda_val}', 
                    (fairness_penalties[i], pehe_values[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    ax1.set_xlabel('Fairness Penalty (PE(A → Ŷ))', fontsize=12)
    ax1.set_ylabel('PEHE', fontsize=12)
    ax1.set_title('Fairness-Accuracy Trade-off\n(PE(A → Ŷ) vs PEHE)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PEHE vs Demographic Parity Gap
    ax2 = axes[1]
    ax2.plot(dp_gaps, pehe_values, 's-', linewidth=2, markersize=8, color='green')
    for i, lambda_val in enumerate(lambdas):
        ax2.annotate(f'λ={lambda_val}', 
                    (dp_gaps[i], pehe_values[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    ax2.set_xlabel('Demographic Parity Gap', fontsize=12)
    ax2.set_ylabel('PEHE', fontsize=12)
    ax2.set_title('Fairness-Accuracy Trade-off\n(Demographic Parity vs PEHE)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Trade-off plot saved to {save_path}")
    plt.close()


def create_loss_components_plot(results_dict, save_path='results/loss_components.png'):
    """
    Plot loss components during training for different lambda values.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping lambda values to results
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot training losses for different lambdas
    for lambda_val in sorted(results_dict.keys()):
        res = results_dict[lambda_val]
        if 'train_losses' in res:
            epochs = range(1, len(res['train_losses']) + 1)
            axes[0, 0].plot(epochs, res['train_losses'], label=f'λ={lambda_val}', linewidth=2)
            axes[0, 1].plot(epochs, res.get('train_pred_losses', []), 
                           label=f'λ={lambda_val}', linewidth=2)
            axes[1, 0].plot(epochs, res.get('train_ipm_losses', []), 
                           label=f'λ={lambda_val}', linewidth=2)
            axes[1, 1].plot(epochs, res.get('train_fairness_penalties', []), 
                           label=f'λ={lambda_val}', linewidth=2)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Prediction Loss')
    axes[0, 1].set_title('Prediction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IPM Loss')
    axes[1, 0].set_title('IPM Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Fairness Penalty')
    axes[1, 1].set_title('Fairness Penalty')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss components plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Load data
    print("Loading IHDP data...")
    data = load_ihdp_data(fold=1)
    T, Y, Y_cf, mu0, mu1, A, X = preprocess_ihdp_data(data)
    
    # Load grid search results
    results_file = 'results/grid_search_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results_summary = json.load(f)
        
        # Convert string keys to float
        results_dict = {float(k): v for k, v in results_summary.items()}
        
        print("\nCreating visualizations...")
        create_tradeoff_plot(results_dict)
        
        # Load full results for loss plots
        full_results = {}
        for lambda_val in results_dict.keys():
            model_path = f'results/fair_cfrnet_lambda_{lambda_val}.pth'
            if os.path.exists(model_path):
                # Re-evaluate to get full results
                model, scaler, config = load_model_results(model_path, model_type='fair')
                results = evaluate_model(model, X, T, Y, Y_cf, mu0, mu1, A, scaler, model_type='fair')
                # Try to get training losses from checkpoint if available
                # Load checkpoint (weights_only=False to allow sklearn StandardScaler)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                # Note: Training losses may not be saved in checkpoint
                full_results[lambda_val] = results
        
        # If we have training data, plot loss components
        # (This would require saving training losses during training)
        
    else:
        print(f"Results file not found: {results_file}")
        print("Please run train_fair_cfrnet.py first to generate results.")
    
    print("\nEvaluation and visualization completed!")

