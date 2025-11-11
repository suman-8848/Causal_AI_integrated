"""
Training script for FairCFRNet with DCE module.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fair_cfr_net import FairCFRNet
from utils.metrics import pehe, compute_ite, demographic_parity_gap_regression, compute_ate_error
from utils.data_loader import load_ihdp_data, preprocess_ihdp_data


class IHDPDataset(Dataset):
    """Dataset class for IHDP data."""
    def __init__(self, X, T, Y, A):
        self.X = torch.FloatTensor(X)
        self.T = torch.FloatTensor(T)
        self.Y = torch.FloatTensor(Y)
        self.A = torch.FloatTensor(A)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx], self.A[idx]


def train_fair_cfrnet(X, T, Y, A, Y_cf=None, mu0=None, mu1=None, lambda_fairness=1.0, val_split=0.2, 
                      batch_size=128, epochs=100, lr=0.001, alpha=1.0,
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      save_path='results/fair_cfrnet.pth'):
    """
    Train FairCFRNet model.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Covariates
    T : np.ndarray
        Treatment indicator
    Y : np.ndarray
        Observed outcomes
    A : np.ndarray
        Sensitive attribute
    Y_cf : np.ndarray, optional
        Counterfactual outcomes (for evaluation)
    lambda_fairness : float
        Weight for fairness penalty
    val_split : float
        Validation split ratio
    batch_size : int
        Batch size
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    alpha : float
        Weight for IPM loss
    device : str
        Device to use ('cuda' or 'cpu')
    save_path : str
        Path to save the trained model
    
    Returns:
    --------
    model : FairCFRNet
        Trained model
    results : dict
        Training and evaluation results
    """
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X)
    T = np.asarray(T)
    Y = np.asarray(Y)
    A = np.asarray(A)
    
    # Normalize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Train-validation split
    indices = np.arange(len(X_scaled))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    T_train, T_val = T[train_idx], T[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]
    A_train, A_val = A[train_idx], A[val_idx]
    
    # Create datasets and data loaders
    train_dataset = IHDPDataset(X_train, T_train, Y_train, A_train)
    val_dataset = IHDPDataset(X_val, T_val, Y_val, A_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = FairCFRNet(
        input_dim=input_dim,
        hidden_dim=200,
        alpha=alpha,
        lambda_fairness=lambda_fairness,
        beta_fairness=0.5,
        m_dim=1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_pred_losses = []
    train_ipm_losses = []
    train_fairness_penalties = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pred_loss = 0.0
        train_ipm_loss = 0.0
        train_fairness_penalty = 0.0
        
        for batch_X, batch_T, batch_Y, batch_A in train_loader:
            batch_X = batch_X.to(device)
            batch_T = batch_T.to(device)
            batch_Y = batch_Y.to(device)
            batch_A = batch_A.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - returns te_unconstrained, te_fair, pe_hat, d_theta, propensity, phi, y_pred
            te_unconstrained, te_fair, pe_hat, d_theta, propensity, phi, y_pred = model(batch_X, batch_A, batch_T)
            
            # Separate treated and control for IPM
            treated_mask = batch_T == 1
            control_mask = batch_T == 0
            
            if treated_mask.sum() > 0 and control_mask.sum() > 0:
                phi_t1 = phi[treated_mask]
                phi_t0 = phi[control_mask]
                ipm_loss = model.compute_ipm_loss(phi_t0, phi_t1)
            else:
                ipm_loss = torch.tensor(0.0, device=device)
            
            # Compute total loss (uses outcomes with mediator)
            total_loss, pred_loss, ipm_loss_val, constraint_loss_val = model.compute_loss(
                te_unconstrained, te_fair, batch_Y, batch_T, phi, pe_hat, ipm_loss, y_pred
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_pred_loss += pred_loss.item()
            train_ipm_loss += ipm_loss_val.item()
            train_fairness_penalty += constraint_loss_val.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pred_loss = 0.0
        val_ipm_loss = 0.0
        val_fairness_penalty = 0.0
        
        with torch.no_grad():
            for batch_X, batch_T, batch_Y, batch_A in val_loader:
                batch_X = batch_X.to(device)
                batch_T = batch_T.to(device)
                batch_Y = batch_Y.to(device)
                batch_A = batch_A.to(device)
                
                te_unconstrained, te_fair, pe_hat, d_theta, propensity, phi, y_pred = model(batch_X, batch_A, batch_T)
                
                # Separate treated and control
                treated_mask = batch_T == 1
                control_mask = batch_T == 0
                
                if treated_mask.sum() > 0 and control_mask.sum() > 0:
                    phi_t1 = phi[treated_mask]
                    phi_t0 = phi[control_mask]
                    ipm_loss = model.compute_ipm_loss(phi_t0, phi_t1)
                else:
                    ipm_loss = torch.tensor(0.0, device=device)
                
                total_loss, pred_loss, ipm_loss_val, constraint_loss_val = model.compute_loss(
                    te_unconstrained, te_fair, batch_Y, batch_T, phi, pe_hat, ipm_loss, y_pred
                )
                
                val_loss += total_loss.item()
                val_pred_loss += pred_loss.item()
                val_ipm_loss += ipm_loss_val.item()
                val_fairness_penalty += constraint_loss_val.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_pred_losses.append(train_pred_loss / len(train_loader))
        train_ipm_losses.append(train_ipm_loss / len(train_loader))
        train_fairness_penalties.append(train_fairness_penalty / len(train_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} "
                  f"(Pred: {train_pred_loss/len(train_loader):.4f}, "
                  f"IPM: {train_ipm_loss/len(train_loader):.4f}, "
                  f"Fair: {train_fairness_penalty/len(train_loader):.4f}) - "
                  f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler_X,
                'input_dim': input_dim,
                'alpha': alpha,
                'lambda_fairness': lambda_fairness,
            }, save_path)
    
    # Load best model (weights_only=False to allow sklearn StandardScaler)
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate on full dataset
    X_scaled_tensor = torch.FloatTensor(scaler_X.transform(X)).to(device)
    A_tensor = torch.FloatTensor(A).to(device)
    T_tensor = torch.FloatTensor(T).to(device)
    
    with torch.no_grad():
        te_unconstrained_full, te_fair_full, pe_hat_full, d_theta_full, propensity_full, phi_full, y_pred_full = model(
            X_scaled_tensor, A_tensor, T_tensor
        )
        ite_pred = model.compute_ite(X_scaled_tensor, A_tensor).cpu().numpy().flatten()
        fairness_penalty_full = pe_hat_full  # Use path-specific effect as fairness metric
    
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_pred_losses': train_pred_losses,
        'train_ipm_losses': train_ipm_losses,
        'train_fairness_penalties': train_fairness_penalties,
        'ite_pred': ite_pred,
        'fairness_penalty': fairness_penalty_full.item(),
        'lambda_fairness': lambda_fairness,
        'scaler': scaler_X,
    }
    
    # Compute metrics using ground truth mu1 - mu0
    # Always use mu1 - mu0 for IHDP dataset
    if mu0 is not None and mu1 is not None:
        # Use ground truth ITE: mu1 - mu0
        ite_true = mu1 - mu0
        pehe_val = pehe(ite_true, ite_pred)
    else:
        raise ValueError("IHDP dataset should have mu0 and mu1 for proper evaluation")
    
    if pehe_val is not None:
        results['pehe'] = pehe_val
        
        # ATE error
        if ite_true is not None:
            ate_true = ite_true.mean()
            ate_pred = ite_pred.mean()
            ate_error = compute_ate_error(ate_true, ate_pred)
            results['ate_error'] = ate_error
            results['ate_true'] = ate_true
            results['ate_pred'] = ate_pred
    
    # Compute fairness metrics
    y_pred_val = []
    with torch.no_grad():
        for batch_X, batch_T, _, batch_A in val_loader:
            batch_X = batch_X.to(device)
            batch_T = batch_T.to(device)
            batch_A = batch_A.to(device)
            te_unconstrained, te_fair, pe_hat, d_theta, propensity, phi, y_pred = model(batch_X, batch_A, batch_T)
            # y_pred already uses mediator in outcome network
            y_pred_val.append(y_pred.cpu().numpy())
    
    y_pred_val = np.concatenate(y_pred_val).flatten()
    # dp_gap = demographic_parity_gap(y_pred_val, A_val)
    # In train_fair_cfrnet.py, right before the call to the function
    print("\n--- DEBUGGING DEMOGRAPHIC PARITY ---")
    print(f"Shape of y_pred_val: {y_pred_val.shape}")
    print(f"Shape of A_val: {A_val.shape}")
    print(f"Unique values in A_val: {np.unique(A_val)}")
    print(f"Mean of y_pred_val for A=0: {y_pred_val[A_val == 0].mean()}")
    print(f"Mean of y_pred_val for A=1: {y_pred_val[A_val == 1].mean()}")
    print("------------------------------------\n")

    dp_gap = demographic_parity_gap_regression(y_pred_val, A_val)
    results['demographic_parity_gap'] = dp_gap
    
    print(f"\nTraining completed (lambda={lambda_fairness})!")
    if 'pehe' in results:
        print(f"PEHE: {results['pehe']:.4f}")
    if 'ate_error' in results:
        print(f"ATE Error: {results['ate_error']:.4f}")
    print(f"Demographic Parity Gap: {results['demographic_parity_gap']:.4f}")
    print(f"Fairness Penalty: {results['fairness_penalty']:.4f}")
    
    return model, results


def grid_search_lambda(X, T, Y, A, Y_cf=None, mu0=None, mu1=None, lambda_values=[0.1],
                       epochs=100, batch_size=128):
    """
    Perform grid search over lambda values.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Covariates
    T : np.ndarray
        Treatment indicator
    Y : np.ndarray
        Observed outcomes
    A : np.ndarray
        Sensitive attribute
    Y_cf : np.ndarray, optional
        Counterfactual outcomes
    lambda_values : list
        List of lambda values to test
    epochs : int
        Number of epochs per lambda
    batch_size : int
        Batch size
    
    Returns:
    --------
    all_results : dict
        Results for each lambda value
    """
    all_results = {}
    
    for lambda_val in lambda_values:
        print(f"\n{'='*60}")
        print(f"Training with lambda={lambda_val}")
        print(f"{'='*60}")
        
        save_path = f'results/fair_cfrnet_lambda_{lambda_val}.pth'
        model, results = train_fair_cfrnet(
            X, T, Y, A, Y_cf=Y_cf, mu0=mu0, mu1=mu1,
            lambda_fairness=lambda_val,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_path
        )
        
        all_results[lambda_val] = results
    
    # Save all results
    os.makedirs('results', exist_ok=True)
    results_summary = {}
    for lambda_val, res in all_results.items():
        results_summary[lambda_val] = {
            'pehe': res.get('pehe', None),
            'ate_error': res.get('ate_error', None),
            'demographic_parity_gap': res.get('demographic_parity_gap', None),
            'fairness_penalty': res.get('fairness_penalty', None),
        }
    
    with open('results/grid_search_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    # Load data
    print("Loading IHDP data...")
    data = load_ihdp_data(fold=1)
    T, Y, Y_cf, mu0, mu1, A, X = preprocess_ihdp_data(data)
    
    print(f"Data loaded: {len(X)} samples")
    print(f"Treatment rate: {T.mean():.2%}")
    print(f"Sensitive attribute distribution: A=0: {(A==0).sum()}, A=1: {(A==1).sum()}")
    
    # Grid search over lambda values
    print("\nStarting grid search over lambda values...")
    all_results = grid_search_lambda(
        X, T, Y, A, Y_cf=Y_cf,
        lambda_values=[0.1],
        epochs=100
    )
    
    print("\nGrid search completed!")
    print("\nSummary of results:")
    for lambda_val in sorted(all_results.keys()):
        res = all_results[lambda_val]
        print(f"\nLambda={lambda_val}:")
        if 'pehe' in res:
            print(f"  PEHE: {res['pehe']:.4f}")
        if 'demographic_parity_gap' in res:
            print(f"  Demographic Parity Gap: {res['demographic_parity_gap']:.4f}")
        print(f"  Fairness Penalty: {res.get('fairness_penalty', 'N/A'):.4f}")

