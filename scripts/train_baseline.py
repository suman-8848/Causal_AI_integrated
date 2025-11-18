"""
Training script for baseline models (TARNet, Naive Fair CFRNet, Adversarial CFRNet).
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

from models.cfr_net import CFRNet
from models.tar_net import TARNet
from models.naive_fair_cfr_net import NaiveFairCFRNet
from models.adversarial_cfr_net import AdversarialCFRNet
from utils.metrics import pehe, compute_ite, demographic_parity_gap_regression, compute_ate_error
from utils.data_loader import load_ihdp_data, preprocess_ihdp_data


class IHDPDataset(Dataset):
    """Dataset class for IHDP data."""
    def __init__(self, X, T, Y, A=None):
        self.X = torch.FloatTensor(X)
        self.T = torch.FloatTensor(T)
        self.Y = torch.FloatTensor(Y)
        if A is not None:
            self.A = torch.FloatTensor(A)
        else:
            self.A = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.A is not None:
            return self.X[idx], self.T[idx], self.Y[idx], self.A[idx]
        else:
            return self.X[idx], self.T[idx], self.Y[idx]


def train_baseline_model(model_type, X, T, Y, A, Y_cf=None, mu0=None, mu1=None, val_split=0.2, 
                        batch_size=128, epochs=100, lr=0.001, alpha=1.0, alpha_adv=1.0,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        save_path='results/baseline_model.pth'):
    """
    Train a baseline model.
    
    Parameters:
    -----------
    model_type : str
        Type of model to train ('cfrnet', 'tarnet', 'naive_fair', 'adversarial')
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
    mu0 : np.ndarray, optional
        Ground truth potential outcome Y(0)
    mu1 : np.ndarray, optional
        Ground truth potential outcome Y(1)
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
    alpha_adv : float
        Weight for adversarial loss (only for adversarial model)
    device : str
        Device to use ('cuda' or 'cpu')
    save_path : str
        Path to save the trained model
    
    Returns:
    --------
    model : torch.nn.Module
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
    
    # Initialize model based on type
    input_dim = X_train.shape[1]
    
    if model_type == 'cfrnet':
        model = CFRNet(input_dim=input_dim, alpha=alpha).to(device)
    elif model_type == 'tarnet':
        model = TARNet(input_dim=input_dim).to(device)
    elif model_type == 'naive_fair':
        # X already has A removed by preprocess_ihdp_data
        # No need to remove again - just use X_scaled as is
        input_dim = X_train.shape[1]
        
        train_dataset = IHDPDataset(X_train, T_train, Y_train, A_train)
        val_dataset = IHDPDataset(X_val, T_val, Y_val, A_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = NaiveFairCFRNet(input_dim=input_dim, alpha=alpha).to(device)
    elif model_type == 'adversarial':
        model = AdversarialCFRNet(input_dim=input_dim, alpha=alpha, alpha_adv=alpha_adv).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pred_loss = 0.0
        train_ipm_loss = 0.0
        train_adv_loss = 0.0
        
        for batch in train_loader:
            if model_type == 'naive_fair':
                batch_X, batch_T, batch_Y, batch_A = batch
                batch_X = batch_X.to(device)
            else:
                batch_X, batch_T, batch_Y, batch_A = batch
                batch_X = batch_X.to(device)
            
            batch_T = batch_T.to(device)
            batch_Y = batch_Y.to(device)
            batch_A = batch_A.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if model_type == 'adversarial':
                y_pred, phi, a_pred = model(batch_X, batch_T, batch_A)
            else:
                y_pred, phi = model(batch_X, batch_T)
            
            # Separate treated and control
            treated_mask = batch_T == 1
            control_mask = batch_T == 0
            
            if treated_mask.sum() > 0 and control_mask.sum() > 0:
                phi_t1 = phi[treated_mask]
                phi_t0 = phi[control_mask]
                if hasattr(model, 'compute_ipm_loss'):
                    ipm_loss = model.compute_ipm_loss(phi_t0, phi_t1)
                else:
                    # For TARNet, use simple MMD
                    ipm_loss = model._compute_mmd(phi_t0, phi_t1)
            else:
                ipm_loss = torch.tensor(0.0, device=device)
            
            # Prediction loss
            pred_loss = nn.functional.mse_loss(y_pred, batch_Y.unsqueeze(1))
            
            # Adversarial loss (only for adversarial model)
            adv_loss = torch.tensor(0.0, device=device)
            if model_type == 'adversarial' and a_pred is not None:
                adv_loss = model.compute_adversarial_loss(a_pred, batch_A)
            
            # Total loss
            if model_type == 'adversarial':
                total_loss = pred_loss + alpha * ipm_loss - alpha_adv * adv_loss
            else:
                total_loss = pred_loss + alpha * ipm_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_pred_loss += pred_loss.item()
            train_ipm_loss += ipm_loss.item()
            train_adv_loss += adv_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pred_loss = 0.0
        val_ipm_loss = 0.0
        val_adv_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'naive_fair':
                    batch_X, batch_T, batch_Y, batch_A = batch
                    batch_X = batch_X.to(device)
                else:
                    batch_X, batch_T, batch_Y, batch_A = batch
                    batch_X = batch_X.to(device)
                
                batch_T = batch_T.to(device)
                batch_Y = batch_Y.to(device)
                batch_A = batch_A.to(device)
                
                if model_type == 'adversarial':
                    y_pred, phi, a_pred = model(batch_X, batch_T, batch_A)
                else:
                    y_pred, phi = model(batch_X, batch_T)
                
                # Separate treated and control
                treated_mask = batch_T == 1
                control_mask = batch_T == 0
                
                if treated_mask.sum() > 0 and control_mask.sum() > 0:
                    phi_t1 = phi[treated_mask]
                    phi_t0 = phi[control_mask]
                    if hasattr(model, 'compute_ipm_loss'):
                        ipm_loss = model.compute_ipm_loss(phi_t0, phi_t1)
                    else:
                        # For TARNet, use simple MMD
                        ipm_loss = model._compute_mmd(phi_t0, phi_t1)
                else:
                    ipm_loss = torch.tensor(0.0, device=device)
                
                pred_loss = nn.functional.mse_loss(y_pred, batch_Y.unsqueeze(1))
                
                # Adversarial loss (only for adversarial model)
                adv_loss = torch.tensor(0.0, device=device)
                if model_type == 'adversarial' and a_pred is not None:
                    adv_loss = model.compute_adversarial_loss(a_pred, batch_A)
                
                if model_type == 'adversarial':
                    total_loss = pred_loss + alpha * ipm_loss - alpha_adv * adv_loss
                else:
                    total_loss = pred_loss + alpha * ipm_loss
                
                val_loss += total_loss.item()
                val_pred_loss += pred_loss.item()
                val_ipm_loss += ipm_loss.item()
                val_adv_loss += adv_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} "
                  f"(Pred: {train_pred_loss/len(train_loader):.4f}, "
                  f"IPM: {train_ipm_loss/len(train_loader):.4f}, "
                  f"Adv: {train_adv_loss/len(train_loader):.4f}) - "
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
                'alpha_adv': alpha_adv if model_type == 'adversarial' else None,
                'model_type': model_type,
            }, save_path)
    
    # Load best model (weights_only=False to allow sklearn StandardScaler)
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate on full dataset
    X_scaled_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        if model_type == 'adversarial':
            _, phi, _ = model(X_scaled_tensor, torch.FloatTensor(T).to(device), torch.FloatTensor(A).to(device))
        else:
            _, phi = model(X_scaled_tensor, torch.FloatTensor(T).to(device))
        ite_pred = model.compute_ite(X_scaled_tensor).cpu().numpy().flatten()
    
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'ite_pred': ite_pred,
        'scaler': scaler_X,
        'model_type': model_type,
    }
    
    # Compute metrics using ground truth mu1 - mu0
    if mu0 is not None and mu1 is not None:
        # Use ground truth ITE: mu1 - mu0
        ite_true = mu1 - mu0
        pehe_val = pehe(ite_true, ite_pred)
    else:
        # Fallback: Use Y_cf to calculate ITE
        ite_true = np.zeros_like(Y)
        treated_mask = T == 1
        control_mask = T == 0
        ite_true[treated_mask] = Y[treated_mask] - Y_cf[treated_mask]
        ite_true[control_mask] = Y_cf[control_mask] - Y[control_mask]
        pehe_val = pehe(ite_true, ite_pred)
    
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
            
            if model_type == 'adversarial':
                y_pred, _, _ = model(batch_X, batch_T, batch_A)
            else:
                y_pred, _ = model(batch_X, batch_T)
            
            y_pred_val.append(y_pred.cpu().numpy())
    
    y_pred_val = np.concatenate(y_pred_val).flatten()
    A_val = A[val_idx]
    
    dp_gap = demographic_parity_gap_regression(y_pred_val, A_val)
    results['demographic_parity_gap'] = dp_gap
    
    print(f"\nTraining completed for {model_type}!")
    if 'pehe' in results:
        print(f"PEHE: {results['pehe']:.4f}")
    if 'ate_error' in results:
        print(f"ATE Error: {results['ate_error']:.4f}")
    if 'demographic_parity_gap' in results:
        print(f"Demographic Parity Gap: {results['demographic_parity_gap']:.4f}")
    
    return model, results


def train_all_baselines(X, T, Y, A, Y_cf=None, mu0=None, mu1=None, epochs=100, batch_size=128):
    """
    Train all baseline models.
    
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
    mu0 : np.ndarray, optional
        Ground truth potential outcome Y(0)
    mu1 : np.ndarray, optional
        Ground truth potential outcome Y(1)
    epochs : int
        Number of epochs per model
    batch_size : int
        Batch size
    
    Returns:
    --------
    all_results : dict
        Results for each model
    """
    all_results = {}
    
    model_types = ['cfrnet', 'tarnet', 'naive_fair', 'adversarial']
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        save_path = f'results/{model_type}_model.pth'
        model, results = train_baseline_model(
            model_type, X, T, Y, A, Y_cf=Y_cf, mu0=mu0, mu1=mu1,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_path
        )
        
        all_results[model_type] = results
    
    # Save all results
    os.makedirs('results', exist_ok=True)
    results_summary = {}
    for model_type, res in all_results.items():
        results_summary[model_type] = {
            'pehe': res.get('pehe', None),
            'ate_error': res.get('ate_error', None),
            'demographic_parity_gap': res.get('demographic_parity_gap', None),
        }
    
    with open('results/baseline_results.json', 'w') as f:
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
    
    # Train all baseline models
    print("\nStarting training of baseline models...")
    all_results = train_all_baselines(
        X, T, Y, A, Y_cf=Y_cf,
        epochs=100
    )
    
    print("\nBaseline training completed!")
    print("\nSummary of results:")
    for model_type in sorted(all_results.keys()):
        res = all_results[model_type]
        print(f"\n{model_type.upper()}:")
        if 'pehe' in res:
            print(f"  PEHE: {res['pehe']:.4f}")
        if 'ate_error' in res:
            print(f"  ATE Error: {res['ate_error']:.4f}")
        if 'demographic_parity_gap' in res:
            print(f"  Demographic Parity Gap: {res['demographic_parity_gap']:.4f}")