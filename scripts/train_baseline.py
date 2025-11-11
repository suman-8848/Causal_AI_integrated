"""
Training script for baseline CFRNet.
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cfr_net import CFRNet
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


def train_baseline_cfrnet(X, T, Y, Y_cf=None, mu0=None, mu1=None, A=None, val_split=0.2, batch_size=128, 
                          epochs=100, lr=0.001, alpha=1.0, device='cuda' if torch.cuda.is_available() else 'cpu',
                          save_path='results/baseline_cfrnet.pth'):
    """
    Train baseline CFRNet model.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Covariates
    T : np.ndarray
        Treatment indicator
    Y : np.ndarray
        Observed outcomes
    Y_cf : np.ndarray, optional
        Counterfactual outcomes (for evaluation)
    A : np.ndarray, optional
        Sensitive attribute (for evaluation)
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
    model : CFRNet
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
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-validation split
    indices = np.arange(len(X_scaled))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    T_train, T_val = T[train_idx], T[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]
    
    # Create datasets and data loaders
    train_dataset = IHDPDataset(X_train, T_train, Y_train)
    val_dataset = IHDPDataset(X_val, T_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = CFRNet(input_dim=input_dim, alpha=alpha).to(device)
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
        
        for batch_X, batch_T, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_T = batch_T.to(device)
            batch_Y = batch_Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred, phi = model(batch_X, batch_T)
            
            # Separate treated and control
            treated_mask = batch_T == 1
            control_mask = batch_T == 0
            
            if treated_mask.sum() > 0 and control_mask.sum() > 0:
                phi_t1 = phi[treated_mask]
                phi_t0 = phi[control_mask]
                ipm_loss = model.compute_ipm_loss(phi_t0, phi_t1)
            else:
                ipm_loss = torch.tensor(0.0, device=device)
            
            # Prediction loss
            pred_loss = nn.functional.mse_loss(y_pred, batch_Y.unsqueeze(1))
            
            # Total loss
            loss = pred_loss + alpha * ipm_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pred_loss += pred_loss.item()
            train_ipm_loss += ipm_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pred_loss = 0.0
        val_ipm_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_T, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                batch_T = batch_T.to(device)
                batch_Y = batch_Y.to(device)
                
                y_pred, phi = model(batch_X, batch_T)
                
                # Separate treated and control
                treated_mask = batch_T == 1
                control_mask = batch_T == 0
                
                if treated_mask.sum() > 0 and control_mask.sum() > 0:
                    phi_t1 = phi[treated_mask]
                    phi_t0 = phi[control_mask]
                    ipm_loss = model.compute_ipm_loss(phi_t0, phi_t1)
                else:
                    ipm_loss = torch.tensor(0.0, device=device)
                
                pred_loss = nn.functional.mse_loss(y_pred, batch_Y.unsqueeze(1))
                loss = pred_loss + alpha * ipm_loss
                
                val_loss += loss.item()
                val_pred_loss += pred_loss.item()
                val_ipm_loss += ipm_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} "
                  f"(Pred: {train_pred_loss/len(train_loader):.4f}, "
                  f"IPM: {train_ipm_loss/len(train_loader):.4f}) - "
                  f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'input_dim': input_dim,
                'alpha': alpha,
            }, save_path)
    
    # Load best model (weights_only=False to allow sklearn StandardScaler)
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate on full dataset
    X_scaled_tensor = torch.FloatTensor(scaler.transform(X)).to(device)
    with torch.no_grad():
        _, phi_full = model(X_scaled_tensor, torch.FloatTensor(T).to(device))
        ite_pred = model.compute_ite(X_scaled_tensor).cpu().numpy().flatten()
    
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'ite_pred': ite_pred,
        'scaler': scaler,
    }
    
    # Compute metrics using ground truth mu1 - mu0
    # Check if mu0 and mu1 are provided, otherwise fall back to Y_cf method
    if 'mu0' in locals() and 'mu1' in locals() and mu0 is not None and mu1 is not None:
        # Use ground truth ITE: mu1 - mu0
        ite_true = mu1 - mu0
        pehe_val = pehe(ite_true, ite_pred)
    elif Y_cf is not None:
        # Fallback: Use Y_cf to calculate ITE
        # ITE_true = Y[T==1] - Y_cf[T==1] for treated, and Y_cf[T==0] - Y[T==0] for control
        ite_true = np.zeros_like(Y)
        treated_mask = T == 1
        control_mask = T == 0
        ite_true[treated_mask] = Y[treated_mask] - Y_cf[treated_mask]
        ite_true[control_mask] = Y_cf[control_mask] - Y[control_mask]
        pehe_val = pehe(ite_true, ite_pred)
    else:
        ite_true = None
        pehe_val = None
    
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
    
    # Compute fairness metrics if A is provided
    if A is not None:
        # For fairness, we need predictions on validation set
        y_pred_val = []
        with torch.no_grad():
            for batch_X, batch_T, _ in val_loader:
                batch_X = batch_X.to(device)
                batch_T = batch_T.to(device)
                y_pred, _ = model(batch_X, batch_T)
                y_pred_val.append(y_pred.cpu().numpy())
        
        y_pred_val = np.concatenate(y_pred_val).flatten()
        A_val = A[val_idx]
        
        dp_gap = demographic_parity_gap_regression(y_pred_val, A_val)
        results['demographic_parity_gap'] = dp_gap
    
    print(f"\nTraining completed!")
    if 'pehe' in results:
        print(f"PEHE: {results['pehe']:.4f}")
    if 'ate_error' in results:
        print(f"ATE Error: {results['ate_error']:.4f}")
    if 'demographic_parity_gap' in results:
        print(f"Demographic Parity Gap: {results['demographic_parity_gap']:.4f}")
    
    return model, results


if __name__ == "__main__":
    # Load data
    print("Loading IHDP data...")
    data = load_ihdp_data(fold=1)
    T, Y, Y_cf, mu0, mu1, A, X = preprocess_ihdp_data(data)
    
    print(f"Data loaded: {len(X)} samples")
    print(f"Treatment rate: {T.mean():.2%}")
    print(f"Sensitive attribute distribution: A=0: {(A==0).sum()}, A=1: {(A==1).sum()}")
    
    # Train baseline
    print("\nTraining baseline CFRNet...")
    model, results = train_baseline_cfrnet(
        X, T, Y, Y_cf=Y_cf, A=A,
        epochs=100,
        batch_size=128,
        alpha=1.0
    )
    
    print("\nBaseline training completed!")

