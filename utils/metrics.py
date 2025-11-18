"""
Metrics for evaluating causal inference models.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def pehe(y_true, y_pred):
    """
    Precision in Estimation of Heterogeneous Effect (PEHE).
    
    Parameters:
    -----------
    y_true : array-like
        True counterfactual outcomes or true ITE
    y_pred : array-like
        Predicted ITE (Y_pred_treated - Y_pred_control)
    
    Returns:
    --------
    pehe : float
        The PEHE metric
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    pehe_value = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return pehe_value


def pehe_torch(y_true, y_pred):
    """
    PEHE metric implemented in PyTorch for use during training.
    
    Parameters:
    -----------
    y_true : torch.Tensor
        True counterfactual outcomes or true ITE
    y_pred : torch.Tensor
        Predicted ITE
    
    Returns:
    --------
    pehe : torch.Tensor
        The PEHE metric
    """
    pehe_value = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return pehe_value


def demographic_parity_gap_regression(y_pred, A):
    """
    Calculate the Demographic Parity Gap for a regression task.
    This is the absolute difference in average predicted outcomes between groups.

    Parameters:
    -----------
    y_pred : array-like
        Predicted continuous outcomes
    A : array-like
        Sensitive attribute

    Returns:
    --------
    gap : float
        The demographic parity gap for regression
    """
    y_pred = np.asarray(y_pred)
    A = np.asarray(A)

    # Calculate average prediction for each group
    group_0_mean = y_pred[A == 0].mean() if (A == 0).sum() > 0 else 0.0
    group_1_mean = y_pred[A == 1].mean() if (A == 1).sum() > 0 else 0.0

    gap = abs(group_0_mean - group_1_mean)
    return gap


def compute_ate_error(ate_true, ate_pred):
    """
    Compute the absolute error in Average Treatment Effect (ATE) estimation.
    
    Parameters:
    -----------
    ate_true : float
        True ATE
    ate_pred : float
        Predicted ATE
    
    Returns:
    --------
    error : float
        Absolute error in ATE estimation
    """
    return abs(ate_true - ate_pred)


def compute_ite(y_treated, y_control):
    """
    Compute Individual Treatment Effect (ITE) from treatment and control outcomes.
    
    Parameters:
    -----------
    y_treated : array-like
        Outcomes under treatment
    y_control : array-like
        Outcomes under control
    
    Returns:
    --------
    ite : np.ndarray
        Individual Treatment Effects
    """
    y_treated = np.asarray(y_treated)
    y_control = np.asarray(y_control)
    return y_treated - y_control


def compute_causal_effect_penalty(y_pred_1, y_pred_0):
    """
    Compute the penalty for the causal effect of sensitive attribute A on outcome Y.
    This measures the difference in predictions when A changes from 0 to 1.
    
    Parameters:
    -----------
    y_pred_1 : torch.Tensor
        Predictions when A=1
    y_pred_0 : torch.Tensor
        Predictions when A=0
    
    Returns:
    --------
    penalty : torch.Tensor
        Mean absolute difference (fairness penalty)
    """
    penalty = torch.mean(torch.abs(y_pred_1 - y_pred_0))
    return penalty


def measure_direct_effect(model, X, A, device='cpu'):
    """
    Measure actual direct effect of A on predictions.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    X : torch.Tensor
        Covariates [batch_size, input_dim]
    A : torch.Tensor
        Sensitive attribute [batch_size]
    device : str
        Device to use ('cuda' or 'cpu')
    
    Returns:
    --------
    direct_effect : float
        Direct effect of A on predictions
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        # Predict with A=0
        A_zero = torch.zeros_like(A)
        if hasattr(model, 'predict'):
            Y_pred_a0 = model.predict(X_tensor, A_zero)
        else:
            # For FairCFRNet, we need to handle differently
            if hasattr(model, 'forward') and 'A' in model.forward.__code__.co_varnames:
                # Model takes A as input
                if hasattr(model, 'compute_ite'):
                    Y_pred_a0 = model.compute_ite(X_tensor, A_zero)
                else:
                    # Create dummy treatment if needed
                    T_dummy = torch.zeros_like(A_zero)
                    _, _, _, _, _, _, Y_pred_a0 = model(X_tensor, A_zero, T_dummy)
            else:
                # Model doesn't take A as input, use default behavior
                Y_pred_a0 = model(X_tensor)
        
        # Predict with A=1
        A_one = torch.ones_like(A)
        if hasattr(model, 'predict'):
            Y_pred_a1 = model.predict(X_tensor, A_one)
        else:
            # For FairCFRNet, we need to handle differently
            if hasattr(model, 'forward') and 'A' in model.forward.__code__.co_varnames:
                # Model takes A as input
                if hasattr(model, 'compute_ite'):
                    Y_pred_a1 = model.compute_ite(X_tensor, A_one)
                else:
                    # Create dummy treatment if needed
                    T_dummy = torch.zeros_like(A_one)
                    _, _, _, _, _, _, Y_pred_a1 = model(X_tensor, A_one, T_dummy)
            else:
                # Model doesn't take A as input, use default behavior
                Y_pred_a1 = model(X_tensor)
        
        # Direct effect
        if isinstance(Y_pred_a0, tuple):
            Y_pred_a0 = Y_pred_a0[0]
        if isinstance(Y_pred_a1, tuple):
            Y_pred_a1 = Y_pred_a1[0]
            
        direct_effect = (Y_pred_a1 - Y_pred_a0).mean().item()
    
    return direct_effect


def measure_counterfactual_fairness(model, X, A, M, device='cpu'):
    """
    Compare Ŷ(A=0) vs Ŷ(A=1) holding M fixed.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    X : torch.Tensor
        Covariates [batch_size, input_dim]
    A : torch.Tensor
        Sensitive attribute [batch_size]
    M : torch.Tensor
        Mediator values [batch_size, m_dim]
    device : str
        Device to use ('cuda' or 'cpu')
    
    Returns:
    --------
    counterfactual_fairness : float
        Difference in predictions when A changes but M is fixed
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    M_tensor = torch.FloatTensor(M).to(device)
    
    with torch.no_grad():
        # Create inputs with A=0 and A=1 but same M
        A_zero = torch.zeros_like(A)
        A_one = torch.ones_like(A)
        
        # Create input tensors with (X, M, A)
        input_a0 = torch.cat([X_tensor, M_tensor, A_zero.unsqueeze(1)], dim=1)
        input_a1 = torch.cat([X_tensor, M_tensor, A_one.unsqueeze(1)], dim=1)
        
        # Get predictions
        if hasattr(model, 'outcome_net'):
            # For FairCFRNet
            Y_pred_a0 = model.outcome_net(input_a0)
            Y_pred_a1 = model.outcome_net(input_a1)
        else:
            # For other models, try to use forward directly
            Y_pred_a0 = model(input_a0)
            Y_pred_a1 = model(input_a1)
        
        # Counterfactual fairness: difference when A changes but M is fixed
        counterfactual_fairness = (Y_pred_a1 - Y_pred_a0).mean().item()
    
    return counterfactual_fairness


def measure_equalized_odds(y_pred, y_true, A, T, threshold=0.5):
    """
    Measure equalized odds: Group fairness on treated/control separately.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted outcomes
    y_true : array-like
        True outcomes
    A : array-like
        Sensitive attribute
    T : array-like
        Treatment indicator
    threshold : float
        Threshold for binary classification
    
    Returns:
    --------
    equalized_odds : dict
        Dictionary with equalized odds metrics for different groups
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    A = np.asarray(A)
    T = np.asarray(T)
    
    # Convert predictions to binary if needed
    if y_pred.max() > 1 or y_pred.min() < 0:
        y_binary = (y_pred > threshold).astype(int)
    else:
        y_binary = y_pred.astype(int)
    
    # Calculate equalized odds for treated and control groups separately
    results = {}
    
    # For treated group (T=1)
    if (T == 1).sum() > 0:
        treated_mask = T == 1
        if (A[treated_mask] == 0).sum() > 0 and (A[treated_mask] == 1).sum() > 0:
            # True positive rate for A=0 and A=1 in treated group
            tpr_a0 = ((y_binary[treated_mask] == 1) & (y_true[treated_mask] == 1) & (A[treated_mask] == 0)).sum() / ((y_true[treated_mask] == 1) & (A[treated_mask] == 0)).sum()
            tpr_a1 = ((y_binary[treated_mask] == 1) & (y_true[treated_mask] == 1) & (A[treated_mask] == 1)).sum() / ((y_true[treated_mask] == 1) & (A[treated_mask] == 1)).sum()
            
            # False positive rate for A=0 and A=1 in treated group
            fpr_a0 = ((y_binary[treated_mask] == 1) & (y_true[treated_mask] == 0) & (A[treated_mask] == 0)).sum() / ((y_true[treated_mask] == 0) & (A[treated_mask] == 0)).sum()
            fpr_a1 = ((y_binary[treated_mask] == 1) & (y_true[treated_mask] == 0) & (A[treated_mask] == 1)).sum() / ((y_true[treated_mask] == 0) & (A[treated_mask] == 1)).sum()
            
            results['treated'] = {
                'tpr_gap': abs(tpr_a0 - tpr_a1),
                'fpr_gap': abs(fpr_a0 - fpr_a1),
                'tpr_a0': tpr_a0,
                'tpr_a1': tpr_a1,
                'fpr_a0': fpr_a0,
                'fpr_a1': fpr_a1
            }
    
    # For control group (T=0)
    if (T == 0).sum() > 0:
        control_mask = T == 0
        if (A[control_mask] == 0).sum() > 0 and (A[control_mask] == 1).sum() > 0:
            # True positive rate for A=0 and A=1 in control group
            tpr_a0 = ((y_binary[control_mask] == 1) & (y_true[control_mask] == 1) & (A[control_mask] == 0)).sum() / ((y_true[control_mask] == 1) & (A[control_mask] == 0)).sum()
            tpr_a1 = ((y_binary[control_mask] == 1) & (y_true[control_mask] == 1) & (A[control_mask] == 1)).sum() / ((y_true[control_mask] == 1) & (A[control_mask] == 1)).sum()
            
            # False positive rate for A=0 and A=1 in control group
            fpr_a0 = ((y_binary[control_mask] == 1) & (y_true[control_mask] == 0) & (A[control_mask] == 0)).sum() / ((y_true[control_mask] == 0) & (A[control_mask] == 0)).sum()
            fpr_a1 = ((y_binary[control_mask] == 1) & (y_true[control_mask] == 0) & (A[control_mask] == 1)).sum() / ((y_true[control_mask] == 0) & (A[control_mask] == 1)).sum()
            
            results['control'] = {
                'tpr_gap': abs(tpr_a0 - tpr_a1),
                'fpr_gap': abs(fpr_a0 - fpr_a1),
                'tpr_a0': tpr_a0,
                'tpr_a1': tpr_a1,
                'fpr_a0': fpr_a0,
                'fpr_a1': fpr_a1
            }
    
    return results