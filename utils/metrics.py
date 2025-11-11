"""
Metrics for evaluating causal inference models.
"""
import numpy as np
import torch


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


# def demographic_parity_gap(y_pred, A, threshold=0.5):
#     """
#     Calculate the Demographic Parity Gap.
    
#     Parameters:
#     -----------
#     y_pred : array-like
#         Predicted outcomes
#     A : array-like
#         Sensitive attribute
#     threshold : float
#         Threshold for binary classification (default: 0.5)
    
#     Returns:
#     --------
#     gap : float
#         The demographic parity gap
#     """
#     y_pred = np.asarray(y_pred)
#     A = np.asarray(A)
    
#     # Convert predictions to binary if needed
#     if y_pred.max() > 1 or y_pred.min() < 0:
#         y_binary = (y_pred > threshold).astype(int)
#     else:
#         y_binary = y_pred.astype(int)
    
#     # Calculate positive prediction rates for each group
#     group_0_rate = y_binary[A == 0].mean() if (A == 0).sum() > 0 else 0.0
#     group_1_rate = y_binary[A == 1].mean() if (A == 1).sum() > 0 else 0.0
    
#     gap = abs(group_0_rate - group_1_rate)
#     return gap

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

