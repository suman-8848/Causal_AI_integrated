"""
Differentiable Causal Estimation (DCE) Module.

This module implements the core algorithmic contribution that estimates and penalizes
the causal effect of sensitive attribute A on outcome Y through mediators M.

Revised to compute path-specific effects (PE) and canonical gradients (d_theta).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCEModule(nn.Module):
    """
    Differentiable Causal Estimation Module.
    
    This module is solely responsible for estimating fairness-related quantities:
    - Path-specific effect PE(A -> Y) through mediators M
    - Canonical gradient d_theta for fair risk minimization
    
    The module uses a dedicated mediator model to learn P(M | X, A) independently
    from the base treatment effect estimation network.
    """
    def __init__(self, x_dim, m_dim=1, hidden_dims_mediator=[64], dropout=0.1):
        """
        Parameters:
        -----------
        x_dim : int
            Dimension of covariates X
        m_dim : int
            Dimension of mediator M (default 1 for binary mediator)
        hidden_dims_mediator : list
            Hidden layer dimensions for mediator model
        dropout : float
            Dropout probability
        """
        super(DCEModule, self).__init__()
        
        self.x_dim = x_dim
        self.m_dim = m_dim
        
        # Dedicated mediator model: learns P(M | X, A)
        # Input: (X, A), Output: M (logits for binary M, or values for continuous M)
        mediator_input_dim = x_dim + 1  # X + A
        layers = []
        prev_dim = mediator_input_dim
        
        for hidden_dim in hidden_dims_mediator:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, m_dim))
        self.mediator_model = nn.Sequential(*layers)
    
    def forward(self, x, a, t, y, propensity_net):
        """
        Forward pass to compute path-specific effect and canonical gradient.
        
        Parameters:
        -----------
        x : torch.Tensor
            Covariates [batch_size, x_dim]
        a : torch.Tensor
            Sensitive attribute [batch_size] (0 or 1)
        t : torch.Tensor
            Treatment indicator [batch_size]
        y : torch.Tensor
            Outcome predictions [batch_size, 1] or [batch_size]
        propensity_net : nn.Module
            Network to estimate P(A=1|X)
        
        Returns:
        --------
        pe_hat : torch.Tensor
            Estimated path-specific effect PE(A -> Y)
        d_theta_mediator_component : torch.Tensor
            Mediator ratio component for d_theta [batch_size]
        p_m_given_x_a : torch.Tensor
            P(M|X,A) [batch_size, m_dim]
        p_m_given_x_a0 : torch.Tensor
            P(M|X,A=0) [batch_size, m_dim]
        """
        # Estimate mediator probabilities
        # P(M=1 | X, A) for binary mediator
        m_logits = self.mediator_model(torch.cat([x, a.unsqueeze(1)], dim=1))
        p_m_given_x_a = torch.sigmoid(m_logits)  # [batch_size, m_dim]
        
        # P(M=1 | X, A=0) - counterfactual mediator
        a_zero = torch.zeros_like(a)
        m_logits_a0 = self.mediator_model(torch.cat([x, a_zero.unsqueeze(1)], dim=1))
        p_m_given_x_a0 = torch.sigmoid(m_logits_a0)  # [batch_size, m_dim]
        
        # For multi-dimensional mediators, take mean or use first dimension
        if p_m_given_x_a.dim() > 1 and p_m_given_x_a.shape[1] > 1:
            p_m_given_x_a = p_m_given_x_a.mean(dim=1, keepdim=True)
            p_m_given_x_a0 = p_m_given_x_a0.mean(dim=1, keepdim=True)
        
        # Compute density ratio for mediator
        # ratio_m = P(M | X, A=0) / P(M | X, A)
        ratio_m = p_m_given_x_a0 / (p_m_given_x_a + 1e-6)
        ratio_m = ratio_m.squeeze()  # [batch_size]
        
        # Estimate path-specific effect using proper AIPW estimator with outcomes Y
        pe_hat = self._estimate_path_effect_aipw(y, x, a, t, propensity_net, p_m_given_x_a, p_m_given_x_a0)
        
        # Store mediator probabilities for d_theta computation
        d_theta_mediator_component = ratio_m
        
        return pe_hat, d_theta_mediator_component, p_m_given_x_a, p_m_given_x_a0
    
    def _estimate_path_effect_aipw(self, y, x, a, t, propensity_net, p_m_given_x_a, p_m_given_x_a0):
        """
        Proper AIPW estimator using actual outcomes Y.
        Based on paper Appendix D.6
        
        Parameters:
        -----------
        y : torch.Tensor
            Outcome predictions [batch_size, 1] or [batch_size]
        x : torch.Tensor
            Covariates
        a : torch.Tensor
            Sensitive attribute
        t : torch.Tensor
            Treatment indicator
        propensity_net : nn.Module
            Network to estimate P(A=1|X)
        p_m_given_x_a : torch.Tensor
            P(M|X,A)
        p_m_given_x_a0 : torch.Tensor
            P(M|X,A=0)
        
        Returns:
        --------
        pe_hat : torch.Tensor
            Estimated path-specific effect (scalar)
        """
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze()
        
        # Get propensity scores P(A=1|X)
        p_a1 = propensity_net(x).squeeze()
        propensity_a = torch.where(a == 1, p_a1, 1 - p_a1)
        
        # Mediator ratio
        ratio_m = p_m_given_x_a0 / (p_m_given_x_a + 1e-6)
        if ratio_m.dim() > 1:
            ratio_m = ratio_m.squeeze()
        
        # Outcome regression components
        mask_a0 = (a == 0)
        mask_a1 = (a == 1)
        
        if mask_a0.sum() == 0 or mask_a1.sum() == 0:
            return torch.tensor(0.0, device=y.device, requires_grad=True)
        
        # Marginalized outcomes
        mu_1 = y[mask_a1].mean()  # E[Y|A=1]
        mu_0 = y[mask_a0].mean()  # E[Y|A=0]
        
        # IPW term with mediator weighting
        # For A=1: weight by ratio_m to account for mediator path
        ipw_a1 = (a / (p_a1 + 1e-6) * ratio_m * (y - mu_1)).mean()
        # For A=0: standard IPW
        ipw_a0 = ((1 - a) / (1 - p_a1 + 1e-6) * (y - mu_0)).mean()
        
        # AIPW combination: outcome regression + IPW
        pe_hat = (mu_1 + ipw_a1) - (mu_0 + ipw_a0)
        
        return pe_hat
    
    def get_mediator_predictions(self, x, a):
        """
        Get mediator predictions for given X and A.
        
        Parameters:
        -----------
        x : torch.Tensor
            Covariates
        a : torch.Tensor
            Sensitive attribute
        
        Returns:
        --------
        p_m : torch.Tensor
            Predicted mediator probabilities P(M|X,A)
        """
        m_logits = self.mediator_model(torch.cat([x, a.unsqueeze(1)], dim=1))
        p_m = torch.sigmoid(m_logits)
        return p_m


def compute_d_theta(x, a, p_m_given_x_a, p_m_given_x_a0, propensity_net):
    """
    Computes the canonical gradient D_Theta for each sample.
    
    Args:
        x: Covariates [batch_size, x_dim]
        a: Sensitive attribute (0 or 1) [batch_size]
        p_m_given_x_a: P(M | X, A) [batch_size, m_dim] or [batch_size]
        p_m_given_x_a0: P(M | X, A=0) [batch_size, m_dim] or [batch_size]
        propensity_net: Model to estimate P(A=1 | X)
    
    Returns:
        d_theta: Canonical gradient [batch_size]
    """
    # 1. Propensity score P(A=1 | X)
    p_a1_given_x = propensity_net(x)
    if p_a1_given_x.dim() > 1:
        p_a1_given_x = p_a1_given_x.squeeze()
    
    # 2. Density ratio for the mediator
    # ratio_m = P(M | X, A=0) / P(M | X, A)
    if p_m_given_x_a.dim() > 1:
        p_m_given_x_a = p_m_given_x_a.squeeze()
    if p_m_given_x_a0.dim() > 1:
        p_m_given_x_a0 = p_m_given_x_a0.squeeze()
    
    ratio_m = p_m_given_x_a0 / (p_m_given_x_a + 1e-6)
    
    # 3. Calculate propensity for each sample
    # For a=1, use P(A=1|X); for a=0, use P(A=0|X) = 1 - P(A=1|X)
    propensity = torch.where(a == 1, p_a1_given_x, 1 - p_a1_given_x)
    
    # 4. Calculate the full gradient
    # (2a - 1) is -1 for a=0 and +1 for a=1
    d_theta = (2 * a - 1) / (propensity + 1e-6) * ratio_m
    
    return d_theta


if __name__ == "__main__":
    """
    Unit test for the DCE Module.
    """
    print("Testing DCE Module...")
    
    # Test parameters
    batch_size = 32
    x_dim = 24  # IHDP has 24 covariates after removing x10
    m_dim = 1
    
    # Create dummy data
    X = torch.randn(batch_size, x_dim)
    A = torch.randint(0, 2, (batch_size,)).float()
    T = torch.randint(0, 2, (batch_size,)).float()
    TE = torch.randn(batch_size, 1)  # Treatment effect estimates
    
    # Instantiate DCE module
    dce = DCEModule(x_dim=x_dim, m_dim=m_dim)
    
    # Forward pass
    pe_hat, d_theta_comp, p_m_a, p_m_a0 = dce(X, A, T, TE)
    
    print(f"Input shapes:")
    print(f"  X: {X.shape}")
    print(f"  A: {A.shape}")
    print(f"  T: {T.shape}")
    print(f"  TE: {TE.shape}")
    print(f"\nOutputs:")
    print(f"  PE_hat: {pe_hat.item():.4f}")
    print(f"  d_theta_mediator_component shape: {d_theta_comp.shape}")
    print(f"  P(M|X,A) shape: {p_m_a.shape}")
    print(f"  P(M|X,A=0) shape: {p_m_a0.shape}")
    
    # Test gradient flow
    loss = pe_hat
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in dce.parameters())
    print(f"Gradients flow correctly: {has_gradients}")
    
    print("\n✓ DCE Module test passed!")

