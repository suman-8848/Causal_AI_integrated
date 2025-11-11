"""
Fair CFRNet: Integration of Fair Risk Minimization with Deep Treatment Effect Estimation.

This implementation integrates the theoretical framework from "Fair Risk Minimization under 
Causal Path-Specific Effect Constraints" into a deep learning model for treatment effect estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dce_module import DCEModule, compute_d_theta


class FairCFRNet(nn.Module):
    """
    Fair CFRNet with integrated path-specific fairness constraints.
    
    Architecture:
    - Representation network takes (X, A) as input
    - Separate treatment and control heads predict Y(1) and Y(0)
    - Propensity network estimates P(T|X)
    - DCE module estimates path-specific effects and canonical gradients
    
    The model optimizes:
    total_loss = pred_loss + alpha * ipm_loss + lambda * L_fair
    where L_fair = |PE(A -> Y)| + beta * ||TE - TE*||^2
    """
    def __init__(self, input_dim, hidden_dim=200, dropout=0.1, alpha=1.0, 
                 lambda_fairness=1.0, beta_fairness=0.5, m_dim=1, use_pmmd=True):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of covariates X (A is provided separately)
        hidden_dim : int
            Hidden dimension for representation network
        dropout : float
            Dropout probability
        alpha : float
            Weight for IPM loss
        lambda_fairness : float
            Weight for fairness constraint loss
        beta_fairness : float
            Weight for fairness adjustment term
        m_dim : int
            Dimension of mediator M (default 1 for binary)
        use_pmmd : bool
            Whether to use MMD for IPM
        """
        super(FairCFRNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda_fairness = lambda_fairness
        self.beta_fairness = beta_fairness
        self.use_pmmd = use_pmmd
        
        # Representation network: takes (X, A) as input
        # +1 for sensitive attribute A
        self.representation = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Treatment heads: predict Y(1) and Y(0) from representation
        self.t_head = nn.Linear(hidden_dim, 1)  # Predicts Y(1)
        self.c_head = nn.Linear(hidden_dim, 1)   # Predicts Y(0)
        
        # Propensity network: estimates P(T|X)
        self.propensity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # DCE Module for fairness: estimates mediators and path-specific effects
        self.dce_module = DCEModule(
            x_dim=input_dim,
            m_dim=m_dim,
            hidden_dims_mediator=[64],
            dropout=dropout
        )
    
    def forward(self, x, a, t):
        """
        Forward pass through FairCFRNet.
        
        Parameters:
        -----------
        x : torch.Tensor
            Covariates [batch_size, input_dim]
        a : torch.Tensor
            Sensitive attribute [batch_size] (0 or 1)
        t : torch.Tensor
            Treatment indicator [batch_size]
        
        Returns:
        --------
        te : torch.Tensor
            Treatment effect estimates [batch_size, 1]
        pe_hat : torch.Tensor
            Estimated path-specific effect PE(A -> Y) (scalar)
        d_theta : torch.Tensor
            Canonical gradient [batch_size]
        propensity : torch.Tensor
            Propensity scores P(T|X) [batch_size, 1]
        phi : torch.Tensor
            Learned representations [batch_size, hidden_dim]
        """
        # Concatenate sensitive attribute for base prediction
        inputs = torch.cat([x, a.unsqueeze(1)], dim=1)
        phi = self.representation(inputs)
        
        # Unconstrained predictions
        y_hat_t = self.t_head(phi)  # Y(1)
        y_hat_c = self.c_head(phi)   # Y(0)
        
        # Unconstrained treatment effect
        te = y_hat_t - y_hat_c
        
        # Fairness components (calculated separately)
        propensity = self.propensity_net(x)  # P(T|X)
        
        # Get path-specific effect and mediator probabilities from DCE module
        pe_hat, d_theta_mediator_comp, p_m_given_x_a, p_m_given_x_a0 = self.dce_module(x, a, t, te)
        
        # Compute full canonical gradient d_theta
        d_theta = compute_d_theta(x, a, p_m_given_x_a, p_m_given_x_a0, self.propensity_net)
        
        return te, pe_hat, d_theta, propensity, phi
    
    def compute_ipm_loss(self, phi_t0, phi_t1):
        """
        Compute IPM (Integral Probability Metric) loss for distribution matching.
        
        Parameters:
        -----------
        phi_t0 : torch.Tensor
            Representations for control group
        phi_t1 : torch.Tensor
            Representations for treatment group
        
        Returns:
        --------
        ipm_loss : torch.Tensor
            IPM loss value
        """
        if self.use_pmmd:
            return self._compute_mmd(phi_t0, phi_t1)
        else:
            return self._compute_wasserstein(phi_t0, phi_t1)
    
    def _compute_mmd(self, x, y, sigma=1.0):
        """
        Compute MMD between two distributions.
        
        Parameters:
        -----------
        x : torch.Tensor
            First distribution samples
        y : torch.Tensor
            Second distribution samples
        sigma : float
            RBF kernel bandwidth
        
        Returns:
        --------
        mmd : torch.Tensor
            MMD value
        """
        def rbf_kernel(x1, x2):
            # RBF kernel: k(x1, x2) = exp(-||x1 - x2||^2 / (2*sigma^2))
            pairwise_dists = torch.cdist(x1, x2) ** 2
            return torch.exp(-pairwise_dists / (2 * sigma ** 2))
        
        K_xx = rbf_kernel(x, x).mean()
        K_yy = rbf_kernel(y, y).mean()
        K_xy = rbf_kernel(x, y).mean()
        
        mmd = K_xx + K_yy - 2 * K_xy
        return mmd
    
    def _compute_wasserstein(self, x, y):
        """
        Approximate Wasserstein distance (simplified version).
        
        Parameters:
        -----------
        x : torch.Tensor
            First distribution samples
        y : torch.Tensor
            Second distribution samples
        
        Returns:
        --------
        wasserstein : torch.Tensor
            Approximate Wasserstein distance
        """
        mean_x = x.mean(dim=0)
        mean_y = y.mean(dim=0)
        return torch.norm(mean_x - mean_y, p=2)
    
    def compute_loss(self, te, y_true, t, phi, pe_hat, d_theta, ipm_loss):
        """
        Compute the total loss for FairCFRNet.
        
        Parameters:
        -----------
        te : torch.Tensor
            Treatment effect estimates
        y_true : torch.Tensor
            True outcomes
        t : torch.Tensor
            Treatment indicator
        phi : torch.Tensor
            Learned representations
        pe_hat : torch.Tensor
            Estimated path-specific effect
        d_theta : torch.Tensor
            Canonical gradient
        ipm_loss : torch.Tensor
            IPM loss value
        
        Returns:
        --------
        total_loss : torch.Tensor
            Total loss = pred_loss + alpha * ipm_loss + lambda * L_fair
        pred_loss : torch.Tensor
            Prediction loss
        ipm_loss_val : torch.Tensor
            IPM loss value
        fairness_loss : torch.Tensor
            Fairness loss value
        """
        # Prediction loss: predict observed outcomes
        # We need to reconstruct y_pred from te and t
        # y_pred = t * y1 + (1-t) * y0
        # But we have te = y1 - y0, so we need y0 and y1 separately
        # For now, use a simplified prediction loss based on treatment effect
        # In practice, you'd want to predict y0 and y1 separately
        
        # Simplified: use MSE on treatment effect if we have ground truth ITE
        # Otherwise, use outcome prediction
        # For this implementation, we'll compute prediction loss on outcomes
        # reconstructed from treatment effects
        
        # Get predictions for observed treatment
        y_pred_t = self.t_head(phi)
        y_pred_c = self.c_head(phi)
        y_pred = torch.where(t.unsqueeze(1) == 1, y_pred_t, y_pred_c)
        
        pred_loss = F.mse_loss(y_pred, y_true.unsqueeze(1))
        
        # Fairness loss: L_fair = |PE(A -> Y)| + beta * adjustment_term
        # Adjustment term: variance of d_theta weighted by treatment effect
        sigma2 = torch.var(d_theta)
        
        # Fairness-adjusted treatment effect (closed-form adjustment)
        # TE* = TE - (PE * d_theta) / (sigma2 + eps)
        te_adj = te.squeeze() - (pe_hat * d_theta) / (sigma2 + 1e-6)
        
        # Constraint loss: minimize path-specific effect
        constraint_loss = torch.abs(pe_hat)
        
        # Adjustment loss: minimize difference between TE and TE*
        adjustment_loss = torch.mean((te.squeeze() - te_adj) ** 2)
        
        # Total fairness loss
        fairness_loss = constraint_loss + self.beta_fairness * adjustment_loss
        
        # Total loss
        total_loss = pred_loss + self.alpha * ipm_loss + self.lambda_fairness * fairness_loss
        
        return total_loss, pred_loss, ipm_loss, fairness_loss
    
    def compute_ite(self, x, a):
        """
        Compute Individual Treatment Effect (ITE) for given inputs.
        
        Parameters:
        -----------
        x : torch.Tensor
            Covariates
        a : torch.Tensor
            Sensitive attribute
        
        Returns:
        --------
        ite : torch.Tensor
            Individual Treatment Effects
        """
        # Concatenate sensitive attribute
        inputs = torch.cat([x, a.unsqueeze(1)], dim=1)
        phi = self.representation(inputs)
        
        # Get predictions
        y_hat_t = self.t_head(phi)
        y_hat_c = self.c_head(phi)
        
        # Treatment effect
        ite = y_hat_t - y_hat_c
        return ite

