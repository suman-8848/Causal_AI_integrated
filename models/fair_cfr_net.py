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
        
        # Representation network: learns from X only (not A)
        # Fairness adjustments are applied based on A separately
        self.representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Treatment heads: predict Y(1) and Y(0) from representation (legacy, kept for compatibility)
        self.t_head = nn.Linear(hidden_dim, 1)  # Predicts Y(1)
        self.c_head = nn.Linear(hidden_dim, 1)   # Predicts Y(0)
        
        # Outcome network: takes (X, M, T) as input to model T -> Y path
        # M is used to block A -> Y path (by using M(A=0) counterfactual)
        # T is the treatment indicator for predicting Y(1) vs Y(0)
        outcome_input_dim = input_dim + m_dim + 1  # X + M + T
        self.outcome_net = nn.Sequential(
            nn.Linear(outcome_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Propensity network for treatment: estimates P(T|X)
        self.propensity_net_t = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Propensity network for sensitive attribute: estimates P(A=1|X)
        self.propensity_net_a = nn.Sequential(
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
    
    def compute_fair_predictions(self, te, pe_hat, d_theta):
        """
        Apply closed-form fair adjustment from Theorem 4.
        
        Parameters:
        -----------
        te : torch.Tensor
            Unconstrained treatment effect estimates
        pe_hat : torch.Tensor
            Estimated path-specific effect PE(A -> Y)
        d_theta : torch.Tensor
            Canonical gradient [batch_size]
        
        Returns:
        --------
        te_fair : torch.Tensor
            Fairness-adjusted treatment effect
        """
        # Compute variance of gradient
        sigma2 = torch.var(d_theta)
        
        # Add minimum threshold to prevent division by very small values
        # This prevents the adjustment from being too aggressive
        sigma2 = torch.clamp(sigma2, min=1e-4)
        
        # Closed-form adjustment (Eq. 6 from paper)
        # Adjust by the path-specific effect weighted by canonical gradient
        adjustment = (pe_hat * d_theta) / sigma2
        
        # Clip adjustment to prevent extreme values (safeguard)
        # Limit adjustment to at most 50% of the unconstrained TE magnitude
        te_magnitude = torch.abs(te.squeeze())
        max_adjustment = 0.5 * te_magnitude
        adjustment = torch.clamp(adjustment, min=-max_adjustment, max=max_adjustment)
        
        te_fair = te.squeeze() - adjustment
        
        return te_fair
    
    def forward(self, x, a, t):
        """
        Forward pass through FairCFRNet.
        Models the causal path A -> M -> Y explicitly.
        
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
        te_unconstrained : torch.Tensor
            Unconstrained treatment effect estimates [batch_size, 1]
        te_fair : torch.Tensor
            Fairness-adjusted treatment effect [batch_size]
        pe_hat : torch.Tensor
            Estimated path-specific effect PE(A -> Y) (scalar)
        d_theta : torch.Tensor
            Canonical gradient [batch_size]
        propensity : torch.Tensor
            Propensity scores P(T|X) [batch_size, 1]
        phi : torch.Tensor
            Learned representations [batch_size, hidden_dim]
        y_pred : torch.Tensor
            Predicted outcomes using mediator [batch_size, 1]
        """
        # Representation learns from X only (not A)
        phi = self.representation(x)
        
        # Get mediator for observed A: M(A)
        m_obs = self.dce_module.get_mediator_predictions(x, a)
        
        # For path-specific effect: use counterfactual mediator M(A=0)
        # This blocks the direct path A -> Y, only allowing A -> M -> Y
        m_a0 = self.dce_module.get_mediator_predictions(x, torch.zeros_like(a))
        
        # Predict Y(1) using counterfactual mediator M(A=0) and T=1
        # Y(1, M(0)): outcome when T=1 but M behaves as if A=0 (blocks A->Y path)
        t_1 = torch.ones_like(t)
        input_y1 = torch.cat([x, m_a0, t_1.unsqueeze(1)], dim=1)
        y1_pred = self.outcome_net(input_y1)
        
        # Predict Y(0) using counterfactual mediator M(A=0) and T=0
        # Y(0, M(0)): outcome when T=0 and M behaves as if A=0 (blocks A->Y path)
        t_0 = torch.zeros_like(t)
        input_y0 = torch.cat([x, m_a0, t_0.unsqueeze(1)], dim=1)
        y0_pred = self.outcome_net(input_y0)
        
        # Unconstrained treatment effect (using mediator to block A->Y path)
        te_unconstrained = y1_pred - y0_pred
        
        # Predict observed outcome using actual mediator M(A) and observed treatment T
        input_y_obs = torch.cat([x, m_obs, t.unsqueeze(1)], dim=1)
        y_pred = self.outcome_net(input_y_obs)
        
        # Fairness components (calculated separately)
        propensity = self.propensity_net_t(x)  # P(T|X)
        
        # Get path-specific effect and mediator probabilities from DCE module
        # Pass actual outcomes y_pred for AIPW estimator
        pe_hat, d_theta_mediator_comp, p_m_given_x_a, p_m_given_x_a0 = self.dce_module(
            x, a, t, y_pred, self.propensity_net_a
        )
        
        # Compute full canonical gradient d_theta
        d_theta = compute_d_theta(x, a, p_m_given_x_a, p_m_given_x_a0, self.propensity_net_a)
        
        # Apply closed-form fair adjustment
        te_fair = self.compute_fair_predictions(te_unconstrained, pe_hat, d_theta)
        
        return te_unconstrained, te_fair, pe_hat, d_theta, propensity, phi, y_pred
    
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
    
    def compute_loss(self, te_unconstrained, te_fair, y_true, t, phi, pe_hat, ipm_loss, y_pred):
        """
        Loss based on fair predictions, with constraint penalty.
        
        Parameters:
        -----------
        te_unconstrained : torch.Tensor
            Unconstrained treatment effect estimates
        te_fair : torch.Tensor
            Fairness-adjusted treatment effect
        y_true : torch.Tensor
            True outcomes
        t : torch.Tensor
            Treatment indicator
        phi : torch.Tensor
            Learned representations
        pe_hat : torch.Tensor
            Estimated path-specific effect
        ipm_loss : torch.Tensor
            IPM loss value
        y_pred : torch.Tensor
            Predicted outcomes using mediator
        
        Returns:
        --------
        total_loss : torch.Tensor
            Total loss = pred_loss + alpha * ipm_loss + lambda * constraint_loss
        pred_loss : torch.Tensor
            Prediction loss using outcomes with mediator
        ipm_loss_val : torch.Tensor
            IPM loss value
        constraint_loss : torch.Tensor
            Constraint penalty (should be small after adjustment)
        """
        # Prediction loss using outcomes with mediator
        # y_pred already uses the mediator M in the outcome network
        pred_loss = F.mse_loss(y_pred, y_true.unsqueeze(1))
        
        # Constraint penalty (should be small after adjustment)
        constraint_loss = torch.abs(pe_hat)
        
        # Total loss
        total_loss = pred_loss + self.alpha * ipm_loss + self.lambda_fairness * constraint_loss
        
        return total_loss, pred_loss, ipm_loss, constraint_loss
    
    def compute_ite(self, x, a):
        """
        Compute Individual Treatment Effect (ITE) for given inputs.
        Returns the fair-adjusted treatment effect using mediators.
        
        Parameters:
        -----------
        x : torch.Tensor
            Covariates
        a : torch.Tensor
            Sensitive attribute
        
        Returns:
        --------
        ite : torch.Tensor
            Fairness-adjusted Individual Treatment Effects
        """
        # Get counterfactual mediator M(A=0) to block A->Y path
        m_a0 = self.dce_module.get_mediator_predictions(x, torch.zeros_like(a))
        
        # Predict Y(1) and Y(0) using counterfactual mediator and treatment T
        # Use T=1 for Y(1) and T=0 for Y(0)
        batch_size = x.shape[0]
        t_1 = torch.ones(batch_size, device=x.device)
        t_0 = torch.zeros(batch_size, device=x.device)
        
        input_y1 = torch.cat([x, m_a0, t_1.unsqueeze(1)], dim=1)
        y1_pred = self.outcome_net(input_y1)
        
        input_y0 = torch.cat([x, m_a0, t_0.unsqueeze(1)], dim=1)
        y0_pred = self.outcome_net(input_y0)
        
        te_unconstrained = y1_pred - y0_pred
        
        # For ITE computation, we need to compute fairness components
        # Use predicted outcome with actual mediator M(A) and average treatment for AIPW
        m_obs = self.dce_module.get_mediator_predictions(x, a)
        # Use T=0.5 as average for computing fairness (or use actual T if available)
        # For now, use T=0.5 to get average outcome
        t_avg = torch.ones(batch_size, device=x.device) * 0.5
        input_y_obs = torch.cat([x, m_obs, t_avg.unsqueeze(1)], dim=1)
        y_pred = self.outcome_net(input_y_obs)
        
        # Create dummy treatment indicator (not used for ITE, but needed for DCE)
        t_dummy = torch.zeros_like(a)
        
        # Get path-specific effect and compute fair adjustment
        pe_hat, _, p_m_given_x_a, p_m_given_x_a0 = self.dce_module(
            x, a, t_dummy, y_pred, self.propensity_net_a
        )
        d_theta = compute_d_theta(x, a, p_m_given_x_a, p_m_given_x_a0, self.propensity_net_a)
        
        # Apply fair adjustment
        te_fair = self.compute_fair_predictions(te_unconstrained, pe_hat, d_theta)
        
        return te_fair.unsqueeze(1)
    
    def compute_ite_unconstrained(self, x, a):
        """
        Compute unconstrained Individual Treatment Effect (ITE) without fairness adjustment.
        Useful for debugging and comparison.
        
        Parameters:
        -----------
        x : torch.Tensor
            Covariates
        a : torch.Tensor
            Sensitive attribute
        
        Returns:
        --------
        ite : torch.Tensor
            Unconstrained Individual Treatment Effects
        """
        # Get counterfactual mediator M(A=0) to block A->Y path
        m_a0 = self.dce_module.get_mediator_predictions(x, torch.zeros_like(a))
        
        # Predict Y(1) and Y(0) using counterfactual mediator and treatment T
        batch_size = x.shape[0]
        t_1 = torch.ones(batch_size, device=x.device)
        t_0 = torch.zeros(batch_size, device=x.device)
        
        input_y1 = torch.cat([x, m_a0, t_1.unsqueeze(1)], dim=1)
        y1_pred = self.outcome_net(input_y1)
        
        input_y0 = torch.cat([x, m_a0, t_0.unsqueeze(1)], dim=1)
        y0_pred = self.outcome_net(input_y0)
        
        te_unconstrained = y1_pred - y0_pred
        
        return te_unconstrained

