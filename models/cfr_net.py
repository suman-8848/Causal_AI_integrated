"""
Baseline CFRNet (Counterfactual Regression Network) implementation.
Based on the original paper: "Learning Representations for Counterfactual Inference" by Shalit et al.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationNetwork(nn.Module):
    """
    Shared representation network used by both treatment and control branches.
    """
    def __init__(self, input_dim, hidden_dims=[200, 200], dropout=0.1):
        super(RepresentationNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class HypothesisNetwork(nn.Module):
    """
    Treatment-specific hypothesis network.
    """
    def __init__(self, input_dim, hidden_dims=[100, 100], dropout=0.1):
        super(HypothesisNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (single outcome prediction)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, phi):
        return self.network(phi)


class CFRNet(nn.Module):
    """
    Counterfactual Regression Network (CFRNet).
    
    This model learns balanced representations to handle selection bias in observational data.
    It uses IPM (Integral Probability Metric) to minimize the distance between treatment and control distributions.
    """
    def __init__(self, input_dim, hidden_dims_rep=[200, 200], hidden_dims_hyp=[100, 100], 
                 dropout=0.1, alpha=1.0, use_pmmd=True):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dims_rep : list
            Hidden layer dimensions for representation network
        hidden_dims_hyp : list
            Hidden layer dimensions for hypothesis networks
        dropout : float
            Dropout probability
        alpha : float
            Weight for IPM loss
        use_pmmd : bool
            Whether to use MMD (Maximum Mean Discrepancy) for IPM
        """
        super(CFRNet, self).__init__()
        
        self.alpha = alpha
        self.use_pmmd = use_pmmd
        
        # Shared representation network
        self.representation_net = RepresentationNetwork(
            input_dim, hidden_dims_rep, dropout
        )
        
        # Treatment-specific hypothesis networks
        self.hypothesis_net_t0 = HypothesisNetwork(
            hidden_dims_rep[-1], hidden_dims_hyp, dropout
        )
        self.hypothesis_net_t1 = HypothesisNetwork(
            hidden_dims_rep[-1], hidden_dims_hyp, dropout
        )
    
    def forward(self, x, t=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features [batch_size, input_dim]
        t : torch.Tensor, optional
            Treatment indicator [batch_size]
        
        Returns:
        --------
        y_pred : torch.Tensor
            Predicted outcomes [batch_size, 1] or [batch_size, 2] if t is None
        phi : torch.Tensor
            Learned representations [batch_size, hidden_dim]
        """
        # Get shared representation
        phi = self.representation_net(x)
        
        # Get predictions from both networks
        y0_pred = self.hypothesis_net_t0(phi)
        y1_pred = self.hypothesis_net_t1(phi)
        
        if t is not None:
            # Return prediction for the observed treatment
            y_pred = torch.where(t.unsqueeze(1) == 1, y1_pred, y0_pred)
            return y_pred, phi
        else:
            # Return both predictions (for counterfactual inference)
            return torch.cat([y0_pred, y1_pred], dim=1), phi
    
    def compute_ite(self, x):
        """
        Compute Individual Treatment Effect (ITE) for given inputs.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features
        
        Returns:
        --------
        ite : torch.Tensor
            Individual Treatment Effects (Y1 - Y0)
        """
        with torch.no_grad():
            _, phi = self.forward(x)
            y0_pred = self.hypothesis_net_t0(phi)
            y1_pred = self.hypothesis_net_t1(phi)
            ite = y1_pred - y0_pred
            return ite
    
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
            # Use MMD (Maximum Mean Discrepancy) as IPM
            return self._compute_mmd(phi_t0, phi_t1)
        else:
            # Use Wasserstein distance approximation
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
        # Simplified: mean distance between distribution means
        mean_x = x.mean(dim=0)
        mean_y = y.mean(dim=0)
        return torch.norm(mean_x - mean_y, p=2)

