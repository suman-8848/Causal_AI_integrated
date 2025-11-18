"""
TARNet (Treatment-Agnostic Representation Network) implementation.
Based on the paper: "Estimating Individual Treatment Effect: Generalization Bounds and Algorithms" by Shalit et al.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TARNet(nn.Module):
    """
    Treatment-Agnostic Representation Network (TARNet).
    
    This model learns a shared representation of covariates that is independent of treatment assignment.
    It uses separate output heads for treatment and control groups.
    """
    def __init__(self, input_dim, hidden_dims_rep=[200, 200], hidden_dims_out=[100, 100], 
                 dropout=0.1):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dims_rep : list
            Hidden layer dimensions for representation network
        hidden_dims_out : list
            Hidden layer dimensions for output networks
        dropout : float
            Dropout probability
        """
        super(TARNet, self).__init__()
        
        # Shared representation network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims_rep:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.representation_net = nn.Sequential(*layers)
        
        # Output networks for treatment and control
        self.treatment_net = nn.Sequential(
            nn.Linear(hidden_dims_rep[-1], hidden_dims_out[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims_out[0], hidden_dims_out[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims_out[1], 1)
        )
        
        self.control_net = nn.Sequential(
            nn.Linear(hidden_dims_rep[-1], hidden_dims_out[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims_out[0], hidden_dims_out[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims_out[1], 1)
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
        y0_pred = self.control_net(phi)
        y1_pred = self.treatment_net(phi)
        
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
            y0_pred = self.control_net(phi)
            y1_pred = self.treatment_net(phi)
            ite = y1_pred - y0_pred
            return ite
    
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
            Bandwidth parameter for RBF kernel
        
        Returns:
        --------
        mmd : torch.Tensor
            MMD value
        """
        def rbf_kernel(x1, x2):
            pairwise_dists = torch.cdist(x1, x2) ** 2
            return torch.exp(-pairwise_dists / (2 * sigma ** 2))
        
        K_xx = rbf_kernel(x, x).mean()
        K_yy = rbf_kernel(y, y).mean()
        K_xy = rbf_kernel(x, y).mean()
        
        mmd = K_xx + K_yy - 2 * K_xy
        return mmd