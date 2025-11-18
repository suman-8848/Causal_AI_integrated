"""
Naive Fair CFRNet implementation.
This is a version of CFRNet where the sensitive attribute A is removed from the input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cfr_net import CFRNet


class NaiveFairCFRNet(CFRNet):
    """
    Naive Fair CFRNet: CFRNet with sensitive attribute A removed from input.
    
    This model learns balanced representations to handle selection bias in observational data,
    but does not explicitly consider fairness constraints. It simply removes the sensitive
    attribute from the input features.
    """
    def __init__(self, input_dim, hidden_dims_rep=[200, 200], hidden_dims_hyp=[100, 100], 
                 dropout=0.1, alpha=1.0, use_pmmd=True):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input features (excluding sensitive attribute)
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
        super(NaiveFairCFRNet, self).__init__(
            input_dim=input_dim,
            hidden_dims_rep=hidden_dims_rep,
            hidden_dims_hyp=hidden_dims_hyp,
            dropout=dropout,
            alpha=alpha,
            use_pmmd=use_pmmd
        )
    
    def forward(self, x, t=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features [batch_size, input_dim] (without sensitive attribute)
        t : torch.Tensor, optional
            Treatment indicator [batch_size]
        
        Returns:
        --------
        y_pred : torch.Tensor
            Predicted outcomes [batch_size, 1] or [batch_size, 2] if t is None
        phi : torch.Tensor
            Learned representations [batch_size, hidden_dim]
        """
        # Use parent class forward method
        return super().forward(x, t)