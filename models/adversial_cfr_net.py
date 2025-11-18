"""
Adversarial Debiasing CFRNet implementation.
This is a version of CFRNet with an adversarial component to reduce bias.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cfr_net import CFRNet


class AdversaryNetwork(nn.Module):
    """
    Adversary network for predicting sensitive attribute from representations.
    """
    def __init__(self, input_dim, hidden_dims=[100, 50], dropout=0.1):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input features (representations)
        hidden_dims : list
            Hidden layer dimensions for adversary network
        dropout : float
            Dropout probability
        """
        super(AdversaryNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer for binary classification (sensitive attribute)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the adversary network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input representations [batch_size, input_dim]
        
        Returns:
        --------
        a_pred : torch.Tensor
            Predicted sensitive attribute [batch_size, 1]
        """
        return self.network(x)


class AdversarialCFRNet(CFRNet):
    """
    Adversarial CFRNet: CFRNet with adversarial debiasing.
    
    This model learns balanced representations to handle selection bias in observational data,
    while also using an adversarial component to reduce bias with respect to sensitive attributes.
    """
    def __init__(self, input_dim, hidden_dims_rep=[200, 200], hidden_dims_hyp=[100, 100], 
                 dropout=0.1, alpha=1.0, use_pmmd=True, alpha_adv=1.0):
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
        alpha_adv : float
            Weight for adversarial loss
        """
        super(AdversarialCFRNet, self).__init__(
            input_dim=input_dim,
            hidden_dims_rep=hidden_dims_rep,
            hidden_dims_hyp=hidden_dims_hyp,
            dropout=dropout,
            alpha=alpha,
            use_pmmd=use_pmmd
        )
        
        self.alpha_adv = alpha_adv
        
        # Adversary network for predicting sensitive attribute
        self.adversary = AdversaryNetwork(
            input_dim=hidden_dims_rep[-1],
            hidden_dims=[100, 50],
            dropout=dropout
        )
    
    def forward(self, x, t=None, a=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features [batch_size, input_dim]
        t : torch.Tensor, optional
            Treatment indicator [batch_size]
        a : torch.Tensor, optional
            Sensitive attribute [batch_size]
        
        Returns:
        --------
        y_pred : torch.Tensor
            Predicted outcomes [batch_size, 1] or [batch_size, 2] if t is None
        phi : torch.Tensor
            Learned representations [batch_size, hidden_dim]
        a_pred : torch.Tensor
            Predicted sensitive attribute [batch_size, 1] (if a is provided)
        """
        # Get shared representation
        phi = self.representation_net(x)
        
        # Get predictions from both networks
        y0_pred = self.hypothesis_net_t0(phi)
        y1_pred = self.hypothesis_net_t1(phi)
        
        if t is not None:
            # Return prediction for the observed treatment
            y_pred = torch.where(t.unsqueeze(1) == 1, y1_pred, y0_pred)
        else:
            # Return both predictions (for counterfactual inference)
            y_pred = torch.cat([y0_pred, y1_pred], dim=1)
        
        # Get adversary prediction if sensitive attribute is provided
        a_pred = None
        if a is not None:
            a_pred = self.adversary(phi)
        
        return y_pred, phi, a_pred
    
    def compute_adversarial_loss(self, a_pred, a_true):
        """
        Compute adversarial loss (binary cross-entropy).
        
        Parameters:
        -----------
        a_pred : torch.Tensor
            Predicted sensitive attribute [batch_size, 1]
        a_true : torch.Tensor
            True sensitive attribute [batch_size]
        
        Returns:
        --------
        adv_loss : torch.Tensor
            Adversarial loss value
        """
        return F.binary_cross_entropy_with_logits(a_pred.squeeze(), a_true)
    
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
            _, phi, _ = self.forward(x)
            y0_pred = self.hypothesis_net_t0(phi)
            y1_pred = self.hypothesis_net_t1(phi)
            ite = y1_pred - y0_pred
            return ite