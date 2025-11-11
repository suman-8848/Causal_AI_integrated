"""
Utility functions for loading PyTorch checkpoints safely.
Handles PyTorch 2.6+ weights_only security feature.
"""
import torch
from sklearn.preprocessing import StandardScaler


def safe_load_checkpoint(filepath, map_location=None):
    """
    Safely load a PyTorch checkpoint that may contain sklearn objects.
    
    This function handles PyTorch 2.6+ default weights_only=True by either:
    1. Using weights_only=False (for trusted checkpoints)
    2. Or allowlisting sklearn classes
    
    Parameters:
    -----------
    filepath : str
        Path to the checkpoint file
    map_location : str or torch.device, optional
        Device to map the checkpoint to
    
    Returns:
    --------
    checkpoint : dict
        Loaded checkpoint dictionary
    """
    try:
        # Try loading with weights_only=True first (safer)
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
    except Exception as e:
        # If that fails (likely contains sklearn objects), use weights_only=False
        # This is safe for checkpoints saved by our own code
        if 'weights_only' in str(e) or 'Unsupported global' in str(e):
            checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        else:
            # If it's a different error, raise it
            raise
    
    return checkpoint


def safe_load_checkpoint_with_allowlist(filepath, map_location=None):
    """
    Load checkpoint using allowlist for sklearn objects (alternative method).
    
    This is the more secure approach if you want to use weights_only=True
    but allow specific sklearn classes.
    
    Parameters:
    -----------
    filepath : str
        Path to the checkpoint file
    map_location : str or torch.device, optional
        Device to map the checkpoint to
    
    Returns:
    --------
    checkpoint : dict
        Loaded checkpoint dictionary
    """
    # Allowlist sklearn preprocessing classes
    torch.serialization.add_safe_globals([StandardScaler])
    
    try:
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
    finally:
        # Note: add_safe_globals is persistent, so we don't need to remove it
        pass
    
    return checkpoint

