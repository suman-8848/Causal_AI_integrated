"""
Data loading utilities for the IHDP dataset.
"""
import pandas as pd
import numpy as np


def load_ihdp_data(fold=1):
    """
    Load the IHDP dataset for a specific fold.
    
    Parameters:
    -----------
    fold : int
        The fold number (1-10 for IHDP dataset)
    
    Returns:
    --------
    data : pd.DataFrame
        The loaded dataset with all columns
    """
    url = f"https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{fold}.csv"
    
    # Define the column names based on the dataset's documentation
    col_names = ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1'] + [f'x{i}' for i in range(1, 26)]
    
    # Load the dataset
    data = pd.read_csv(url, header=None, names=col_names)
    
    return data


def preprocess_ihdp_data(data):
    """
    Preprocess the IHDP data and extract relevant variables.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw IHDP dataset
    
    Returns:
    --------
    T : np.ndarray
        Treatment variable
    Y : np.ndarray
        Outcome variable (factual)
    Y_cf : np.ndarray
        Counterfactual outcome
    mu0 : np.ndarray
        Ground truth potential outcome under T=0
    mu1 : np.ndarray
        Ground truth potential outcome under T=1
    A : np.ndarray
        Sensitive attribute (x10)
    X : pd.DataFrame
        Covariates (excluding treatment, outcomes, and sensitive attribute)
    """
    # Extract variables
    T = data['treatment'].values
    Y = data['y_factual'].values
    Y_cf = data['y_cfactual'].values
    mu0 = data['mu0'].values  # Ground truth potential outcome under T=0
    mu1 = data['mu1'].values  # Ground truth potential outcome under T=1
    A = data['x10'].values  # Sensitive attribute
    X = data.drop(columns=['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1', 'x10'])
    
    return T, Y, Y_cf, mu0, mu1, A, X


if __name__ == "__main__":
    # Test the data loading
    data = load_ihdp_data(fold=1)
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nDataset Information:")
    print(data.info())
    print("\nDataset Shape:", data.shape)
    
    # Test preprocessing
    T, Y, Y_cf, mu0, mu1, A, X = preprocess_ihdp_data(data)
    print("\nPreprocessed data shapes:")
    print(f"T (treatment): {T.shape}")
    print(f"Y (outcome): {Y.shape}")
    print(f"Y_cf (counterfactual): {Y_cf.shape}")
    print(f"mu0 (ground truth Y0): {mu0.shape}")
    print(f"mu1 (ground truth Y1): {mu1.shape}")
    print(f"A (sensitive attribute): {A.shape}")
    print(f"X (covariates): {X.shape}")

