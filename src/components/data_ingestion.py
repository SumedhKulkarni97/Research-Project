"""
data_ingestion.py

This script handles all data loading and preprocessing tasks for the MNIST dataset.
It fetches the data, binarizes it, and separates it into positive and
negative example sets based on a target digit.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from tqdm import tqdm

def load_and_prepare_data(target_digit_str, n_positive, n_negative):
    """
    Loads, binarizes, and subsets the MNIST dataset.

    Args:
        target_digit_str (str): The digit to learn (e.g., '7').
        n_positive (int): The number of positive examples to include in the subset.
        n_negative (int): The number of negative examples to include in the subset.

    Returns:
        tuple: A tuple containing (E_pos, E_neg, n_features).
               E_pos: A numpy array of positive examples.
               E_neg: A numpy array of negative examples.
               n_features: The number of features (pixels) in each example.
    """
    print("Loading MNIST dataset...")
    # Using 'auto' parser is often more robust
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data
    y = mnist.target
    print("Dataset loaded.")

    # --- Binarize the data ---
    # Convert pixel values to 0 or 1 based on a threshold
    threshold = 128
    X_bin = (X > threshold).astype(int)
    n_features = X_bin.shape[1]

    # --- Create Positive and Negative Example Sets ---
    positive_indices = np.where(y == target_digit_str)[0]
    negative_indices = np.where(y != target_digit_str)[0]

    # --- Subsampling for feasibility ---
    # Shuffle and select the subset
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)
    
    pos_subset_indices = positive_indices[:n_positive]
    neg_subset_indices = negative_indices[:n_negative]

    E_pos = X_bin[pos_subset_indices]
    E_neg = X_bin[neg_subset_indices]

    print(f"\nData prepared for digit '{target_digit_str}':")
    print(f" - Number of features (pixels): {n_features}")
    print(f" - Positive examples in subset: {len(E_pos)}")
    print(f" - Negative examples in subset: {len(E_neg)}")
    
    return E_pos, E_neg, n_features
