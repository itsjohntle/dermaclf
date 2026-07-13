from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_dermatology_data():
    """Fetch the UCI dermatology dataset.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Feature matrix (X) and target labels (y).
    """
    dermatology_data = fetch_ucirepo(id=33)

    X = dermatology_data.data.features
    y = dermatology_data.data.targets

    return X, y

def split_data(X, y, test_size=0.2, stratify=y, random_state=42):
    """Splits features and targets into training and test sets.

    Args:
        X (pandas.DataFrame): Feature matrix.
        y (pandas.DataFrame): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        straify (array_like): Labels used to straify the split.
        random-state (int): Seed for reproducibile shuffling/splitting

    Returns:
        list: [X_train, X_test, y_train, y_test]
    """
    return train_test_split(X, y, test_size, stratify, random_state)
