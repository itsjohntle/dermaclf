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


