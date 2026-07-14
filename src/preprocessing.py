from sklearn.model_selection import train_test_split
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

    return (X, y)

def handle_missing_age(X):
    """Handles missing data in age feature.
    
    Args:
        X (pandas.DataFrame): Feature matrix.
    
    Returns:
        pandas.DataFrame: Copy of feature matrix (X) with missing values in age filled in with age median.
    """
    X = X.copy()
    X['age'] = X['age'].fillna(X['age'].median()) 

    return X

def validate_dataset(X, y):
    """Validates the dermatology dataset against expected data quality constraints
    
    Args:
        X (pandas.DataFrame): Feature matrix.
        y (pandas.DataFrame): Target labels.
        
    Raises:
        AssertionError: If any quality constraint is violated.
    """
    assert X.duplicated().sum() == 0
    assert X.drop(columns='age').apply(lambda c: c.between(0, 3)).all().all()
    assert X['family history'].isin([0, 1]).all()
    assert (X['age'].dropna() >= 0).all()
    assert X.index.equals(y.index)
    assert y['class'].isin(range(1, 7)).all()

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits features and targets into training and test sets.

    Args:
        X (pandas.DataFrame): Feature matrix.
        y (pandas.DataFrame): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducible shuffling/splitting

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)