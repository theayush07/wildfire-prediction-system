"""
Data preprocessing utilities for wildfire prediction.
Handles data cleaning, splitting, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path
from typing import Tuple, List, Optional
import config


def load_dataset(dataset_name: str = "synthetic") -> pd.DataFrame:
    """
    Load a processed dataset.
    
    Args:
        dataset_name: Name of the dataset ('synthetic' or 'uci')
    
    Returns:
        DataFrame with the dataset
    """
    if dataset_name == "synthetic":
        file_path = config.FEATURES_DIR / "engineered_synthetic_data.csv"
    elif dataset_name == "uci":
        file_path = config.FEATURES_DIR / "engineered_uci_data.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {dataset_name} dataset: {df.shape}")
    return df


def select_features(df: pd.DataFrame, feature_groups: Optional[List[str]] = None) -> List[str]:
    """
    Select features for modeling.
    
    Args:
        df: Input DataFrame
        feature_groups: List of feature group names to include
    
    Returns:
        List of feature column names
    """
    if feature_groups is None:
        # Use all available numeric features except target and identifiers
        exclude_cols = [config.TARGET_COLUMN, config.DATE_COLUMN, 'date', 
                       'month', 'day', 'year', 'burned_area', 'area', 'X', 'Y']
        features = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    else:
        features = []
        for group in feature_groups:
            if group == 'weather':
                features.extend([f for f in config.BASE_WEATHER_FEATURES if f in df.columns])
            elif group == 'environmental':
                features.extend([f for f in config.ENVIRONMENTAL_FEATURES if f in df.columns])
            elif group == 'temporal':
                features.extend([f for f in config.TEMPORAL_FEATURES if f in df.columns])
    
    # Remove duplicates and ensure they exist
    features = list(dict.fromkeys(features))
    features = [f for f in features if f in df.columns]
    
    return features


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = None,
    feature_cols: List[str] = None,
    test_size: float = None,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        feature_cols: List of feature columns
        test_size: Test set proportion
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    # Select features
    if feature_cols is None:
        feature_cols = select_features(df)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape}, Fire rate: {y_train.mean()*100:.2f}%")
    print(f"Test set: {X_test.shape}, Fire rate: {y_test.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = 'standard',
    handle_missing: bool = True
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Preprocess features: handle missing values and scale.
    
    Args:
        X_train: Training features
        X_test: Test features
        scaler_type: Type of scaler ('standard', 'robust', or 'none')
        handle_missing: Whether to impute missing values
    
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Handle missing values
    if handle_missing:
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
    # Scale features
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'none':
        return X_train.values, X_test.values, None
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessor(scaler, feature_names: List[str], output_path: Path):
    """Save scaler and feature names for later use."""
    preprocessor_data = {
        'scaler': scaler,
        'feature_names': feature_names
    }
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessor_data, f)
    print(f"Saved preprocessor to {output_path}")


def load_preprocessor(input_path: Path):
    """Load saved scaler and feature names."""
    with open(input_path, 'rb') as f:
        preprocessor_data = pickle.load(f)
    return preprocessor_data['scaler'], preprocessor_data['feature_names']


def create_class_weights(y: pd.Series) -> dict:
    """
    Create class weights for imbalanced datasets.
    
    Args:
        y: Target series
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    print(f"Class weights: {class_weights}")
    return class_weights
