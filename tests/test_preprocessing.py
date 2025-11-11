"""
Unit tests for preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    select_features, prepare_train_test_split, 
    preprocess_features, create_class_weights
)


@pytest.fixture
def sample_data():
    """Create sample wildfire data for testing."""
    np.random.seed(42)
    n = 100
    
    data = {
        'date': pd.date_range('2020-01-01', periods=n),
        'temp_max': np.random.uniform(20, 40, n),
        'temp_min': np.random.uniform(10, 25, n),
        'relative_humidity': np.random.uniform(20, 80, n),
        'wind_speed': np.random.uniform(0, 20, n),
        'precipitation': np.random.uniform(0, 10, n),
        'fire_occurred': np.random.binomial(1, 0.3, n)
    }
    
    return pd.DataFrame(data)


def test_select_features(sample_data):
    """Test feature selection."""
    features = select_features(sample_data)
    
    assert isinstance(features, list)
    assert len(features) > 0
    assert 'fire_occurred' not in features
    assert 'date' not in features


def test_prepare_train_test_split(sample_data):
    """Test train-test split."""
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        sample_data, test_size=0.2, random_state=42
    )
    
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]


def test_preprocess_features(sample_data):
    """Test feature preprocessing."""
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        sample_data, test_size=0.2, random_state=42
    )
    
    X_train_scaled, X_test_scaled, scaler = preprocess_features(
        X_train, X_test, scaler_type='standard'
    )
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert scaler is not None
    
    # Check that scaling was applied (mean ~ 0, std ~ 1)
    assert abs(X_train_scaled.mean()) < 0.5
    assert abs(X_train_scaled.std() - 1.0) < 0.5


def test_create_class_weights(sample_data):
    """Test class weight calculation."""
    y = sample_data['fire_occurred']
    weights = create_class_weights(y)
    
    assert isinstance(weights, dict)
    assert 0 in weights
    assert 1 in weights
    assert all(w > 0 for w in weights.values())
