"""
Unit tests for model training and prediction.
"""

import pytest
import numpy as np
import pandas as pd
from src.models import WildfireModel


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 200
    
    X_train = np.random.randn(n, 5)
    y_train = np.random.binomial(1, 0.3, n)
    X_test = np.random.randn(50, 5)
    y_test = np.random.binomial(1, 0.3, 50)
    
    feature_names = [f'feature_{i}' for i in range(5)]
    
    return X_train, y_train, X_test, y_test, feature_names


def test_logistic_regression(sample_training_data):
    """Test logistic regression model."""
    X_train, y_train, X_test, y_test, feature_names = sample_training_data
    
    model = WildfireModel('logistic')
    model.feature_names = feature_names
    model.train(X_train, y_train)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    assert len(predictions) == len(X_test)
    assert len(probabilities) == len(X_test)
    assert probabilities.shape[1] == 2
    assert all(0 <= p <= 1 for p in probabilities[:, 1])


def test_random_forest(sample_training_data):
    """Test random forest model."""
    X_train, y_train, X_test, y_test, feature_names = sample_training_data
    
    model = WildfireModel('random_forest', n_estimators=10)
    model.feature_names = feature_names
    model.train(X_train, y_train)
    
    predictions = model.predict(X_test)
    importance = model.get_feature_importance()
    
    assert len(predictions) == len(X_test)
    assert importance is not None
    assert len(importance) == len(feature_names)


def test_feature_importance(sample_training_data):
    """Test feature importance extraction."""
    X_train, y_train, X_test, y_test, feature_names = sample_training_data
    
    model = WildfireModel('random_forest', n_estimators=10)
    model.feature_names = feature_names
    model.train(X_train, y_train)
    
    importance_df = model.get_feature_importance()
    
    assert importance_df is not None
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert len(importance_df) == len(feature_names)
