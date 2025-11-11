"""
Model training and prediction for wildfire prediction.
Implements baseline and ensemble models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import optuna
from optuna.samplers import TPESampler
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import config
from src.preprocessing import (
    load_dataset, prepare_train_test_split, preprocess_features,
    select_features, save_preprocessor
)


class WildfireModel:
    """Base class for wildfire prediction models."""
    
    def __init__(self, model_type: str, **kwargs):
        """
        Initialize model.
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'xgboost', 'lightgbm')
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.feature_names = None
        self.scaler = None
    
    def _create_model(self, **kwargs):
        """Create model instance based on type."""
        if self.model_type == 'logistic':
            params = {**config.LOGISTIC_REGRESSION_PARAMS, **kwargs}
            return LogisticRegression(**params)
        elif self.model_type == 'random_forest':
            params = {**config.RANDOM_FOREST_PARAMS, **kwargs}
            return RandomForestClassifier(**params)
        elif self.model_type == 'xgboost':
            params = {**config.XGBOOST_PARAMS, **kwargs}
            return XGBClassifier(**params)
        elif self.model_type == 'lightgbm':
            params = {**config.LIGHTGBM_PARAMS, **kwargs}
            return LGBMClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print(f"Training {self.model_type} model...")
        
        # XGBoost with early stopping
        if self.model_type == 'xgboost' and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        # LightGBM with early stopping
        elif self.model_type == 'lightgbm' and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lightgbm.early_stopping(50), lightgbm.log_evaluation(0)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        print(f"  {self.model_type} training complete")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (for tree-based models)."""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def save(self, filepath: Path):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.scaler = model_data['scaler']
        return instance


def train_all_models(dataset_name: str = "synthetic") -> Dict[str, WildfireModel]:
    """
    Train all models on the specified dataset.
    
    Args:
        dataset_name: Name of dataset to use
    
    Returns:
        Dictionary of trained models
    """
    print("="*60)
    print(f"TRAINING ALL MODELS - {dataset_name.upper()} DATASET")
    print("="*60)
    print()
    
    # Load data
    df = load_dataset(dataset_name)
    
    # Prepare train/test split
    feature_cols = select_features(df)
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df, feature_cols=feature_cols
    )
    
    # Split training set for validation
    from sklearn.model_selection import train_test_split
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_SIZE, 
        random_state=config.RANDOM_STATE, stratify=y_train
    )
    
    # Preprocess features (different scaling for different models)
    models = {}
    
    # Logistic Regression (needs scaling)
    print("\n1. Training Logistic Regression...")
    X_train_scaled, X_test_scaled, scaler = preprocess_features(
        X_train, X_test, scaler_type='standard'
    )
    X_train_sub_scaled, X_val_scaled, _ = preprocess_features(
        X_train_sub, X_val, scaler_type='standard'
    )
    
    lr_model = WildfireModel('logistic')
    lr_model.feature_names = feature_cols
    lr_model.scaler = scaler
    lr_model.train(X_train_scaled, y_train)
    models['logistic_regression'] = lr_model
    
    # Tree-based models (no scaling needed)
    X_train_raw = X_train.values
    X_test_raw = X_test.values
    X_train_sub_raw = X_train_sub.values
    X_val_raw = X_val.values
    
    # Random Forest
    print("\n2. Training Random Forest...")
    rf_model = WildfireModel('random_forest')
    rf_model.feature_names = feature_cols
    rf_model.train(X_train_raw, y_train)
    models['random_forest'] = rf_model
    
    # XGBoost
    print("\n3. Training XGBoost...")
    xgb_model = WildfireModel('xgboost')
    xgb_model.feature_names = feature_cols
    xgb_model.train(X_train_sub_raw, y_train_sub, X_val_raw, y_val)
    models['xgboost'] = xgb_model
    
    # LightGBM
    print("\n4. Training LightGBM...")
    lgb_model = WildfireModel('lightgbm')
    lgb_model.feature_names = feature_cols
    lgb_model.train(X_train_sub_raw, y_train_sub, X_val_raw, y_val)
    models['lightgbm'] = lgb_model
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    for name, model in models.items():
        model_path = config.MODELS_DIR / f"{name}_{dataset_name}.pkl"
        model.save(model_path)
    
    # Save preprocessor
    preprocessor_path = config.MODELS_DIR / f"preprocessor_{dataset_name}.pkl"
    save_preprocessor(scaler, feature_cols, preprocessor_path)
    
    # Save test data for evaluation
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_cols
    }
    test_path = config.MODELS_DIR / f"test_data_{dataset_name}.pkl"
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"Test data saved to {test_path}")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return models


def hyperparameter_tuning_xgboost(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Hyperparameter tuning for XGBoost using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of trials for optimization
    
    Returns:
        Best parameters
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'random_state': config.RANDOM_STATE
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        from sklearn.metrics import roc_auc_score
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        
        return score
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train wildfire prediction models')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'uci'],
                       help='Dataset to use for training')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Train models
    models = train_all_models(args.dataset)
    
    # Optional: hyperparameter tuning
    if args.tune:
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Load data for tuning
        df = load_dataset(args.dataset)
        feature_cols = select_features(df)
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df, feature_cols=feature_cols
        )
        
        from sklearn.model_selection import train_test_split
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=config.VAL_SIZE,
            random_state=config.RANDOM_STATE, stratify=y_train
        )
        
        best_params = hyperparameter_tuning_xgboost(
            X_train_sub.values, y_train_sub, X_val.values, y_val,
            n_trials=config.OPTUNA_N_TRIALS
        )
        
        # Train model with best parameters
        print("\nTraining XGBoost with optimized parameters...")
        tuned_model = WildfireModel('xgboost', **best_params)
        tuned_model.feature_names = feature_cols
        tuned_model.train(X_train.values, y_train)
        
        # Save tuned model
        tuned_path = config.MODELS_DIR / f"xgboost_tuned_{args.dataset}.pkl"
        tuned_model.save(tuned_path)


if __name__ == "__main__":
    main()
