"""
Model evaluation and explainability for wildfire prediction.
Includes metrics, visualizations, and SHAP analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import shap
import pickle
from pathlib import Path
from typing import Dict, List
import config


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        model_name: Name of the model
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
    
    return metrics


def print_evaluation_report(metrics: Dict):
    """Print formatted evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT - {metrics['model']}")
    print(f"{'='*60}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        print(f"PR-AUC:        {metrics['pr_auc']:.4f}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add accuracy for each class
    for i in range(2):
        accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        plt.text(i, i, f'\n{accuracy:.2%}', ha='center', va='center', 
                color='white' if cm[i, i] > cm.max()/2 else 'black', fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path.name}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path.name}")
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path.name}")
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, model_name="Model", 
                           top_n=20, save_path=None):
    """Plot feature importance."""
    if importance_df is None or len(importance_df) == 0:
        return
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, max(6, len(top_features) * 0.3)))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'{model_name} - Top {len(top_features)} Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path.name}")
    plt.close()


def plot_shap_summary(model, X_sample, feature_names, model_name="Model", save_path=None):
    """
    Create SHAP summary plot.
    
    Args:
        model: Trained model
        X_sample: Sample of features for SHAP analysis
        feature_names: List of feature names
        model_name: Name of the model
        save_path: Path to save the plot
    """
    print(f"Generating SHAP values for {model_name}...")
    
    # Create explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.Explainer(model, X_sample)
    else:
        explainer = shap.Explainer(model, X_sample)
    
    # Calculate SHAP values
    shap_values = explainer(X_sample)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                     show=False, plot_size=(10, 8))
    plt.title(f'{model_name} - SHAP Summary Plot', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path.name}")
    plt.close()
    
    return shap_values


def plot_shap_waterfall(shap_values, sample_idx=0, save_path=None):
    """Plot SHAP waterfall for a single prediction."""
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[sample_idx], show=False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path.name}")
    plt.close()


def compare_models(results: List[Dict], save_path=None):
    """
    Compare multiple models side by side.
    
    Args:
        results: List of metric dictionaries
        save_path: Path to save comparison plot
    """
    df = pd.DataFrame(results)
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    if not available_metrics:
        return
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4*len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        axes[idx].bar(df['model'], df[metric], color='steelblue', alpha=0.7)
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylim([0, 1.0])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(df[metric]):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path.name}")
    plt.close()


def evaluate_all_models(dataset_name: str = "synthetic"):
    """
    Evaluate all trained models and generate reports.
    
    Args:
        dataset_name: Name of dataset used for training
    """
    print("="*60)
    print(f"EVALUATING ALL MODELS - {dataset_name.upper()} DATASET")
    print("="*60)
    
    # Load test data
    test_path = config.MODELS_DIR / f"test_data_{dataset_name}.pkl"
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    feature_names = test_data['feature_names']
    
    # Create results directory
    results_dir = config.RESULTS_DIR / f"evaluation_{dataset_name}"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Evaluate each model
    from src.models import WildfireModel
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    all_metrics = []
    
    for model_name in model_names:
        model_path = config.MODELS_DIR / f"{model_name}_{dataset_name}.pkl"
        
        if not model_path.exists():
            print(f"\nSkipping {model_name} - model file not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*60}")
        
        # Load model
        model = WildfireModel.load(model_path)
        
        # Prepare data
        if model.scaler is not None:
            X_test_prepared = model.scaler.transform(X_test)
        else:
            X_test_prepared = X_test.values
        
        # Make predictions
        y_pred = model.predict(X_test_prepared)
        y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
        all_metrics.append(metrics)
        print_evaluation_report(metrics)
        
        # Generate visualizations
        print("Generating visualizations...")
        plot_confusion_matrix(y_test, y_pred, model_name,
                            save_path=results_dir / f"{model_name}_confusion_matrix.png")
        plot_roc_curve(y_test, y_pred_proba, model_name,
                      save_path=results_dir / f"{model_name}_roc_curve.png")
        plot_precision_recall_curve(y_test, y_pred_proba, model_name,
                                   save_path=results_dir / f"{model_name}_pr_curve.png")
        
        # Feature importance
        importance_df = model.get_feature_importance()
        if importance_df is not None:
            plot_feature_importance(importance_df, model_name,
                                  save_path=results_dir / f"{model_name}_feature_importance.png")
            
            # Save feature importance to CSV
            importance_path = results_dir / f"{model_name}_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"  Saved: {importance_path.name}")
        
        # SHAP analysis (for tree-based models)
        if model_name in ['random_forest', 'xgboost', 'lightgbm']:
            # Sample data for SHAP (to reduce computation time)
            sample_size = min(config.SHAP_SAMPLE_SIZE, len(X_test))
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
            
            if not hasattr(X_sample, 'values'):
                X_sample = pd.DataFrame(X_sample, columns=feature_names)
            
            shap_values = plot_shap_summary(
                model.model, X_sample, feature_names, model_name,
                save_path=results_dir / f"{model_name}_shap_summary.png"
            )
    
    # Compare all models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    compare_models(all_metrics, save_path=results_dir / "model_comparison.png")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = results_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {results_dir.absolute()}")
    print(f"{'='*60}")
    
    return all_metrics


def main():
    """Main evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate wildfire prediction models')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'uci'],
                       help='Dataset used for training')
    
    args = parser.parse_args()
    
    evaluate_all_models(args.dataset)


if __name__ == "__main__":
    main()
