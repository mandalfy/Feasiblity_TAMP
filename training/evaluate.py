"""
Model evaluation utilities.

Computes classification metrics and generates analysis plots.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "auto",
    use_images: bool = False,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run on
        use_images: Whether model uses image inputs
        threshold: Classification threshold
        
    Returns:
        Dict of metrics
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        
    model = model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if use_images:
                images = batch['image'].to(device)
                actions = batch['action'].to(device)
                probs = model(images, actions)
            else:
                inputs = batch['input'].to(device)
                probs = model(inputs)
                
            labels = batch['label']
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs >= threshold).astype(float)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    metrics['auc'] = auc(fpr, tpr)
    
    # Store raw outputs for plotting
    metrics['_probs'] = all_probs
    metrics['_labels'] = all_labels
    metrics['_preds'] = all_preds
    metrics['_fpr'] = fpr
    metrics['_tpr'] = tpr
    
    return metrics


def print_evaluation_report(metrics: Dict[str, float]):
    """Print a formatted evaluation report."""
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"AUC-ROC:   {metrics['auc']*100:.2f}%")
    print("="*50)
    
    # Confusion matrix
    cm = confusion_matrix(metrics['_labels'], metrics['_preds'])
    print("\nConfusion Matrix:")
    print(f"  True Neg (Correct rejection): {cm[0,0]}")
    print(f"  False Pos (False alarm):      {cm[0,1]}")
    print(f"  False Neg (Miss):             {cm[1,0]}")
    print(f"  True Pos (Hit):               {cm[1,1]}")


def plot_evaluation(
    metrics: Dict[str, float],
    save_dir: str = "plots/",
    show: bool = False
):
    """
    Generate evaluation plots.
    
    Args:
        metrics: Metrics dict from evaluate_model
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROC Curve
    ax = axes[0, 0]
    ax.plot(metrics['_fpr'], metrics['_tpr'], 'b-', lw=2, 
            label=f'ROC (AUC = {metrics["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confusion Matrix
    ax = axes[0, 1]
    cm = confusion_matrix(metrics['_labels'], metrics['_preds'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Infeasible', 'Feasible'])
    ax.set_yticklabels(['Infeasible', 'Feasible'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    fontsize=14, color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=ax)
    
    # Probability Distribution
    ax = axes[1, 0]
    probs = metrics['_probs'].flatten()
    labels = metrics['_labels'].flatten()
    
    ax.hist(probs[labels == 0], bins=30, alpha=0.7, label='Infeasible', color='red')
    ax.hist(probs[labels == 1], bins=30, alpha=0.7, label='Feasible', color='green')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Distribution')
    ax.legend()
    
    # Metrics Bar Chart
    ax = axes[1, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['auc']
    ]
    
    bars = ax.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    
    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation.png'), dpi=150)
    
    if show:
        plt.show()
    plt.close()
    
    print(f"Plots saved to {save_dir}")


def find_optimal_threshold(
    metrics: Dict[str, float],
    optimize_for: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        metrics: Metrics dict with raw predictions
        optimize_for: Metric to optimize ('f1', 'accuracy', 'balanced')
        
    Returns:
        (optimal_threshold, best_score)
    """
    probs = metrics['_probs']
    labels = metrics['_labels']
    
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(float)
        
        if optimize_for == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        elif optimize_for == 'accuracy':
            score = accuracy_score(labels, preds)
        elif optimize_for == 'balanced':
            # Balance precision and recall
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec + 1e-8)
        else:
            raise ValueError(f"Unknown optimization target: {optimize_for}")
            
        if score > best_score:
            best_score = score
            best_threshold = thresh
            
    return best_threshold, best_score
