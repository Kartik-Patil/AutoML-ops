import json
import os
from datetime import datetime
import csv
from pathlib import Path

def log_metrics(metrics):
    """
    Log and return model performance metrics
    
    Args:
        metrics (dict): Dictionary containing model metrics
    Returns:
        dict: The same metrics in a format ready for frontend display
    """
    timestamp = datetime.now().isoformat()
    
    # Create metrics storage directory if it doesn't exist
    storage_dir = Path('metrics_storage')
    storage_dir.mkdir(exist_ok=True)
    
    metrics_file = storage_dir / 'metrics_history.json'
    
    # Initialize or load existing metrics
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            try:
                metrics_history = json.load(f)
            except json.JSONDecodeError:
                metrics_history = []
    else:
        metrics_history = []
    
    # Add new metrics with timestamp
    metrics_entry = {
        'timestamp': timestamp,
        'metrics': metrics
    }
    metrics_history.append(metrics_entry)
    
    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    
    # Format metrics for frontend display
    display_metrics = {}
    for model_name, model_metrics in metrics.items():
        display_metrics[model_name] = {
            'accuracy': round(float(model_metrics.get('accuracy', 0)), 4),
            'precision': round(float(model_metrics.get('precision', 0)), 4),
            'recall': round(float(model_metrics.get('recall', 0)), 4),
            'f1': round(float(model_metrics.get('f1', 0)), 4),
            'auc': round(float(model_metrics.get('auc', 0)), 4),
            'logloss': round(float(model_metrics.get('logloss', 0)), 4)
        }
    
    return display_metrics

def get_latest_metrics():
    """
    Retrieve the most recent metrics
    Returns:
        dict: Latest metrics in a format ready for frontend display
    """
    storage_dir = Path('metrics_storage')
    metrics_file = storage_dir / 'metrics_history.json'
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            try:
                metrics_history = json.load(f)
                if metrics_history:
                    return metrics_history[-1]['metrics']
            except json.JSONDecodeError:
                return {}
    return {}