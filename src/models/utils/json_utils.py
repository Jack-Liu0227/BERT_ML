import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_metrics(metrics, save_path):
    """
    Save metrics to JSON file
    
    Args:
        metrics (dict): Metrics dictionary to save
        save_path (str): Path to save the JSON file
    """
    with open(save_path, 'w',encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder,ensure_ascii=False)

def load_metrics(load_path):
    """
    Load metrics from JSON file
    
    Args:
        load_path (str): Path to load the JSON file from
    
    Returns:
        dict: Loaded metrics dictionary
    """
    with open(load_path, 'r',encoding='utf-8') as f:
        return json.load(f) 