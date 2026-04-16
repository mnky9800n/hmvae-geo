"""CLI entry points for the LilyVAE pipeline."""

import numpy as np


def make_serializable(obj):
    """Recursively convert numpy types to Python built-ins for json.dump."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj
