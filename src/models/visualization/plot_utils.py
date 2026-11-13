"""Compatibility shim forwarding to Scripts.figures.plot_utils."""
from __future__ import annotations

try:
    from Scripts.figures.plot_utils import (
        plot_training_curves,
        plot_cv_curves,
        plot_compare_scatter,
        plot_prediction_scatter,
        plot_all_sets_compare_scatter,
        plot_cv_error_boxplot,
        plot_feature_importance,
        plot_loss_r2_curves,
        plot_true_vs_pred_scatter,
    )
except ImportError:
    def plot_training_curves(*args, **kwargs): pass
    def plot_cv_curves(*args, **kwargs): pass
    def plot_compare_scatter(*args, **kwargs): pass
    def plot_prediction_scatter(*args, **kwargs): pass
    def plot_all_sets_compare_scatter(*args, **kwargs): pass
    def plot_cv_error_boxplot(*args, **kwargs): pass
    def plot_feature_importance(*args, **kwargs): pass
    def plot_loss_r2_curves(*args, **kwargs): pass
    def plot_true_vs_pred_scatter(*args, **kwargs): pass
