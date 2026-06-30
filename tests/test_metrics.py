"""Tests for Metrics Calculation"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import MetricCalculator


class TestMetricCalculator:
    """Test metric calculations"""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions"""
        return {
            'y_true': np.array([0, 1, 1, 0, 1, 0, 0, 1]),
            'y_pred': np.array([0, 1, 1, 0, 0, 0, 1, 1]),
            'y_prob': np.array([
                [0.9, 0.1],  # True 0, Pred 0 ✓
                [0.2, 0.8],  # True 1, Pred 1 ✓
                [0.1, 0.9],  # True 1, Pred 1 ✓
                [0.8, 0.2],  # True 0, Pred 0 ✓
                [0.6, 0.4],  # True 1, Pred 0 ✗
                [0.7, 0.3],  # True 0, Pred 0 ✓
                [0.3, 0.7],  # True 0, Pred 1 ✗
                [0.1, 0.9],  # True 1, Pred 1 ✓
            ])
        }

    def test_accuracy(self, sample_predictions):
        """Test accuracy calculation"""
        calc = MetricCalculator()
        metrics = calc.calculate_all_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_prob']
        )

        # 6 corretos de 8 = 75%
        assert np.isclose(metrics['accuracy'], 0.75), f"Expected 0.75, got {metrics['accuracy']}"

    def test_precision(self, sample_predictions):
        """Test precision calculation"""
        calc = MetricCalculator()
        metrics = calc.calculate_all_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_prob']
        )

        # Should be a valid precision value
        assert 0 <= metrics['precision'] <= 1, f"Precision {metrics['precision']} out of range"

    def test_recall(self, sample_predictions):
        """Test recall calculation"""
        calc = MetricCalculator()
        metrics = calc.calculate_all_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_prob']
        )

        # Should be a valid recall value
        assert 0 <= metrics['recall'] <= 1, f"Recall {metrics['recall']} out of range"

    def test_f1_score(self, sample_predictions):
        """Test F1 score calculation"""
        calc = MetricCalculator()
        metrics = calc.calculate_all_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_prob']
        )

        # Should be a valid F1 value
        assert 0 <= metrics['f1_score'] <= 1, f"F1-Score {metrics['f1_score']} out of range"

    def test_confusion_matrix(self, sample_predictions):
        """Test confusion matrix"""
        calc = MetricCalculator()
        metrics = calc.calculate_all_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_prob']
        )

        # Should have correct shape
        assert metrics['confusion_matrix'].shape == (2, 2)

    def test_auc_roc(self, sample_predictions):
        """Test AUC-ROC calculation"""
        calc = MetricCalculator()
        metrics = calc.calculate_all_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_prob']
        )

        # Should be between 0 and 1
        if metrics['auc_roc'] is not None:
            assert 0 <= metrics['auc_roc'] <= 1, f"AUC-ROC {metrics['auc_roc']} out of range [0, 1]"

    def test_all_metrics(self, sample_predictions):
        """Test all metrics at once"""
        calc = MetricCalculator()
        metrics = calc.calculate_all_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_prob']
        )

        required_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'confusion_matrix']
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
            if key != 'auc_roc' and key != 'confusion_matrix':  # auc_roc can be None, confusion_matrix is array
                assert isinstance(metrics[key], (float, np.floating))

    def test_metrics_in_range(self, sample_predictions):
        """Test all metrics are in valid range"""
        calc = MetricCalculator()
        metrics = calc.calculate_all_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_prob']
        )

        # Skip confusion_matrix and None values
        for metric_name, metric_value in metrics.items():
            if metric_name == 'confusion_matrix' or metric_value is None:
                continue
            assert 0 <= metric_value <= 1, \
                f"{metric_name}={metric_value} out of range [0, 1]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
