"""Tests for Alert System"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alert_system import (
    StressLevel, AlertThresholds, StressAlert, StressDetector
)


class TestStressLevel:
    """Test StressLevel enum"""

    def test_stress_levels_exist(self):
        """Test all stress levels are defined"""
        assert hasattr(StressLevel, 'NORMAL')
        assert hasattr(StressLevel, 'MILD')
        assert hasattr(StressLevel, 'MODERATE')
        assert hasattr(StressLevel, 'SEVERE')

    def test_stress_level_values(self):
        """Test stress level values"""
        assert StressLevel.NORMAL.value == 0
        assert StressLevel.MILD.value == 1
        assert StressLevel.MODERATE.value == 2
        assert StressLevel.SEVERE.value == 3


class TestAlertThresholds:
    """Test alert thresholds"""

    def test_default_thresholds(self):
        """Test default thresholds are set"""
        thresholds = AlertThresholds()

        assert hasattr(thresholds, 'f1_max_threshold')
        assert hasattr(thresholds, 'youden_threshold')
        assert hasattr(thresholds, 'severe_threshold')

    def test_threshold_values(self):
        """Test threshold values are in correct range"""
        thresholds = AlertThresholds()

        assert 0 < thresholds.f1_max_threshold < 1
        assert 0 < thresholds.youden_threshold < 1
        assert 0 < thresholds.severe_threshold < 1
        # Youden is more lenient than severe
        assert thresholds.youden_threshold < thresholds.severe_threshold

    def test_classify_stress_level(self):
        """Test stress level classification"""
        thresholds = AlertThresholds()

        # Low confidence = NORMAL
        assert thresholds.classify_stress_level(0.2) == StressLevel.NORMAL
        # Mid confidence = MILD or MODERATE
        level_mid = thresholds.classify_stress_level(0.5)
        assert level_mid in [StressLevel.MILD, StressLevel.MODERATE]
        # High confidence = SEVERE
        assert thresholds.classify_stress_level(0.75) == StressLevel.SEVERE


class TestStressAlert:
    """Test StressAlert dataclass"""

    def test_alert_creation(self):
        """Test creating an alert"""
        from datetime import datetime

        alert = StressAlert(
            timestamp=datetime.now(),
            plant_id='plant_001',
            stress_level=StressLevel.MODERATE,
            confidence=0.85,
            detected_anomalies=['color_shift', 'wilting', 'temperature_spike'],
            visual_indicators=['color_shift', 'wilting'],
            temporal_indicators=['temperature_spike'],
            recommendation='Adjust temperature and increase ventilation'
        )

        assert alert.plant_id == 'plant_001'
        assert alert.confidence == 0.85
        assert alert.stress_level == StressLevel.MODERATE
        assert len(alert.visual_indicators) == 2
        assert len(alert.temporal_indicators) == 1
        assert len(alert.detected_anomalies) == 3


class TestStressDetector:
    """Test stress detection logic"""

    def test_detector_creation(self):
        """Test creating a detector"""
        detector = StressDetector()
        assert detector is not None

    def test_detect_visual_anomalies(self):
        """Test visual anomaly detection"""
        detector = StressDetector()

        visual_features = {
            'color_shift_green': 0.4,
            'texture_variance': 0.6,
            'wilting_index': 0.35
        }

        anomalies = detector.detect_visual_anomalies(visual_features)

        # Should detect color shift and texture
        assert len(anomalies) >= 1

    def test_detect_temporal_anomalies(self):
        """Test temporal anomaly detection"""
        detector = StressDetector()

        temporal_features = {
            'temperature_volatility': 4.0,
            'humidity_variance': 0.3,
            'co2_deviation': 150
        }

        anomalies = detector.detect_temporal_anomalies(temporal_features)

        # Should detect temperature and humidity anomalies
        assert len(anomalies) >= 1

    def test_no_anomalies(self):
        """Test when no anomalies are present"""
        detector = StressDetector()

        visual_features = {
            'color_shift_green': 0.1,
            'texture_variance': 0.2,
            'wilting_index': 0.1
        }

        temporal_features = {
            'temperature_volatility': 1.0,
            'humidity_variance': 0.1,
            'co2_deviation': 50
        }

        visual_anom = detector.detect_visual_anomalies(visual_features)
        temporal_anom = detector.detect_temporal_anomalies(temporal_features)

        # Should not detect anomalies
        assert len(visual_anom) == 0
        assert len(temporal_anom) == 0

    def test_alert_generation(self):
        """Test alert generation with high confidence"""
        detector = StressDetector()

        # High stress features
        visual_features = {
            'color_shift_green': 0.5,
            'texture_variance': 0.7,
            'wilting_index': 0.6
        }

        temporal_features = {
            'temperature_volatility': 5.0,
            'humidity_variance': 0.4,
            'co2_deviation': 200
        }

        visual_anom = detector.detect_visual_anomalies(visual_features)
        temporal_anom = detector.detect_temporal_anomalies(temporal_features)

        # Should detect multiple anomalies
        assert len(visual_anom) > 0 or len(temporal_anom) > 0


class TestAlertThresholdLogic:
    """Test threshold-based alert levels"""

    def test_alert_level_normal(self):
        """Test normal level detection"""
        thresholds = AlertThresholds()
        confidence = 0.2  # Below all thresholds = NORMAL

        level = thresholds.classify_stress_level(confidence)
        assert level == StressLevel.NORMAL

    def test_alert_level_severe(self):
        """Test severe level detection"""
        thresholds = AlertThresholds()
        confidence = 0.95  # Above severe threshold

        level = thresholds.classify_stress_level(confidence)
        assert level == StressLevel.SEVERE

    def test_all_threshold_levels(self):
        """Test all threshold levels"""
        thresholds = AlertThresholds()

        test_cases = [
            (0.1, StressLevel.NORMAL),  # Below mild_threshold (0.4208)
            (0.421, StressLevel.MILD),  # Just above mild_threshold
            (0.522, StressLevel.MODERATE),  # Between moderate_threshold (0.5213) and severe_threshold (0.70)
            (0.75, StressLevel.SEVERE),  # Above severe_threshold (0.70)
        ]

        for confidence, expected_level in test_cases:
            level = thresholds.classify_stress_level(confidence)
            assert level == expected_level, \
                f"Confidence {confidence} should be {expected_level}, got {level}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
