"""Tests for Deep Learning Models"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    PhenotypicFeatureExtractor,
    TemporalSensorAnalyzer,
    MultimodalFusionModel,
    create_multimodal_model
)


class TestPhenotypicFeatureExtractor:
    """Test CNN feature extractor"""

    def test_forward_pass(self, device, sample_image):
        """Test CNN forward pass"""
        sample_image = sample_image.to(device)
        model = PhenotypicFeatureExtractor(input_channels=3, num_features=256).to(device)
        model.eval()

        with torch.no_grad():
            output = model(sample_image)

        assert output.shape == (1, 256), f"Expected shape (1, 256), got {output.shape}"

    def test_output_dtype(self, device, sample_image):
        """Test output data type"""
        sample_image = sample_image.to(device)
        model = PhenotypicFeatureExtractor().to(device)
        model.eval()

        with torch.no_grad():
            output = model(sample_image)

        assert output.dtype == torch.float32

    def test_gradients_disabled(self, device, sample_image):
        """Test gradients in eval mode"""
        sample_image = sample_image.to(device)
        model = PhenotypicFeatureExtractor().to(device)
        model.eval()

        with torch.no_grad():
            output = model(sample_image)

        assert output.grad is None

    def test_batch_processing(self, device):
        """Test batch processing"""
        batch_images = torch.randn(8, 3, 224, 224).to(device)
        model = PhenotypicFeatureExtractor().to(device)
        model.eval()

        with torch.no_grad():
            output = model(batch_images)

        assert output.shape == (8, 256)


class TestTemporalSensorAnalyzer:
    """Test LSTM temporal analyzer"""

    def test_forward_pass(self, device, sample_sensors):
        """Test LSTM forward pass"""
        sample_sensors = sample_sensors.to(device)
        model = TemporalSensorAnalyzer(
            input_size=4, output_size=128, hidden_size=64, num_layers=2
        ).to(device)
        model.eval()

        with torch.no_grad():
            output = model(sample_sensors)

        assert output.shape == (1, 128), f"Expected (1, 128), got {output.shape}"

    def test_batch_temporal(self, device):
        """Test batch temporal processing"""
        batch_sensors = torch.randn(8, 24, 4).to(device)
        model = TemporalSensorAnalyzer(input_size=4, output_size=128).to(device)
        model.eval()

        with torch.no_grad():
            output = model(batch_sensors)

        assert output.shape == (8, 128)

    def test_different_sequence_lengths(self, device):
        """Test with different sequence lengths"""
        for seq_len in [12, 24, 48]:
            sensors = torch.randn(1, seq_len, 4).to(device)
            model = TemporalSensorAnalyzer(input_size=4, output_size=128).to(device)
            model.eval()

            with torch.no_grad():
                output = model(sensors)

            assert output.shape == (1, 128)


class TestMultimodalFusionModel:
    """Test multimodal fusion"""

    def test_fusion_forward(self, device):
        """Test fusion forward pass"""
        visual_feat = torch.randn(1, 256).to(device)
        temporal_feat = torch.randn(1, 128).to(device)

        model = MultimodalFusionModel(
            visual_feature_size=256,
            temporal_feature_size=128,
            num_classes=2,
            fusion_type='hybrid'
        ).to(device)
        model.eval()

        with torch.no_grad():
            logits = model(visual_feat, temporal_feat)

        assert logits.shape == (1, 2), f"Expected (1, 2), got {logits.shape}"

    def test_fusion_types(self, device):
        """Test different fusion types"""
        visual_feat = torch.randn(1, 256).to(device)
        temporal_feat = torch.randn(1, 128).to(device)

        for fusion_type in ['early', 'late', 'hybrid']:
            model = MultimodalFusionModel(
                visual_feature_size=256,
                temporal_feature_size=128,
                num_classes=2,
                fusion_type=fusion_type
            ).to(device)
            model.eval()

            with torch.no_grad():
                logits = model(visual_feat, temporal_feat)

            assert logits.shape == (1, 2)

    def test_softmax_output(self, device):
        """Test softmax probabilities"""
        visual_feat = torch.randn(1, 256).to(device)
        temporal_feat = torch.randn(1, 128).to(device)

        model = MultimodalFusionModel(
            visual_feature_size=256,
            temporal_feature_size=128,
            num_classes=2
        ).to(device)
        model.eval()

        with torch.no_grad():
            logits = model(visual_feat, temporal_feat)
            probs = torch.softmax(logits, dim=1)

        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(1).to(device), atol=1e-6)
        # All probabilities should be between 0 and 1
        assert (probs >= 0).all() and (probs <= 1).all()


class TestCreateMultimodalModel:
    """Test factory function"""

    def test_model_creation(self, device):
        """Test creating all models"""
        visual, temporal, fusion = create_multimodal_model(
            visual_feature_size=256,
            temporal_feature_size=128,
            num_sensor_vars=4,
            device=device
        )

        assert visual is not None
        assert temporal is not None
        assert fusion is not None

    def test_end_to_end_forward(self, device, sample_batch):
        """Test complete end-to-end forward pass"""
        visual, temporal, fusion = create_multimodal_model(device=device)

        visual.eval()
        temporal.eval()
        fusion.eval()

        images = sample_batch['images'].to(device)
        sensors = sample_batch['sensors'].to(device)

        with torch.no_grad():
            visual_feat = visual(images)
            temporal_feat = temporal(sensors)
            logits = fusion(visual_feat, temporal_feat)

        assert logits.shape == (8, 2)

    def test_model_parameters(self, device):
        """Test model has correct number of parameters"""
        visual, temporal, fusion = create_multimodal_model(device=device)

        # Count trainable parameters
        visual_params = sum(p.numel() for p in visual.parameters() if p.requires_grad)
        temporal_params = sum(p.numel() for p in temporal.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)

        total_params = visual_params + temporal_params + fusion_params

        # Should have millions of parameters
        assert total_params > 1_000_000, f"Expected > 1M params, got {total_params}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
