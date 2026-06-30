"""Pytest configuration and fixtures"""

import sys
from pathlib import Path
import torch
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Get device (cuda or cpu)"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_image():
    """Create a sample image tensor"""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_sensors():
    """Create a sample sensor tensor"""
    return torch.randn(1, 24, 4)  # 24 timesteps, 4 variables


@pytest.fixture
def sample_batch():
    """Create a sample batch"""
    return {
        'images': torch.randn(8, 3, 224, 224),
        'sensors': torch.randn(8, 24, 4),
        'labels': torch.randint(0, 2, (8,))
    }
