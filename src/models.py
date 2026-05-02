"""
Deep Learning Models for Multimodal Abiotic Stress Detection

Este módulo contém as arquiteturas de:
- CNN para extração de features fenotípicas
- LSTM/Transformer para análise de séries temporais
- Módulo de fusão multimodal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Tuple, Optional


class PhenotypicFeatureExtractor(nn.Module):
    """
    Rede Neural Convolucional para extração de features fenotípicas.

    Baseada em ResNet18 para análise de características visuais sutis
    que indicam estresse abiótico em plantas.
    """

    def __init__(self, input_channels: int = 3, num_features: int = 256):
        """
        Args:
            input_channels: Número de canais da imagem (3 para RGB)
            num_features: Dimensionalidade do vetor de features (embedding)
        """
        super().__init__()
        self.num_features = num_features

        # Bloco convolucional inicial
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Blocos residuais (simplificado ResNet18)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature projection
        self.fc = nn.Linear(256, num_features)
        self.bn_fc = nn.BatchNorm1d(num_features)

    def _make_layer(self, in_channels: int, out_channels: int,
                   num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Constrói um bloco residual."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processa imagem e extrai features fenotípicas.

        Args:
            x: Tensor de imagem (B, C, H, W)

        Returns:
            Tensor de features (B, num_features)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.bn_fc(x)

        return x


class ResidualBlock(nn.Module):
    """Bloco residual básico (skip connection)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class TemporalSensorAnalyzer(nn.Module):
    """
    Rede LSTM para análise de séries temporais de sensores IoT.

    Modela a dinâmica temporal de temperatura, umidade, CO2 e outras
    variáveis ambientais para identificar padrões de estresse.
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 128):
        """
        Args:
            input_size: Número de variáveis de sensores (ex: temp, umidade, CO2)
            hidden_size: Dimensionalidade dos estados LSTM
            num_layers: Número de camadas LSTM
            output_size: Dimensionalidade do output
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x: torch.Tensor,
               seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processa série temporal e extrai features temporais.

        Args:
            x: Tensor de séries temporais (B, T, input_size)
            seq_lengths: Tensor com comprimentos de sequência (B,)

        Returns:
            Tensor de features temporais (B, output_size)
        """
        # Pack padded sequence se houver comprimentos variáveis
        if seq_lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed_x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        # Aplicar mecanismo de atenção
        attention_weights = self.attention(lstm_out)  # (B, T, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (B, hidden_size)

        # Projetar para dimensionalidade de output
        out = self.fc(context)
        out = self.bn(out)

        return out


class MultimodalFusionModel(nn.Module):
    """
    Arquitetura de fusão multimodal para predição de estresse abiótico.

    Combina:
    - Features fenotípicas de CNNs (imagens)
    - Features temporais de LSTMs (sensores IoT)
    - Em um espaço latente comum para predição
    """

    def __init__(self, visual_feature_size: int = 256,
                 temporal_feature_size: int = 128,
                 hidden_size: int = 256,
                 num_classes: int = 2,
                 fusion_type: str = 'late'):
        """
        Args:
            visual_feature_size: Dimensionalidade de features visuais
            temporal_feature_size: Dimensionalidade de features temporais
            hidden_size: Tamanho das camadas ocultas
            num_classes: Número de classes de predição (ex: 2 para stress/normal)
            fusion_type: Tipo de fusão ('early', 'late', 'hybrid')
        """
        super().__init__()
        self.fusion_type = fusion_type

        # Camadas de projeção para mesmo espaço latente
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3)
        )

        self.temporal_projection = nn.Sequential(
            nn.Linear(temporal_feature_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3)
        )

        # Módulo de fusão
        if fusion_type == 'early':
            fusion_input_size = hidden_size * 2
        elif fusion_type == 'late':
            fusion_input_size = hidden_size
        else:  # hybrid
            fusion_input_size = hidden_size * 3

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, visual_features: torch.Tensor,
               temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Processa features multimodais e prediz estresse abiótico.

        Args:
            visual_features: Features de imagens (B, visual_feature_size)
            temporal_features: Features de sensores (B, temporal_feature_size)

        Returns:
            Logits de classificação (B, num_classes)
        """
        # Projetar para espaço latente comum
        visual_proj = self.visual_projection(visual_features)
        temporal_proj = self.temporal_projection(temporal_features)

        # Fusão baseada no tipo selecionado
        if self.fusion_type == 'early':
            fused = torch.cat([visual_proj, temporal_proj], dim=1)
        elif self.fusion_type == 'late':
            fused = visual_proj + temporal_proj  # Fusão por adição (element-wise)
        else:  # hybrid
            # Combina as duas abordagens
            element_wise = visual_proj * temporal_proj
            fused = torch.cat([visual_proj, temporal_proj, element_wise], dim=1)

        # Classificação
        logits = self.classifier(fused)

        return logits


def create_multimodal_model(visual_feature_size: int = 256,
                           temporal_feature_size: int = 128,
                           num_sensor_vars: int = 4,
                           num_classes: int = 2,
                           fusion_type: str = 'hybrid',
                           device: str = 'cpu') -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Factory function para criar o sistema completo multimodal.

    Returns:
        Tupla (visual_extractor, temporal_analyzer, fusion_model)
    """
    visual_model = PhenotypicFeatureExtractor(input_channels=3,
                                            num_features=visual_feature_size).to(device)
    temporal_model = TemporalSensorAnalyzer(input_size=num_sensor_vars,
                                          output_size=temporal_feature_size).to(device)
    fusion_model = MultimodalFusionModel(visual_feature_size=visual_feature_size,
                                        temporal_feature_size=temporal_feature_size,
                                        num_classes=num_classes,
                                        fusion_type=fusion_type).to(device)

    return visual_model, temporal_model, fusion_model


if __name__ == "__main__":
    # Teste das arquiteturas
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Teste CNN
    visual_model, temporal_model, fusion_model = create_multimodal_model(device=device)

    batch_size = 4
    visual_input = torch.randn(batch_size, 3, 224, 224).to(device)
    temporal_input = torch.randn(batch_size, 24, 4).to(device)  # 24 timesteps, 4 sensores

    visual_features = visual_model(visual_input)
    print(f"Visual features shape: {visual_features.shape}")

    temporal_features = temporal_model(temporal_input)
    print(f"Temporal features shape: {temporal_features.shape}")

    logits = fusion_model(visual_features, temporal_features)
    print(f"Prediction logits shape: {logits.shape}")

    probabilities = F.softmax(logits, dim=1)
    print(f"Probability shape: {probabilities.shape}")
    print(f"Sample predictions: {probabilities[0]}")
