"""
Pipeline de Preparação de Dados

Este módulo coordena:
- Carregamento de dados multimodais
- Pré-processamento (normalização, augmentação)
- Divisão em treino/validação/teste
- Criação de DataLoaders PyTorch
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from PIL import Image

from src.data_loader import MultimodalDataLoader, ImagePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    Dataset PyTorch para dados multimodais (imagens + séries temporais).

    Carrega pares de:
    - Imagem fenotípica de planta
    - Série temporal de sensores (temperatura, umidade, CO2, etc)
    - Label de estresse abiótico (binário: 0=normal, 1=stress)
    """

    def __init__(self, image_paths: List[str],
                temporal_sequences: np.ndarray,
                labels: np.ndarray,
                image_size: Tuple[int, int] = (224, 224),
                num_sensor_vars: int = 4,
                sequence_length: int = 24,
                transform=None):
        """
        Args:
            image_paths: Lista de caminhos para imagens
            temporal_sequences: Array (N, sequence_length, num_sensor_vars)
            labels: Array (N,) com labels binários
            image_size: Tamanho alvo para imagens (H, W)
            num_sensor_vars: Número de variáveis de sensores
            sequence_length: Comprimento de sequência temporal
            transform: Transformações a aplicar (augmentação)
        """
        self.image_paths = image_paths
        self.temporal_sequences = temporal_sequences
        self.labels = labels
        self.image_size = image_size
        self.num_sensor_vars = num_sensor_vars
        self.sequence_length = sequence_length
        self.transform = transform

        assert len(image_paths) == len(temporal_sequences) == len(labels), \
            "Image paths, temporal sequences e labels devem ter mesmo comprimento"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna: (image_tensor, temporal_tensor, label)
        """
        # Carregar imagem
        try:
            img = Image.open(self.image_paths[idx])
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Pré-processar imagem
            img = ImagePreprocessor.resize_image(np.array(img), self.image_size)
            img = ImagePreprocessor.normalize_image(img)

            # Converter para tensor (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        except Exception as e:
            logger.error(f"Erro ao carregar imagem {self.image_paths[idx]}: {e}")
            # Retornar imagem vazia em caso de erro
            img_tensor = torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        # Série temporal (já normalizada esperadamente)
        temporal_tensor = torch.from_numpy(
            self.temporal_sequences[idx].astype(np.float32)
        )

        # Garantir tamanho esperado
        if temporal_tensor.shape[0] < self.sequence_length:
            # Padding
            pad = torch.zeros((self.sequence_length - temporal_tensor.shape[0],
                             self.num_sensor_vars), dtype=torch.float32)
            temporal_tensor = torch.cat([temporal_tensor, pad], dim=0)
        elif temporal_tensor.shape[0] > self.sequence_length:
            # Crop
            temporal_tensor = temporal_tensor[:self.sequence_length, :]

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, temporal_tensor, label


class DataPipeline:
    """
    Pipeline completo de preparação de dados.

    Orquestra: carregamento → pré-processamento → divisão → criação de dataloaders
    """

    def __init__(self, data_dir: str, random_state: int = 42):
        """
        Args:
            data_dir: Diretório raiz de dados
            random_state: Seed para reprodutibilidade
        """
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.loader = MultimodalDataLoader(str(self.data_dir))

        # Escaladores para normalização
        self.sensor_scaler = StandardScaler()
        self.image_stats = {}

    def prepare_training_data(self,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             batch_size: int = 32,
                             num_workers: int = 4,
                             limit_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepara dataloaders para treino, validação e teste.

        Args:
            train_ratio: Proporção treino (0-1)
            val_ratio: Proporção validação
            test_ratio: Proporção teste
            batch_size: Tamanho do batch
            num_workers: Número de workers para data loading
            limit_samples: Limitar número de amostras (para teste rápido)

        Returns:
            (train_loader, val_loader, test_loader)
        """
        logger.info("Iniciando pipeline de preparação de dados...")

        # 1. Carregar imagens e sensores
        image_paths, temporal_sequences, labels = self._load_and_preprocess_data(limit_samples)

        logger.info(f"Amostras carregadas: {len(image_paths)}")
        logger.info(f"Shape temporal sequences: {temporal_sequences.shape}")
        logger.info(f"Distribuição de labels: {np.bincount(labels)}")

        # 2. Normalizar dados de sensores
        temporal_sequences = self._normalize_temporal_data(temporal_sequences)

        # 3. Criar dataset
        dataset = MultimodalDataset(
            image_paths=image_paths,
            temporal_sequences=temporal_sequences,
            labels=labels,
            image_size=(224, 224),
            num_sensor_vars=temporal_sequences.shape[2] if len(temporal_sequences.shape) > 2 else 1,
            sequence_length=temporal_sequences.shape[1] if len(temporal_sequences.shape) > 1 else 24
        )

        # 4. Dividir em treino/validação/teste
        train_size = int(len(dataset) * train_ratio)
        val_size = int(len(dataset) * val_ratio)
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_state)
        )

        logger.info(f"Tamanhos: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        # 5. Criar dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info("✓ Dataloaders criados com sucesso")

        return train_loader, val_loader, test_loader

    def _load_and_preprocess_data(self, limit_samples: Optional[int] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Carrega e pré-processa dados multimodais.

        Returns:
            (image_paths, temporal_sequences, labels)
        """
        plants = self.loader.get_plant_list()

        if not plants:
            logger.warning("Nenhuma planta encontrada!")
            plants = ['sigrow']  # Default

        logger.info(f"Processando plantas: {plants}")

        all_image_paths = []
        all_temporal_sequences = []
        all_labels = []

        for plant in plants:
            # Carregar imagens
            images_dict = self.loader.load_images_for_plant(plant, limit=limit_samples)

            if not images_dict:
                logger.warning(f"Nenhuma imagem encontrada para {plant}")
                continue

            plant_image_paths = list(images_dict.keys())

            # Carregar dados de sensores (simulado por enquanto)
            temporal_seqs = self._create_synthetic_temporal_data(len(plant_image_paths))

            # Criar labels (simulado: 50% normal, 50% stress)
            labels = np.random.binomial(n=1, p=0.5, size=len(plant_image_paths))

            all_image_paths.extend(plant_image_paths)
            all_temporal_sequences.append(temporal_seqs)
            all_labels.extend(labels)

        # Concatenar dados de todas as plantas
        temporal_sequences = np.vstack(all_temporal_sequences) if all_temporal_sequences else np.array([])
        all_labels = np.array(all_labels)

        return all_image_paths, temporal_sequences, all_labels

    def _create_synthetic_temporal_data(self, num_samples: int,
                                       num_vars: int = 4,
                                       sequence_length: int = 24) -> np.ndarray:
        """
        Cria dados temporais sintéticos para teste.

        Args:
            num_samples: Número de amostras
            num_vars: Número de variáveis (temperatura, umidade, CO2, luz)
            sequence_length: Comprimento de sequência

        Returns:
            Array (num_samples, sequence_length, num_vars)
        """
        # Simular série temporal: valores entre -1 e 1
        data = np.random.randn(num_samples, sequence_length, num_vars)

        # Suavizar com média móvel para mais realismo
        from scipy.ndimage import uniform_filter1d
        for i in range(num_samples):
            for j in range(num_vars):
                data[i, :, j] = uniform_filter1d(data[i, :, j], size=3)

        return data

    def _normalize_temporal_data(self, temporal_sequences: np.ndarray) -> np.ndarray:
        """Normaliza dados de sensores usando StandardScaler."""

        num_samples, seq_length, num_vars = temporal_sequences.shape

        # Reshape para (num_samples * seq_length, num_vars)
        reshaped = temporal_sequences.reshape(-1, num_vars)

        # Fit e transform
        reshaped_normalized = self.sensor_scaler.fit_transform(reshaped)

        # Reshape de volta
        normalized = reshaped_normalized.reshape(num_samples, seq_length, num_vars)

        return normalized


def create_dataloaders(data_dir: str,
                      batch_size: int = 32,
                      limit_samples: Optional[int] = None,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function para criar dataloaders.

    Args:
        data_dir: Diretório de dados
        batch_size: Tamanho do batch
        limit_samples: Limitar amostras (para testes rápidos)
        num_workers: Workers para loading

    Returns:
        (train_loader, val_loader, test_loader)
    """
    pipeline = DataPipeline(data_dir)
    return pipeline.prepare_training_data(
        batch_size=batch_size,
        limit_samples=limit_samples,
        num_workers=num_workers
    )


if __name__ == "__main__":
    # Teste do pipeline
    data_dir = "/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data"

    logger.info("Testando pipeline de dados...")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        limit_samples=50,  # Teste rápido
        num_workers=2
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Visualizar um batch
    for images, temporal, labels in train_loader:
        logger.info(f"Batch shapes:")
        logger.info(f"  Images: {images.shape}")
        logger.info(f"  Temporal: {temporal.shape}")
        logger.info(f"  Labels: {labels.shape}")
        break

    logger.info("✓ Pipeline testado com sucesso!")
