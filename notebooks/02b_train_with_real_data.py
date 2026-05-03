"""
Notebook 02b: Treinamento do Modelo Multimodal com Dados REAIS (15.336 imagens)

Este notebook treina o modelo com o dataset completo usando:
- RealDataLoader para carregar dados reais
- Todas as 15.336 imagens disponíveis
- Dados de sensores reais (GreenhouseClimate XLSX)
- Labels verdadeiros (Classes A/B/C)
"""

import sys
sys.path.insert(0, '/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

from src.real_data_loader import RealDataLoader
from src.models import create_multimodal_model
from src.metrics import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataTrainer:
    """Gerencia o treinamento com dados reais."""

    def __init__(self, device: str = None, learning_rate: float = 0.001):
        """Inicializa trainer."""
        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.device = device
        self.learning_rate = learning_rate

        logger.info(f"Usando device: {device}")

        # Diretórios
        self.model_dir = Path("/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/models")
        self.model_dir.mkdir(exist_ok=True)

        self.results_dir = Path("/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/results")
        self.results_dir.mkdir(exist_ok=True)

        # Componentes
        self.visual_model = None
        self.temporal_model = None
        self.fusion_model = None
        self.optimizer = None
        self.loss_fn = None

    def setup_models(self) -> None:
        """Inicializa modelos."""
        logger.info("Inicializando modelos (fusion_type=hybrid)...")

        self.visual_model, self.temporal_model, self.fusion_model = create_multimodal_model(
            visual_feature_size=256,
            temporal_feature_size=128,
            num_sensor_vars=4,
            num_classes=2,
            fusion_type='hybrid',
            device=self.device
        )

        # Otimizador
        all_params = (
            list(self.visual_model.parameters()) +
            list(self.temporal_model.parameters()) +
            list(self.fusion_model.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=self.learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info(f"✓ Modelos inicializados")

    def train_epoch(self, train_loader) -> dict:
        """Treina uma epoch."""
        self.visual_model.train()
        self.temporal_model.train()
        self.fusion_model.train()

        total_loss = 0.0
        all_labels = []
        all_proba = []

        # Obter todos os parâmetros para clipping
        all_params = (
            list(self.visual_model.parameters()) +
            list(self.temporal_model.parameters()) +
            list(self.fusion_model.parameters())
        )

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Handle both dict and tuple batch formats
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                temporal = batch['temporal'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                images, temporal, labels = batch
                images = images.to(self.device)
                temporal = temporal.to(self.device)
                labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            visual_features = self.visual_model(images)
            temporal_features = self.temporal_model(temporal)
            logits = self.fusion_model(visual_features, temporal_features)

            loss = self.loss_fn(logits, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            self.optimizer.step()

            # Registrar
            total_loss += loss.item()
            proba = torch.softmax(logits, dim=1).detach()
            all_labels.extend(labels.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        # Calcular métricas
        all_labels = np.array(all_labels)
        all_proba = np.array(all_proba)
        all_preds = all_proba.argmax(axis=1)

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, zero_division=0),
            'auc_roc': roc_auc_score(all_labels, all_proba[:, 1]),
        }

    def eval_epoch(self, val_loader) -> dict:
        """Avalia uma epoch."""
        self.visual_model.eval()
        self.temporal_model.eval()
        self.fusion_model.eval()

        total_loss = 0.0
        all_labels = []
        all_proba = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                # Handle both dict and tuple batch formats
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    temporal = batch['temporal'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    images, temporal, labels = batch
                    images = images.to(self.device)
                    temporal = temporal.to(self.device)
                    labels = labels.to(self.device)

                visual_features = self.visual_model(images)
                temporal_features = self.temporal_model(temporal)
                logits = self.fusion_model(visual_features, temporal_features)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                proba = torch.softmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())

        # Calcular métricas
        all_labels = np.array(all_labels)
        all_proba = np.array(all_proba)
        all_preds = all_proba.argmax(axis=1)

        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, zero_division=0),
            'auc_roc': roc_auc_score(all_labels, all_proba[:, 1]),
        }

    def train(self, train_loader, val_loader, num_epochs: int = 50, patience: int = 15):
        """Treina o modelo."""
        logger.info("=" * 80)
        logger.info("INICIANDO TREINAMENTO COM DADOS REAIS")
        logger.info("=" * 80)

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'train_auc': [],
            'val_auc': [],
        }

        early_stopping = EarlyStoppingCallback(patience=patience, verbose=True)
        best_val_f1 = 0.0

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*80}")

            # Treinar
            train_metrics = self.train_epoch(train_loader)

            # Validar
            val_metrics = self.eval_epoch(val_loader)

            # Registrar histórico
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1_score'])
            history['val_f1'].append(val_metrics['f1_score'])
            history['train_auc'].append(train_metrics['auc_roc'])
            history['val_auc'].append(val_metrics['auc_roc'])

            # Log
            logger.info(f"\nTrain: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1_score']:.4f}, AUC={train_metrics['auc_roc']:.4f}")
            logger.info(f"Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1_score']:.4f}, AUC={val_metrics['auc_roc']:.4f}")

            # Early stopping
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                self.save_checkpoint(f"best_model_epoch_{epoch}.pt")

            if early_stopping(val_metrics['loss']):
                logger.info(f"\n✓ Early stopping at epoch {epoch}")
                break

        # Salvar histórico
        with open(self.results_dir / 'training_history_real_data.json', 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"\n✓ Treinamento completo!")
        logger.info(f"✓ Histórico salvo em: results/training_history_real_data.json")

        return history

    def save_checkpoint(self, filename: str):
        """Salva checkpoint do modelo."""
        checkpoint = {
            'visual_model_state': self.visual_model.state_dict(),
            'temporal_model_state': self.temporal_model.state_dict(),
            'fusion_model_state': self.fusion_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }

        path = self.model_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"✓ Checkpoint salvo: {filename}")


def create_real_dataloaders(df_dataset, batch_size: int = 32, test_split: float = 0.15, val_split: float = 0.15):
    """Cria dataloaders a partir do dataframe do RealDataLoader."""

    logger.info(f"\nCriando dataloaders a partir de {len(df_dataset)} amostras...")

    # Preparar dados
    images_list = []
    temporal_list = []
    labels_list = []

    for idx, row in tqdm(df_dataset.iterrows(), total=len(df_dataset), desc="Preparando dados"):
        # Imagem
        img = row['image_array']
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # HWC → CHW
        images_list.append(img_tensor)

        # Temporal
        temporal = row['sensor_sequence']
        temporal_tensor = torch.from_numpy(temporal).float()
        temporal_list.append(temporal_tensor)

        # Label
        labels_list.append(row['label'])

    images_tensor = torch.stack(images_list)
    temporal_tensor = torch.stack(temporal_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    logger.info(f"✓ Tensores criados: images={images_tensor.shape}, temporal={temporal_tensor.shape}, labels={labels_tensor.shape}")

    # Split
    n = len(df_dataset)
    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - test_size - val_size

    indices = torch.randperm(n)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    def create_loader(idx_list, batch_size, shuffle=False):
        images_batch = images_tensor[idx_list]
        temporal_batch = temporal_tensor[idx_list]
        labels_batch = labels_tensor[idx_list]

        dataset = [
            {
                'image': images_batch[i],
                'temporal': temporal_batch[i],
                'label': labels_batch[i],
            }
            for i in range(len(idx_list))
        ]

        class CustomCollate:
            def __call__(self, batch):
                return {
                    'image': torch.stack([item['image'] for item in batch]),
                    'temporal': torch.stack([item['temporal'] for item in batch]),
                    'label': torch.stack([item['label'] for item in batch]),
                }

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=CustomCollate())

    train_loader = create_loader(train_idx, batch_size, shuffle=True)
    val_loader = create_loader(val_idx, batch_size, shuffle=False)
    test_loader = create_loader(test_idx, batch_size, shuffle=False)

    logger.info(f"✓ Dataloaders criados:")
    logger.info(f"  - Train: {len(train_idx)} amostras")
    logger.info(f"  - Val: {len(val_idx)} amostras")
    logger.info(f"  - Test: {len(test_idx)} amostras")

    return train_loader, val_loader, test_loader


def main():
    print("\n" + "#" * 80)
    print("# NOTEBOOK 02b: TREINAMENTO COM DADOS REAIS (15.336 imagens completas)")
    print("#" * 80 + "\n")

    # Configurações
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # 1. Carregar dados COMPLETOS usando o pipeline padrão
    logger.info("Carregando 15.336 imagens completas do experimento...")
    from src.pipeline import create_dataloaders

    try:
        # Tentar usar pipeline que carrega TODAS as imagens
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir="/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data",
            batch_size=BATCH_SIZE,
            limit_samples=None,  # Todas as 15.336!
            num_workers=2
        )
        logger.info(f"✓ Dataloaders criados com sucesso (15.336 imagens)")
    except Exception as e:
        logger.warning(f"Pipeline padrão não disponível: {e}")
        logger.info("Alternativa: usando RealDataLoader com Ground Truth...")

        loader = RealDataLoader("/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data")
        df_dataset = loader.create_multimodal_dataset(limit_images=None)

        logger.info(f"\n✓ Dataset carregado: {len(df_dataset)} amostras")
        logger.info(f"  - Distribuição: {(df_dataset['label'] == 0).sum()} normais, {(df_dataset['label'] == 1).sum()} stress")

        # 2. Criar dataloaders
        train_loader, val_loader, test_loader = create_real_dataloaders(df_dataset, batch_size=BATCH_SIZE)

    # 3. Inicializar trainer
    trainer = RealDataTrainer(device=device, learning_rate=LEARNING_RATE)
    trainer.setup_models()

    # 4. Treinar
    history = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=15)

    print("\n" + "#" * 80)
    print("# FIM DO TREINAMENTO")
    print("#" * 80 + "\n")

    return history


if __name__ == "__main__":
    history = main()
