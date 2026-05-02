"""
Notebook 02: Treinamento do Modelo Multimodal

Executa o pipeline completo de treinamento:
- CNN para extração de features visuais
- LSTM para análise temporal
- Fusão multimodal
- Otimização com Adam
"""

import sys
sys.path.insert(0, '/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json

from src.pipeline import create_dataloaders
from src.models import create_multimodal_model
from src.metrics import MetricCalculator, EarlyStoppingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalTrainer:
    """Gerencia o treinamento do modelo multimodal."""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        """
        Args:
            device: 'cuda' ou 'cpu'
            learning_rate: Taxa de aprendizado
            weight_decay: L2 regularization
        """
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        logger.info(f"Usando device: {device}")

        # Diretórios
        self.model_dir = Path("/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/models")
        self.model_dir.mkdir(exist_ok=True)

        self.results_dir = Path("/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/results")
        self.results_dir.mkdir(exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(str(self.results_dir / "runs"))

        # Modelos e otimizador
        self.visual_model = None
        self.temporal_model = None
        self.fusion_model = None
        self.optimizer = None
        self.loss_fn = None

    def setup_models(self, num_classes: int = 2,
                    fusion_type: str = 'hybrid') -> None:
        """Inicializa modelos."""
        logger.info(f"Inicializando modelos (fusion_type={fusion_type})...")

        self.visual_model, self.temporal_model, self.fusion_model = create_multimodal_model(
            visual_feature_size=256,
            temporal_feature_size=128,
            num_sensor_vars=4,
            num_classes=num_classes,
            fusion_type=fusion_type,
            device=self.device
        )

        # Otimizador (aplica a todos os parâmetros)
        all_params = list(self.visual_model.parameters()) + \
                     list(self.temporal_model.parameters()) + \
                     list(self.fusion_model.parameters())

        self.optimizer = optim.Adam(all_params,
                                   lr=self.learning_rate,
                                   weight_decay=self.weight_decay)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info("✓ Modelos inicializados")
        logger.info(f"  Visual CNN: {sum(p.numel() for p in self.visual_model.parameters())} params")
        logger.info(f"  Temporal LSTM: {sum(p.numel() for p in self.temporal_model.parameters())} params")
        logger.info(f"  Fusion: {sum(p.numel() for p in self.fusion_model.parameters())} params")

    def train_epoch(self, train_loader) -> float:
        """Treina uma época."""
        self.visual_model.train()
        self.temporal_model.train()
        self.fusion_model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Treinamento", leave=False)
        for images, temporal, labels in pbar:
            images = images.to(self.device)
            temporal = temporal.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            visual_features = self.visual_model(images)
            temporal_features = self.temporal_model(temporal)
            logits = self.fusion_model(visual_features, temporal_features)

            # Loss
            loss = self.loss_fn(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.visual_model.parameters()) +
                list(self.temporal_model.parameters()) +
                list(self.fusion_model.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.update(1)

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader) -> dict:
        """Valida modelo e retorna métricas."""
        self.visual_model.eval()
        self.temporal_model.eval()
        self.fusion_model.eval()

        all_preds = []
        all_proba = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validação", leave=False)
            for images, temporal, labels in pbar:
                images = images.to(self.device)
                temporal = temporal.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                visual_features = self.visual_model(images)
                temporal_features = self.temporal_model(temporal)
                logits = self.fusion_model(visual_features, temporal_features)

                # Loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                num_batches += 1

                # Predictions
                proba = torch.softmax(logits, dim=1)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                pbar.update(1)

        all_preds = np.array(all_preds)
        all_proba = np.array(all_proba)
        all_targets = np.array(all_targets)

        # Métricas
        metrics = MetricCalculator.calculate_all_metrics(all_targets, all_preds, all_proba)
        metrics['loss'] = total_loss / num_batches

        return metrics

    def train(self, train_loader, val_loader,
             num_epochs: int = 100,
             early_stopping_patience: int = 15) -> dict:
        """
        Treinamento completo com early stopping.

        Returns:
            Dicionário com histórico de treinamento
        """
        logger.info(f"\n{'='*80}")
        logger.info("INICIANDO TREINAMENTO")
        logger.info(f"{'='*80}\n")

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        early_stopping = EarlyStoppingCallback(patience=early_stopping_patience,
                                             metric='val_f1_score',
                                             mode='max')

        best_f1 = 0.0
        global_step = 0

        for epoch in range(num_epochs):
            logger.info(f"\nÉpoca {epoch+1}/{num_epochs}")
            logger.info("-" * 80)

            # Treino
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            # Validação
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_score'])

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss:   {val_metrics['loss']:.4f}")
            logger.info(f"Val Acc:    {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1:     {val_metrics['f1_score']:.4f}")
            logger.info(f"Val Prec:   {val_metrics['precision']:.4f}")
            logger.info(f"Val Rec:    {val_metrics['recall']:.4f}")

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, global_step)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], global_step)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], global_step)
            self.writer.add_scalar('F1/val', val_metrics['f1_score'], global_step)

            global_step += 1

            # Salvar melhor modelo
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                self.save_checkpoint(epoch, val_metrics)
                logger.info(f"✓ Novo melhor modelo salvo (F1={best_f1:.4f})")

            # Early stopping
            if early_stopping(val_metrics['f1_score']):
                logger.info(f"\n✓ Early stopping ativado na época {epoch+1}")
                break

        logger.info(f"\n{'='*80}")
        logger.info("TREINAMENTO CONCLUÍDO")
        logger.info(f"Melhor F1-Score: {best_f1:.4f}")
        logger.info(f"{'='*80}\n")

        self.writer.close()

        return history

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Salva checkpoint do modelo."""
        checkpoint = {
            'epoch': epoch,
            'visual_model_state': self.visual_model.state_dict(),
            'temporal_model_state': self.temporal_model.state_dict(),
            'fusion_model_state': self.fusion_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics
        }

        path = self.model_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # Também salvar como 'best_model.pt'
        best_path = self.model_dir / "best_model.pt"
        torch.save(checkpoint, best_path)

        logger.info(f"Checkpoint salvo: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Carrega checkpoint pré-treinado."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.visual_model.load_state_dict(checkpoint['visual_model_state'])
        self.temporal_model.load_state_dict(checkpoint['temporal_model_state'])
        self.fusion_model.load_state_dict(checkpoint['fusion_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        logger.info(f"Checkpoint carregado: {checkpoint_path}")


def main():
    """Executa pipeline completo de treinamento."""

    print("\n" + "#"*80)
    print("# NOTEBOOK 02: TREINAMENTO DO MODELO MULTIMODAL")
    print("#"*80 + "\n")

    # Configurações - FASE 1: Dataset COMPLETO + AMBAS PLANTAS (28 Abril 2026)
    # ✅ Modificado para treino com TODAS as amostras disponíveis
    # ✅ Incluindo: sigrow (90) + raspberry (2.166) = 2.256 imagens!
    BATCH_SIZE = 32                # Aumentado de 8 (mais estável)
    LEARNING_RATE = 0.0005         # Reduzido de 0.001 (convergência suave)
    NUM_EPOCHS = 100               # Aumentado de 30 (mais iterações)
    EARLY_STOPPING_PATIENCE = 10
    LIMIT_SAMPLES = None           # None = TODAS as imagens disponíveis!

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Carregar dados
    logger.info("Preparando dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data",
        batch_size=BATCH_SIZE,
        limit_samples=LIMIT_SAMPLES,
        num_workers=2
    )

    # 2. Inicializar trainer
    trainer = MultimodalTrainer(device=device,
                               learning_rate=LEARNING_RATE)

    # 3. Setup modelos
    trainer.setup_models(num_classes=2, fusion_type='hybrid')

    # 4. Treinar
    history = trainer.train(train_loader=train_loader,
                           val_loader=val_loader,
                           num_epochs=NUM_EPOCHS,
                           early_stopping_patience=EARLY_STOPPING_PATIENCE)

    # 5. Salvar histórico
    history_path = trainer.results_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"✓ Histórico salvo: {history_path}")

    # 6. Avaliar em teste (opcional)
    logger.info("\nAvaliando em dataset de teste...")
    test_metrics = trainer.validate(test_loader)
    logger.info("\nMétricas de Teste:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")

    print("\n" + "#"*80)
    print("# FIM DO TREINAMENTO")
    print("#"*80 + "\n")

    print("Próximas etapas:")
    print("  1. Executar: python notebooks/03_evaluate_and_visualize.py")
    print("  2. Executar: python notebooks/04_alert_system_demo.py")
    print("\n")


if __name__ == "__main__":
    main()
