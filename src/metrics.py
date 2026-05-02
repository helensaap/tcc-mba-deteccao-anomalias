"""
Módulo de Métricas e Avaliação

Implementa métricas consolidadas para avaliação de desempenho:
- Acurácia
- Precisão (Precision)
- Recall (Sensibilidade)
- F1-Score
- AUC-ROC
- Matriz de Confusão
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import torch


class MetricCalculator:
    """Calcula métricas de desempenho para classificação binária e multiclasse."""

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calcula todas as métricas de desempenho.

        Args:
            y_true: Labels verdadeiros
            y_pred: Predições (classe)
            y_proba: Probabilidades das predições (opcional, para AUC-ROC)

        Returns:
            Dicionário com todas as métricas
        """
        metrics = {}

        # Métricas básicas
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # AUC-ROC (apenas para binário ou se probabilidades fornecidas)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics['auc_roc'] = None

        # Matriz de confusão
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], verbose: bool = True) -> str:
        """
        Formata e exibe métricas de forma legível.

        Returns:
            String formatada com as métricas
        """
        output = "\n" + "="*50 + "\n"
        output += "MÉTRICAS DE DESEMPENHO\n"
        output += "="*50 + "\n"

        output += f"Acurácia:  {metrics['accuracy']:.4f}\n"
        output += f"Precisão:  {metrics['precision']:.4f}\n"
        output += f"Recall:    {metrics['recall']:.4f}\n"
        output += f"F1-Score:  {metrics['f1_score']:.4f}\n"

        if metrics['auc_roc'] is not None:
            output += f"AUC-ROC:   {metrics['auc_roc']:.4f}\n"

        output += "\nMatriz de Confusão:\n"
        output += str(metrics['confusion_matrix']) + "\n"
        output += "="*50 + "\n"

        if verbose:
            print(output)

        return output

    @staticmethod
    def calculate_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calcula curva ROC e retorna FPR, TPR, AUC.

        Args:
            y_true: Labels verdadeiros (binário)
            y_proba: Probabilidades para a classe positiva

        Returns:
            (fpr, tpr, auc_score)
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score


class EarlyStoppingCallback:
    """
    Callback para parar o treinamento cedo se a métrica de validação não melhorar.
    """

    def __init__(self, patience: int = 10, metric: str = 'val_loss',
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience: Número de epochs sem melhora antes de parar
            metric: Métrica a monitorar
            mode: 'min' para minimizar, 'max' para maximizar
            verbose: Se True, imprime mensagens
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Verifica se deve parar o treinamento.

        Args:
            current_value: Valor atual da métrica

        Returns:
            True se deve parar, False caso contrário
        """
        if self.mode == 'min':
            if current_value < self.best_value:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        else:  # max
            if current_value > self.best_value:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            if self.verbose:
                print(f"Early stopping: {self.metric} não melhorou por {self.patience} epochs")
            self.should_stop = True
            return True

        return False


class ConfusionMatrixAnalyzer:
    """Analisa matriz de confusão em detalhes."""

    @staticmethod
    def analyze(cm: np.ndarray, class_names: list = None) -> Dict:
        """
        Analisa matriz de confusão.

        Args:
            cm: Matriz de confusão
            class_names: Nomes das classes (opcional)

        Returns:
            Dicionário com análises detalhadas
        """
        analysis = {}

        # Para caso binário
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            analysis['true_negatives'] = int(tn)
            analysis['false_positives'] = int(fp)
            analysis['false_negatives'] = int(fn)
            analysis['true_positives'] = int(tp)

            analysis['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            analysis['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            analysis['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        return analysis


def evaluate_model(model, data_loader, device: str = 'cpu',
                  num_classes: int = 2) -> Dict:
    """
    Avalia modelo em dataset de teste.

    Args:
        model: Modelo PyTorch
        data_loader: DataLoader com dados de teste
        device: 'cuda' ou 'cpu'
        num_classes: Número de classes

    Returns:
        Dicionário com métricas de avaliação
    """
    model.eval()
    all_preds = []
    all_proba = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            outputs = model(x)
            proba = torch.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_proba.extend(proba.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_proba = np.array(all_proba)
    all_targets = np.array(all_targets)

    metrics = MetricCalculator.calculate_all_metrics(all_targets, all_preds, all_proba)

    return metrics
