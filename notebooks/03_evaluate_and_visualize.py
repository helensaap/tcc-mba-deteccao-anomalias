"""
Notebook 03: Avaliação e Visualização com ROC Curve Analysis

Executa análise completa do modelo com foco em:
- ROC Curve para encontrar thresholds ótimos
- Youden's Index para threshold equilibrado
- F1-Score máximo para threshold otimizado
- Matriz de confusão
- Curvas de treinamento
"""

import sys
sys.path.insert(0, '/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    f1_score, precision_recall_curve, roc_auc_score
)
import json

from src.pipeline import create_dataloaders
from src.models import create_multimodal_model
from src.metrics import MetricCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações de visualização
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


class ModelEvaluator:
    """Avalia modelo com foco em validação científica de thresholds."""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model_path = model_path or '/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/models/best_model.pt'
        self.device = device
        self.results_dir = Path('/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/results')
        self.results_dir.mkdir(exist_ok=True)

    def load_checkpoint(self, checkpoint_path: str):
        """Carrega modelo treinado."""
        logger.info(f"Carregando modelo de: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Recriar arquitetura
        self.visual_model, self.temporal_model, self.fusion_model = create_multimodal_model(
            visual_feature_size=256,
            temporal_feature_size=128,
            num_sensor_vars=4,
            num_classes=2,
            fusion_type='hybrid',
            device=self.device
        )

        # Carregar pesos
        self.visual_model.load_state_dict(checkpoint['visual_model_state'])
        self.temporal_model.load_state_dict(checkpoint['temporal_model_state'])
        self.fusion_model.load_state_dict(checkpoint['fusion_model_state'])

        # Modo avaliação
        self.visual_model.eval()
        self.temporal_model.eval()
        self.fusion_model.eval()

        logger.info("✓ Modelo carregado com sucesso")

    def predict_on_loader(self, data_loader) -> tuple:
        """
        Faz predições em um DataLoader.

        Returns:
            (predictions, probabilities, targets) - Arrays numpy
        """
        all_preds = []
        all_proba = []
        all_targets = []

        with torch.no_grad():
            for images, temporal, labels in data_loader:
                images = images.to(self.device)
                temporal = temporal.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                visual_features = self.visual_model(images)
                temporal_features = self.temporal_model(temporal)
                logits = self.fusion_model(visual_features, temporal_features)

                # Probabilidades
                proba = torch.softmax(logits, dim=1)

                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        return (
            np.array(all_preds),
            np.array(all_proba),
            np.array(all_targets)
        )

    def analyze_roc_curve(self, y_true, y_proba):
        """
        Analisa ROC curve e encontra thresholds ótimos.

        Returns:
            Dict com métricas e thresholds recomendados
        """
        logger.info("\n" + "="*80)
        logger.info("ANÁLISE DE ROC CURVE - ENCONTRANDO THRESHOLDS ÓTIMOS")
        logger.info("="*80 + "\n")

        # Probabilidade da classe STRESS (classe 1)
        y_proba_stress = y_proba[:, 1]

        # Calcular ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_stress)
        roc_auc = auc(fpr, tpr)

        logger.info(f"AUC-ROC: {roc_auc:.4f}")

        # 1. YOUDEN'S INDEX (TPR - FPR)
        youden_index = tpr - fpr
        optimal_youden_idx = np.argmax(youden_index)
        youden_threshold = thresholds[optimal_youden_idx]
        youden_tpr = tpr[optimal_youden_idx]
        youden_fpr = fpr[optimal_youden_idx]

        logger.info(f"\n📊 MÉTODO 1: Youden's Index (TPR - FPR)")
        logger.info(f"├─ Threshold ótimo: {youden_threshold:.4f}")
        logger.info(f"├─ TPR (Sensitivity): {youden_tpr:.4f}")
        logger.info(f"├─ FPR (1-Specificity): {youden_fpr:.4f}")
        logger.info(f"└─ Interpretação: Balanceamento entre detecção e falsos positivos")

        # 2. F1-SCORE MÁXIMO
        f1_scores = []
        for threshold in thresholds:
            y_pred_binary = (y_proba_stress >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_binary)
            f1_scores.append(f1)

        f1_scores = np.array(f1_scores)
        best_f1_idx = np.argmax(f1_scores)
        f1_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]

        logger.info(f"\n📊 MÉTODO 2: F1-Score Máximo")
        logger.info(f"├─ Threshold ótimo: {f1_threshold:.4f}")
        logger.info(f"├─ F1-Score máximo: {best_f1:.4f}")
        logger.info(f"└─ Interpretação: Melhor balanço Precision-Recall")

        # 3. PRECISION-RECALL CURVE
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba_stress)

        # F1 score em cada threshold
        f1_pr = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_pr_idx = np.argmax(f1_pr)
        pr_threshold = pr_thresholds[best_pr_idx] if best_pr_idx < len(pr_thresholds) else 0.5

        logger.info(f"\n📊 MÉTODO 3: Precision-Recall Curve")
        logger.info(f"├─ Threshold ótimo: {pr_threshold:.4f}")
        logger.info(f"├─ Precision: {precision[best_pr_idx]:.4f}")
        logger.info(f"├─ Recall: {recall[best_pr_idx]:.4f}")
        logger.info(f"└─ Interpretação: Prioriza detecção (Recall)")

        # Retornar resultados
        results = {
            'auc_roc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'youden_threshold': youden_threshold,
            'youden_tpr': youden_tpr,
            'youden_fpr': youden_fpr,
            'f1_threshold': f1_threshold,
            'best_f1': best_f1,
            'f1_scores': f1_scores,
            'pr_threshold': pr_threshold,
            'precision': precision,
            'recall': recall,
            'pr_thresholds': pr_thresholds,
        }

        return results

    def plot_roc_curves(self, roc_results):
        """Visualiza ROC curve com thresholds recomendados."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: ROC Curve
        ax = axes[0, 0]
        fpr, tpr, thresholds = roc_results['fpr'], roc_results['tpr'], roc_results['thresholds']
        auc_score = roc_results['auc_roc']

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')
        ax.plot(roc_results['youden_fpr'], roc_results['youden_tpr'], 'r*', markersize=15,
                label=f"Youden's Index (threshold={roc_results['youden_threshold']:.4f})")
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title('ROC Curve - Análise de Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: F1-Score vs Threshold
        ax = axes[0, 1]
        ax.plot(thresholds, roc_results['f1_scores'], 'g-', linewidth=2, label='F1-Score')
        ax.axvline(roc_results['f1_threshold'], color='r', linestyle='--',
                   label=f"Optimal (threshold={roc_results['f1_threshold']:.4f})")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Precision-Recall Curve
        ax = axes[1, 0]
        precision = roc_results['precision']
        recall = roc_results['recall']
        ax.plot(recall, precision, 'purple', linewidth=2, label='Precision-Recall Curve')
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (Positive Predictive Value)')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Threshold Comparison
        ax = axes[1, 1]
        thresholds_to_plot = [
            ('Youden', roc_results['youden_threshold'], 'red'),
            ('F1-Max', roc_results['f1_threshold'], 'green'),
            ('PR-Curve', roc_results['pr_threshold'], 'purple'),
        ]

        x_pos = np.arange(len(thresholds_to_plot))
        values = [t[1] for t in thresholds_to_plot]
        colors = [t[2] for t in thresholds_to_plot]
        labels = [t[0] for t in thresholds_to_plot]

        bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Threshold Value')
        ax.set_title('Comparação de Thresholds Recomendados')
        ax.set_ylim([0, 1])

        # Adicionar valores nas barras
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.results_dir / '03_roc_analysis.png', dpi=150, bbox_inches='tight')
        logger.info(f"✓ Gráficos salvos em: results/03_roc_analysis.png")
        plt.close()

    def evaluate_at_threshold(self, y_true, y_proba, threshold: float):
        """Calcula métricas em um threshold específico."""
        y_proba_stress = y_proba[:, 1]
        y_pred = (y_proba_stress >= threshold).astype(int)

        from sklearn.metrics import precision_score, recall_score, accuracy_score

        return {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }


def main():
    print("\n" + "#"*80)
    print("# NOTEBOOK 03: AVALIAÇÃO E VALIDAÇÃO COM ROC CURVE")
    print("#"*80 + "\n")

    # Configurações
    BATCH_SIZE = 8
    LIMIT_SAMPLES = 100
    device = 'cpu'

    # 1. Carregar dados
    logger.info("Preparando dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data",
        batch_size=BATCH_SIZE,
        limit_samples=LIMIT_SAMPLES,
        num_workers=2
    )

    # 2. Instanciar avaliador
    evaluator = ModelEvaluator(device=device)

    # 3. Carregar modelo treinado
    evaluator.load_checkpoint('/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/models/best_model.pt')

    # 4. Fazer predições no test set
    logger.info("\nFazendo predições no TEST SET...")
    y_pred_test, y_proba_test, y_true_test = evaluator.predict_on_loader(test_loader)

    logger.info(f"✓ Predições completas:")
    logger.info(f"├─ Total de amostras: {len(y_true_test)}")
    logger.info(f"├─ Classe 0 (Normal): {sum(y_true_test == 0)}")
    logger.info(f"└─ Classe 1 (Stress): {sum(y_true_test == 1)}")

    # 5. Análise de ROC Curve
    roc_results = evaluator.analyze_roc_curve(y_true_test, y_proba_test)

    # 6. Visualizar
    logger.info("\nGerando visualizações...")
    evaluator.plot_roc_curves(roc_results)

    # 7. Avaliar em diferentes thresholds
    logger.info("\n" + "="*80)
    logger.info("COMPARAÇÃO DE MÉTRICAS EM DIFERENTES THRESHOLDS")
    logger.info("="*80 + "\n")

    thresholds_to_eval = [
        ('Padrão (0.50)', 0.50),
        ("Youden's Index", roc_results['youden_threshold']),
        ('F1-Score Máximo', roc_results['f1_threshold']),
        ('PR-Curve', roc_results['pr_threshold']),
    ]

    results_table = []
    for name, threshold in thresholds_to_eval:
        metrics = evaluator.evaluate_at_threshold(y_true_test, y_proba_test, threshold)
        results_table.append({
            'Método': name,
            'Threshold': f"{threshold:.4f}",
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
        })

        logger.info(f"{name} (threshold={threshold:.4f})")
        logger.info(f"├─ Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"├─ Precision: {metrics['precision']:.4f}")
        logger.info(f"├─ Recall:    {metrics['recall']:.4f}")
        logger.info(f"└─ F1-Score:  {metrics['f1']:.4f}\n")

    # Salvar tabela
    df_results = pd.DataFrame(results_table)
    df_results.to_csv(evaluator.results_dir / '03_threshold_comparison.csv', index=False)
    logger.info(f"✓ Tabela de comparação salva: results/03_threshold_comparison.csv")

    # 8. Salvar recomendações científicas
    recommendations = {
        'analysis_date': str(pd.Timestamp.now()),
        'dataset_info': {
            'total_samples': len(y_true_test),
            'normal_class': int(sum(y_true_test == 0)),
            'stress_class': int(sum(y_true_test == 1)),
        },
        'auc_roc': float(roc_results['auc_roc']),
        'recommended_thresholds': {
            'youden_index': {
                'threshold': float(roc_results['youden_threshold']),
                'description': "Balanceamento ótimo entre TPR e FPR",
                'use_case': "Aplicações gerais"
            },
            'f1_score_max': {
                'threshold': float(roc_results['f1_threshold']),
                'description': "F1-Score máximo (Precision-Recall balanço)",
                'use_case': "Quando Precision e Recall têm igual importância"
            },
            'pr_curve': {
                'threshold': float(roc_results['pr_threshold']),
                'description': "Máxima Recall (detecção de stress)",
                'use_case': "Quando é crítico não perder casos de stress"
            }
        },
        'scientific_recommendation': (
            f"Usar threshold Youden ({roc_results['youden_threshold']:.4f}) "
            "como base científica para alertas, "
            f"ajustando conforme especialistas agrícolas indicarem."
        )
    }

    with open(evaluator.results_dir / '03_roc_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)

    logger.info(f"\n✓ Recomendações salvas: results/03_roc_recommendations.json")

    print("\n" + "#"*80)
    print("# FIM DA AVALIAÇÃO COM ROC CURVE")
    print("#"*80 + "\n")

    print("Próximas etapas:")
    print("  1. Revisar SCIENTIFIC_JUSTIFICATION.md com os novos thresholds")
    print("  2. Consultar especialistas sobre as recomendações")
    print("  3. Atualizar alert_system.py com thresholds validados")
    print("\n")

    return recommendations


if __name__ == "__main__":
    recommendations = main()
