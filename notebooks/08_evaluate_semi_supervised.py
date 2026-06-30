"""
FASE 8: AVALIAÇÃO - COMPARAR MODELO SEMI-SUPERVISED vs FASE 6
=============================================================

Avalia o novo modelo (best_model_semi_supervised.pt) no test set
e compara com o modelo de Fase 6 (best_model_advanced_real_data.pt)

Autor: Helen Paixão
Data: Maio 2026
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from src.models import TemporalSensorAnalyzer, MultimodalFusionModel
from src.real_data_loader import RealDataLoader

print("\n" + "="*80)
print("FASE 8: AVALIAÇÃO - MODELO SEMI-SUPERVISED vs FASE 6")
print("="*80)

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,
    'visual_feature_size': 256,
    'temporal_feature_size': 128,
    'data_dir': 'data',
}

device = CONFIG['device']

# ============================================================================
# RESNET18 COM FINE-TUNING
# ============================================================================

class ResNet18(nn.Module):
    """ResNet18 com fine-tuning"""
    def __init__(self, num_features=256):
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        for param in list(self.features.parameters())[:-16]:
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_features),
            nn.BatchNorm1d(num_features),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================================
# DATASET
# ============================================================================

class EvaluationDataset(torch.utils.data.Dataset):
    """Dataset para avaliação"""
    def __init__(self, dataframe):
        self.df = dataframe
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_array = row['image_array'].astype(np.uint8)
        img_tensor = self.transform(img_array)
        sensor_tensor = torch.from_numpy(row['sensor_sequence']).float()
        label = torch.tensor(row['label']).long()
        return img_tensor, sensor_tensor, label

# ============================================================================
# CARREGAR DADOS
# ============================================================================

print("\n📂 Carregando dados...")

loader = RealDataLoader(CONFIG['data_dir'])
df_real = loader.create_multimodal_dataset(limit_images=None)

# Dividir dados
np.random.seed(42)
indices = np.random.permutation(len(df_real))
train_size = int(0.7 * len(df_real))
val_size = int(0.15 * len(df_real))

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

test_df = df_real.iloc[test_idx].reset_index(drop=True)

test_dataset = EvaluationDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"   ✅ Test set: {len(test_df)} amostras")

# ============================================================================
# FUNÇÃO DE AVALIAÇÃO
# ============================================================================

def evaluate_model(model_path, model_name):
    """Avalia um modelo"""
    print(f"\n🤖 Carregando modelo: {model_name}")
    print(f"   Path: {model_path}")

    # Criar modelos
    visual_model = ResNet18(CONFIG['visual_feature_size']).to(device)
    temporal_model = TemporalSensorAnalyzer(
        input_size=4, output_size=CONFIG['temporal_feature_size'], hidden_size=64, num_layers=2
    ).to(device)
    fusion_model = MultimodalFusionModel(
        visual_feature_size=CONFIG['visual_feature_size'],
        temporal_feature_size=CONFIG['temporal_feature_size'],
        num_classes=2, fusion_type='hybrid'
    ).to(device)

    # Carregar pesos
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        visual_model.load_state_dict(checkpoint['visual_model'])
        temporal_model.load_state_dict(checkpoint['temporal_model'])
        fusion_model.load_state_dict(checkpoint['fusion_model'])
        print(f"   ✅ Modelo carregado com sucesso")
    else:
        print(f"   ❌ Modelo não encontrado!")
        return None

    # Modo avaliação
    visual_model.eval()
    temporal_model.eval()
    fusion_model.eval()

    # Avaliar
    test_loss = 0
    test_preds, test_labels = [], []
    test_probs = []

    criterion = nn.CrossEntropyLoss()

    print(f"\n🧪 Avaliando no test set...")

    with torch.no_grad():
        for images, sensors, labels in tqdm(test_loader, desc="Avaliação", leave=False):
            images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)

            visual_feat = visual_model(images)
            temporal_feat = temporal_model(sensors)
            logits = fusion_model(visual_feat, temporal_feat)
            loss = criterion(logits, labels)

            test_loss += loss.item()
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))

    # Métricas
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)

    try:
        test_auc = roc_auc_score(test_labels, test_probs)
    except:
        test_auc = 0.0

    cm = confusion_matrix(test_labels, test_preds)

    results = {
        'model_name': model_name,
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_recall,
        'f1': test_f1,
        'auc_roc': test_auc,
        'loss': test_loss,
        'confusion_matrix': cm,
        'predictions': test_preds,
        'labels': test_labels,
        'probabilities': test_probs
    }

    return results

# ============================================================================
# AVALIAR AMBOS OS MODELOS
# ============================================================================

print("\n" + "="*80)
print("AVALIANDO MODELOS")
print("="*80)

# Fase 6
resultado_fase6 = evaluate_model(
    'models/best_model_advanced_real_data.pt',
    'FASE 6 (Supervised + 107 imagens)'
)

# Fase 7
resultado_fase7 = evaluate_model(
    'models/best_model_semi_supervised.pt',
    'FASE 7 (Semi-Supervised + 2 pseudo-labels)'
)

# ============================================================================
# COMPARAÇÃO
# ============================================================================

if resultado_fase6 and resultado_fase7:
    print("\n" + "="*80)
    print("📊 COMPARAÇÃO: FASE 6 vs FASE 7")
    print("="*80)

    print(f"\n{'Métrica':<20} {'Fase 6':<15} {'Fase 7':<15} {'Mudança':<15}")
    print("-" * 65)

    # Accuracy
    acc6 = resultado_fase6['accuracy']
    acc7 = resultado_fase7['accuracy']
    mudanca = acc7 - acc6
    sinal = "⬆️" if mudanca > 0 else ("⬇️" if mudanca < 0 else "→")
    print(f"{'Accuracy':<20} {acc6*100:>6.2f}%        {acc7*100:>6.2f}%        {mudanca*100:>+6.2f}% {sinal}")

    # Precision
    prec6 = resultado_fase6['precision']
    prec7 = resultado_fase7['precision']
    mudanca = prec7 - prec6
    sinal = "⬆️" if mudanca > 0 else ("⬇️" if mudanca < 0 else "→")
    print(f"{'Precision':<20} {prec6*100:>6.2f}%        {prec7*100:>6.2f}%        {mudanca*100:>+6.2f}% {sinal}")

    # Recall
    rec6 = resultado_fase6['recall']
    rec7 = resultado_fase7['recall']
    mudanca = rec7 - rec6
    sinal = "⬆️" if mudanca > 0 else ("⬇️" if mudanca < 0 else "→")
    print(f"{'Recall':<20} {rec6*100:>6.2f}%        {rec7*100:>6.2f}%        {mudanca*100:>+6.2f}% {sinal}")

    # F1-Score
    f16 = resultado_fase6['f1']
    f17 = resultado_fase7['f1']
    mudanca = f17 - f16
    sinal = "⬆️" if mudanca > 0 else ("⬇️" if mudanca < 0 else "→")
    print(f"{'F1-Score':<20} {f16*100:>6.2f}%        {f17*100:>6.2f}%        {mudanca*100:>+6.2f}% {sinal}")

    # AUC-ROC
    auc6 = resultado_fase6['auc_roc']
    auc7 = resultado_fase7['auc_roc']
    mudanca = auc7 - auc6
    sinal = "⬆️" if mudanca > 0 else ("⬇️" if mudanca < 0 else "→")
    print(f"{'AUC-ROC':<20} {auc6:>6.4f}        {auc7:>6.4f}        {mudanca:>+6.4f} {sinal}")

    # Loss
    loss6 = resultado_fase6['loss']
    loss7 = resultado_fase7['loss']
    mudanca = loss7 - loss6
    sinal = "⬆️" if mudanca > 0 else ("⬇️" if mudanca < 0 else "→")
    print(f"{'Loss':<20} {loss6:>6.4f}        {loss7:>6.4f}        {mudanca:>+6.4f} {sinal}")

    # Matrizes de confusão
    print(f"\n{'Matriz de Confusão':<20}")
    print(f"{'Fase 6:':<20}")
    print(f"{resultado_fase6['confusion_matrix']}")
    print(f"\n{'Fase 7:':<20}")
    print(f"{resultado_fase7['confusion_matrix']}")

    # Conclusão
    print("\n" + "="*80)
    print("📋 CONCLUSÃO")
    print("="*80)

    if acc7 > acc6:
        melhoria = ((acc7 - acc6) / acc6) * 100
        print(f"\n✅ MELHORIA DETECTADA!")
        print(f"   Acurácia melhorou de {acc6*100:.2f}% para {acc7*100:.2f}%")
        print(f"   Ganho: +{melhoria:.2f}% relativo")
        print(f"\n   🎯 Semi-supervised learning COM benefício!")
        print(f"   Mesmo com poucos pseudo-labels, modelo aprendeu melhor.")
    elif acc7 < acc6:
        piora = ((acc6 - acc7) / acc6) * 100
        print(f"\n⚠️  PIORA DETECTADA")
        print(f"   Acurácia piorou de {acc6*100:.2f}% para {acc7*100:.2f}%")
        print(f"   Perda: -{piora:.2f}% relativo")
        print(f"\n   Motivos possíveis:")
        print(f"   • Só 2 pseudo-labels (muito poucos)")
        print(f"   • Pseudo-labels podem estar incorretos")
        print(f"   • Threshold 0.85 selecionou amostras ruins")
        print(f"\n   💡 Sugestão: Tentar com threshold 0.70-0.75")
    else:
        print(f"\n→ RESULTADOS EQUIVALENTES")
        print(f"   Ambos com acurácia {acc6*100:.2f}%")
        print(f"   Semi-supervised não prejudicou, apenas manteve performance")

    # Salvar resultados
    print(f"\n💾 Salvando resultados...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    comparison = {
        'fase6': {
            'accuracy': float(resultado_fase6['accuracy']),
            'precision': float(resultado_fase6['precision']),
            'recall': float(resultado_fase6['recall']),
            'f1': float(resultado_fase6['f1']),
            'auc_roc': float(resultado_fase6['auc_roc']),
            'loss': float(resultado_fase6['loss']),
        },
        'fase7': {
            'accuracy': float(resultado_fase7['accuracy']),
            'precision': float(resultado_fase7['precision']),
            'recall': float(resultado_fase7['recall']),
            'f1': float(resultado_fase7['f1']),
            'auc_roc': float(resultado_fase7['auc_roc']),
            'loss': float(resultado_fase7['loss']),
        },
        'melhoria_accuracy': float(resultado_fase7['accuracy'] - resultado_fase6['accuracy']),
        'melhoria_f1': float(resultado_fase7['f1'] - resultado_fase6['f1']),
    }

    with open(results_dir / "08_evaluation_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"   ✅ Resultados salvos em results/08_evaluation_comparison.json")

print("\n" + "="*80 + "\n")
