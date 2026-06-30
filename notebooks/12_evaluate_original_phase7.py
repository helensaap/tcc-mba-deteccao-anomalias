"""
AVALIAÇÃO: Phase 7 ORIGINAL (não-melhorado)
============================================

Objetivo: Confirmar se o Phase 7 Original realmente é 64.71%
Arquivo: models/best_model_semi_supervised.pt
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
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score
)
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from src.models import TemporalSensorAnalyzer, MultimodalFusionModel
from src.real_data_loader import RealDataLoader

print("\n" + "="*80)
print("AVALIAÇÃO: PHASE 7 ORIGINAL (SEM MELHORIAS)")
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
# RESNET18
# ============================================================================

class ResNet18(nn.Module):
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

np.random.seed(42)
indices = np.random.permutation(len(df_real))
train_size = int(0.7 * len(df_real))
val_size = int(0.15 * len(df_real))

test_idx = indices[train_size+val_size:]
test_df = df_real.iloc[test_idx].reset_index(drop=True)

test_dataset = EvaluationDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"✅ Test set: {len(test_df)} amostras")

# ============================================================================
# AVALIAR PHASE 7 ORIGINAL
# ============================================================================

print(f"\n🤖 Carregando modelo: PHASE 7 ORIGINAL")
print(f"   Path: models/best_model_semi_supervised.pt")

visual_model = ResNet18(CONFIG['visual_feature_size']).to(device)
temporal_model = TemporalSensorAnalyzer(
    input_size=4, output_size=CONFIG['temporal_feature_size'], hidden_size=64, num_layers=2
).to(device)
fusion_model = MultimodalFusionModel(
    visual_feature_size=CONFIG['visual_feature_size'],
    temporal_feature_size=CONFIG['temporal_feature_size'],
    num_classes=2, fusion_type='hybrid'
).to(device)

model_path = Path("models") / "best_model_semi_supervised.pt"
if model_path.exists():
    checkpoint = torch.load(model_path, map_location=device)
    visual_model.load_state_dict(checkpoint['visual_model'])
    temporal_model.load_state_dict(checkpoint['temporal_model'])
    fusion_model.load_state_dict(checkpoint['fusion_model'])
    print(f"   ✅ Modelo carregado com sucesso")
else:
    print(f"   ❌ Modelo não encontrado!")
    exit(1)

visual_model.eval()
temporal_model.eval()
fusion_model.eval()

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

test_prec = precision_score(test_labels, test_preds, zero_division=0)
test_recall = recall_score(test_labels, test_preds, zero_division=0)
test_f1 = f1_score(test_labels, test_preds, zero_division=0)

try:
    test_auc = roc_auc_score(test_labels, test_probs)
except:
    test_auc = 0.0

cm = confusion_matrix(test_labels, test_preds)

# ============================================================================
# EXIBIR RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("📊 RESULTADOS: PHASE 7 ORIGINAL")
print("="*80)

print(f"\n{'Métrica':<20} {'Valor':<15}")
print("-" * 35)
print(f"{'Accuracy':<20} {test_acc*100:>6.2f}%")
print(f"{'Precision':<20} {test_prec*100:>6.2f}%")
print(f"{'Recall':<20} {test_recall*100:>6.2f}%")
print(f"{'F1-Score':<20} {test_f1*100:>6.2f}%")
print(f"{'AUC-ROC':<20} {test_auc:>6.4f}")
print(f"{'Loss':<20} {test_loss:>6.4f}")

print(f"\n📋 MATRIZ DE CONFUSÃO:")
print(f"   {'':>10} Pred Normal  Pred Stress")
print(f"   Real Normal     {cm[0,0]:>4d}         {cm[0,1]:>4d}")
print(f"   Real Stress     {cm[1,0]:>4d}         {cm[1,1]:>4d}")

print(f"\n📈 BREAKDOWN:")
print(f"   Total test samples: {len(test_labels)}")
print(f"   Acertos: {int(test_acc * len(test_labels))}/{len(test_labels)}")
print(f"   Stress samples: {sum(test_labels)}")
print(f"   Stress detectados: {cm[1,1]}")

print("\n" + "="*80 + "\n")
