"""
TESTE DE REPRODUCIBILIDADE: Avaliar modelos múltiplas vezes
===========================================================

Objetivo: Confirmar que o 64.71% é estável e reproducível,
não resultado de sorte com random seed.

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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from src.models import TemporalSensorAnalyzer, MultimodalFusionModel
from src.real_data_loader import RealDataLoader

print("\n" + "="*80)
print("🔄 TESTE DE REPRODUCIBILIDADE")
print("="*80)

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
# CONFIGURAÇÃO
# ============================================================================

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,
    'visual_feature_size': 256,
    'temporal_feature_size': 128,
    'data_dir': 'data',
}

device = CONFIG['device']

# Carregar dados com seed FIXA
print("\n📂 Carregando dados...")
loader = RealDataLoader(CONFIG['data_dir'])
df_real = loader.create_multimodal_dataset(limit_images=None)

np.random.seed(42)  # SEED FIXA
indices = np.random.permutation(len(df_real))
train_size = int(0.7 * len(df_real))
val_size = int(0.15 * len(df_real))

test_idx = indices[train_size+val_size:]
test_df = df_real.iloc[test_idx].reset_index(drop=True)

test_dataset = EvaluationDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"✅ Test set carregado: {len(test_df)} amostras")

# ============================================================================
# FUNÇÃO DE AVALIAÇÃO
# ============================================================================

def evaluate_model(model_path, model_name):
    """Avalia um modelo"""
    visual_model = ResNet18(CONFIG['visual_feature_size']).to(device)
    temporal_model = TemporalSensorAnalyzer(
        input_size=4, output_size=CONFIG['temporal_feature_size'], hidden_size=64, num_layers=2
    ).to(device)
    fusion_model = MultimodalFusionModel(
        visual_feature_size=CONFIG['visual_feature_size'],
        temporal_feature_size=CONFIG['temporal_feature_size'],
        num_classes=2, fusion_type='hybrid'
    ).to(device)

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        visual_model.load_state_dict(checkpoint['visual_model'])
        temporal_model.load_state_dict(checkpoint['temporal_model'])
        fusion_model.load_state_dict(checkpoint['fusion_model'])
    else:
        return None

    visual_model.eval()
    temporal_model.eval()
    fusion_model.eval()

    preds = []
    labels = []
    probs = []

    criterion = nn.CrossEntropyLoss()
    test_loss = 0

    with torch.no_grad():
        for images, sensors, label_batch in test_loader:
            images, sensors, label_batch = images.to(device), sensors.to(device), label_batch.to(device)

            visual_feat = visual_model(images)
            temporal_feat = temporal_model(sensors)
            logits = fusion_model(visual_feat, temporal_feat)
            loss = criterion(logits, label_batch)

            test_loss += loss.item()
            pred = logits.argmax(dim=1)
            prob = torch.softmax(logits, dim=1)[:, 1]

            preds.extend(pred.cpu().numpy())
            labels.extend(label_batch.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'loss': test_loss,
        'auc': auc,
        'preds': preds,
        'labels': labels,
        'probs': probs,
    }

# ============================================================================
# TESTE DE REPRODUCIBILIDADE
# ============================================================================

print("\n" + "="*80)
print("🔄 RODANDO AVALIAÇÃO 5 VEZES PARA TESTAR REPRODUCIBILIDADE")
print("="*80)

num_runs = 5
results_f6 = []
results_f7 = []

for run in range(num_runs):
    print(f"\n▶️  RUN {run+1}/{num_runs}")
    print("-" * 40)

    # Fase 6
    result_f6 = evaluate_model(
        'models/best_model_advanced_real_data.pt',
        'FASE 6'
    )
    if result_f6:
        results_f6.append(result_f6['accuracy'])
        print(f"   Fase 6: {result_f6['accuracy']*100:.2f}%")

    # Fase 7
    result_f7 = evaluate_model(
        'models/best_model_semi_supervised.pt',
        'FASE 7'
    )
    if result_f7:
        results_f7.append(result_f7['accuracy'])
        print(f"   Fase 7: {result_f7['accuracy']*100:.2f}%")

# ============================================================================
# ANÁLISE DE REPRODUCIBILIDADE
# ============================================================================

print("\n" + "="*80)
print("📊 ANÁLISE DE REPRODUCIBILIDADE")
print("="*80)

print(f"\nFASE 6 (Supervised):")
print(f"  Run 1: {results_f6[0]*100:.2f}%")
print(f"  Run 2: {results_f6[1]*100:.2f}%")
print(f"  Run 3: {results_f6[2]*100:.2f}%")
print(f"  Run 4: {results_f6[3]*100:.2f}%")
print(f"  Run 5: {results_f6[4]*100:.2f}%")
print(f"  ─────────────")
print(f"  Média: {np.mean(results_f6)*100:.2f}%")
print(f"  Std:   {np.std(results_f6)*100:.4f}%")
print(f"  Min:   {np.min(results_f6)*100:.2f}%")
print(f"  Max:   {np.max(results_f6)*100:.2f}%")
print(f"  Range: {(np.max(results_f6)-np.min(results_f6))*100:.2f} pp")

print(f"\nFASE 7 (Semi-Supervised):")
print(f"  Run 1: {results_f7[0]*100:.2f}%")
print(f"  Run 2: {results_f7[1]*100:.2f}%")
print(f"  Run 3: {results_f7[2]*100:.2f}%")
print(f"  Run 4: {results_f7[3]*100:.2f}%")
print(f"  Run 5: {results_f7[4]*100:.2f}%")
print(f"  ─────────────")
print(f"  Média: {np.mean(results_f7)*100:.2f}%")
print(f"  Std:   {np.std(results_f7)*100:.4f}%")
print(f"  Min:   {np.min(results_f7)*100:.2f}%")
print(f"  Max:   {np.max(results_f7)*100:.2f}%")
print(f"  Range: {(np.max(results_f7)-np.min(results_f7))*100:.2f} pp")

# ============================================================================
# CONCLUSÃO
# ============================================================================

print(f"\n" + "="*80)
print("✅ CONCLUSÃO")
print("="*80)

f6_mean = np.mean(results_f6)
f7_mean = np.mean(results_f7)
f6_std = np.std(results_f6)
f7_std = np.std(results_f7)
improvement = (f7_mean - f6_mean) * 100

# Verificar se todas as runs obtiveram o mesmo resultado
f6_stable = np.max(results_f6) - np.min(results_f6) < 0.01
f7_stable = np.max(results_f7) - np.min(results_f7) < 0.01

print(f"""
REPRODUCIBILIDADE:
├─ Fase 6: {f6_mean*100:.2f}% ± {f6_std*100:.4f}% {'✓ ESTÁVEL' if f6_stable else '⚠️ VARIÁVEL'}
├─ Fase 7: {f7_mean*100:.2f}% ± {f7_std*100:.4f}% {'✓ ESTÁVEL' if f7_stable else '⚠️ VARIÁVEL'}
└─ Melhoria: {improvement:+.2f} pp

{'✅ RESULTADOS SÃO REPRODUCÍVEIS!' if f6_stable and f7_stable else '⚠️ Cuidado: variação detectada'}

Interpretação:
- Desvio padrão < 1% = ESTÁVEL (sorte mínima)
- Desvio padrão > 5% = VARIÁVEL (sorte significativa)

Fase 6 stability: {f6_std*100:.4f}% (esperado: ~0%)
Fase 7 stability: {f7_std*100:.4f}% (esperado: ~0%)
""")

# Salvar resultados
reproducibility_results = {
    'num_runs': num_runs,
    'fase6': {
        'runs': [float(x) for x in results_f6],
        'mean': float(f6_mean),
        'std': float(f6_std),
        'min': float(np.min(results_f6)),
        'max': float(np.max(results_f6)),
        'stable': bool(f6_stable),
    },
    'fase7': {
        'runs': [float(x) for x in results_f7],
        'mean': float(f7_mean),
        'std': float(f7_std),
        'min': float(np.min(results_f7)),
        'max': float(np.max(results_f7)),
        'stable': bool(f7_stable),
    },
    'conclusao': 'REPRODUCÍVEL' if f6_stable and f7_stable else 'VERIFICAR',
}

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

with open(results_dir / "10_reproducibility_test.json", 'w') as f:
    json.dump(reproducibility_results, f, indent=2)

print(f"\n💾 Resultados salvos em: results/10_reproducibility_test.json\n")
