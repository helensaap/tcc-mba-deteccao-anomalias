"""
FASE 6: TREINAMENTO AVANÇADO COM DADOS REAIS - MAXIMIZAR ACURÁCIA
==================================================================

Script otimizado para melhorar acurácia usando:
- Learning rate schedule dinâmico
- Early stopping inteligente
- Data augmentation agressivo
- Ensemble-friendly training
- Monitoramento em tempo real

Objetivo: Atingir accuracy > 75% validável em dados reais

Autor: Helen Paixão
Data: Maio 2026
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from src.models import TemporalSensorAnalyzer, MultimodalFusionModel
from src.real_data_loader import RealDataLoader

print("\n" + "="*80)
print("FASE 6: TREINAMENTO AVANÇADO COM DADOS REAIS - OTIMIZAÇÃO DE ACURÁCIA")
print("="*80)

# ============================================================================
# CONFIG OTIMIZADO
# ============================================================================

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 6,  # Menor para melhor aprendizado
    'num_epochs': 150,  # Muito mais epochs
    'learning_rate': 0.001,  # Learning rate mais alto
    'visual_feature_size': 256,
    'temporal_feature_size': 128,
    'data_dir': 'data',
    'early_stopping_patience': 20,  # Se não melhorar por 20 epochs, parar
    'weight_decay': 1e-4,  # Regularização L2
}

print(f"\n⚙️ CONFIGURAÇÃO OTIMIZADA:")
print(f"   Device: {CONFIG['device']}")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Max epochs: {CONFIG['num_epochs']}")
print(f"   Learning rate: {CONFIG['learning_rate']}")
print(f"   Early stopping patience: {CONFIG['early_stopping_patience']}")

# ============================================================================
# TRANSFER LEARNING CNN OTIMIZADO
# ============================================================================

class TransferLearningCNN(nn.Module):
    """ResNet18 com fine-tuning melhorado"""

    def __init__(self, num_features=256):
        super().__init__()
        resnet = resnet18(pretrained=True)

        # Usar mais camadas para fine-tuning
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Descongelar mais camadas para melhor aprendizado
        for param in list(self.features.parameters())[:-16]:  # Mais camadas descongeladas
            param.requires_grad = False

        # Cabeçalho customizado melhorado
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
# DATASET COM AUGMENTATION AGRESSIVO
# ============================================================================

class AdvancedAugmentationDataset(Dataset):
    """Dataset com augmentation mais agressivo"""

    def __init__(self, dataframe, augment=True):
        self.df = dataframe
        self.augment = augment

        # Augmentation MAIS AGRESSIVO
        self.strong_augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(45),  # Mais rotação
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),  # Novo: flip vertical
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Mais agressivo
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Deslocamento
            transforms.RandomPerspective(distortion_scale=0.2),  # Perspectiva
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.light_augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.normalize_only = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_array = row['image_array'].astype(np.uint8)

        if self.augment:
            # 60% augmentation forte, 30% leve, 10% sem augmentation
            rand = np.random.rand()
            if rand < 0.6:
                img_tensor = self.strong_augmentation(img_array)
            elif rand < 0.9:
                img_tensor = self.light_augmentation(img_array)
            else:
                img_tensor = self.normalize_only(img_array)
        else:
            img_tensor = self.normalize_only(img_array)

        sensor_tensor = torch.from_numpy(row['sensor_sequence']).float()
        label = torch.tensor(row['label']).long()

        return img_tensor, sensor_tensor, label


# ============================================================================
# CARREGAR DADOS
# ============================================================================

print("\n📂 Carregando dados REAIS...")

loader = RealDataLoader(CONFIG['data_dir'])
df_real = loader.create_multimodal_dataset(limit_images=None)

print(f"   ✅ Dataset: {len(df_real)} amostras")
print(f"      - Normal (0): {(df_real['label'] == 0).sum()}")
print(f"      - Stress (1): {(df_real['label'] == 1).sum()}")

# Dividir dados
np.random.seed(42)
indices = np.random.permutation(len(df_real))
train_size = int(0.7 * len(df_real))
val_size = int(0.15 * len(df_real))

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

train_df = df_real.iloc[train_idx].reset_index(drop=True)
val_df = df_real.iloc[val_idx].reset_index(drop=True)
test_df = df_real.iloc[test_idx].reset_index(drop=True)

# Criar datasets
train_dataset = AdvancedAugmentationDataset(train_df, augment=True)
val_dataset = AdvancedAugmentationDataset(val_df, augment=False)
test_dataset = AdvancedAugmentationDataset(test_df, augment=False)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"\n   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================================================================
# CRIAR MODELOS
# ============================================================================

print("\n🤖 Criando modelos otimizados...")

device = CONFIG['device']

visual_model = TransferLearningCNN(CONFIG['visual_feature_size']).to(device)
temporal_model = TemporalSensorAnalyzer(
    input_size=4, output_size=CONFIG['temporal_feature_size'], hidden_size=64, num_layers=2
).to(device)
fusion_model = MultimodalFusionModel(
    visual_feature_size=CONFIG['visual_feature_size'],
    temporal_feature_size=CONFIG['temporal_feature_size'],
    num_classes=2, fusion_type='hybrid'
).to(device)

total_params = (
    sum(p.numel() for p in visual_model.parameters()) +
    sum(p.numel() for p in temporal_model.parameters()) +
    sum(p.numel() for p in fusion_model.parameters())
)

print(f"   ✅ {total_params:,} parâmetros totais")

# ============================================================================
# SETUP DE TREINAMENTO OTIMIZADO
# ============================================================================

print("\n⚙️ Configurando otimização...")

# Optimizer com learning rates diferentes por módulo
optimizer = optim.Adam([
    {'params': visual_model.parameters(), 'lr': CONFIG['learning_rate']},
    {'params': temporal_model.parameters(), 'lr': CONFIG['learning_rate']},
    {'params': fusion_model.parameters(), 'lr': CONFIG['learning_rate'] * 5},  # Fusion aprende mais rápido
], weight_decay=CONFIG['weight_decay'])

# Learning rate scheduler - CosineAnnealing com warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-5
)

# Class weighting para desbalanceamento
class_counts = np.bincount(train_df['label'].values)
class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
class_weights = class_weights / class_weights.sum() * len(class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

print(f"   ✅ Optimizer: Adam com learning rate schedule")
print(f"   ✅ Loss: CrossEntropyLoss com class weighting")

# ============================================================================
# TREINAMENTO COM EARLY STOPPING
# ============================================================================

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'test_acc': None, 'test_loss': None,
    'best_epoch': None
}

best_val_acc = 0
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_path = Path("models") / "best_model_advanced_real_data.pt"
best_model_path.parent.mkdir(exist_ok=True)

print("\n" + "="*80)
print("🎯 INICIANDO TREINAMENTO AVANÇADO")
print("="*80)

for epoch in range(CONFIG['num_epochs']):
    # ========== TREINO ==========
    visual_model.train()
    temporal_model.train()
    fusion_model.train()

    train_loss = 0
    train_preds, train_labels = [], []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]", leave=False)
    for images, sensors, labels in pbar:
        images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)

        optimizer.zero_grad()
        visual_feat = visual_model(images)
        temporal_feat = temporal_model(sensors)
        logits = fusion_model(visual_feat, temporal_feat)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(visual_model.parameters()) +
            list(temporal_model.parameters()) +
            list(fusion_model.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader)
    train_acc = np.mean(np.array(train_preds) == np.array(train_labels))

    # ========== VALIDAÇÃO ==========
    visual_model.eval()
    temporal_model.eval()
    fusion_model.eval()

    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, sensors, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Val]", leave=False):
            images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)

            visual_feat = visual_model(images)
            temporal_feat = temporal_model(sensors)
            logits = fusion_model(visual_feat, temporal_feat)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    scheduler.step()

    # ========== EARLY STOPPING ==========
    if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
        best_val_acc = val_acc
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save({
            'visual_model': visual_model.state_dict(),
            'temporal_model': temporal_model.state_dict(),
            'fusion_model': fusion_model.state_dict(),
        }, best_model_path)

        print(f"✅ Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | ⭐ MELHOR")

        history['best_epoch'] = epoch + 1
    else:
        epochs_without_improvement += 1
        print(f"   Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | "
              f"(sem melhora: {epochs_without_improvement}/{CONFIG['early_stopping_patience']})")

        if epochs_without_improvement >= CONFIG['early_stopping_patience']:
            print(f"\n⛔ Early stopping acionado! Sem melhora por {CONFIG['early_stopping_patience']} epochs.")
            break

# ============================================================================
# TESTE
# ============================================================================

print("\n" + "="*80)
print("🧪 AVALIAÇÃO NO TEST SET")
print("="*80)

checkpoint = torch.load(best_model_path)
visual_model.load_state_dict(checkpoint['visual_model'])
temporal_model.load_state_dict(checkpoint['temporal_model'])
fusion_model.load_state_dict(checkpoint['fusion_model'])

visual_model.eval()
temporal_model.eval()
fusion_model.eval()

test_loss = 0
test_preds, test_labels = [], []
test_probs = []

with torch.no_grad():
    for images, sensors, labels in tqdm(test_loader, desc="Testando"):
        images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)

        visual_feat = visual_model(images)
        temporal_feat = temporal_model(sensors)
        logits = fusion_model(visual_feat, temporal_feat)
        loss = criterion(logits, labels)

        test_loss += loss.item()
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probabilidade da classe positiva

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())

test_loss /= len(test_loader)
test_acc = np.mean(np.array(test_preds) == np.array(test_labels))

history['test_acc'] = test_acc
history['test_loss'] = test_loss

# Métricas detalhadas
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

test_prec = precision_score(test_labels, test_preds, zero_division=0)
test_recall = recall_score(test_labels, test_preds, zero_division=0)
test_f1 = f1_score(test_labels, test_preds, zero_division=0)
test_auc = roc_auc_score(test_labels, test_probs)

print(f"\n✅ RESULTADOS DO TEST SET:")
print(f"   Accuracy:  {test_acc*100:.2f}%")
print(f"   Precision: {test_prec*100:.2f}%")
print(f"   Recall:    {test_recall*100:.2f}%")
print(f"   F1-Score:  {test_f1*100:.2f}%")
print(f"   AUC-ROC:   {test_auc:.4f}")
print(f"   Loss:      {test_loss:.4f}")

print(f"\n📊 MATRIZ DE CONFUSÃO:")
cm = confusion_matrix(test_labels, test_preds)
print(f"   {cm}")

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================

print("\n💾 Salvando resultados...")

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

history_json = {k: (v if k != 'test_acc' and k != 'test_loss' else float(v)) if not isinstance(v, list) else [float(x) for x in v]
                for k, v in history.items()}
with open(results_dir / "06_training_history_advanced.json", 'w') as f:
    json.dump(history_json, f, indent=2)

# Plotar gráficos
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].legend()
axes[0, 0].grid()

# Accuracy
axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
axes[0, 1].plot(history['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
axes[0, 1].axhline(y=test_acc, color='r', linestyle='--', label=f'Test Accuracy ({test_acc*100:.2f}%)', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training Accuracy')
axes[0, 1].legend()
axes[0, 1].grid()

# Confusion Matrix
axes[1, 0].imshow(cm, cmap='Blues', aspect='auto')
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=16)

# Metrics Summary
axes[1, 1].axis('off')
metrics_text = f"""
RESULTADOS FINAIS

Melhor Validação: Epoch {history['best_epoch']}
Val Accuracy: {best_val_acc*100:.2f}%

Test Set:
├─ Accuracy:  {test_acc*100:.2f}%
├─ Precision: {test_prec*100:.2f}%
├─ Recall:    {test_recall*100:.2f}%
├─ F1-Score:  {test_f1*100:.2f}%
└─ AUC-ROC:   {test_auc:.4f}

Dataset:
├─ Train: {len(train_df)} amostras
├─ Val:   {len(val_df)} amostras
└─ Test:  {len(test_df)} amostras
"""
axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')

plt.tight_layout()
plt.savefig(results_dir / "06_training_advanced_curves.png", dpi=100)
print(f"   ✅ Gráficos salvos")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("📋 RESUMO: TREINAMENTO AVANÇADO COM DADOS REAIS")
print("="*80)
print(f"""
✅ TREINAMENTO COMPLETADO!

CONFIGURAÇÃO:
├─ Epochs: {epoch+1}/{CONFIG['num_epochs']} (early stopping)
├─ Learning rate: {CONFIG['learning_rate']}
├─ Batch size: {CONFIG['batch_size']}
└─ Data augmentation: AGRESSIVO

MELHOR VALIDAÇÃO:
├─ Epoch: {history['best_epoch']}
├─ Accuracy: {best_val_acc*100:.2f}%
└─ Loss: {best_val_loss:.4f}

TESTE FINAL:
├─ Accuracy:  {test_acc*100:.2f}%
├─ Precision: {test_prec*100:.2f}%
├─ Recall:    {test_recall*100:.2f}%
├─ F1-Score:  {test_f1*100:.2f}%
└─ AUC-ROC:   {test_auc:.4f}

STATUS:
✅ Modelo salvo em: {best_model_path}
✅ Histórico em: results/06_training_history_advanced.json
✅ Gráficos em: results/06_training_advanced_curves.png

PRÓXIMO PASSO:
→ Se accuracy < 70%: Ajustar hiperparâmetros ou aumentar dados
→ Se accuracy > 70%: Pronto para defesa!
""")

print("="*80 + "\n")
