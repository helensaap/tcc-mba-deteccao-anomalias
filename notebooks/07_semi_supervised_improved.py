"""
FASE 7 MELHORADO: SEMI-SUPERVISED LEARNING COM OTIMIZAÇÕES
===========================================================

Melhorias implementadas:
1. Reduzir threshold de pseudo-label: 0.85 → 0.70 (mais pseudo-labels)
2. Aumentar lambda (unlabeled weight): 1.0 → 2.0 (aprende mais com unlabeled)
3. Melhorar data augmentation: 60/30/10 → 80/15/5 (mais regularização)

Objetivo: Atingir 75%+ de acurácia

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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from src.models import TemporalSensorAnalyzer, MultimodalFusionModel
from src.real_data_loader import RealDataLoader

# ============================================================================
# RESNET18 COM FINE-TUNING
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

print("\n" + "="*80)
print("FASE 7 MELHORADO: SEMI-SUPERVISED LEARNING COM OTIMIZAÇÕES")
print("="*80)

# ============================================================================
# CONFIG MELHORADA
# ============================================================================

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 0.0005,
    'visual_feature_size': 256,
    'temporal_feature_size': 128,
    'data_dir': 'data',
    'pseudo_label_threshold': 0.70,  # ⬇️ REDUZIDO (era 0.85)
    'lambda_u': 2.0,                  # ⬆️ AUMENTADO (era 1.0)
    'early_stopping_patience': 15,
}

device = CONFIG['device']

print(f"\n⚙️ CONFIGURAÇÃO OTIMIZADA:")
print(f"   Device: {CONFIG['device']}")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Pseudo-label threshold: {CONFIG['pseudo_label_threshold']} (⬇️ era 0.85)")
print(f"   Lambda (unlabeled weight): {CONFIG['lambda_u']} (⬆️ era 1.0)")
print(f"   Max epochs: {CONFIG['num_epochs']}")

# ============================================================================
# CARREGAR MODELO PRÉ-TREINADO (FASE 6)
# ============================================================================

print("\n📂 Carregando modelo pré-treinado (Fase 6)...")

visual_model = ResNet18(CONFIG['visual_feature_size']).to(device)
temporal_model = TemporalSensorAnalyzer(
    input_size=4, output_size=CONFIG['temporal_feature_size'], hidden_size=64, num_layers=2
).to(device)
fusion_model = MultimodalFusionModel(
    visual_feature_size=CONFIG['visual_feature_size'],
    temporal_feature_size=CONFIG['temporal_feature_size'],
    num_classes=2, fusion_type='hybrid'
).to(device)

model_path = Path("models") / "best_model_advanced_real_data.pt"
if model_path.exists():
    checkpoint = torch.load(model_path, map_location=device)
    visual_model.load_state_dict(checkpoint['visual_model'])
    temporal_model.load_state_dict(checkpoint['temporal_model'])
    fusion_model.load_state_dict(checkpoint['fusion_model'])
    print(f"   ✅ Modelo carregado de {model_path}")
else:
    print(f"   ⚠️  Modelo não encontrado")

visual_model.eval()
temporal_model.eval()
fusion_model.eval()

# ============================================================================
# DAILY IMAGE LOADER
# ============================================================================

class DailyImageLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.daily_images_dir = self.data_dir / 'raw' / '1st Experiment' / 'Images_1stExperiment' / '1stExperiment_Daily_Images'

    def load_unlabeled_images(self):
        print("\n📸 Carregando imagens diárias não-rotuladas...")

        unlabeled_images = []
        crops = ['cva', 'digital-cucumbers', 'koala', 'monday-lettuce', 'reference', 'veggie-might']
        varieties = ['raspberry', 'sigrow']

        for crop in crops:
            crop_dir = self.daily_images_dir / crop
            if crop_dir.exists():
                crop_total = 0

                for variety in varieties:
                    variety_dir = crop_dir / variety
                    if variety_dir.exists():
                        image_files = list(variety_dir.glob('*.png')) + list(variety_dir.glob('*.jpg'))
                        crop_total += len(image_files)

                        for img_path in tqdm(image_files, desc=f"Carregando {crop}/{variety}", leave=False):
                            try:
                                import cv2
                                img = cv2.imread(str(img_path))
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = cv2.resize(img, (224, 224))
                                    unlabeled_images.append({
                                        'image': img,
                                        'path': str(img_path),
                                        'crop': crop,
                                        'variety': variety
                                    })
                            except Exception as e:
                                pass

                if crop_total > 0:
                    print(f"   ✅ {crop}: {crop_total} imagens")

        print(f"\n   📊 Total de imagens não-rotuladas: {len(unlabeled_images)}")
        return unlabeled_images

# ============================================================================
# PSEUDO-LABELING COM THRESHOLD REDUZIDO
# ============================================================================

print("\n🤖 Executando pseudo-labeling nas 15k imagens...")

daily_loader = DailyImageLoader(CONFIG['data_dir'])
unlabeled_images = daily_loader.load_unlabeled_images()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

pseudo_labeled_data = []
pseudo_confidences = []

print("\n🔮 Gerando pseudo-labels com threshold REDUZIDO (0.70)...")

with torch.no_grad():
    for i, img_data in enumerate(tqdm(unlabeled_images, desc="Pseudo-labeling", total=len(unlabeled_images))):
        img_array = img_data['image']
        img_tensor = transform(img_array).unsqueeze(0).to(device)

        visual_feat = visual_model(img_tensor)
        temporal_feat = torch.zeros(1, CONFIG['temporal_feature_size']).to(device)
        logits = fusion_model(visual_feat, temporal_feat)
        probs = torch.softmax(logits, dim=1)
        pred_label = probs.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()

        # ⬇️ THRESHOLD REDUZIDO
        if confidence > CONFIG['pseudo_label_threshold']:
            pseudo_labeled_data.append({
                'image': img_array,
                'label': pred_label,
                'confidence': confidence,
                'path': img_data['path']
            })
            pseudo_confidences.append(confidence)

print(f"   ✅ {len(pseudo_labeled_data)} pseudo-labels gerados (confidence > {CONFIG['pseudo_label_threshold']})")
if pseudo_confidences:
    print(f"   📊 Confidence média: {np.mean(pseudo_confidences):.4f}")
    print(f"   📊 Label distribution: {sum(1 for d in pseudo_labeled_data if d['label']==0)} normal, {sum(1 for d in pseudo_labeled_data if d['label']==1)} stress")

# ============================================================================
# DATASET COM AUGMENTAÇÃO MELHORADA
# ============================================================================

class MixMatchDatasetImproved(Dataset):
    """Dataset com augmentação melhorada"""

    def __init__(self, labeled_df, pseudo_data, augment=True):
        self.labeled_df = labeled_df
        self.pseudo_data = pseudo_data
        self.augment = augment

        # ⬆️ AUGMENTAÇÃO MELHORADA (mais agressiva)
        self.strong_augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.weak_augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.no_augmentation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labeled_df) + len(self.pseudo_data)

    def __getitem__(self, idx):
        if idx < len(self.labeled_df):
            row = self.labeled_df.iloc[idx]
            img_array = row['image_array'].astype(np.uint8)
            label = torch.tensor(row['label']).long()
            is_labeled = torch.tensor(1.0)

            if self.augment:
                rand = np.random.rand()
                # ⬆️ DISTRIBUIÇÃO MELHORADA: 80% strong, 15% weak, 5% none
                if rand < 0.80:
                    img_tensor = self.strong_augmentation(img_array)
                elif rand < 0.95:
                    img_tensor = self.weak_augmentation(img_array)
                else:
                    img_tensor = self.no_augmentation(img_array)
            else:
                img_tensor = self.no_augmentation(img_array)

            sensor_tensor = torch.from_numpy(row['sensor_sequence']).float()

        else:
            pseudo_idx = idx - len(self.labeled_df)
            pseudo_sample = self.pseudo_data[pseudo_idx]
            img_array = pseudo_sample['image']
            label = torch.tensor(pseudo_sample['label']).long()
            is_labeled = torch.tensor(0.0)

            img_tensor = self.strong_augmentation(img_array)
            sensor_tensor = torch.zeros(24, 4)

        return img_tensor, sensor_tensor, label, is_labeled

# ============================================================================
# CARREGAR DADOS LABELED
# ============================================================================

print("\n📂 Carregando dados labeled...")

loader = RealDataLoader(CONFIG['data_dir'])
df_real = loader.create_multimodal_dataset(limit_images=None)

labeled_df = df_real.reset_index(drop=True)

print(f"   ✅ {len(labeled_df)} amostras labeled")

# ============================================================================
# CRIAR DATASET COMBINADO
# ============================================================================

print("\n🔗 Combinando labeled ({}) + pseudo-labeled ({})...".format(
    len(labeled_df), len(pseudo_labeled_data)))

combined_dataset = MixMatchDatasetImproved(labeled_df, pseudo_labeled_data, augment=True)
combined_loader = DataLoader(combined_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

print(f"   ✅ Dataset combinado: {len(combined_dataset)} amostras")

# ============================================================================
# PREPARAR MODELOS PARA TREINAMENTO
# ============================================================================

print("\n🤖 Preparando modelos para fine-tuning...")

visual_model.train()
temporal_model.train()
fusion_model.train()

optimizer = optim.Adam([
    {'params': visual_model.parameters(), 'lr': CONFIG['learning_rate']},
    {'params': temporal_model.parameters(), 'lr': CONFIG['learning_rate']},
    {'params': fusion_model.parameters(), 'lr': CONFIG['learning_rate'] * 2},
], weight_decay=1e-5)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=1e-6
)

class_weights = torch.FloatTensor([1.0, 1.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print(f"   ✅ Optimizer: Adam com learning rate schedule")
print(f"   ✅ Loss: CrossEntropyLoss (com class weighting)")

# ============================================================================
# TREINAMENTO MELHORADO
# ============================================================================

print("\n" + "="*80)
print("🎯 INICIANDO SEMI-SUPERVISED TRAINING MELHORADO")
print("="*80)

history = {
    'labeled_loss': [],
    'unlabeled_loss': [],
    'total_loss': [],
    'epoch': []
}

best_loss = float('inf')
epochs_without_improvement = 0
best_model_path = Path("models") / "best_model_semi_supervised_improved.pt"
best_model_path.parent.mkdir(exist_ok=True)

for epoch in range(CONFIG['num_epochs']):
    labeled_loss_sum = 0
    unlabeled_loss_sum = 0
    total_loss_sum = 0
    num_labeled = 0
    num_unlabeled = 0

    pbar = tqdm(combined_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}", leave=True)

    for images, sensors, labels, is_labeled in pbar:
        images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)
        is_labeled = is_labeled.to(device)

        optimizer.zero_grad()

        visual_feat = visual_model(images)
        temporal_feat = temporal_model(sensors)
        logits = fusion_model(visual_feat, temporal_feat)

        # Loss para labeled data
        labeled_mask = is_labeled == 1
        if labeled_mask.sum() > 0:
            labeled_logits = logits[labeled_mask]
            labeled_labels = labels[labeled_mask]
            labeled_loss = criterion(labeled_logits, labeled_labels)
            num_labeled += labeled_mask.sum().item()
        else:
            labeled_loss = torch.tensor(0.0).to(device)

        # Loss para unlabeled data (com lambda aumentado)
        unlabeled_mask = is_labeled == 0
        if unlabeled_mask.sum() > 0:
            unlabeled_logits = logits[unlabeled_mask]
            unlabeled_labels = labels[unlabeled_mask]
            unlabeled_loss = criterion(unlabeled_logits, unlabeled_labels)
            # ⬆️ LAMBDA AUMENTADO
            unlabeled_loss = unlabeled_loss * CONFIG['lambda_u']
            num_unlabeled += unlabeled_mask.sum().item()
        else:
            unlabeled_loss = torch.tensor(0.0).to(device)

        total_loss = labeled_loss + unlabeled_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(visual_model.parameters()) +
            list(temporal_model.parameters()) +
            list(fusion_model.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        labeled_loss_sum += labeled_loss.item() * labeled_mask.sum().item() if labeled_mask.sum() > 0 else 0
        unlabeled_loss_sum += unlabeled_loss.item() * unlabeled_mask.sum().item() if unlabeled_mask.sum() > 0 else 0
        total_loss_sum += total_loss.item()

        pbar.set_postfix({
            'labeled_loss': labeled_loss_sum / max(num_labeled, 1),
            'unlabeled_loss': unlabeled_loss_sum / max(num_unlabeled, 1),
            'total_loss': total_loss_sum / (pbar.n + 1)
        })

    scheduler.step()

    avg_labeled_loss = labeled_loss_sum / max(num_labeled, 1)
    avg_unlabeled_loss = unlabeled_loss_sum / max(num_unlabeled, 1)
    avg_total_loss = total_loss_sum / len(combined_loader)

    history['labeled_loss'].append(avg_labeled_loss)
    history['unlabeled_loss'].append(avg_unlabeled_loss)
    history['total_loss'].append(avg_total_loss)
    history['epoch'].append(epoch + 1)

    print(f"\n📊 Epoch {epoch+1:3d} | "
          f"Labeled Loss: {avg_labeled_loss:.4f} | "
          f"Unlabeled Loss: {avg_unlabeled_loss:.4f} | "
          f"Total Loss: {avg_total_loss:.4f}")

    if avg_total_loss < best_loss:
        best_loss = avg_total_loss
        epochs_without_improvement = 0

        torch.save({
            'visual_model': visual_model.state_dict(),
            'temporal_model': temporal_model.state_dict(),
            'fusion_model': fusion_model.state_dict(),
        }, best_model_path)

        print(f"   ✅ MELHOR MODELO SALVO (Loss: {best_loss:.4f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= CONFIG['early_stopping_patience']:
            print(f"\n⛔ Early stopping acionado!")
            break

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("💾 SALVANDO RESULTADOS")
print("="*80)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

with open(results_dir / "07_semi_supervised_improved_history.json", 'w') as f:
    json.dump({
        'labeled_loss': [float(x) for x in history['labeled_loss']],
        'unlabeled_loss': [float(x) for x in history['unlabeled_loss']],
        'total_loss': [float(x) for x in history['total_loss']],
        'epochs': history['epoch'],
        'final_epoch': epoch + 1,
        'pseudo_labels_used': len(pseudo_labeled_data),
        'labeled_samples': len(labeled_df),
        'total_samples': len(combined_dataset),
        'best_loss': float(best_loss),
        'config': {
            'pseudo_label_threshold': CONFIG['pseudo_label_threshold'],
            'lambda_u': CONFIG['lambda_u'],
        }
    }, f, indent=2)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['labeled_loss'], label='Labeled Loss', marker='o')
plt.plot(history['epoch'], history['unlabeled_loss'], label='Unlabeled Loss', marker='s')
plt.plot(history['epoch'], history['total_loss'], label='Total Loss', marker='^', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Semi-Supervised Training Loss (MELHORADO)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.text(0.1, 0.9, 'RESUMO SEMI-SUPERVISED MELHORADO', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, f'Labeled samples: {len(labeled_df)}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.7, f'Pseudo-labeled: {len(pseudo_labeled_data)} (threshold: 0.70)', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.6, f'Total training: {len(combined_dataset)}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.5, f'Epochs: {epoch+1}/{CONFIG["num_epochs"]}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.4, f'Best loss: {best_loss:.4f}', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.3, f'Lambda (SSL weight): {CONFIG["lambda_u"]} (⬆️ era 1.0)', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.2, f'Augmentation: 80% strong / 15% weak / 5% none', fontsize=11, transform=plt.gca().transAxes)
plt.axis('off')

plt.tight_layout()
plt.savefig(results_dir / "07_semi_supervised_improved_training.png", dpi=100)
print(f"   ✅ Gráficos salvos")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("📋 RESUMO: SEMI-SUPERVISED LEARNING MELHORADO")
print("="*80)
print(f"""
✅ TREINAMENTO COMPLETADO!

DADOS:
├─ Labeled samples: {len(labeled_df)}
├─ Pseudo-labeled: {len(pseudo_labeled_data)} (threshold: {CONFIG['pseudo_label_threshold']} ⬇️)
└─ Total training: {len(combined_dataset)}

TREINAMENTO:
├─ Epochs: {epoch+1}/{CONFIG['num_epochs']} (early stopping)
├─ Learning rate: {CONFIG['learning_rate']}
├─ Lambda (unlabeled weight): {CONFIG['lambda_u']} (⬆️ era 1.0)
├─ Augmentation: 80% strong / 15% weak / 5% none (⬆️)
└─ Best loss: {best_loss:.4f}

MODELO:
├─ Visual: ResNet18 + fine-tuning
├─ Temporal: LSTM 2-layer
├─ Fusion: Hybrid
└─ Salvo em: {best_model_path}

PRÓXIMOS PASSOS:
→ Avaliar modelo em test set (notebooks/08_evaluate_semi_supervised.py)
→ Comparar com Fase 6 (52.94%)
→ Espera-se melhoria de 10-15% com otimizações!
""")

print("="*80 + "\n")
