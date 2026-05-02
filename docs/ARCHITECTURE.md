# Arquitetura de IA Multimodal para Predição de Estresse Abiótico

## Visão Geral do Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SISTEMA MULTIMODAL DE DETECÇÃO DE ESTRESSE            │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   Dados Brutos   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Data Pipeline  │
                    │  (Normalização)  │
                    └────┬─────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼────────┐            ┌────────▼──────┐
    │   IMAGENS   │            │  SENSORES IoT │
    │  (600x800)  │            │ (Séries Temp) │
    └────┬────────┘            └────────┬──────┘
         │                             │
    ┌────▼─────────────────────────────▼──┐
    │   Pré-processamento Multimodal      │
    │ - Resize & Normalização (imagens)   │
    │ - StandardScaler (sensores)         │
    └────┬──────────────────────────────┬─┘
         │                              │
    ┌────▼──────────────┐     ┌─────────▼──────┐
    │   CNN (ResNet18)  │     │   LSTM + ATT   │
    │                   │     │   (Temporal)   │
    │ Conv Layers       │     │                │
    │ ↓                 │     │ Attention      │
    │ Visual Features   │     │ ↓              │
    │ (256-dim)         │     │ Temporal Feats │
    │                   │     │ (128-dim)      │
    └────┬──────────────┘     └─────────┬──────┘
         │                              │
         └──────────────┬───────────────┘
                        │
             ┌──────────▼─────────────┐
             │  FUSION MODULE         │
             │ ┌────────────────────┐ │
             │ │ Early: Concat      │ │
             │ │ Late: Addition     │ │
             │ │ Hybrid: Mixed      │ │
             │ └────────────────────┘ │
             │      ↓                 │
             │  (256 + 128 = 384 dim) │
             └──────────┬──────────────┘
                        │
             ┌──────────▼─────────────┐
             │   CLASSIFIER           │
             │  Dense Layers          │
             │  Softmax               │
             │      ↓                 │
             │  [P(Normal)            │
             │   P(Stress)]           │
             └──────────┬──────────────┘
                        │
             ┌──────────▼─────────────┐
             │   ALERT SYSTEM         │
             │                        │
             │ IF confidence > 0.90   │
             │   → ALERTA CRÍTICO     │
             │ ELIF confidence > 0.75 │
             │   → ALERTA MODERADO    │
             │ ELIF confidence > 0.60 │
             │   → ALERTA LEVE        │
             └────────────────────────┘
```

## 1. Componentes Principais

### 1.1 Data Loader (`src/data_loader.py`)
- **MultimodalDataLoader**: Coordena carregamento de:
  - Imagens PNG (15.336 imagens, 600x800)
  - Metadados JSON (RealSense D415)
  - Dados XLSX (Weather, Crop, Controls)
- **ImagePreprocessor**: Resize, normalização, augmentação

### 1.2 Pipeline de Dados (`src/pipeline.py`)
- **DataPipeline**: Orquestra todo o fluxo
- **MultimodalDataset**: Dataset PyTorch com suporte a (imagem, série temporal, label)
- Divisão: 70% treino, 15% validação, 15% teste
- Normalização com StandardScaler

### 1.3 Modelos de Deep Learning (`src/models.py`)

#### CNN para Visão Computacional
```python
PhenotypicFeatureExtractor (ResNet18)
├── Conv Layer (64 channels)
├── Residual Blocks (64 → 128 → 256)
├── Global Average Pooling
├── Fully Connected (256-dim embedding)
└── Output: 256-dimensional feature vector
```

**Objetivo**: Extrair características visuais sutis que indicam estresse:
- Alteração de coloração (deficiência nutricional)
- Textura foliar anômala
- Sinais de murchamento incipiente

#### LSTM para Análise Temporal
```python
TemporalSensorAnalyzer (2-layer LSTM)
├── Input: (batch_size, seq_length=24, num_sensors=4)
├── LSTM Layers (hidden_size=128)
├── Attention Mechanism
│   ├── Compute attention weights (T,)
│   └── Weighted sum of LSTM outputs
├── Fully Connected (128-dim embedding)
└── Output: 128-dimensional feature vector
```

**Objetivo**: Modelar dependências temporais em:
- Temperatura (oscilações, volatilidade)
- Umidade relativa (estabilidade)
- CO₂ (concentração, desvios)
- Radiação/Luz (padrões irregulares)

### 1.4 Fusão Multimodal (`src/models.py`)
```python
MultimodalFusionModel
├── Visual Projection: 256 → 256 (hidden_size)
├── Temporal Projection: 128 → 256
├── Fusion Strategies:
│   ├── EARLY:  [proj_visual, proj_temporal] → concat
│   ├── LATE:   proj_visual + proj_temporal → element-wise add
│   ├── HYBRID: concat([proj_visual, proj_temporal, visual*temporal])
└── Classifier:
    ├── Dense(384 → 256)
    ├── ReLU + BatchNorm + Dropout
    ├── Dense(256 → 128)
    ├── Dense(128 → 2)  [num_classes=2: Normal/Stress]
    └── Softmax → Probability
```

### 1.5 Sistema de Alertas (`src/alert_system.py`)
```python
StressDetector
├── Detect Visual Anomalies:
│   ├── Color shift > 0.30
│   ├── Wilting index > 0.40
│   └── Texture variance > 0.50
├── Detect Temporal Anomalies:
│   ├── Temperature volatility > 3.0°C
│   ├── Humidity variance > 0.25
│   ├── CO₂ deviation > 150 ppm
│   └── Temporal irregularity > 0.60
└── Generate Alerts:
    ├── NORMAL (confidence < 0.60)
    ├── MILD (0.60-0.75) → "Monitoramento aumentado"
    ├── MODERATE (0.75-0.90) → "Intervenção necessária"
    └── SEVERE (>0.90) → "ALERTA CRÍTICO"
```

### 1.6 Métricas (`src/metrics.py`)
- Acurácia
- Precisão (Precision)
- Recall (Sensibilidade)
- F1-Score
- AUC-ROC
- Matriz de Confusão
- Early Stopping Callback

## 2. Fluxo de Treinamento

```
┌─ Inicializar modelos (CNN, LSTM, Fusion)
│
├─ Para cada época:
│  ├─ Para cada batch de treino:
│  │  ├─ Forward pass (imagem → CNN, série temporal → LSTM)
│  │  ├─ Fusão multimodal
│  │  ├─ Classificação
│  │  ├─ Calcular Loss (CrossEntropyLoss)
│  │  ├─ Backward pass (gradientes)
│  │  └─ Optimizer step (Adam)
│  │
│  ├─ Validação:
│  │  ├─ Eval mode (sem dropout, BN statistics)
│  │  ├─ Calcular métricas
│  │  └─ Early stopping check
│  │
│  └─ Log: Loss, Acurácia, F1-Score
│
└─ Salvar modelo com melhor F1-Score
```

## 3. Dataset e Dados

### Dataset Disponível
- **Planta**: Sigrow (leafy greens em ambiente controlado)
- **Período**: Fevereiro 2022 - Março 2022
- **Imagens**: 15.336 fotos (600x800 pixels)
- **Sensores**:
  - Temperatura externa/interna
  - Umidade relativa
  - Radiação solar
  - CO₂ (implícito em GreenhouseControls)
- **Ground Truth**: Classificação de plantas em GreenhouseCrop.xlsx
  - Class A: Alta qualidade de bioativos
  - Class B: Qualidade média
  - Class C: Baixa qualidade / Estresse

### Mapeamento de Labels
- Class A, B → Label 0 (Normal)
- Class C → Label 1 (Stress)

## 4. Hiperparâmetros

```python
# Modelo CNN
image_size = (224, 224)
visual_feature_size = 256
num_residual_blocks = 3

# Modelo LSTM
num_sensor_vars = 4  # Temperatura, Umidade, CO2, Radiação
sequence_length = 24  # 24 timesteps (1 dia de sensores)
hidden_size_lstm = 128
num_layers_lstm = 2

# Fusão
fusion_type = 'hybrid'  # early, late, ou hybrid
hidden_size_fusion = 256

# Treinamento
batch_size = 32
learning_rate = 0.001
optimizer = 'Adam'
loss_function = 'CrossEntropyLoss'
num_epochs = 100
early_stopping_patience = 15

# Regularização
dropout = 0.3
weight_decay = 1e-4
```

## 5. Estrutura de Arquivos

```
tcc-mba-deteccao-anomalias/
├── data/
│   ├── raw/                          # Dados brutos extraídos
│   │   └── 1st Experiment/
│   │       ├── Images_1stExperiment/
│   │       │   └── 1stExperiment_Daily_Images/
│   │       │       └── cva/sigrow/    # 15.336 imagens PNG
│   │       └── Excel files (XLSX)
│   └── processed/                    # Dados processados (opcional)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # MultimodalDataLoader
│   ├── pipeline.py                  # DataPipeline
│   ├── models.py                    # CNN, LSTM, Fusion
│   ├── metrics.py                   # Métricas e avaliação
│   └── alert_system.py              # Sistema de alertas
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.py
│   ├── 02_train_multimodal_model.py
│   ├── 03_evaluate_and_visualize.py
│   └── 04_alert_system_demo.py
│
├── models/                          # Modelos treinados (.pth)
├── results/                         # Resultados e gráficos
├── docs/                            # Documentação
├── requirements.txt
└── README.md
```

## 6. Workflow de Execução

1. **Preparação de Dados**
   ```bash
   python notebooks/01_exploratory_data_analysis.py
   ```

2. **Treinamento**
   ```bash
   python notebooks/02_train_multimodal_model.py
   ```

3. **Avaliação**
   ```bash
   python notebooks/03_evaluate_and_visualize.py
   ```

4. **Sistema de Alertas**
   ```bash
   python notebooks/04_alert_system_demo.py
   ```

## 7. Inovações Técnicas

### Fusão Multimodal
Diferentemente dos sistemas existentes que analisam imagens OU sensores isoladamente, este sistema:
- **Correlaciona síncrona** entre fluxo contínuo de sensores (causa) e características fenotípicas (efeito incipiente)
- **Identifica padrões combinados** que sinalizem perda de qualidade química
- **Permite alertas precoces** antes de danos irreversíveis

### Detecção de "Fenótipo Silencioso"
- Planta visualmente saudável mas metabolicamente comprometida
- Detectável via combinação de:
  - Micro-oscilações ambientais (sensores)
  - Sutis alterações de coloração/textura (imagens)
  - Mecanismo de atenção temporal (LSTM)

### Implementação Eficiente
- ResNet18 (leve) em vez de modelos pesados
- Attention mechanism reduz comprimento efetivo de sequência
- Fusão híbrida balanceia complexidade e performance

## 8. Referências Teóricas

- **Visão Computacional**: He et al. (2015) - ResNet
- **Séries Temporais**: Vaswani et al. (2017) - Attention is All You Need
- **Fusão Multimodal**: Baltrušaitis et al. (2018)
- **Estresse Abiótico**: Taiz et al. (2015) - Fisiologia Vegetal
- **IA na Agricultura**: Hughes & Salathe (2016) - PlantVillage Dataset

---

**Última atualização**: Abril 2026
**Status**: Em desenvolvimento
**Próximos passos**: Treinamento completo, validação e deployment
