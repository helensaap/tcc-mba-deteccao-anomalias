# 🌱 IA Multimodal para Predição de Estresse Abiótico

**Uso de Inteligência Artificial Multimodal para Predição de Estresse Abiótico visando a Maximização de Bioativos em Cultivos Farmacêuticos Indoor**

## Visão Geral

Este projeto implementa um sistema de **Deep Learning multimodal** para detecção precoce de estresse abiótico em plantas cultivadas em ambientes controlados (indoor farming). O sistema funde dados de:

- 🖼️ **Imagens fenotípicas** (Câmera de profundidade RealSense D415)
- 📊 **Séries temporais de sensores IoT** (Temperatura, Umidade, CO₂, Radiação)
- 🎯 **Redes Neurais Convolucionais (CNN)** para visão computacional
- 🔄 **LSTM com Mecanismo de Atenção** para análise temporal
- 🔗 **Fusão Multimodal** (Early, Late ou Hybrid) para correlação de modalidades

## 📌 Problema de Pesquisa

Em cultivos de alto valor agregado (Cannabis medicinal, fitoterápicos), **micro-oscilações ambientais** causam:

- ❌ Síntese reduzida de bioativos (metabólitos secundários)
- ❌ Plantas visualmente saudáveis mas quimicamente pobres ("fenótipo silencioso")
- ❌ Falha em padronização exigida por agências regulatórias (ANVISA)
- ❌ Prejuízos econômicos severos

**Nossa solução**: Detectar essas anomalias **antes que ocorram danos irreversíveis** combinando sinais visuais sutis com padrões temporais dos sensores.

## 🎯 Objetivos

### Objetivo Geral
Desenvolver uma arquitetura de **IA Multimodal** baseada em Deep Learning para predição precoce de estresse abiótico em cultivos indoor, possibilitando alertas automáticos.

### Objetivos Específicos
1. ✅ Fusão computacional de dados heterogêneos (imagens + séries temporais)
2. ✅ Extração automática de features fenotípicas via CNN
3. ✅ Modelagem temporal com LSTM + Atenção
4. ✅ Detecção de padrões combinados que sinalizem perda de qualidade
5. ✅ Sistema integrado de alertas precoces
6. ✅ Validação com métricas consolidadas (Acurácia, Precisão, Recall, F1, AUC-ROC)

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone <seu-repo>
cd tcc-mba-deteccao-anomalias

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 2. Prepare os Dados

```bash
# O arquivo de dados será extraído automaticamente
# Ele deve estar em: /Users/helen.paixao/Downloads/1st Experiment.zip
# Ou você pode copiar manualmente para: ./data/raw/

# Execute a exploração inicial
python notebooks/01_exploratory_data_analysis.py
```

### 3. Treine o Modelo

```bash
python notebooks/02_train_multimodal_model.py
```

Isso irá:
- Carregar imagens e dados de sensores
- Criar dataloaders (treino/validação/teste)
- Treinar CNN + LSTM + Fusion conjuntamente
- Salvar melhor modelo em `./models/best_model.pt`

### 4. Avalie e Visualize Resultados

```bash
python notebooks/03_evaluate_and_visualize.py
```

### 5. Teste o Sistema de Alertas

```bash
python notebooks/04_alert_system_demo.py
```

## 📁 Estrutura do Projeto

```
tcc-mba-deteccao-anomalias/
│
├── data/
│   ├── raw/
│   │   └── 1st Experiment/           # 24.46 GB extraídos
│   │       ├── Images_1stExperiment/
│   │       │   └── 15,336 fotos PNG (600x800)
│   │       └── Excel files XLSX
│   └── processed/                     # (Opcional)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # MultimodalDataLoader
│   ├── pipeline.py                   # DataPipeline + Dataset
│   ├── models.py                     # CNN + LSTM + Fusion
│   ├── metrics.py                    # Métricas e Early Stopping
│   └── alert_system.py               # Sistema de alertas
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.py    # EDA
│   ├── 02_train_multimodal_model.py       # Treinamento
│   ├── 03_evaluate_and_visualize.py       # Avaliação
│   └── 04_alert_system_demo.py            # Demo de alertas
│
├── models/
│   ├── best_model.pt                 # Melhor modelo
│   └── checkpoint_epoch_*.pt         # Checkpoints
│
├── results/
│   ├── 01_sample_images.png
│   ├── 02_image_dimensions.png
│   ├── training_history.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── docs/
│   ├── ARCHITECTURE.md               # Arquitetura detalhada
│   └── README.md                     # Este arquivo
│
├── requirements.txt
└── README.md
```

## 🏗️ Arquitetura do Sistema

### Pipeline Geral

```
INPUT
  ├── Imagens (600×800 PNG)
  └── Sensores IoT (série temporal)
      ↓
PRÉ-PROCESSAMENTO
  ├── CNN (Resize 224×224 + Normalização)
  └── StandardScaler (Sensores)
      ↓
EXTRAÇÃO DE FEATURES
  ├── CNN ResNet18 → 256-dim visual features
  └── LSTM (2-layer) + Attention → 128-dim temporal features
      ↓
FUSÃO MULTIMODAL
  ├── Early: concat[256, 128]
  ├── Late: 256 + 128 (element-wise add)
  └── Hybrid: concat[256, 128, 256⊙128]
      ↓
CLASSIFICAÇÃO
  ├── Dense(384 → 256)
  ├── Dense(256 → 128)
  └── Dense(128 → 2) [Normal/Stress]
      ↓
SISTEMA DE ALERTAS
  ├── Normal (conf < 0.60)
  ├── Mild (0.60-0.75)
  ├── Moderate (0.75-0.90)
  └── Severe (>0.90)
```

### Componentes Chave

| Componente | Descrição | Arquivo |
|-----------|-----------|---------|
| **DataLoader** | Carrega imagens, XLSX, JSON | `src/data_loader.py` |
| **Pipeline** | Pré-processamento e DataLoaders | `src/pipeline.py` |
| **CNN** | ResNet18 para features visuais | `src/models.py` |
| **LSTM** | 2-layer com Attention temporal | `src/models.py` |
| **Fusion** | Early/Late/Hybrid | `src/models.py` |
| **Metrics** | Acurácia, F1, AUC-ROC, etc | `src/metrics.py` |
| **Alerts** | Sistema de detecção e alertas | `src/alert_system.py` |

## 📊 Dataset

| Atributo | Valor |
|----------|-------|
| **Planta** | Sigrow (leafy greens) |
| **Período** | Fevereiro - Março 2022 |
| **Imagens** | 15.336 fotos (600×800) |
| **Sensores** | Temperatura, Umidade, Radiação, CO₂ |
| **Ground Truth** | Classes A/B (Normal) vs C (Stress) |
| **Câmera** | RealSense D415 + metadados calibração |

## 🔧 Hiperparâmetros Padrão

```python
# Modelo
visual_feature_size = 256
temporal_feature_size = 128
num_sensor_vars = 4
sequence_length = 24
fusion_type = 'hybrid'

# Treinamento
batch_size = 32
learning_rate = 0.001
optimizer = Adam
loss = CrossEntropyLoss
num_epochs = 100

# Regularização
dropout = 0.3
weight_decay = 1e-4
early_stopping_patience = 15
```

## 📈 Métricas de Desempenho

O sistema calcula:

- **Acurácia**: % de predições corretas
- **Precisão**: % de alertas corretos (evita falsos positivos)
- **Recall**: % de casos de stress detectados (evita falsos negativos)
- **F1-Score**: Média harmônica Precisão-Recall
- **AUC-ROC**: Curva ROC para análise de trade-offs
- **Confusion Matrix**: Detalhamento de erros

### Exemplo de Saída
```
╔════════════════════════════════════════════════════════════════╗
║                    MÉTRICAS DE DESEMPENHO                     ║
╠════════════════════════════════════════════════════════════════╣
║ Acurácia:  0.8743
║ Precisão:  0.8621
║ Recall:    0.8965
║ F1-Score:  0.8791
║ AUC-ROC:   0.9234
║
║ Matriz de Confusão:
║ [[1243   45]
║  [ 38  674]]
╚════════════════════════════════════════════════════════════════╝
```

## 🚨 Sistema de Alertas

Quando uma predição é feita, o sistema:

1. **Detecta anomalias visuais**:
   - Redução de pigmentação verde
   - Textura foliar anômala
   - Sinais de murchamento

2. **Detecta anomalias temporais**:
   - Oscilações bruscas de temperatura
   - Instabilidade de umidade
   - Desvios em CO₂
   - Padrões irregulares

3. **Gera alertas graduados**:
   ```
   ╔════════════════════════════════════════════════════════════════╗
   ║                     ALERTA DE ESTRESSE                         ║
   ╠════════════════════════════════════════════════════════════════╣
   ║ Planta:            raspberry_001
   ║ Nível de Estresse: MODERATE
   ║ Confiança:         82.34%
   ║ Timestamp:         2026-04-20 20:45:30
   ╠════════════════════════════════════════════════════════════════╣
   ║ Indicadores Visuais:
   ║   • Redução anormal de pigmentação verde
   ║   • Sinais incipientes de murchamento
   ║
   ║ Indicadores Temporais:
   ║   • Oscilações bruscas de temperatura
   ║   • Instabilidade na umidade relativa
   ║
   ║ Recomendação: Intervenção necessária. Ajustar temperatura/
   ║   umidade/CO2. Aumentar frequência de irrigação.
   ║   Estabilizar sistema de controle climático.
   ╚════════════════════════════════════════════════════════════════╝
   ```

## 🔬 Inovações Técnicas

### 1. Fusão Multimodal Síncrona
Diferentemente dos sistemas atuais:
- ❌ Antigos: Analisam imagens OU sensores separadamente
- ✅ Nosso: Correlaciona síncrona entre ambas as modalidades

### 2. Detecção de "Fenótipo Silencioso"
- Identifica plantas visualmente saudáveis mas metabolicamente comprometidas
- Combina: micro-oscilações (sensores) + sutis alterações (imagens) + atenção temporal

### 3. Mecanismo de Atenção Temporal
- Reduz comprimento efetivo de sequência
- Foca em timesteps críticos
- Eficiência computacional vs. ResNet pesados

## 🔗 Referências

### Teóricas
- **ResNet**: He et al. (2015) - Deep Residual Learning
- **Attention**: Vaswani et al. (2017) - Attention is All You Need
- **Multimodal Fusion**: Baltrušaitis et al. (2018)
- **Estresse Abiótico**: Taiz et al. (2015) - Fisiologia Vegetal
- **IA na Agricultura**: Hughes & Salathe (2016) - PlantVillage

### Contexto Brasileiro
- AGÊNCIA BRASIL (2024) - Brasil importa 90% de IFAs
- EMBRAPA (2023) - Bioeconomia pode gerar US$ 284 bi/ano
- ANVISA - Regulação de produtos farmacêuticos

## 📝 Próximas Etapas

- [ ] Expandir dataset com múltiplas espécies
- [ ] Fine-tuning com backbone pré-treinados (ImageNet)
- [ ] Implementar Transformer completo (em vez de LSTM)
- [ ] Deploy em servidor IoT (Edge Computing)
- [ ] Dashboard em tempo real (WebApp + WebSocket)
- [ ] Validação em campo com produtores reais
- [ ] Publicação em conferência (SIBGRAPI, ANALITICA EXPO)

## 📧 Contato e Dúvidas

**Autora**: Helen Paixão
**Instituição**: MBA em Inteligência Artificial e Big Data
**Data**: Abril 2026

Para dúvidas sobre a implementação, consulte os notebooks e o arquivo `ARCHITECTURE.md`.

---

## 📜 Licença

Este projeto é para fins acadêmicos e de pesquisa.

## ⭐ Agradecimentos

- Ao dataset "1st Experiment" disponibilizado
- À comunidade de IA e agricultura de precisão
- À ANVISA e ao contexto de soberania farmacêutica brasileira

---

**Status**: ✅ Em desenvolvimento | Última atualização: Abril 2026
