# 🌱 IA Multimodal para Predição de Estresse Abiótico

**TCC MBA — Inteligência Artificial e Big Data**
**Uso de IA Multimodal para Predição de Estresse Abiótico visando a Maximização de Bioativos em Cultivos Farmacêuticos Indoor**

[![status](https://img.shields.io/badge/status-defesa--ready-success)]()
[![test_acc](https://img.shields.io/badge/test_acc-64.71%25-blue)]()
[![reproducible](https://img.shields.io/badge/reproducible-σ%3D0.0-brightgreen)]()

---

## 🎯 Em uma frase

Sistema de Deep Learning **multimodal** que combina **imagens fenotípicas** (CNN) e **séries temporais de sensores IoT** (LSTM+Attention) para detectar **estresse abiótico precoce** em plantas cultivadas indoor — o chamado **"fenótipo silencioso"** (planta visualmente saudável, quimicamente comprometida).

---

## ⚡ Quick Start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Frontend (predição em tempo real)
streamlit run app.py
# → http://localhost:8501

# 3. Reproduzir o treino oficial (Fase 7, semi-supervised)
python notebooks/07_semi_supervised_learning.py

# 4. Avaliar
python notebooks/08_evaluate_semi_supervised.py
```

---

## 📊 Resultados Oficiais

| Métrica | Valor |
|---|---|
| **Test Accuracy** | 64.71% |
| **F1-Score** | 0.40 |
| **AUC-ROC** | 0.729 |
| **Reprodutibilidade** | σ=0.0 em 5 runs |

**Modelo oficial:** `models/best_model_semi_supervised.pt` (Fase 7).
Fonte única e detalhada em **[RESULTS.md](RESULTS.md)**. Card formal em **[MODEL_CARD.md](MODEL_CARD.md)**.

> Test set tem n=17. Cada amostra vale ~5.9 pp. Resultado é prova de conceito acadêmica, **não produção**.
> Limitações documentadas honestamente em [AUDITORIA_CIENTIFICA.md](AUDITORIA_CIENTIFICA.md).

---

## 🏗️ Arquitetura

```
Imagem RGB (224×224)          Sensores (24 × 4)
        │                              │
   ResNet18 fine-tuned            LSTM 2-layer
   (256-dim)                      + Attention (128-dim)
        │                              │
        └──────────┬───────────────────┘
                   ▼
         Fusion Hybrid: [v, t, v⊙t]  (640-dim)
                   ▼
         Classifier → Softmax → P(Normal) | P(Stress)
                   ▼
              Sistema de Alertas (4 níveis)
```

Detalhes em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 📁 Estrutura do Projeto

```
.
├── src/                       # Código fonte
│   ├── models.py              # CNN + LSTM-Attention + Fusion
│   ├── real_data_loader.py    # Parser de dados reais (XLSX/JSON)
│   ├── pipeline.py            # Dataset PyTorch
│   ├── metrics.py             # Acc, P, R, F1, AUC, EarlyStop
│   └── alert_system.py        # 4 níveis de severidade
├── notebooks/                 # Pipeline 01→12 (EDA → train → eval)
│   └── _archive/              # Notebooks descontinuados
├── models/                    # Checkpoints .pt (manifest em MODELS_MANIFEST.md)
├── data/raw/1st Experiment/   # Dataset original (15.336 imagens + sensores)
├── results/                   # Métricas .json + gráficos .png
├── tests/                     # pytest (metrics, models, alert_system)
├── scripts/                   # Scripts auxiliares (monitor_epochs etc.)
├── app.py                     # Frontend Streamlit
└── docs/                      # Documentação técnica
```

---

## 📚 Documentação

| Documento | Quando ler |
|---|---|
| **[METODOLOGIA.md](METODOLOGIA.md)** | Apresentar o trabalho (banca, slides) |
| **[RESULTS.md](RESULTS.md)** | Citar métricas — fonte única de verdade |
| **[MODEL_CARD.md](MODEL_CARD.md)** | Detalhes do modelo oficial |
| **[ANALISE_PROFUNDA.md](ANALISE_PROFUNDA.md)** | Diagnóstico técnico + roadmap futuro |
| **[AUDITORIA_CIENTIFICA.md](AUDITORIA_CIENTIFICA.md)** | Verificações de honestidade científica |
| **[CHANGELOG_FASES.md](CHANGELOG_FASES.md)** | Cronologia das 7 fases |
| **[POLITICA_DADOS.md](POLITICA_DADOS.md)** | Decisão de não usar dados sintéticos |
| **[SCIENTIFIC_JUSTIFICATION.md](SCIENTIFIC_JUSTIFICATION.md)** | Origem científica dos parâmetros |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitetura detalhada |
| [docs/FRONTEND.md](docs/FRONTEND.md) | Guia do Streamlit |
| [models/MODELS_MANIFEST.md](models/MODELS_MANIFEST.md) | Catálogo de `.pt` |

---

## 📦 Dataset

| Atributo | Valor |
|---|---|
| Origem | "1st Experiment" (Fev–Mar 2022) |
| Imagens | 15.336 RGB (RealSense D415) |
| Sensores | Tair, Rhair, CO2air, PARin (24 timesteps) |
| Ground truth | 239 imagens labeled (A/B=Normal, C=Stress) |
| Split | 74 train · 16 val · 17 test (seed=42) |

Política em [POLITICA_DADOS.md](POLITICA_DADOS.md). **Apenas dados reais — sem componentes sintéticos.**

---

## ✅ Status (29-Jun-2026)

- [x] Pipeline multimodal CNN + LSTM-Attention + Fusion
- [x] Treinamento semi-supervised com 15.229 imagens unlabeled
- [x] Métricas validadas e reprodutíveis (σ=0.0)
- [x] Threshold de alerta validado por ROC curve (Youden=0.4482)
- [x] Frontend Streamlit funcional
- [x] Auditoria de honestidade científica completa
- [x] Documentação consolidada (28 arquivos → 9 enxutos)

---

## 🚧 Limitações Honestas

- Test set n=17 (cada amostra ~5.9 pp)
- Recall=0.286 — modelo conservador, perde casos reais de stress
- Domínio restrito (leafy greens, não Cannabis)
- 107 labels / 15.336 imagens (0.7% supervisão)

Veja [MODEL_CARD.md §8](MODEL_CARD.md) e roadmap em [ANALISE_PROFUNDA.md §5](ANALISE_PROFUNDA.md).

---

## 👤 Autora & Licença

**Helen Paixão** · MBA IA e Big Data · 2026
Projeto acadêmico — fins de pesquisa e ensino.

---

## 📖 Referências Núcleo

He et al. (2015) · Vaswani et al. (2017) · Baltrušaitis et al. (2018) · Taiz & Zeiger (2015) · Youden (1950) · Lee (2013). Lista completa em [METODOLOGIA.md](METODOLOGIA.md).
