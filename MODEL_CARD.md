# 🧠 Model Card — IA Multimodal de Detecção de Estresse Abiótico

**Versão:** v1.0 (Fase 7 — Semi-Supervised)
**Data de release:** 2026-05-17
**Autora:** Helen Paixão (MBA IA e Big Data)

---

## 1. Descrição Geral

Modelo de Deep Learning multimodal que combina imagens fenotípicas de plantas e séries temporais de sensores ambientais para classificar plantas em **Normal (0)** ou **Stress (1)**.

- **Tipo:** Classificação binária multimodal
- **Modalidades:** Imagem RGB (224×224) + série temporal (24 timesteps × 4 sensores)
- **Framework:** PyTorch 2.0+
- **Tamanho:** ~46 MB (`.pt` serializado)

---

## 2. Arquitetura

| Bloco | Componente | Output dim |
|---|---|---|
| Visual encoder | ResNet18 fine-tuned (ImageNet pre-trained, últimas 16 camadas descongeladas) | 256 |
| Temporal encoder | LSTM bidirecional 2 layers + attention sobre timesteps | 128 |
| Fusion | Hybrid: `concat([visual, temporal, visual ⊙ temporal])` | 640 |
| Classifier | Dense 640 → 384 → 256 → 128 → 2 (softmax) | 2 |

Detalhes em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) e [src/models.py](src/models.py).

---

## 3. Treinamento

| Parâmetro | Valor |
|---|---|
| Estratégia | Semi-supervised (pseudo-labeling com threshold 0.70, λ_unlabeled=2.0) |
| Otimizador | Adam (lr=5e-4, weight_decay=1e-4) |
| Loss | CrossEntropyLoss (supervised) + λ·MSE(pseudo-labels) |
| Batch size | 8 |
| Epochs | 100 (early stopping patience=20) |
| Best epoch | (registrado em `results/07_semi_supervised_history.json`) |
| Hardware | CPU/GPU compatível (treino realizado em CPU local) |

**Notebook gerador:** [notebooks/07_semi_supervised_learning.py](notebooks/07_semi_supervised_learning.py)

---

## 4. Dataset

| Item | Valor |
|---|---|
| Origem | "1st Experiment" (Fev–Mar 2022) |
| Imagens labeled | 107 RGB (RealSense D415) |
| Imagens unlabeled (pseudo) | 15.229 |
| Sensores | Tair, Rhair, CO2air, PARin (4 vars × 24 timesteps, ciclo circadiano) |
| Classes | A/B → Normal (0); C → Stress (1) |
| Splits | 74 train · 16 val · 17 test |

Política de dados em [POLITICA_DADOS.md](docs/POLITICA_DADOS.md).

---

## 5. Métricas (Test Set, n=17)

| Métrica | Valor |
|---|---|
| Accuracy | **64.71%** |
| Precision | 0.667 |
| Recall | 0.286 |
| F1-Score | 0.400 |
| AUC-ROC | 0.729 |
| Loss | 0.601 |

Fonte: `results/08_evaluation_comparison.json`. Análise completa em [RESULTS.md](docs/RESULTS.md).

---

## 6. Threshold de Decisão (Produção)

**Recomendado:** `0.4482` (Youden's Index, validado por ROC curve em 29 amostras balanceadas).

Níveis de severidade do alerta (`src/alert_system.py`):
- `< 0.4156` → **NORMAL**
- `0.4156 – 0.4482` → **MILD**
- `0.4482 – 0.70` → **MODERATE**
- `≥ 0.70` → **SEVERE**

---

## 7. Casos de Uso Pretendidos

✅ **Adequado para:**
- Prova de conceito acadêmica em IA multimodal aplicada a agricultura de precisão
- Pesquisa em detecção de "fenótipo silencioso" (planta visualmente saudável, quimicamente comprometida)
- Demonstração de fusão CNN + LSTM-Attention em ambiente real

❌ **NÃO adequado para:**
- Decisões agronômicas em produção sem validação adicional
- Generalização para outras espécies sem retreino
- Substituição de análise química (HPLC, SPAD, etc.)
- Operação sem revisão humana

---

## 8. Limitações Conhecidas

1. **Test set pequeno (n=17):** intervalo de confiança amplo. Cada amostra vale ~5.9 pp.
2. **Domínio restrito:** treinado em *leafy greens* (sigrow) — não validado em Cannabis ou fitoterápicos.
3. **Recall baixo (0.286):** o modelo detecta apenas ~29% dos casos reais de stress. Para uso real, o threshold precisaria ser reduzido aumentando falsos positivos.
4. **Dataset desbalanceado labelado:** 107 labels para 15.336 imagens (0.7% supervisão).
5. **Validação ex situ:** sem teste em estufa operacional ainda.

Roadmap de mitigação em [ANALISE_PROFUNDA.md](docs/ANALISE_PROFUNDA.md).

---

## 9. Considerações Éticas e Científicas

- **Honestidade:** este card declara o desempenho real do modelo, sem maquiar números. Veja [AUDITORIA_CIENTIFICA.md](docs/AUDITORIA_CIENTIFICA.md).
- **Sem dados sintéticos:** modelo treinado 100% com dados reais. Notebooks que usavam `np.random` foram descontinuados (`.deprecated`).
- **Reprodutível:** σ=0.0 entre 5 runs (`results/10_reproducibility_test.json`).

---

## 10. Como Carregar

```python
import torch
from src.models import create_multimodal_model

visual, temporal, fusion = create_multimodal_model(fusion_type='hybrid')
fusion.load_state_dict(torch.load('models/best_model_semi_supervised.pt', map_location='cpu'))
fusion.eval()
```

Para predição passo a passo, ver [app.py](app.py) (função `predict_real`).

---

## 11. Manifest de Modelos Disponíveis

Todos os checkpoints `.pt` em [models/](models/) estão catalogados em [models/MODELS_MANIFEST.md](models/MODELS_MANIFEST.md).

---

## 12. Citação

```
Paixão, H. (2026). IA Multimodal para Predição de Estresse Abiótico em
Cultivos Farmacêuticos Indoor. Trabalho de Conclusão de Curso, MBA em
Inteligência Artificial e Big Data.
```
