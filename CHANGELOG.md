# 📜 Changelog das Fases — Cronologia Técnica

Linha do tempo única e auditável do projeto. Substitui: `PROGRESS_SUMMARY.md`, `PROJECT_SUMMARY.md`, `TRAINING_LOG.md`, `TRAINING_PROGRESS_REPORT.md`, `TRAINING_SUMMARY.txt`, `FASE_7_STATUS.txt`, `FASE_7_EXPLAINED.md`, `SUMMARY_FASE_7.md`, `INDEX_DOCUMENTOS_ANALISE.md`.

---

## Fase 1 — Setup & Exploração (Abr/2026)

- Estrutura inicial do repositório (`src/`, `notebooks/`, `data/`, `models/`)
- Notebook [01_exploratory_data_analysis.py](notebooks/01_exploratory_data_analysis.py): EDA do dataset, distribuição de imagens
- Dependências em [requirements.txt](requirements.txt)

**Saída:** dataset mapeado (15.336 imagens · sensores Tair/Rhair/CO2/PAR).

---

## Fase 2 — Pipeline Multimodal Base (Abr/2026)

- [src/models.py](src/models.py): CNN-residual + LSTM+Attention + Fusion Hybrid
- [src/data_loader.py](src/data_loader.py) e [src/pipeline.py](src/pipeline.py): Dataset PyTorch
- [src/metrics.py](src/metrics.py): Accuracy, Precision, Recall, F1, AUC, EarlyStopping
- Notebook [02_train_multimodal_model.py](notebooks/02_train_multimodal_model.py): primeiro treino

**Saída:** modelo treinado com dados sintéticos. ⚠️ Acurácia artificial (100% epoch 6 → red flag).

---

## Fase 3 — Avaliação + ROC (Abr/2026)

- Notebook [03_evaluate_and_visualize.py](notebooks/03_evaluate_and_visualize.py)
- Análise ROC com 29 amostras → thresholds validados (`results/03_roc_recommendations.json`)
- Youden's Index = 0.4482 substituiu thresholds inventados (0.60/0.75/0.90)

**Saída:** [SCIENTIFIC_JUSTIFICATION.md](docs/SCIENTIFIC_JUSTIFICATION.md) seção 6 documentando origem de cada threshold.

---

## Fase 4 — Integração Dados Reais (Mai/2026)

- [src/real_data_loader.py](src/real_data_loader.py): parser de `GreenhouseCrop.xlsx` (labels) e `GreenhouseClimate.xlsx` (13.825 sensores)
- Notebook [02b_train_with_real_data.py](notebooks/02b_train_with_real_data.py)
- **Decisão crítica:** descontinuar notebooks com `np.random.randn()` — formalizado em [POLITICA_DADOS.md](docs/POLITICA_DADOS.md)
- Arquivos movidos para `.deprecated`:
  - `notebooks/04_retrain_improved.py.deprecated`
  - `notebooks/04_retrain_transfer_learning.py.deprecated`

**Saída:** dataset real consolidado — 107 labeled + 15.229 unlabeled.

---

## Fase 5 — Retreino com Dados Reais (Mai/2026)

- Notebook [05_retrain_with_real_data.py](notebooks/05_retrain_with_real_data.py)
- Curvas em `results/05_training_curves_real_data.png`, métricas em `05_training_history_real_data.json`
- Acurácia caiu para ~56% (esperado: realismo, sem separabilidade artificial)

**Lição:** acurácias altas em fases anteriores eram artefato dos dados sintéticos.

---

## Fase 6 — Treinamento Avançado Supervised (17-Mai-2026)

- Notebook [06_advanced_training_real_data.py](notebooks/06_advanced_training_real_data.py)
- Transfer learning com ResNet18 pretreinado (ImageNet)
- Data augmentation agressivo, scheduler dinâmico, early stopping patience=20
- 32 epochs · best_epoch=12 · val_acc pico = 75%

**Resultado:** test acc **52.94%**, F1=**0.00**, AUC=0.486 → ⚠️ **modelo degenerou** (prediz sempre Normal).
**Diagnóstico:** apenas 107 imagens labeled em arquitetura grande demais. Necessário aproveitar as 15.229 unlabeled.

Modelo: [models/best_model_advanced_real_data.pt](models/best_model_advanced_real_data.pt)

---

## Fase 7 — Semi-Supervised Learning (17-Mai-2026) ⭐

**Hipótese:** se 99.3% do dataset (15.229 imagens) está sem label, pseudo-labeling pode recuperar essa informação.

- Notebook [07_semi_supervised_learning.py](notebooks/07_semi_supervised_learning.py)
- Pseudo-label threshold inicial: 0.85
- λ_unlabeled: 1.0
- 80 epochs

**Resultado:** test acc **64.71%**, F1=**0.40**, AUC=**0.729** → ✅ **modelo aprende a classe positiva**.

Modelo oficial: [models/best_model_semi_supervised.pt](models/best_model_semi_supervised.pt)

### Tentativa de melhoria ("Fase 7 improved")
- Notebook [07_semi_supervised_improved.py](notebooks/07_semi_supervised_improved.py)
- Threshold reduzido para 0.70 · λ aumentado para 2.0 · augmentation 80/15/5

**Resultado:** test acc 52.94%, F1=0.00 → ❌ regressão (mais pseudo-labels = mais ruído).
Modelo: [models/best_model_semi_supervised_improved.pt](models/best_model_semi_supervised_improved.pt) (descartado).

**Conclusão:** Fase 7 original (threshold 0.85) é o ponto ótimo.

---

## Fase Auditoria — Validação de Honestidade (17-Mai-2026)

Realizada após questionamento de honestidade científica.

| Verificação | Notebook | Resultado |
|---|---|---|
| Auditoria de melhoria | [09_audit_accuracy_improvement.py](notebooks/09_audit_accuracy_improvement.py) | ✅ 5/5 passes |
| Reprodutibilidade | [10_test_reproducibility.py](notebooks/10_test_reproducibility.py) | ✅ σ=0.0 em 5 runs |
| Completude do dataset | [11_verify_all_images_loaded.py](notebooks/11_verify_all_images_loaded.py) | ✅ 15.336 imagens carregadas |
| Reavaliação Fase 7 original | [12_evaluate_original_phase7.py](notebooks/12_evaluate_original_phase7.py) | ✅ 64.71% reproduzido |

Detalhes em [AUDITORIA_CIENTIFICA.md](docs/AUDITORIA_CIENTIFICA.md).

---

## Fase Documentação — Consolidação (29-Jun-2026)

- 28 arquivos `.md/.txt` na raiz → 9 arquivos enxutos
- Métricas unificadas em [RESULTS.md](docs/RESULTS.md) (fonte única de verdade)
- Card de modelo formal em [MODEL_CARD.md](MODEL_CARD.md)
- Scripts soltos movidos para `scripts/`
- Notebooks `.deprecated` movidos para `notebooks/_archive/`

---

## Estado Atual (29-Jun-2026)

| Dimensão | Status |
|---|---|
| Código (`src/`) | ✅ Sólido, com 3 arquivos de teste cobrindo metrics/models/alerts |
| Modelo oficial | ✅ `best_model_semi_supervised.pt` — 64.71% test acc, AUC 0.729 |
| Frontend Streamlit | ✅ Funcional ([app.py](app.py)) |
| Documentação | ✅ Consolidada — 9 arquivos enxutos |
| Reprodutibilidade | ✅ σ=0.0 verificada |
| Defesa acadêmica | ✅ Material pronto em [METODOLOGIA.md](docs/METODOLOGIA.md) |

---

## Próximas Fases Possíveis (Backlog)

Não obrigatório para defesa; veja [ANALISE_PROFUNDA.md](docs/ANALISE_PROFUNDA.md):

- Coleta de mais labels reais (de 107 → 500+)
- Validação em estufa operacional (Cannabis medicinal, fitoterápicos)
- Calibração de thresholds visuais com SPAD meter / espectrofotometria
- Substituir LSTM por Transformer (overkill mas publicável)
- Deploy edge (Jetson Nano + câmera IoT)
