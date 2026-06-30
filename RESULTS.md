# 📊 RESULTS — Fonte Única de Verdade

**Última atualização:** 2026-06-29
**Métricas extraídas diretamente de:** `results/*.json`

Este é o **único documento autorizado** a reportar métricas do projeto. Qualquer outra fonte (slides, posts, conversas) deve referenciar esta tabela.

---

## 🎯 Métricas Oficiais (Modelo de Produção)

| Métrica | Valor | Origem |
|---|---|---|
| **Test Accuracy** | **64.71%** | `results/08_evaluation_comparison.json` (chave `fase7`) |
| **F1-Score (stress)** | 0.40 | idem |
| **Recall (stress)** | 0.286 | idem |
| **Precision (stress)** | 0.667 | idem |
| **AUC-ROC** | 0.729 | idem |
| **Loss (test)** | 0.601 | idem |
| **Reprodutibilidade** | 100% (σ=0.0 em 5 runs) | `results/10_reproducibility_test.json` |

**Modelo oficial:** [models/best_model_semi_supervised.pt](models/best_model_semi_supervised.pt) (Fase 7, semi-supervised learning).
Detalhes do checkpoint em [MODEL_CARD.md](MODEL_CARD.md).

---

## 📐 Composição do Test Set

| Item | Valor |
|---|---|
| Amostras totais (test) | **17** |
| Amostras (train) | 74 |
| Amostras (val) | 16 |
| Classes | 2 (Normal=0, Stress=1) |
| Origem | "1st Experiment" — GroundTruth de 239 imagens RGB anotadas |

> ⚠️ **Honestidade científica:** com apenas 17 amostras de teste, cada predição correta vale ~5.9 pp.
> A diferença entre 52.94% e 64.71% corresponde a **2 amostras**. As conclusões devem ser tratadas como
> evidência preliminar, não como performance final de produção. Veja [AUDITORIA_CIENTIFICA.md](AUDITORIA_CIENTIFICA.md).

---

## 🔬 Histórico de Fases (Cronologia das Métricas)

| Fase | Modelo `.pt` | Test Acc | F1 | AUC | Status |
|---|---|---|---|---|---|
| **Fase 6** (supervised) | `best_model_advanced_real_data.pt` | 52.94% | **0.00** | 0.486 | ⚠️ Degenerado — prediz só "Normal" |
| **Fase 7** (semi-supervised) | `best_model_semi_supervised.pt` | **64.71%** ⭐ | 0.40 | 0.729 | ✅ **OFICIAL** |
| Fase 7 "improved" | `best_model_semi_supervised_improved.pt` | 52.94% | 0.00 | 0.500 | ❌ Voltou a degenerar |

**Por que a Fase 7 é a oficial e não as outras:**
1. Apenas a Fase 7 produziu um modelo com F1 > 0 (modelo realmente aprende a classe positiva).
2. AUC=0.729 > 0.5 confirma poder discriminativo real.
3. Reprodutível: σ=0.0 entre 5 runs com seeds diferentes.

**Por que Fase 6 e Fase 7 "improved" reportam 52.94% mas devem ser desconsideradas:**
Esse número corresponde à baseline trivial de predizer sempre a classe majoritária (Normal = 9/17 = 52.94%).
F1=0 e AUC≈0.5 confirmam que o modelo não aprendeu nada.

---

## 🎚️ Thresholds Validados por ROC Curve

Análise em `results/03_roc_recommendations.json` (28-Abr-2026):

| Método | Threshold | Uso |
|---|---|---|
| **Youden's Index** ⭐ | **0.4482** | Balanço TPR/FPR — recomendado para alertas gerais |
| F1-Score Max | 0.4156 | Quando recall é crítico (não perder casos de stress) |
| PR-Curve | 0.4156 | Idem F1-max |

> Nota: thresholds anteriores de 0.5213 / 0.4208 (citados em docs históricos) vinham de uma análise de
> 15 amostras feita antes da Fase 7. Os valores acima refletem a análise mais recente (29 amostras balanceadas).
> Para produção, usar **0.4482 (Youden)**.

Os 4 níveis de severidade do sistema de alertas ([src/alert_system.py](src/alert_system.py)) operam sobre esses thresholds:

```
NORMAL    < 0.4156
MILD      0.4156 – 0.4482
MODERATE  0.4482 – 0.70
SEVERE    ≥ 0.70
```

---

## 📈 Curvas de Treinamento

- Fase 6: `results/06_training_advanced_curves.png` · histórico em `06_training_history_advanced.json` (32 epochs, best_epoch=12, val_acc pico 75%)
- Fase 7: `results/07_semi_supervised_training.png` · histórico em `07_semi_supervised_history.json`
- Fase 7 improved: `results/07_semi_supervised_improved_training.png` · histórico em `07_semi_supervised_improved_history.json`

**Observação sobre val_acc=75% na Fase 6:** o pico foi no epoch 12 mas o modelo final (epoch 32) generalizou para 52.94% no test set. Isso indica overfitting no val set pequeno (16 amostras). Por isso a Fase 7 foi necessária.

---

## ✅ Auditoria de Legitimidade

`results/09_auditoria_legitimidade.json` confirma:

```json
{
  "dados_reais": true,
  "test_set_legítimo": true,
  "sem_data_leakage": true,
  "melhoria_realistica": true,
  "fase7_diferente_fase6": true
}
```

Procedimento de auditoria detalhado em [AUDITORIA_CIENTIFICA.md](AUDITORIA_CIENTIFICA.md).

---

## 🚫 Métricas que NÃO devem ser citadas

Estas métricas aparecem em documentos antigos e **estão erradas ou são enganosas**:

| Métrica citada | Onde aparece (histórico) | Por que descartar |
|---|---|---|
| 98% accuracy | FINAL_SUMMARY.md (deletado) | Transfer learning em dataset trivial — overfitting |
| 100% accuracy (epoch 6) | logs de notebooks `.deprecated` | Dados sintéticos (np.random) — descontinuados |
| ~50% accuracy | README antigo | Vago; o número exato é 52.94% (Fase 6 degenerada) |

Veja [CHANGELOG_FASES.md](CHANGELOG_FASES.md) para a cronologia completa.

---

## 🔁 Como Reproduzir

```bash
# Avaliação do modelo oficial
python notebooks/08_evaluate_semi_supervised.py

# Teste de reprodutibilidade (5 runs)
python notebooks/10_test_reproducibility.py

# Auditoria de legitimidade
python notebooks/09_audit_accuracy_improvement.py
```

Todos esses scripts leem os mesmos `.pt` e geram os mesmos JSONs em `results/`.
