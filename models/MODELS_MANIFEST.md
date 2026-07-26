# 🗃️ Manifest de Modelos Treinados

> ⚠️ **Os pesos `.pt` NÃO são versionados no Git** (vide `.gitignore`).
> São distribuídos via **GitHub Releases** (tag `v1.0`) e/ou **Hugging Face Hub**.

**Modelo oficial:** `best_model_semi_supervised.pt` — Fase 7, 64.71% acc / F1=0.40 / AUC=0.729.

---

## Como obter os pesos

### Opção A — GitHub Release
```bash
# Substitua <USER>/<REPO> pela URL do repo
curl -L -o models/best_model_semi_supervised.pt \
  https://github.com/<USER>/<REPO>/releases/download/v1.0/best_model_semi_supervised.pt
```

### Opção B — Hugging Face Hub
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="helenpaixao/tcc-stress-abiotico",
    filename="best_model_semi_supervised.pt",
    local_dir="models/",
)
```

---

## Modelos de Produção / Defesa

| Arquivo | Fase | Notebook gerador | Test Acc | F1 | AUC | Uso |
|---|---|---|---|---|---|---|
| `best_model_semi_supervised.pt` | **7 (oficial)** ⭐ | [07_semi_supervised_learning.py](../notebooks/07_semi_supervised_learning.py) | **64.71%** | 0.40 | 0.729 | **OFICIAL — DEFESA** |
| `best_model_advanced_real_data.pt` | 6 | [06_advanced_training_real_data.py](../notebooks/06_advanced_training_real_data.py) | 52.94% | 0.00 | 0.486 | Comparação histórica (Fase 6 baseline) |
| `best_model_semi_supervised_improved.pt` | 7-improved | [07_semi_supervised_improved.py](../notebooks/07_semi_supervised_improved.py) | 52.94% | 0.00 | 0.500 | Descartado (regressão) |
| `best_model_real_data.pt` | 5 | [05_retrain_with_real_data.py](../notebooks/05_retrain_with_real_data.py) | ~56% | — | — | Primeira integração de dados reais |
| `best_model.pt` | 2 | [02_train_multimodal_model.py](../notebooks/02_train_multimodal_model.py) | ~50% | — | — | Modelo inicial (dados sintéticos parciais) |

---

## Checkpoints intermediários

Movidos para `models/archive/` (gitignored). São snapshots de treino sem valor
para inferência; mantidos localmente apenas para auditoria. Podem ser deletados
sem perda científica.

---

## Como carregar o modelo oficial

```python
import torch
from src.models import create_multimodal_model

visual, temporal, fusion = create_multimodal_model(fusion_type='hybrid')
fusion.load_state_dict(
    torch.load('models/best_model_semi_supervised.pt', map_location='cpu')
)
fusion.eval()
```

---

## Total em disco

~225 MB nos 5 modelos públicos. Distribuir apenas
`best_model_semi_supervised.pt` (46 MB) cobre 100% do que [RESULTS.md](../docs/RESULTS.md)
e [MODEL_CARD.md](../MODEL_CARD.md) descrevem.
