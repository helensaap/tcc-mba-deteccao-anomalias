# 📋 POLÍTICA DE DADOS: APENAS DADOS REAIS

**Data**: 17 de Maio de 2026
**Decisão**: APENAS DADOS REAIS DO EXPERIMENTO
**Status**: ✅ IMPLEMENTADO

---

## 🔴 SCRIPTS DESCONTINUADOS (Dados Sintéticos)

Estes scripts foram **DESCONTINUADOS** e não devem ser usados:

```bash
❌ notebooks/04_retrain_improved.py
   - Motivo: Usa np.random.randn() (ruído gaussiano puro)
   - Problema: Acurácia artificial (100% em Epoch 6)
   - Status: DEPRECATED

❌ notebooks/04_retrain_transfer_learning.py
   - Motivo: Similar, dados sintéticos
   - Status: DEPRECATED
```

---

## 🟢 SCRIPT OFICIAL (Dados Reais)

```bash
✅ notebooks/05_retrain_with_real_data.py
   - Origem: Experimento "1st Experiment"
   - Dataset: 107 imagens RGB REAIS + 180 labels REAIS + 13.825 sensores REAIS
   - Acurácia: REALISTA (~56%, não 100%)
   - Status: OFICIAL
```

---

## 📊 DADOS REAIS DO EXPERIMENTO

### Fonte
```
data/raw/1st Experiment/
├── Images_1stExperiment/
│   ├── 1stExperiment_Ground_Truth/
│   │   └── GroundTruth_All_239_Images.json (metadados)
│   └── 1stExperiment_RGB_Images/ (107 imagens)
├── TimeSeries_1stExperiment/
│   ├── monday-lettuce/
│   ├── cva/
│   ├── koala/
│   ├── veggie-might/
│   └── reference/
│       ├── GreenhouseCrop.xlsx (labels A/B/C)
│       ├── GreenhouseClimate.xlsx (13.825 sensores)
│       └── GreenhouseControls.xlsx
```

### Características
- **107 imagens RGB** do câmera RealSense D415
- **180 labels verdadeiros** (Classes A=Normal, B=Normal, C=Stress)
  - Classe A: 76 plantas normais
  - Classe B: 17 plantas normais
  - Classe C: 87 plantas com stress (perda qualidade)
- **13.825 registros de sensores** por crop
  - Temperatura ambiente (Tair)
  - Umidade relativa (Rhair)
  - Concentração CO₂ (CO2air)
  - Radiação PAR (PARin)

---

## 🎯 GARANTIAS

### Reprodutibilidade
- ✅ Dados são públicos e imutáveis
- ✅ Labels vêm de experimento controlado
- ✅ Sensores são medidos, não simulados
- ✅ Totalmente rastreável

### Honestidade Científica
- ✅ Sem ruído artificial
- ✅ Sem separabilidade trivial
- ✅ Acurácia realista (não 100%)
- ✅ Resultados validáveis em campo

### Pronto para Defesa
- ✅ Dados reais
- ✅ Base científica sólida
- ✅ Análise crítica madura
- ✅ Limitações bem documentadas

---

## 📝 REGRA DE OURO

```
🎯 REGRA FUNDAMENTAL:

SE ACURÁCIA > 95% EM PRIMEIRAS EPOCHS
└─ INVESTIGAR IMEDIATAMENTE
   ├─ Dados são realmente complexos?
   ├─ Separabilidade é artificial?
   ├─ Há overfitting?
   └─ Usar DADOS REAIS ao invés de sintéticos
```

---

## ✅ CONFIRMAÇÃO

```
DECISÃO TOMADA: 17 de Maio de 2026
APROVADO POR: Helen Paixão
IMPLEMENTADO: Sim ✅

De agora em diante:
✅ APENAS dados reais
✅ NUNCA dados sintéticos
✅ SEMPRE validável em campo
```

---

**Próximas Execuções**: Usar SEMPRE `notebooks/05_retrain_with_real_data.py`
