# 📋 Política de Dados — Dados Reais do Experimento

**Data**: 17 de Maio de 2026
**Status**: Implementado

---

## 🔴 Scripts Descontinuados (Dados Sintéticos)

Estes scripts foram descontinuados e não devem ser utilizados:

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

## 🟢 Script Oficial (Dados Reais)

```bash
✅ notebooks/05_retrain_with_real_data.py
   - Origem: Experimento "1st Experiment"
   - Dataset: 107 imagens RGB REAIS + 180 labels REAIS + 13.825 sensores REAIS
   - Acurácia: REALISTA (~56%, não 100%)
   - Status: OFICIAL
```

---

## 📊 Dados Reais do Experimento

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

## 🎯 Garantias de Reprodutibilidade e Honestidade

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

## 📝 Critério de Validação

Durante o desenvolvimento, acurácias acima de 95% nas primeiras épocas devem ser investigadas como possível indicador de dados sintéticos ou separabilidade artificial. O uso exclusivo de dados reais (sem `np.random`) evita esse viés.

---

## ✅ Decisão Registrada

**Data:** 17 de Maio de 2026  
**Responsável:** Helen Paixão

A partir desta data, o projeto adota exclusivamente dados reais do experimento "1st Experiment". Scripts com componentes sintéticos (`np.random.randn()`) foram movidos para `notebooks/_archive/` e não devem ser utilizados para resultados de defesa.

---
