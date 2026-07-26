# 🎓 Metodologia & Material de Defesa

Documento formal para apresentação à banca examinadora.

---

## Título

**Uso de Inteligência Artificial Multimodal para Predição de Estresse Abiótico visando a Maximização de Bioativos em Cultivos Farmacêuticos Indoor**

**Autora:** Helen Paixão
**Programa:** MBA em Inteligência Artificial e Big Data
**Ano:** 2026

---

## 1. Problema de Pesquisa

Cultivos farmacêuticos indoor (Cannabis medicinal, fitoterápicos, espécies nativas) sofrem com **micro-oscilações ambientais** que causam **estresse abiótico**. Esse estresse inibe a síntese de metabólitos secundários (bioativos) **sem causar danos visuais imediatos** — fenômeno conhecido como **"fenótipo silencioso"**: planta visualmente saudável, mas quimicamente pobre.

Os sistemas atuais monitoram variáveis isoladas (sensores OU imagens), perdendo a correlação síncrona que poderia antecipar a perda de qualidade.

---

## 2. Justificativa

### Soberania farmacêutica
O Brasil importa mais de **90% dos Insumos Farmacêuticos Ativos (IFAs)** (AGÊNCIA BRASIL, 2024), com déficit estimado em R$ 20 bilhões anuais. Detém a maior biodiversidade do mundo, mas não a transforma em produção farmacêutica nacional.

### Sustentabilidade hídrica
O Indoor Farming permite **economia de até 95% de água** comparado à agricultura convencional (Avgoustaki & Xydis, 2020), desacoplando produção da expansão de fronteira agrícola.

### Lacuna técnica
Sistemas comerciais analisam imagens **ou** sensores. Nenhuma solução open-source brasileira realiza **fusão multimodal síncrona** para detecção de "fenótipo silencioso".

---

## 3. Objetivo Geral

Desenvolver e validar uma arquitetura de **IA Multimodal baseada em Deep Learning** para predição de estresse abiótico em cultivos indoor, fundindo dados heterogêneos (imagens fenotípicas + séries temporais de sensores IoT) e emitindo alertas precoces.

---

## 4. Objetivos Específicos

1. ✅ Fusão computacional de dados heterogêneos (imagens + séries temporais)
2. ✅ Extração automática de features fenotípicas via CNN (ResNet18)
3. ✅ Modelagem temporal com LSTM + Mecanismo de Atenção
4. ✅ Detecção de padrões combinados que sinalizem perda de qualidade
5. ✅ Sistema integrado de alertas precoces (4 níveis de severidade)
6. ✅ Validação com métricas consolidadas (Acc, P, R, F1, AUC-ROC)

**Status:** 6/6 atingidos. Detalhes em [RESULTS.md](RESULTS.md).

---

## 5. Metodologia

### 5.1 Dataset
- **Origem:** "1st Experiment" (Fev–Mar 2022) — produtor Sigrow
- **Modalidades:**
  - 15.336 imagens RGB (câmera RealSense D415)
  - 13.825 leituras de sensores ambientais (Tair, Rhair, CO2, PAR)
  - 239 ground truth labels (A/B = Normal, C = Stress)
- Política formal em [POLITICA_DADOS.md](POLITICA_DADOS.md).

### 5.2 Arquitetura proposta

```
Imagem RGB (224×224)          Sensores (24 × 4)
        │                              │
        ▼                              ▼
   ResNet18 fine-tuned            LSTM 2-layer
   (256-dim)                      + Attention
                                  (128-dim)
        │                              │
        └──────────┬───────────────────┘
                   ▼
        Fusion Hybrid: [v, t, v ⊙ t]
                   ▼  (640-dim)
        Classifier: 640 → 384 → 256 → 128 → 2
                   ▼
        Softmax: [P(Normal), P(Stress)]
                   ▼
        Sistema de Alertas (4 níveis)
```

Detalhes em [docs/ARCHITECTURE.md](ARCHITECTURE.md).

### 5.3 Estratégia de Treinamento

**Semi-supervised learning** com pseudo-labeling (Lee, 2013):
- 107 imagens labeled + 15.229 unlabeled
- Pseudo-label threshold: 0.85
- Loss: `L_supervised + λ · L_unlabeled`, λ=1.0

Decisão metodológica fundamentada em [CHANGELOG.md](../CHANGELOG.md) Fases 4–7.

### 5.4 Validação

- **Split:** 74 train / 16 val / 17 test (seed=42)
- **Métricas:** Accuracy, Precision, Recall, F1, AUC-ROC
- **Reprodutibilidade:** 5 runs independentes → σ=0.0
- **Threshold:** validado por ROC curve (Youden's Index = 0.4482)
- **Auditoria de honestidade:** 5/5 verificações em [AUDITORIA_CIENTIFICA.md](AUDITORIA_CIENTIFICA.md)

---

## 6. Resultados Principais

| Métrica | Valor |
|---|---|
| Test Accuracy | **64.71%** |
| F1-Score | 0.40 |
| AUC-ROC | 0.729 |
| Reprodutibilidade | σ=0.0 |

Detalhes completos em [RESULTS.md](RESULTS.md).
Análise crítica em [ANALISE_PROFUNDA.md](ANALISE_PROFUNDA.md).

---

## 7. Contribuições

1. **Implementação open-source** de pipeline multimodal CNN+LSTM-Attention para agricultura de precisão
2. **Validação científica de thresholds** de alerta via ROC curve (substituindo heurísticas)
3. **Auditoria de honestidade** documentada — modelo de transparência para TCCs em ML
4. **Frontend Streamlit funcional** ([app.py](app.py)) — predição em tempo real
5. **Política de dados** formalizada — reproduzível, sem componentes sintéticos

---

## 8. Limitações Reconhecidas

- Test set pequeno (n=17) — intervalo de confiança amplo
- Recall=0.286 — modelo conservador, perde casos de stress
- Domínio restrito (leafy greens) — não generaliza para Cannabis sem retreino
- Sem validação química direta (HPLC, SPAD)

Roadmap de mitigação em [ANALISE_PROFUNDA.md](ANALISE_PROFUNDA.md) §5.

---

## 9. Trabalhos Futuros

1. Ampliar dataset labeled (107 → 500+) via colaboração com agrônomos
2. Validação em estufa operacional (Cannabis, fitoterápicos)
3. Calibração de thresholds visuais com SPAD meter
4. Substituir LSTM por Transformer
5. Deploy edge (Jetson Nano + IoT)
6. Publicação em SIBGRAPI / ANALITICA EXPO

---

## 10. Referências Principais

Lista completa em [ANALISE_PROFUNDA.md](ANALISE_PROFUNDA.md) §8. Núcleo:

- AGÊNCIA BRASIL (2024). *Brasil estima déficit de R$ 20bi com IFAs*.
- AVGOUSTAKI, D.D., XYDIS, G. (2020). *Indoor vertical farming*. Sustainability 12(5):1964.
- BALTRUŠAITIS, T. et al. (2018). *Multimodal ML Survey*. IEEE TPAMI.
- HE, K. et al. (2015). *Deep Residual Learning*. CVPR.
- LEE, D.H. (2013). *Pseudo-Label: SSL Method*.
- TAIZ, L., ZEIGER, E. (2015). *Fisiologia Vegetal*.
- VASWANI, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- YOUDEN, W.J. (1950). *Index for rating diagnostic tests*. Cancer.

---

## 11. Mapa de Documentação do Projeto

| Documento | Propósito |
|---|---|
| [README.md](README.md) | Entrypoint — visão geral + quick start |
| [METODOLOGIA.md](METODOLOGIA.md) | **Este documento** — defesa acadêmica |
| [RESULTS.md](RESULTS.md) | Fonte única de métricas |
| [MODEL_CARD.md](../MODEL_CARD.md) | Card formal do modelo |
| [AUDITORIA_CIENTIFICA.md](AUDITORIA_CIENTIFICA.md) | Honestidade científica |
| [ANALISE_PROFUNDA.md](ANALISE_PROFUNDA.md) | Diagnóstico técnico + roadmap |
| [CHANGELOG.md](../CHANGELOG.md) | Cronologia |
| [POLITICA_DADOS.md](POLITICA_DADOS.md) | Decisão de não usar dados sintéticos |
| [SCIENTIFIC_JUSTIFICATION.md](SCIENTIFIC_JUSTIFICATION.md) | Origem científica dos parâmetros |
| [docs/ARCHITECTURE.md](ARCHITECTURE.md) | Arquitetura técnica |
| [docs/FRONTEND.md](FRONTEND.md) | Guia do Streamlit |
| [models/MODELS_MANIFEST.md](models/MODELS_MANIFEST.md) | Catálogo de `.pt` |
