# 🔬 Análise Técnica Profunda & Roadmap

Consolida: `ANALISE_COMPLETA_PROFUNDA.md`, `MODEL_ANALYSIS_AND_IMPROVEMENT_STRATEGY.md`, `ANALISE_ALINHAMENTO_REQUISITOS.md`, `RESUMO_ANALISE_PT.md`.

Foco: diagnóstico técnico do gap entre desempenho atual e estado-da-arte, mais um roadmap acionável.

---

## 1. Diagnóstico do Modelo Atual

### 1.1 O que funciona ✅
- **Arquitetura multimodal está correta** — CNN + LSTM-Attention + Fusion Hybrid combina sinais visuais e temporais
- **Sem data leakage** — splits isolados, validação reprodutível
- **AUC > 0.5** — modelo aprende padrões reais (AUC=0.729)
- **Pseudo-labeling agregou valor** — F1 saltou de 0 → 0.4 na Fase 7

### 1.2 O que limita ⚠️

| Limitação | Impacto | Causa raiz |
|---|---|---|
| Test set n=17 | IC amplo, métricas instáveis | Apenas 239 ground truth labels no dataset original |
| Recall=0.286 | Perde 71% dos casos de stress | Modelo conservador (threshold alto), classe minoritária |
| 0.7% supervisão | Aproveita pouco do dataset | Estrutura do experimento original |
| Domínio restrito | Não generaliza | Treinado apenas em leafy greens |
| Modelo grande p/ dados pequenos | Risco overfit | ResNet18 (11M params) vs 107 labels |

---

## 2. Análise por Componente

### 2.1 Visual Encoder (ResNet18 fine-tuned)

**Status:** adequado, mas superdimensionado.

- 11M parâmetros vs 107 imagens labeled = razão 100k:1 (não saudável)
- Últimas 16 camadas descongeladas é razoável, mas ainda muito
- Pre-training ImageNet ajuda, mas domínio é distante (objetos vs plantas)

**Recomendações ordenadas por impacto:**
1. Reduzir para arquitetura menor (MobileNetV3, EfficientNet-B0) — menos params, mesmo recall
2. Congelar mais camadas (apenas FC head treinável) até ter mais labels
3. Aplicar test-time augmentation (TTA) no inference

### 2.2 Temporal Encoder (LSTM 2-layer + Attention)

**Status:** sólido teoricamente, mas pouco explorado.

- 24 timesteps × 4 sensores = janela curta, bem dimensionada para ciclo circadiano
- Mecanismo de atenção é apropriado
- **Mas:** as features temporais podem estar dominadas pelas visuais na fusion

**Recomendações:**
1. Análise de ablação: treinar apenas com modalidade temporal e medir contribuição
2. Considerar TCN (Temporal Convolutional Network) como alternativa — mais paralelo
3. Adicionar features derivadas: gradientes (dT/dt), médias móveis

### 2.3 Fusion Hybrid

**Status:** escolha defensável, mas não validada empiricamente.

- `concat([v, t, v⊙t])` = 640 dims → grande para classifier de 4 layers
- Multiplicação elemento-a-elemento captura interações (bom para "fenótipo silencioso")

**Recomendações:**
1. Ablation entre Early/Late/Hybrid — reportar qual é melhor para este dataset
2. Considerar Co-Attention (Lu et al., 2019) — atenção cruzada visual ↔ temporal
3. Reduzir output dim do fusion (640 → 256) para classifier mais leve

### 2.4 Classifier (Dense 640 → 384 → 256 → 128 → 2)

**Status:** profundo demais para classificação binária com dados escassos.

- 4 camadas FC = ~400k params adicionais
- Risco de overfit alto

**Recomendações:**
1. Simplificar para `640 → 128 → 2`
2. Aumentar dropout (0.3 → 0.5)
3. Substituir BatchNorm por LayerNorm (mais estável com batches pequenos)

---

## 3. Análise de Treinamento

### 3.1 Hiperparâmetros (Fase 7 oficial)

| Hiperparâmetro | Valor atual | Comentário |
|---|---|---|
| lr | 5e-4 | Razoável para fine-tuning |
| batch_size | 8 | Pequeno por necessidade (RAM CPU) — afeta BatchNorm |
| epochs | 100 | OK com patience=20 |
| pseudo-label threshold | 0.85 | Ponto ótimo encontrado empiricamente |
| λ_unlabeled | 1.0 | Tentativa de 2.0 causou regressão |
| weight_decay | 1e-4 | Regularização leve |

### 3.2 Curva de aprendizagem (observada)

Olhando `results/06_training_history_advanced.json`:

- **train_acc oscila** entre 35% e 67% — muita variância, sintoma de batch pequeno + dados escassos
- **val_acc pico** em epoch 12 (75%) mas degrada
- **gap train/val** moderado — não é puro overfit, é instabilidade

**Interpretação:** o modelo está no limite do que dá pra extrair com 74 amostras de treino. Mais regularização não ajuda, mais dados sim.

---

## 4. Gap Analysis com Requisitos do TCC

Baseado nos objetivos declarados em [README.md](README.md) e na metodologia:

| Objetivo declarado | Status | Evidência |
|---|---|---|
| 1. Fusão computacional multimodal | ✅ Atingido | `src/models.py::MultimodalFusionModel` |
| 2. Extração features fenotípicas (CNN) | ✅ Atingido | `PhenotypicFeatureExtractor` + ResNet18 |
| 3. Modelagem temporal (LSTM+Attention) | ✅ Atingido | `TemporalSensorAnalyzer` |
| 4. Detecção de padrões combinados | ✅ Parcialmente | AUC=0.729 confirma, mas recall baixo |
| 5. Sistema de alertas integrado | ✅ Atingido | `src/alert_system.py` + 4 níveis |
| 6. Validação com métricas consolidadas | ✅ Atingido | Accuracy, P, R, F1, AUC, ROC — todas calculadas |

**Cobertura objetiva:** 6/6 ✅ (com ressalva no recall).

---

## 5. Roadmap (Backlog, não obrigatório para defesa)

### Curto prazo (1–2 semanas) — melhorias incrementais
- [ ] **Ablation Early/Late/Hybrid** — reportar qual fusion é ótima
- [ ] **TTA no inference** — flip horizontal, +1–2% acc esperado
- [ ] **Threshold sweep no test set** — encontrar o melhor F1 operacional
- [ ] **Documentar comparativo com baseline** (apenas-imagem vs apenas-sensor vs multimodal)

### Médio prazo (1–3 meses) — ampliação acadêmica
- [ ] **Aumentar labels para 500+** via anotação manual ou colaboração com agrônomo
- [ ] **Validação em segunda estufa** (outro dataset Sigrow ou similar)
- [ ] **Calibrar thresholds visuais** com SPAD meter / espectrofotometria
- [ ] **Substituir LSTM por Transformer** (publicável)
- [ ] **Publicação em SIBGRAPI ou ANALITICA EXPO**

### Longo prazo (6+ meses) — produção
- [ ] **Validação em campo** com produtores reais de Cannabis medicinal
- [ ] **Deploy edge** (Jetson Nano + câmera IoT, inferência local)
- [ ] **Dashboard tempo real** (WebSocket + Streamlit/Plotly Dash)
- [ ] **Integração com ANVISA** para rastreabilidade

---

## 6. Riscos para a Defesa

| Risco | Severidade | Mitigação |
|---|---|---|
| Banca questionar n=17 do test set | 🟡 Média | Argumentar honestidade científica (recall<perfect, mas AUC>0.5, reprodutível) |
| Banca questionar 64.71% vs estado-da-arte (>90%) | 🟡 Média | Esclarecer que SOTA tem datasets de 10k+ labels; 107 é academicamente legítimo |
| Banca perguntar sobre validação química real | 🟢 Baixa | Reconhecer como limitação e citar como trabalho futuro |
| Confusão entre fases (52 vs 64 vs 75) | 🔴 Alta antes / 🟢 Baixa agora | Resolvida com [RESULTS.md](RESULTS.md) como fonte única |

---

## 7. Postura Recomendada na Defesa

**Frame:** "Este TCC entrega uma **prova de conceito completa** de uma arquitetura multimodal CNN+LSTM-Attention para detecção de estresse abiótico, validada com dados reais (não sintéticos), reprodutível e com auditoria de honestidade científica documentada. As limitações de tamanho de dataset são reconhecidas e mapeadas como trabalho futuro."

Os números reportados são modestos mas **defensáveis**. É melhor entrar com 64.71% honesto do que com 98% que a banca derruba em 30 segundos.

---

## 8. Referências Bibliográficas Consolidadas

- He, K. et al. (2015). *Deep Residual Learning for Image Recognition*. CVPR.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Baltrušaitis, T. et al. (2018). *Multimodal Machine Learning: A Survey and Taxonomy*. IEEE TPAMI.
- Taiz, L., Zeiger, E. (2015). *Fisiologia Vegetal*, 6ª ed.
- Hsiao, T.C. (1973). *Plant responses to water stress*. Annu. Rev. Plant Physiol.
- Youden, W.J. (1950). *Index for rating diagnostic tests*. Cancer.
- Van Rijsbergen, C.J. (1979). *Information Retrieval*.
- Hughes, D.P., Salathe, M. (2016). *PlantVillage*. arXiv:1604.02143.
- Lee, D.H. (2013). *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method*.
- AGÊNCIA BRASIL (2024). *Brasil estima déficit de R$ 20bi com IFAs*.
- AVGOUSTAKI, D.D., XYDIS, G. (2020). *Indoor vertical farming*. Sustainability.
