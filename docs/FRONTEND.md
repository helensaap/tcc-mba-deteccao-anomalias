# 🌐 Frontend Streamlit — Guia de Uso

Arquivo principal: [../app.py](../app.py)
Configuração: [../.streamlit/](../.streamlit/)

---

## Instalação & Execução

```bash
# 1. Ative o venv
source venv/bin/activate

# 2. Instale dependências (se ainda não instaladas)
pip install -r requirements.txt

# 3. Execute
streamlit run app.py
```

Abre em `http://localhost:8501`.

Para servidor remoto: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.

---

## Páginas Disponíveis

| Página | Propósito |
|---|---|
| 📊 **Dashboard** | Visão geral: 15.336 imagens, 4 sensores, gráficos de treino |
| 🧠 **Modelo** | Arquitetura CNN+LSTM-Attention+Fusion, thresholds recomendados |
| 📈 **Treinamento** | Hiperparâmetros (lr=5e-4, batch=8, epochs=100), curvas F1 & AUC |
| 🔮 **Predições** | Upload de imagem + input de sensores → predição em tempo real |
| 🚨 **Alertas** | 4 níveis (NORMAL/MILD/MODERATE/SEVERE) com recomendações |
| ℹ️ **Sobre** | Metadata do projeto, autora, dataset, tecnologias |

---

## Função-chave: `predict_real()`

Localizada em [../app.py](../app.py). Fluxo:

1. Recebe imagem + 4 valores de sensor
2. Pré-processa: resize 224×224, normalize, expand seq temporal
3. Forward pass: `fusion(visual_features, temporal_features)`
4. Softmax → probabilidade de stress
5. Aplica threshold (0.4482 — Youden) → classifica
6. Gera `StressAlert` via `src/alert_system.py`

---

## Troubleshooting

| Erro | Causa | Solução |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Executando fora do root | `cd` para raiz do projeto antes |
| `FileNotFoundError: results/training_history.json` | Histórico não gerado | `python notebooks/07_semi_supervised_learning.py` |
| `FileNotFoundError: models/best_model.pt` | Modelo ausente | Treinar ou copiar `best_model_semi_supervised.pt` |
| Streamlit roda mas predição falha | Mismatch dim sensores | Confirmar 4 valores no input (T, RH, CO2, PAR) |

---

## Roadmap do Frontend

Implementadas hoje:
- ✅ Dashboard de métricas
- ✅ Visualização de arquitetura
- ✅ Predição em tempo real
- ✅ Sistema de alertas com níveis

Backlog (não obrigatório para defesa):
- [ ] Histórico persistente de predições (SQLite)
- [ ] Exportar predições em CSV
- [ ] Integração com câmeras IoT (RealSense)
- [ ] Dashboard de monitoramento contínuo (auto-refresh)
- [ ] Sistema de notificações (email/webhook)
