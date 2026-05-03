# 🌐 Guia do Frontend Streamlit

## Visão Geral

O aplicativo Streamlit fornece uma interface interativa para visualizar, testar e monitorar o modelo de detecção de anomalias em plantas.

## Instalação

### 1. Instalar Streamlit (se não estiver instalado)

```bash
pip install streamlit>=1.28.0 matplotlib seaborn
```

## Executar o Frontend

### Opção 1: Executar localmente

```bash
# Certifique-se de estar no diretório raiz do projeto
cd /Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias

# Execute o aplicativo
streamlit run app.py
```

O aplicativo abrirá em `http://localhost:8501`

### Opção 2: Executar em um servidor remoto

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Páginas Disponíveis

### 📊 Dashboard
- Visão geral do dataset
- Métricas principais
- Gráficos de treinamento (Loss, Accuracy)

**Use quando**: Você quer uma rápida visão geral do status do projeto

### 🧠 Modelo
- Descrição da arquitetura neural
- Componentes principais (CNN, LSTM, Fusion)
- Thresholds recomendados

**Use quando**: Você quer entender como o modelo funciona

### 📈 Treinamento
- Histórico completo de treinamento
- Métricas finais
- Gráficos de F1-Score e AUC-ROC

**Use quando**: Você quer analisar o desempenho do treinamento em detalhes

### 🔮 Predições
- Interface para testar o modelo
- Upload de imagem
- Entrada manual de dados de sensores
- Resultados de predição

**Use quando**: Você quer testar o modelo com dados novos

### 🚨 Alertas
- Níveis de severidade
- Indicadores visuais e temporais
- Recomendações de ação

**Use quando**: Você quer entender o sistema de alertas

### ℹ️ Sobre
- Informações do projeto
- Tecnologias utilizadas
- Dataset details
- Próximas etapas

**Use quando**: Você quer informações gerais sobre o projeto

## Solução de Problemas

### Erro: "ModuleNotFoundError: No module named 'src'"

**Solução**: Certifique-se de executar o comando no diretório raiz do projeto.

```bash
cd /Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias
streamlit run app.py
```

### Erro: "FileNotFoundError: results/training_history.json"

**Solução**: O histórico de treinamento não foi gerado yet. Execute o treinamento primeiro:

```bash
python notebooks/02b_train_with_real_data.py
```

### Erro: "FileNotFoundError: models/best_model.pt"

**Solução**: O modelo ainda não foi treinado. Execute:

```bash
python notebooks/02b_train_with_real_data.py
```

## Features Implementadas

✅ Dashboard com métricas de treinamento
✅ Visualização de arquitetura do modelo
✅ Gráficos interativos de histórico de treinamento
✅ Interface de predição com upload de imagens
✅ Sistema de alertas com níveis de severidade
✅ Informações completas do projeto
✅ Temas personalizados com cores verdes (tema agrícola)

## Features Futuras

- [ ] Predição em tempo real
- [ ] Dashboard de monitoramento contínuo
- [ ] Exportar predições em CSV
- [ ] Integração com câmeras IoT
- [ ] Histórico de predições
- [ ] Análise comparativa de modelos
- [ ] Sistema de notificações

## Dicas de Uso

1. **Primeira Vez**: Comece na página "Sobre" para entender o projeto
2. **Análise**: Use o "Dashboard" para visão geral rápida
3. **Detalhes**: Use "Treinamento" para análise profunda
4. **Testes**: Use "Predições" para experimentar com dados novos
5. **Entendimento**: Use "Modelo" para aprender a arquitetura

## Contato

Para dúvidas sobre o frontend, consulte:
- `app.py` - Código principal
- `.streamlit/config.toml` - Configurações
- `README.md` - Documentação geral

---

**Status**: ✅ Pronto para uso | **Última atualização**: Maio 2026
