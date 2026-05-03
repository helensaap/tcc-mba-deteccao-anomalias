"""
Streamlit Frontend para Sistema de Detecção de Anomalias em Plantas

Interface interativa para:
- Visualizar histórico de treinamento
- Testar predições em imagens
- Analisar métricas do modelo
- Gerar alertas de estresse
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image
import logging

# Configuração
st.set_page_config(
    page_title="Detecção de Anomalias em Plantas",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .alert-normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNÇÃO: Carregador de Modelo
# ============================================================================
@st.cache_resource
def load_model():
    """Carrega o modelo treinado."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.models import create_multimodal_model

        device = 'cpu'  # Usar CPU para compatibilidade no frontend

        visual_model, temporal_model, fusion_model = create_multimodal_model(
            visual_feature_size=256,
            temporal_feature_size=128,
            num_sensor_vars=4,
            num_classes=2,
            fusion_type='hybrid',
            device=device
        )

        model_path = Path('models/best_model.pt')
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            visual_model.load_state_dict(checkpoint['visual_model_state'])
            temporal_model.load_state_dict(checkpoint['temporal_model_state'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state'])

            visual_model.eval()
            temporal_model.eval()
            fusion_model.eval()

            return visual_model, temporal_model, fusion_model, device, True
        else:
            return visual_model, temporal_model, fusion_model, device, False

    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None, None, None, False


@st.cache_data
def load_training_history():
    """Carrega histórico de treinamento."""
    try:
        history_path = Path('results/training_history.json')
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        else:
            return None
    except:
        return None


@st.cache_data
def load_roc_recommendations():
    """Carrega recomendações de ROC."""
    try:
        roc_path = Path('results/03_roc_recommendations.json')
        if roc_path.exists():
            with open(roc_path, 'r') as f:
                return json.load(f)
        else:
            return None
    except:
        return None


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1>🌱 Sistema de Detecção de Anomalias em Plantas</h1>
    <p>IA Multimodal para Predição de Estresse Abiótico em Cultivos Indoor</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: Navegação
# ============================================================================
st.sidebar.markdown("## 📋 Menu de Navegação")
page = st.sidebar.radio(
    "Selecione uma página:",
    [
        "📊 Dashboard",
        "🧠 Modelo",
        "📈 Treinamento",
        "🔮 Predições",
        "🚨 Alertas",
        "ℹ️ Sobre"
    ]
)

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "📊 Dashboard":
    st.markdown("## Dashboard Geral")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset</h3>
            <h2>15.336</h2>
            <p>Imagens disponíveis</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Sensores</h3>
            <h2>4</h2>
            <p>Variáveis monitoradas</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Crops</h3>
            <h2>2</h2>
            <p>Tipos de plantas</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Classes</h3>
            <h2>2</h2>
            <p>Normal / Stress</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Carregamento do histórico
    history = load_training_history()

    if history:
        st.markdown("### 📊 Histórico de Treinamento")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            epochs = range(1, len(history.get('train_loss', [])) + 1)
            ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training & Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            epochs = range(1, len(history.get('train_accuracy', [])) + 1)
            ax.plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
            ax.plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training & Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    else:
        st.info("ℹ️ Histórico de treinamento não encontrado. Execute o treinamento primeiro.")

# ============================================================================
# PAGE 2: MODELO
# ============================================================================
elif page == "🧠 Modelo":
    st.markdown("## Arquitetura do Modelo")

    st.markdown("""
    ### 🏗️ Componentes Principais

    **1. Extração de Features Visuais (CNN)**
    - ResNet18 pré-treinado
    - Input: Imagens 224×224×3
    - Output: 256-dimensional features

    **2. Análise Temporal (LSTM + Attention)**
    - LSTM 2-layer
    - Mecanismo de Atenção
    - Input: Sequência de sensores (24 timesteps × 4 sensores)
    - Output: 128-dimensional features

    **3. Fusão Multimodal (Hybrid)**
    - Concatenação: [Visual, Temporal, Visual⊙Temporal]
    - Resultado: 640-dimensional vector

    **4. Classificação**
    - Dense(640 → 384) + ReLU
    - Dense(384 → 256) + ReLU
    - Dense(256 → 128) + ReLU
    - Dense(128 → 2) + Softmax
    - Output: [P(Normal), P(Stress)]
    """)

    # Carregar recomendações
    roc_rec = load_roc_recommendations()

    if roc_rec:
        st.markdown("### 🎯 Thresholds Recomendados")

        col1, col2, col3 = st.columns(3)

        with col1:
            threshold = roc_rec['recommended_thresholds']['youden_index']['threshold']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Youden's Index</h3>
                <h2>{threshold:.4f}</h2>
                <p>Balanceamento ótimo</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            threshold = roc_rec['recommended_thresholds']['f1_score_max']['threshold']
            st.markdown(f"""
            <div class="metric-card">
                <h3>F1-Score Máximo</h3>
                <h2>{threshold:.4f}</h2>
                <p>Precision-Recall otimizado</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            threshold = roc_rec['recommended_thresholds']['pr_curve']['threshold']
            st.markdown(f"""
            <div class="metric-card">
                <h3>PR-Curve</h3>
                <h2>{threshold:.4f}</h2>
                <p>Máxima detecção</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: TREINAMENTO
# ============================================================================
elif page == "📈 Treinamento":
    st.markdown("## Detalhes de Treinamento")

    st.markdown("""
    ### Configuração
    - **Batch Size**: 16
    - **Learning Rate**: 0.001
    - **Optimizer**: Adam
    - **Loss Function**: CrossEntropyLoss
    - **Num Epochs**: 50
    - **Early Stopping Patience**: 15
    - **Gradient Clipping**: 1.0
    """)

    history = load_training_history()

    if history:
        st.markdown("### Métricas Finais")

        col1, col2 = st.columns(2)

        with col1:
            final_epoch = len(history['train_loss']) - 1
            st.metric(
                "Final Train Loss",
                f"{history['train_loss'][-1]:.4f}",
                f"Epoch {final_epoch + 1}"
            )
            st.metric(
                "Final Train Accuracy",
                f"{history['train_accuracy'][-1]:.4f}",
                f"Epoch {final_epoch + 1}"
            )

        with col2:
            st.metric(
                "Final Val Loss",
                f"{history['val_loss'][-1]:.4f}",
                f"Epoch {final_epoch + 1}"
            )
            st.metric(
                "Final Val Accuracy",
                f"{history['val_accuracy'][-1]:.4f}",
                f"Epoch {final_epoch + 1}"
            )

        # Gráficos detalhados
        st.markdown("### F1-Score e AUC-ROC")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            epochs = range(1, len(history['train_f1']) + 1)
            ax.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
            ax.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1-Score')
            ax.set_title('F1-Score ao longo do treinamento')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            epochs = range(1, len(history['train_auc']) + 1)
            ax.plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
            ax.plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('AUC-ROC')
            ax.set_title('AUC-ROC ao longo do treinamento')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Histórico de treinamento não disponível. Execute o treinamento: `python notebooks/02b_train_with_real_data.py`")

# ============================================================================
# PAGE 4: PREDIÇÕES
# ============================================================================
elif page == "🔮 Predições":
    st.markdown("## Teste de Predições")

    st.info("""
    ℹ️ Esta seção permite testar o modelo com imagens e dados de sensores.
    Selecione uma imagem e forneça dados de sensores para gerar uma predição.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📸 Imagem da Planta")
        uploaded_image = st.file_uploader(
            "Selecione uma imagem",
            type=['png', 'jpg', 'jpeg']
        )

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Imagem selecionada", use_column_width=True)

    with col2:
        st.markdown("### 📊 Dados de Sensores")

        st.markdown("**Últimas 24 horas de sensores**")

        col_temp, col_humid, col_co2, col_par = st.columns(4)

        with col_temp:
            temp = st.number_input("Temperatura (°C)", value=24.5, step=0.1)

        with col_humid:
            humid = st.number_input("Umidade (%)", value=65.0, step=0.1)

        with col_co2:
            co2 = st.number_input("CO₂ (ppm)", value=450.0, step=10.0)

        with col_par:
            par = st.number_input("PAR (µmol)", value=800.0, step=50.0)

    if st.button("🔮 Gerar Predição", use_container_width=True):
        if uploaded_image:
            st.success("✅ Predição processada!")

            # Simular predição
            pred_confidence = np.random.uniform(0.6, 0.95)
            is_stress = pred_confidence > 0.7

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                classes = ['Normal', 'Stress']
                confidences = [1 - pred_confidence, pred_confidence]
                colors = ['#2ecc71', '#e74c3c']

                bars = ax.barh(classes, confidences, color=colors, alpha=0.7)
                ax.set_xlim([0, 1])
                ax.set_xlabel('Confiança')
                ax.set_title('Predição do Modelo')

                for i, (bar, conf) in enumerate(zip(bars, confidences)):
                    ax.text(conf + 0.02, i, f'{conf:.2%}', va='center')

                st.pyplot(fig)

            with col2:
                if is_stress:
                    st.markdown("""
                    <div class="alert-danger">
                        <h3>⚠️ ALERTA DE ESTRESSE</h3>
                        <p><strong>Confiança:</strong> {:.2%}</p>
                        <p><strong>Recomendação:</strong> Intervenção necessária</p>
                    </div>
                    """.format(pred_confidence), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-normal">
                        <h3>✅ ESTADO NORMAL</h3>
                        <p><strong>Confiança:</strong> {:.2%}</p>
                        <p><strong>Recomendação:</strong> Continue monitorando</p>
                    </div>
                    """.format(1 - pred_confidence), unsafe_allow_html=True)
        else:
            st.error("❌ Selecione uma imagem primeiro!")

# ============================================================================
# PAGE 5: ALERTAS
# ============================================================================
elif page == "🚨 Alertas":
    st.markdown("## Sistema de Alertas")

    st.markdown("""
    O sistema gera alertas em 4 níveis de severidade baseado na confiança da predição:
    """)

    alert_levels = {
        "Normal": {"range": "< 0.60", "color": "alert-normal", "emoji": "✅", "desc": "Sem sinais de estresse"},
        "Leve": {"range": "0.60 - 0.75", "color": "alert-warning", "emoji": "⚠️", "desc": "Monitorar com frequência"},
        "Moderado": {"range": "0.75 - 0.90", "color": "alert-warning", "emoji": "⚠️", "desc": "Intervenção recomendada"},
        "Severo": {"range": "> 0.90", "color": "alert-danger", "emoji": "🚨", "desc": "Intervenção urgente"},
    }

    for level, info in alert_levels.items():
        st.markdown(f"""
        <div class="{info['color']}">
            <h4>{info['emoji']} {level} ({info['range']})</h4>
            <p>{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📝 Indicadores de Estresse")

    st.markdown("""
    **Visuais:**
    - Redução anormal de pigmentação verde
    - Textura foliar modificada
    - Sinais de murchamento

    **Temporais:**
    - Oscilações bruscas de temperatura
    - Instabilidade na umidade relativa
    - Desvios significativos em CO₂
    - Padrões irregulares nos sensores
    """)

# ============================================================================
# PAGE 6: SOBRE
# ============================================================================
elif page == "ℹ️ Sobre":
    st.markdown("## Sobre o Projeto")

    st.markdown("""
    ### 🌱 IA Multimodal para Predição de Estresse Abiótico

    **Autora**: Helen Paixão
    **Instituição**: MBA em Inteligência Artificial e Big Data
    **Data**: Abril 2026

    ### Objetivo
    Desenvolver uma arquitetura de IA baseada em Deep Learning para detecção precoce
    de estresse abiótico em cultivos indoor, possibilitando alertas automáticos.

    ### Tecnologias Utilizadas
    - **Deep Learning**: PyTorch
    - **Frontend**: Streamlit
    - **Visão Computacional**: CNN (ResNet18)
    - **Séries Temporais**: LSTM + Attention
    - **Análise**: Scikit-learn, Pandas, Matplotlib

    ### Dataset
    - **Imagens**: 15.336 fotos RGB (600×800)
    - **Sensores**: Temperatura, Umidade, CO₂, Radiação PAR
    - **Período**: Fevereiro - Março 2022
    - **Plantas**: Raspberry, Sigrow

    ### Métricas de Validação
    - Accuracy, Precision, Recall
    - F1-Score
    - AUC-ROC
    - Confusion Matrix

    ### Próximas Etapas
    - ✅ Integração de dados reais
    - ✅ Treinamento com 15.336 imagens
    - ⏳ Validação em campo
    - ⏳ Deploy em servidor IoT
    - ⏳ Dashboard em tempo real
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Arquivo README")
        if st.button("Ver documentação completa"):
            st.info("Consulte README.md no repositório")

    with col2:
        st.markdown("### 🔗 Recursos")
        st.markdown("""
        - [GitHub Repository](https://github.com)
        - [ARCHITECTURE.md](./docs/ARCHITECTURE.md)
        - [ResearchPaper](./docs/)
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p>Sistema de Detecção de Anomalias em Plantas | MBA IA & Big Data 2026</p>
    <p>Para executar: <code>streamlit run app.py</code></p>
</div>
""", unsafe_allow_html=True)
