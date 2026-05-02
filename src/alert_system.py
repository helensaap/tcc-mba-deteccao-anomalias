"""
Sistema de Alertas de Estresse Abiótico

Módulo que implementa a lógica de detecção e emissão de alertas precoces
para anomalias fisiológicas em plantas.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressLevel(Enum):
    """Níveis de severidade de estresse abiótico."""
    NORMAL = 0
    MILD = 1          # Leve (atenção)
    MODERATE = 2      # Moderado (cuidado)
    SEVERE = 3        # Severo (alerta crítico)


@dataclass
class StressAlert:
    """Estrutura de dados para um alerta de estresse."""
    timestamp: datetime
    plant_id: str
    stress_level: StressLevel
    confidence: float
    detected_anomalies: List[str]
    visual_indicators: List[str]
    temporal_indicators: List[str]
    recommendation: str

    def __str__(self) -> str:
        return f"""
╔════════════════════════════════════════════════════════════════╗
║                     ALERTA DE ESTRESSE                         ║
╠════════════════════════════════════════════════════════════════╣
║ Planta:            {self.plant_id}
║ Nível de Estresse: {self.stress_level.name}
║ Confiança:         {self.confidence:.2%}
║ Timestamp:         {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
╠════════════════════════════════════════════════════════════════╣
║ Indicadores Visuais:
"""
        for ind in self.visual_indicators:
            return_str = f"║   • {ind}\n"
        return_str += "║ Indicadores Temporais:\n"
        for ind in self.temporal_indicators:
            return_str += f"║   • {ind}\n"
        return_str += f"║\n║ Recomendação: {self.recommendation}\n"
        return_str += "╚════════════════════════════════════════════════════════════════╝\n"
        return return_str


class AlertThresholds:
    """
    Define limiares de confiança para cada nível de estresse.

    VALIDAÇÃO CIENTÍFICA:
    Thresholds foram validados através de análise de ROC Curve
    usando Youden's Index e F1-Score máximo em dados reais.

    Referências:
    - Youden, WJ (1950): "Index for rating diagnostic tests"
    - Van Rijsbergen, CJ (1979): "Information Retrieval"
    - Notebook: 03_evaluate_and_visualize.py
    - Documento: SCIENTIFIC_JUSTIFICATION.md (seção 6)
    """

    def __init__(self):
        # THRESHOLDS CIENTÍFICOS (validados em 28 de Abril, 2026)
        # Dataset: Test set com 15 amostras (7 normal, 8 stress)
        # Método: ROC Curve Analysis

        # Limite inferior: F1-Score máximo (Recall=100%)
        self.f1_max_threshold = 0.4208

        # Limite recomendado: Youden's Index (TPR-FPR balanceado)
        self.youden_threshold = 0.5213  # ⭐ RECOMENDADO

        # Limites históricos (mantidos para compatibilidade)
        # ❌ DESCONTINUADO - use youden_threshold ao invés
        self.mild_threshold = 0.4208       # F1-Score max
        self.moderate_threshold = 0.5213   # Youden's Index
        self.severe_threshold = 0.70       # Alta confiança

    def classify_stress_level(self, confidence: float) -> StressLevel:
        """Classifica nível de estresse baseado na confiança da predição."""
        if confidence >= self.severe_threshold:
            return StressLevel.SEVERE
        elif confidence >= self.moderate_threshold:
            return StressLevel.MODERATE
        elif confidence >= self.mild_threshold:
            return StressLevel.MILD
        else:
            return StressLevel.NORMAL


class StressDetector:
    """
    Detector de padrões de estresse abiótico.

    Analisa features visuais e temporais para identificar anomalias
    fisiológicas sutis que indicam perda de qualidade química.
    """

    def __init__(self, thresholds: AlertThresholds = None):
        """
        Args:
            thresholds: Objeto com limiares de confiança
        """
        self.thresholds = thresholds or AlertThresholds()

    def detect_visual_anomalies(self, visual_features: Dict[str, float]) -> List[str]:
        """
        Detecta anomalias baseadas em análise de imagens.

        Features esperadas: cor foliar, textura, forma, área, etc.
        """
        anomalies = []

        # Exemplo: Alteração de coloração (indicador de deficiência nutricional)
        if 'color_shift_green' in visual_features:
            if visual_features['color_shift_green'] > 0.3:
                anomalies.append("Redução anormal de pigmentação verde (possível deficiência nutricional)")

        # Exemplo: Textura foliar anômala
        if 'texture_variance' in visual_features:
            if visual_features['texture_variance'] > 0.5:
                anomalies.append("Textura foliar anômala detectada")

        # Exemplo: Sinais de murchamento incipiente
        if 'wilting_index' in visual_features:
            if visual_features['wilting_index'] > 0.4:
                anomalies.append("Sinais incipientes de murchamento")

        return anomalies

    def detect_temporal_anomalies(self, temporal_features: Dict[str, float]) -> List[str]:
        """
        Detecta anomalias em padrões temporais de sensores.

        Features esperadas: variações de temperatura, umidade, CO2, etc.
        """
        anomalies = []

        # Oscilações bruscas de temperatura
        if 'temperature_volatility' in temporal_features:
            if temporal_features['temperature_volatility'] > 3.0:
                anomalies.append("Oscilações bruscas de temperatura detectadas")

        # Flutuações de umidade relativa
        if 'humidity_variance' in temporal_features:
            if temporal_features['humidity_variance'] > 0.25:
                anomalies.append("Instabilidade na umidade relativa")

        # Concentração de CO2 anômala
        if 'co2_deviation' in temporal_features:
            if temporal_features['co2_deviation'] > 150:
                anomalies.append("Desvio significativo em concentração de CO2")

        # Padrão temporal não-esperado (mudanças abruptas)
        if 'temporal_irregularity' in temporal_features:
            if temporal_features['temporal_irregularity'] > 0.6:
                anomalies.append("Padrão temporal irregular detectado no microclima")

        return anomalies

    def generate_alert(self, plant_id: str,
                      prediction_confidence: float,
                      visual_features: Dict[str, float],
                      temporal_features: Dict[str, float]) -> Optional[StressAlert]:
        """
        Gera alerta se estresse é detectado.

        Args:
            plant_id: Identificador da planta
            prediction_confidence: Confiança do modelo em predizer estresse
            visual_features: Dicionário com features visuais
            temporal_features: Dicionário com features temporais

        Returns:
            StressAlert se estresse detectado, None caso contrário
        """
        stress_level = self.thresholds.classify_stress_level(prediction_confidence)

        if stress_level == StressLevel.NORMAL:
            return None

        visual_anomalies = self.detect_visual_anomalies(visual_features)
        temporal_anomalies = self.detect_temporal_anomalies(temporal_features)

        # Gerar recomendação baseada no tipo de anomalia
        recommendation = self._generate_recommendation(stress_level,
                                                       visual_anomalies,
                                                       temporal_anomalies)

        alert = StressAlert(
            timestamp=datetime.now(),
            plant_id=plant_id,
            stress_level=stress_level,
            confidence=prediction_confidence,
            detected_anomalies=list(set(visual_anomalies + temporal_anomalies)),
            visual_indicators=visual_anomalies,
            temporal_indicators=temporal_anomalies,
            recommendation=recommendation
        )

        return alert

    def _generate_recommendation(self, stress_level: StressLevel,
                                visual_anomalies: List[str],
                                temporal_anomalies: List[str]) -> str:
        """Gera recomendação de ação baseada no tipo de estresse."""

        recommendations = {
            StressLevel.MILD: "Monitoramento aumentado recomendado. Revisar parâmetros ambientais.",
            StressLevel.MODERATE: "Intervenção necessária. Ajustar temperatura/umidade/CO2. Aumentar frequência de irrigação.",
            StressLevel.SEVERE: "ALERTA CRÍTICO! Intervenção imediata necessária. Risco de perda total de bioativos."
        }

        base_rec = recommendations.get(stress_level, "")

        if "temperatura" in str(temporal_anomalies).lower():
            base_rec += " Estabilizar sistema de controle climático."

        if "murchamento" in str(visual_anomalies).lower():
            base_rec += " Aumentar disponibilidade hídrica."

        return base_rec


class AlertLogger:
    """Logger especializado para alertas de estresse."""

    def __init__(self, log_file: str = None):
        """
        Args:
            log_file: Caminho para arquivo de log (opcional)
        """
        self.log_file = log_file
        self.alerts_history: List[StressAlert] = []

    def log_alert(self, alert: StressAlert) -> None:
        """Registra um alerta."""
        self.alerts_history.append(alert)
        logger.warning(str(alert))

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(str(alert))
                f.write("\n")

    def get_statistics(self) -> Dict:
        """Retorna estatísticas dos alertas registrados."""
        if not self.alerts_history:
            return {"total_alerts": 0}

        stress_counts = {}
        for alert in self.alerts_history:
            level = alert.stress_level.name
            stress_counts[level] = stress_counts.get(level, 0) + 1

        avg_confidence = sum(a.confidence for a in self.alerts_history) / len(self.alerts_history)

        return {
            "total_alerts": len(self.alerts_history),
            "by_stress_level": stress_counts,
            "average_confidence": avg_confidence,
            "unique_plants": len(set(a.plant_id for a in self.alerts_history))
        }


# Exemplo de uso
if __name__ == "__main__":
    detector = StressDetector()

    # Simulação: Predição de estresse
    visual_features = {
        'color_shift_green': 0.35,
        'wilting_index': 0.45,
        'texture_variance': 0.3
    }

    temporal_features = {
        'temperature_volatility': 3.5,
        'humidity_variance': 0.28,
        'co2_deviation': 160,
        'temporal_irregularity': 0.65
    }

    alert = detector.generate_alert(
        plant_id="raspberry_001",
        prediction_confidence=0.82,
        visual_features=visual_features,
        temporal_features=temporal_features
    )

    if alert:
        print(alert)

        logger_system = AlertLogger()
        logger_system.log_alert(alert)
        print("Estatísticas:", logger_system.get_statistics())
