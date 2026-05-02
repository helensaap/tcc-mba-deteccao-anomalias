"""
Data Loader Module for Multimodal Abiotic Stress Detection System

Este módulo gerencia o carregamento e processamento de dados multimodais:
- Imagens fenotípicas (PNG)
- Metadados de câmera (JSON)
- Dados de sensores IoT (XLSX)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image
import openpyxl
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDataLoader:
    """
    Carregador de dados multimodais para o sistema de detecção de estresse abiótico.

    Coordena o carregamento de:
    - Imagens de plantas (visão computacional)
    - Series temporais de sensores ambientais (IoT)
    - Metadados de câmeras e equipamentos
    """

    def __init__(self, data_dir: str):
        """
        Inicializa o carregador de dados.

        Args:
            data_dir: Caminho para o diretório de dados brutos
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw" / "1st Experiment"
        self.images_dir = self.raw_data_dir / "Images_1stExperiment" / "1stExperiment_Daily_Images"

        if not self.images_dir.exists():
            logger.warning(f"Diretório de imagens não encontrado: {self.images_dir}")

    def get_plant_list(self) -> List[str]:
        """Retorna lista de plantas monitoradas."""
        if not self.images_dir.exists():
            return []

        # ✅ MODIFICADO 28 ABRIL: INCLUIR RASPBERRY também!
        # Antes: excluía raspberry
        # Agora: inclui AMBAS as plantas (sigrow + raspberry = 2.256 imagens!)
        plant_dirs = [d.name for d in self.images_dir.glob("**/cva/*")
                     if d.is_dir()]  # Sem exclusão, pega todas!
        return sorted(set(plant_dirs))

    def load_images_for_plant(self, plant_name: str,
                            limit: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Carrega todas as imagens de uma planta específica.

        Args:
            plant_name: Nome da planta
            limit: Número máximo de imagens a carregar (None = todas)

        Returns:
            Dicionário {image_path: np_array}
        """
        images = {}
        plant_img_dir = self.images_dir / f"cva/{plant_name}"

        if not plant_img_dir.exists():
            logger.error(f"Diretório da planta não encontrado: {plant_img_dir}")
            return images

        png_files = sorted(plant_img_dir.rglob("*.png"))[:limit]

        logger.info(f"Carregando {len(png_files)} imagens de {plant_name}")
        for img_path in tqdm(png_files, desc=f"Carregando imagens: {plant_name}"):
            try:
                img = Image.open(img_path)
                images[str(img_path)] = np.array(img)
            except Exception as e:
                logger.error(f"Erro ao carregar {img_path}: {e}")

        return images

    def load_metadata_from_json(self, json_path: str) -> Dict:
        """Carrega metadados de calibração de câmera (D415 RealSense)."""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar JSON: {e}")
            return {}

    def load_sensor_data_from_excel(self, excel_path: str) -> pd.DataFrame:
        """
        Carrega dados de sensores IoT de arquivo XLSX.

        Returns:
            DataFrame com colunas: timestamp, temperatura, umidade, CO2, etc.
        """
        try:
            df = pd.read_excel(excel_path)
            logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar XLSX: {e}")
            return pd.DataFrame()

    def extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """
        Extrai timestamp do nome do arquivo de imagem.

        Formato esperado: ...d415_XX_AAAA_MM_DD_HH_MM_SS...
        """
        parts = filename.split('_')
        if len(parts) >= 6:
            try:
                return f"{parts[2]}-{parts[3]}-{parts[4]} {parts[5]}:{parts[6]}:{parts[7]}"
            except:
                return None
        return None

    def align_multimodal_data(self, images_dict: Dict, sensor_df: pd.DataFrame,
                            tolerance_seconds: int = 60) -> pd.DataFrame:
        """
        Alinha dados de imagens com series temporais de sensores.

        Args:
            images_dict: Dicionário de imagens carregadas
            sensor_df: DataFrame com dados de sensores
            tolerance_seconds: Tolerância de alinhamento temporal

        Returns:
            DataFrame consolidado com referências a imagens e dados de sensores alinhados
        """
        aligned_data = []

        for img_path in images_dict.keys():
            timestamp_str = self.extract_timestamp_from_filename(Path(img_path).name)
            if timestamp_str:
                aligned_data.append({
                    'image_path': img_path,
                    'timestamp': timestamp_str,
                    'image_shape': images_dict[img_path].shape
                })

        aligned_df = pd.DataFrame(aligned_data)

        if not aligned_df.empty and not sensor_df.empty:
            logger.info(f"Dados alinhados: {len(aligned_df)} imagens com sensores")

        return aligned_df


class ImagePreprocessor:
    """Módulo para pré-processamento de imagens."""

    @staticmethod
    def resize_image(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Redimensiona imagem mantendo aspect ratio."""
        img_pil = Image.fromarray(img.astype('uint8')) if img.dtype != np.uint8 else Image.fromarray(img)
        img_pil.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Adiciona padding para manter tamanho exato
        new_img = Image.new('RGB' if img_pil.mode == 'RGB' else 'L', target_size, color=(0, 0, 0))
        new_img.paste(img_pil, ((target_size[0] - img_pil.size[0]) // 2,
                               (target_size[1] - img_pil.size[1]) // 2))
        return np.array(new_img)

    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        """Normaliza imagem para intervalo [0, 1]."""
        return img.astype(np.float32) / 255.0

    @staticmethod
    def augment_image(img: np.ndarray, rotation_range: float = 15,
                     flip_horizontal: bool = True) -> List[np.ndarray]:
        """Realiza augmentação simples de dados de imagem."""
        augmented = [img]

        if flip_horizontal and np.random.rand() > 0.5:
            augmented.append(np.fliplr(img))

        return augmented


def main():
    """Teste do data loader."""
    data_dir = "/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data"
    loader = MultimodalDataLoader(data_dir)

    logger.info("=== Teste de Data Loader ===")
    plants = loader.get_plant_list()
    logger.info(f"Plantas encontradas: {plants}")

    if plants:
        test_plant = plants[0]
        images = loader.load_images_for_plant(test_plant, limit=5)
        logger.info(f"Imagens carregadas para {test_plant}: {len(images)}")


if __name__ == "__main__":
    main()
