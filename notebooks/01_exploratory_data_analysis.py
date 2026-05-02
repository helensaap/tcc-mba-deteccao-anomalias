"""
Notebook 01: Análise Exploratória de Dados (EDA)

Este notebook realiza a exploração inicial dos dados multimodais:
- Estrutura e volume de dados
- Visualização de imagens de plantas
- Análise de séries temporais de sensores
- Correlações entre modalidades
"""

import sys
sys.path.insert(0, '/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image
import logging

from src.data_loader import MultimodalDataLoader, ImagePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações de visualização
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def explore_data_structure():
    """Explora a estrutura geral dos dados extraídos."""
    print("\n" + "="*80)
    print("FASE 1.2: ANÁLISE EXPLORATÓRIA DE DADOS")
    print("="*80)

    data_dir = "/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data"
    loader = MultimodalDataLoader(data_dir)

    raw_data_dir = loader.raw_data_dir
    print(f"\nDiretório de dados: {raw_data_dir}")
    print(f"Existe: {raw_data_dir.exists()}")

    # Contar arquivos por tipo
    images_dir = loader.images_dir
    if images_dir.exists():
        png_count = len(list(images_dir.rglob("*.png")))
        json_count = len(list(images_dir.rglob("*.json")))
        print(f"\n--- IMAGENS ---")
        print(f"Arquivos PNG: {png_count}")
        print(f"Arquivos JSON: {json_count}")

    # Procurar XLSX (dados de sensores)
    xlsx_files = list(raw_data_dir.rglob("*.xlsx"))
    print(f"\n--- DADOS DE SENSORES ---")
    print(f"Arquivos XLSX encontrados: {len(xlsx_files)}")
    for xlsx in xlsx_files[:3]:
        print(f"  • {xlsx.name}")


def load_and_visualize_sample_images():
    """Carrega e visualiza amostras de imagens de plantas."""
    print("\n" + "="*80)
    print("VISUALIZAÇÃO DE AMOSTRAS DE IMAGENS")
    print("="*80)

    data_dir = "/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data"
    loader = MultimodalDataLoader(data_dir)

    plants = loader.get_plant_list()
    print(f"\nPlantas monitoradas: {plants}")

    # Carregar poucas imagens de cada planta
    fig, axes = plt.subplots(len(plants), 3, figsize=(15, 5 * len(plants)))
    if len(plants) == 1:
        axes = axes.reshape(1, -1)

    for row, plant in enumerate(plants):
        images_dict = loader.load_images_for_plant(plant, limit=3)

        for col, (img_path, img_array) in enumerate(list(images_dict.items())[:3]):
            ax = axes[row, col]

            # Converter para uint8 se necessário
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                img_to_display = (img_array * 255).astype(np.uint8)
            else:
                img_to_display = img_array

            # Tratar imagens em grayscale
            if len(img_to_display.shape) == 2:
                ax.imshow(img_to_display, cmap='gray')
            else:
                ax.imshow(img_to_display)

            ax.set_title(f"{plant} - {Path(img_path).name[:30]}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/results/01_sample_images.png', dpi=150)
    print("✓ Imagens salvas em: results/01_sample_images.png")
    plt.close()


def analyze_image_dimensions():
    """Analisa dimensões das imagens."""
    print("\n" + "="*80)
    print("ANÁLISE DE DIMENSÕES DE IMAGENS")
    print("="*80)

    data_dir = "/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data"
    loader = MultimodalDataLoader(data_dir)

    plants = loader.get_plant_list()
    dimensions_data = []

    for plant in plants:
        images_dict = loader.load_images_for_plant(plant, limit=50)

        for img_path, img_array in images_dict.items():
            height, width = img_array.shape[:2]
            dimensions_data.append({
                'plant': plant,
                'height': height,
                'width': width,
                'channels': img_array.shape[2] if len(img_array.shape) > 2 else 1
            })

    df_dims = pd.DataFrame(dimensions_data)

    print(f"\nResumo de dimensões de imagens:")
    print(df_dims.groupby('plant')[['height', 'width']].describe())

    # Visualizar distribuição de dimensões
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_dims.boxplot(column='height', by='plant', ax=axes[0])
    axes[0].set_title("Distribuição de Altura de Imagens")
    axes[0].set_ylabel("Altura (pixels)")

    df_dims.boxplot(column='width', by='plant', ax=axes[1])
    axes[1].set_title("Distribuição de Largura de Imagens")
    axes[1].set_ylabel("Largura (pixels)")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig('/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/results/02_image_dimensions.png', dpi=150)
    print("✓ Análise salva em: results/02_image_dimensions.png")
    plt.close()


def explore_sensor_data():
    """Carrega e analisa dados de sensores XLSX."""
    print("\n" + "="*80)
    print("ANÁLISE DE DADOS DE SENSORES IoT")
    print("="*80)

    data_dir = "/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data"
    loader = MultimodalDataLoader(data_dir)

    xlsx_files = list(loader.raw_data_dir.rglob("*.xlsx"))

    if not xlsx_files:
        print("Nenhum arquivo XLSX encontrado para análise.")
        return

    for xlsx_path in xlsx_files[:2]:  # Analisar apenas 2 primeiros
        print(f"\n--- Arquivo: {xlsx_path.name} ---")

        try:
            df = loader.load_sensor_data_from_excel(str(xlsx_path))

            if not df.empty:
                print(f"Dimensões: {df.shape}")
                print(f"\nColunas: {list(df.columns)}")
                print(f"\nPrimeiras linhas:")
                print(df.head())
                print(f"\nEstatísticas descritivas:")
                print(df.describe())

        except Exception as e:
            print(f"Erro ao processar {xlsx_path.name}: {e}")


def explore_json_metadata():
    """Analisa arquivos JSON com metadados de câmera."""
    print("\n" + "="*80)
    print("ANÁLISE DE METADADOS JSON (Câmera RealSense D415)")
    print("="*80)

    data_dir = "/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data"
    loader = MultimodalDataLoader(data_dir)

    json_files = list(loader.raw_data_dir.rglob("*.json"))

    if not json_files:
        print("Nenhum arquivo JSON encontrado.")
        return

    for json_path in json_files[:2]:
        print(f"\n--- Arquivo: {json_path.name} ---")
        metadata = loader.load_metadata_from_json(str(json_path))

        if metadata:
            print(f"Chaves no JSON: {list(metadata.keys())[:10]}")
            print(f"Amostra de dados:")
            for key in list(metadata.keys())[:3]:
                value = metadata[key]
                if isinstance(value, dict):
                    print(f"  {key}: {list(value.keys())}")
                else:
                    print(f"  {key}: {str(value)[:100]}")


def create_summary_statistics():
    """Cria resumo estatístico geral dos dados."""
    print("\n" + "="*80)
    print("RESUMO ESTATÍSTICO DOS DADOS")
    print("="*80)

    data_dir = "/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/data"
    loader = MultimodalDataLoader(data_dir)

    summary = {
        'plants': loader.get_plant_list(),
        'total_images': len(list(loader.images_dir.rglob("*.png"))) if loader.images_dir.exists() else 0,
        'total_json_files': len(list(loader.images_dir.rglob("*.json"))) if loader.images_dir.exists() else 0,
        'total_sensor_files': len(list(loader.raw_data_dir.rglob("*.xlsx"))),
    }

    print(f"\nResumo Geral:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return summary


def main():
    """Executa todas as análises exploratórias."""

    # Criar diretório de resultados
    results_dir = Path("/Users/helen.paixao/Desktop/tcc-mba-deteccao-anomalias/results")
    results_dir.mkdir(exist_ok=True)

    print("\n" + "#"*80)
    print("# NOTEBOOK 01: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("#"*80)

    # Executar análises
    explore_data_structure()
    analyze_image_dimensions()

    try:
        load_and_visualize_sample_images()
    except Exception as e:
        logger.error(f"Erro ao visualizar imagens: {e}")

    try:
        explore_sensor_data()
    except Exception as e:
        logger.error(f"Erro ao analisar dados de sensores: {e}")

    try:
        explore_json_metadata()
    except Exception as e:
        logger.error(f"Erro ao analisar JSON: {e}")

    summary = create_summary_statistics()

    print("\n" + "#"*80)
    print("# FIM DA ANÁLISE EXPLORATÓRIA")
    print("#"*80)
    print("\nPróximas etapas:")
    print("  1. Preparar pipeline de dados (treino/validação/teste)")
    print("  2. Treinar modelos CNN e LSTM")
    print("  3. Validar fusão multimodal")
    print("  4. Implementar sistema de alertas")
    print("\n")


if __name__ == "__main__":
    main()
