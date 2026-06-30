#!/usr/bin/env python3
"""
Monitor em Tempo Real - FASE 7: Semi-Supervised Learning
=========================================================

Monitora o progresso do treinamento semi-supervisionado em tempo real
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess

def clear_screen():
    """Limpar terminal"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_process_status():
    """Verificar se processo está rodando"""
    result = subprocess.run(
        "ps aux | grep '07_semi_supervised' | grep -v grep",
        shell=True,
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())

def get_training_history():
    """Carregar histórico de treinamento se disponível"""
    history_file = Path("results/07_semi_supervised_history.json")
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def display_status():
    """Exibir status em tempo real"""
    clear_screen()

    is_running = get_process_status()
    history = get_training_history()

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║         🚀 MONITOR - FASE 7: SEMI-SUPERVISED LEARNING COM 15K IMAGENS     ║")
    print("║                                                                            ║")
    print(f"║  Status: {'✅ RODANDO' if is_running else '⏹️  PARADO'}                                                                      ║")
    print(f"║  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                              ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    if is_running:
        print("\n📊 TREINAMENTO EM PROGRESSO...\n")

        if history:
            epochs = history.get('epochs', [])
            labeled_loss = history.get('labeled_loss', [])
            unlabeled_loss = history.get('unlabeled_loss', [])
            total_loss = history.get('total_loss', [])

            if epochs:
                current_epoch = len(epochs)
                max_epochs = history.get('epochs', [])[-1] if history.get('epochs') else 100

                print(f"⏳ Epoch: {current_epoch}")
                print(f"📈 Labeled Loss (últimas 5): {[f'{x:.4f}' for x in labeled_loss[-5:]]}")
                print(f"📈 Unlabeled Loss (últimas 5): {[f'{x:.4f}' for x in unlabeled_loss[-5:]]}")
                print(f"📈 Total Loss (últimas 5): {[f'{x:.4f}' for x in total_loss[-5:]]}")

                # Progresso
                progress_bar_len = 40
                filled = int(progress_bar_len * current_epoch / 100)
                bar = '█' * filled + '░' * (progress_bar_len - filled)
                print(f"\n📊 Progresso: [{bar}] {current_epoch}%\n")

        print("📂 Arquivos sendo gerados:")
        print("   └─ models/best_model_semi_supervised.pt")
        print("   └─ results/07_semi_supervised_history.json")
        print("   └─ results/07_semi_supervised_training.png")

    else:
        print("\n✅ TREINAMENTO FINALIZADO!\n")

        if history:
            print("📊 RESULTADOS FINAIS:\n")
            print(f"   Labeled samples: {history.get('labeled_samples', 'N/A')}")
            print(f"   Pseudo-labeled: {history.get('pseudo_labels_used', 'N/A')}")
            print(f"   Total training: {history.get('total_samples', 'N/A')}")
            print(f"   Epochs: {history.get('final_epoch', 'N/A')}/{history.get('num_epochs', 100)}")
            print(f"   Best Loss: {history.get('best_loss', 'N/A')}")

            print("\n🎯 PRÓXIMOS PASSOS:")
            print("   1. Avaliar modelo em test set")
            print("   2. Comparar com Fase 6 (52.94%)")
            print("   3. Atualizar app.py se houver melhoria")
            print("   4. Preparar resultados para defesa")

            print("\n📊 Gráficos disponíveis:")
            if Path("results/07_semi_supervised_training.png").exists():
                print("   ✅ results/07_semi_supervised_training.png")

            print("\n📂 Modelo salvo:")
            if Path("models/best_model_semi_supervised.pt").exists():
                print("   ✅ models/best_model_semi_supervised.pt")

        print("\n" + "="*80)
        print("Digite 'exit' para sair do monitor")
        print("="*80)

if __name__ == "__main__":
    print("Iniciando monitor em tempo real...\n")

    try:
        while True:
            display_status()

            user_input = input("\nPressione ENTER para atualizar (ou 'exit' para sair): ").strip().lower()
            if user_input == 'exit':
                print("\n👋 Monitor finalizado")
                break

            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Monitor finalizado")
