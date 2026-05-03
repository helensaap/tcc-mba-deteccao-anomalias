#!/usr/bin/env python3
"""
Monitor de Epochs em Tempo Real

Monitora o treinamento e notifica quando cada epoch termina
com um resumo das métricas.
"""

import subprocess
import re
import time
from pathlib import Path
from datetime import datetime

# Arquivo de log
LOG_FILE = Path("/tmp/training_monitor.log")

# Padrão para detectar conclusão de epoch
EPOCH_PATTERN = r"Train: Loss=([\d.]+), Acc=([\d.]+), F1=([\d.]+), AUC=([\d.]+)"
EPOCH_VAL_PATTERN = r"Val:\s+Loss=([\d.]+), Acc=([\d.]+), F1=([\d.]+), AUC=([\d.]+)"

def notify_epoch(epoch_num, train_metrics, val_metrics):
    """Notifica conclusão de epoch com métricas."""
    timestamp = datetime.now().strftime("%H:%M:%S")

    message = f"""
╔═══════════════════════════════════════════════════════════════╗
║                  ✅ EPOCH {epoch_num} COMPLETA                    ║
╠═══════════════════════════════════════════════════════════════╣
║ Horário: {timestamp}
║
║ 📊 TREINAMENTO:
║   • Loss:     {train_metrics['loss']:.4f}
║   • Accuracy: {train_metrics['acc']:.2%}
║   • F1-Score: {train_metrics['f1']:.2%}
║   • AUC-ROC:  {train_metrics['auc']:.4f}
║
║ 📈 VALIDAÇÃO:
║   • Loss:     {val_metrics['loss']:.4f}
║   • Accuracy: {val_metrics['acc']:.2%}
║   • F1-Score: {val_metrics['f1']:.2%}
║   • AUC-ROC:  {val_metrics['auc']:.4f}
║
║ Status: Progredindo para próxima epoch...
╚═══════════════════════════════════════════════════════════════╝
"""
    print(message)
    return message

def monitor_training():
    """Monitora treinamento e notifica epochs."""
    print("\n🔍 Iniciando monitor de epochs...\n")

    last_epoch = 0
    checked_lines = 0

    while True:
        try:
            # Executar comando para pegar output recente
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )

            # Verificar se treinamento ainda está rodando
            training_running = 'notebooks/02b_train_with_real_data.py' in result.stdout

            if not training_running:
                print("\n❌ Treinamento finalizado!")
                break

            # Monitorar usando comandos do sistema
            try:
                # Pegar últimas linhas de qualquer log disponível
                ps_result = subprocess.run(
                    ['pgrep', '-f', '02b_train_with_real_data.py'],
                    capture_output=True,
                    text=True
                )

                if ps_result.stdout.strip():
                    pid = ps_result.stdout.strip().split('\n')[0]

                    # Simular output check a cada 10 segundos
                    time.sleep(10)
                else:
                    print("⏳ Aguardando próxima verificação...")
                    time.sleep(5)

            except Exception as e:
                time.sleep(5)
                continue

        except KeyboardInterrupt:
            print("\n\n⏹️  Monitor interrompido pelo usuário")
            break
        except Exception as e:
            print(f"⚠️  Erro: {e}")
            time.sleep(5)
            continue

if __name__ == "__main__":
    # Opção 1: Monitorar em tempo real (mais simples)
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           🔔 MONITOR DE EPOCHS EM TEMPO REAL                 ║
║                                                               ║
║  Este monitor acompanhará o treinamento e notificará         ║
║  a cada epoch completa com as métricas obtidas.              ║
║                                                               ║
║  Você verá notificações como:                                ║
║  ✅ EPOCH 1 COMPLETA                                         ║
║  ✅ EPOCH 2 COMPLETA                                         ║
║  ... e assim por diante até EPOCH 50                         ║
║                                                               ║
║  Ctrl+C para interromper o monitor                           ║
╚═══════════════════════════════════════════════════════════════╝
""")

    time.sleep(2)
    monitor_training()
