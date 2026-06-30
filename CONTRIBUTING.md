# Contribuindo

Este é um projeto acadêmico (TCC MBA), mas contribuições e issues são
bem-vindas — especialmente para reproduzir ou estender os resultados.

## Setup

```bash
git clone <repo-url>
cd tcc-mba-deteccao-anomalias
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Obter o modelo treinado

Os pesos `.pt` **não** estão no Git (são pesados). Baixe via GitHub Release
ou Hugging Face Hub conforme [models/MODELS_MANIFEST.md](models/MODELS_MANIFEST.md).

## Obter os dados

O dataset "1st Experiment" não é redistribuído. Veja
[POLITICA_DADOS.md](POLITICA_DADOS.md) para origem e termos.

## Rodar os testes

```bash
pytest tests/ -v
```

## Reproduzir o treino oficial

```bash
python notebooks/07_semi_supervised_learning.py
python notebooks/08_evaluate_semi_supervised.py
```

Resultado esperado: 64.71% test acc, σ=0.0 em 5 runs (seed=42).

## Estilo

- Python 3.10+
- Sem comentários redundantes — código fala por si
- Docs em PT-BR (alinhado ao TCC)

## Reportar problemas

Abra uma issue descrevendo:
- Comando executado
- Saída completa (stdout + stderr)
- Versão do Python e `pip freeze` relevante

## Pull requests

PRs pequenos e focados são preferidos. Para mudanças grandes, abra uma
issue antes para alinhar.
