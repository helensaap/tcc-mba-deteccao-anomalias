# Plano de Release — v1.0 (Defesa TCC)

## Checklist pré-publicação

- [x] `.gitignore` cobre `.pt`, `.agents/`, `venv/`, `data/`, `results/`
- [x] `LICENSE` (MIT + CC-BY-4.0 para pesos/docs)
- [x] `CONTRIBUTING.md`
- [x] Checkpoints intermediários movidos para `models/archive/` (gitignored)
- [x] `models/MODELS_MANIFEST.md` aponta para Release/HF
- [ ] Repo remoto criado no GitHub (`gh repo create`)
- [ ] Modelo oficial subido como GitHub Release asset
- [ ] (opcional) Mirror no Hugging Face Hub

## Passos

### 1. Criar repositório no GitHub

```bash
# Pública desde o início
gh repo create helen-paixao/tcc-stress-abiotico \
  --public \
  --description "IA Multimodal para Predição de Estresse Abiótico (TCC MBA)" \
  --homepage "https://github.com/helen-paixao/tcc-stress-abiotico"

git remote add origin https://github.com/helen-paixao/tcc-stress-abiotico.git
```

### 2. Push inicial

```bash
git push -u origin main
```

### 3. Criar tag e Release com o modelo oficial

```bash
git tag -a v1.0 -m "v1.0 — Defesa TCC (Fase 7 semi-supervised)"
git push origin v1.0

gh release create v1.0 \
  models/best_model_semi_supervised.pt \
  --title "v1.0 — Modelo oficial (defesa)" \
  --notes-file RESULTS.md
```

Isso anexa o `.pt` ao Release sem inflar o repo.

### 4. (Opcional) Hugging Face Hub

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload helen-paixao/tcc-stress-abiotico \
  models/best_model_semi_supervised.pt \
  best_model_semi_supervised.pt
```

Adicionar `MODEL_CARD.md` como `README.md` do repo HF.

### 5. Verificação pós-publicação

- [ ] Clone limpo em outra pasta funciona: `pip install -r requirements.txt`
- [ ] Download do modelo via Release URL funciona
- [ ] `streamlit run app.py` roda com o modelo baixado
- [ ] `pytest tests/` passa
- [ ] README badges renderizam
- [ ] LICENSE aparece no GitHub sidebar

## Riscos / pontos de atenção

- **Tamanho do histórico:** confirmar que `.pt` não foi commitado por engano
  em algum commit antigo (`git log --stat --all -- 'models/*.pt'`). Se sim,
  considerar `git filter-repo` antes de tornar público.
- **PDF da metodologia:** 256 KB — confirmar que pode ser distribuído (verificar
  termos da disciplina).
- **`.agents/`:** ferramentas internas — gitignored, não vai vazar.
- **Dataset:** não redistribuído. README e `POLITICA_DADOS.md` documentam origem.
