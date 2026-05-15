# Interface (Vite + Vue)

Regra essencial: sem estilos inline. O contrato CSS detalhado vive em `.sangoi/reference/ui/frontend-css-contracts.md`; o style guide em `.sangoi/frontend/guidelines/frontend-style-guide.md` é orientação de autoria, não a fonte normativa do contrato.

## Dev
- WSL/Linux: `./run-webui.sh` (na raiz do repo) — sobe API + Vite juntos usando `.venv`.
- Ou manualmente (na raiz do repositório):
  - API: `CODEX_ROOT="$(pwd)" PYTHONPATH="$(pwd)" API_PORT_OVERRIDE=7850 .venv/bin/python apps/backend/interfaces/api/run_api.py`
  - UI: `cd apps/interface && npm install && npm run dev -- --host`
- Proxy `/api` aponta para `API_HOST:API_PORT` (env vars).

## Estrutura
- `src/main.ts` → importa `src/styles.css`; é o único bootstrap CSS de runtime.
- `src/styles.css` → entrypoint CSS de runtime; importa os módulos de estilo declarados.
- `src/views` → telas (txt2img, img2img, extras, settings).
- `src/router.ts` → rotas simples.

## Verificação
- `cd apps/interface && npm run verify:css-contracts` — único gate CSS direto.
- `cd apps/interface && npm run verify` — wrapper-only; encadeia `verify:css-contracts`, `typecheck` e `build`.

## Padrões de estilo
- Sem `style="..."` e sem `:style` fora das exceções tipadas sancionadas pelo contrato.
- Estados por classes/atributos (`data-state`), não por estilos ad-hoc.
- Não use classes internas de terceiros (ex.: `.svelte-xxxx`).
- Para topology/budgets/exceções tipadas, consulte `.sangoi/reference/ui/frontend-css-contracts.md`.
- Para guidance de naming/layout/tokens/Tailwind, consulte `.sangoi/frontend/guidelines/frontend-style-guide.md`.
