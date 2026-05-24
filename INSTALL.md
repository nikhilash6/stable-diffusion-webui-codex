# Install — Codex WebUI (Backend + Vue UI)

This repo ships:
- **Backend API**: FastAPI (`apps/backend/interfaces/api/run_api.py`)
- **Frontend UI**: Vue 3 + Vite (`apps/interface`)

## Prerequisites
- Git
- Internet access (first install: downloads `uv`, CPython 3.12.10, Node.js (via nodeenv), and wheels)

Video runtime note:
- `install-webui.(cmd|sh)` provisions repo-local `ffmpeg` + `ffprobe` binaries and the default RIFE checkpoint automatically (no manual PATH/model setup required).

## Quick install (recommended)

### Windows (PowerShell or CMD)
1) Run the installer (downloads `uv`, installs managed CPython **3.12.10** into `.uv/python`, syncs deps from `uv.lock` into `.venv`, installs Node.js into `.nodeenv` (via nodeenv), runs lock-preserving `npm ci`; keeps `uv`/`npm` caches repo-local under `.uv/cache` and `.npm-cache`):
```powershell
.\install-webui.cmd
```

```bat
install-webui.cmd
```

`install-webui.ps1` is the Windows installer source of truth.
`install-webui.cmd` is a thin wrapper that invokes the PowerShell installer and propagates its exit code.

On Windows, the installer prompts for **Simple / Advanced / Check / Reinstall** by default.
Advanced mode includes CPU/CUDA/skip and a **Windows-only custom PyTorch** entry that loads options from a remote JSON manifest.
For automation, pass `--no-menu` and use flags/env vars.

Windows installer flags:
- `install-webui.cmd --check` (verify installer-managed dependencies only; non-destructive)
- `install-webui.cmd --reinstall-deps` (reinstall dependencies in-place, without deleting `.venv` / `.nodeenv`)

Windows one-liner (Win11Debloat style, but safe: download-to-file + execute; no `iex`):
```powershell
$ref = "main" # recommend: replace with a release tag or commit SHA
$url = "https://raw.githubusercontent.com/sangoi-exe/stable-diffusion-webui-codex/$ref/install-webui.ps1"
$dst = Join-Path $env:TEMP "codex-install-webui.ps1"
irm $url -OutFile $dst
powershell -NoProfile -ExecutionPolicy Bypass -File $dst
```

2) Launch the GUI launcher:
```bat
run-webui.bat
```

3) Safe update (fail-closed):
```bat
update-webui.bat
```

### Linux / WSL
1) Run the installer (downloads `uv`, installs managed CPython **3.12.10** into `.uv/python`, syncs deps from `uv.lock` into `.venv`, installs Node.js into `.nodeenv` (via nodeenv), runs lock-preserving `npm ci`; keeps `uv`/`npm` caches repo-local under `.uv/cache` and `.npm-cache`):
```bash
bash install-webui.sh
```

Linux installer flags:
- `bash install-webui.sh --check` (verify installer-managed dependencies only; non-destructive)
- `bash install-webui.sh --reinstall-deps` (reinstall dependencies in-place, without deleting `.venv` / `.nodeenv`)

2) Start API + UI:
```bash
./run-webui.sh
```

3) Safe update (fail-closed):
```bash
bash update-webui.sh
# Optional: ignore untracked-path preflight checks (tracked changes still abort)
bash update-webui.sh --force
```

### Docker (Linux / WSL)
1) Build image:
```bash
docker build -t codex-webui:latest .
```

Optional build overrides:
- `--build-arg CODEX_TORCH_MODE=cpu|cuda|rocm|skip`
- `--build-arg CODEX_TORCH_BACKEND=cpu|cu126|cu128|cu130|rocm64`
- `--build-arg CODEX_CUDA_VARIANT=12.6|12.8|13`

2) Run container (GPU example):
```bash
docker run --rm -it --gpus all \
  -p 7850:7850 -p 7860:7860 \
  -v "$(pwd)/models:/opt/stable-diffusion-webui-codex/models" \
  -v "$(pwd)/output:/opt/stable-diffusion-webui-codex/output" \
  codex-webui:latest
```

Runtime behavior:
- Entry point is `run-webui-docker.sh`.
- Interactive sessions open a terminal TUI (`apps/docker_tui_launcher.py`) to configure launcher env/profile values without manual `KEY=VALUE` typing.
- Disable TUI with `-e CODEX_DOCKER_TUI=0` or container args `--no-tui`.
- Configure-only mode: `codex-webui:latest --configure-only --tui`.
- If `API_PORT_OVERRIDE` / `WEB_PORT` are changed in TUI/profile, host `-p` mappings must use the same ports.
- Docker defaults are preseeded for this project profile (CUDA runtime path, SDPA flash, LoRA online, WAN22 `ram+hd`) and can be overridden with `-e KEY=VALUE` or compose `.env`.

3) Compose (GPU + persisted volumes):
```bash
docker compose up --build
```

First-time interactive profile configuration:
```bash
docker compose run --rm webui --tui --configure-only
```

## Manual install (no installer scripts)

Use this if you want full manual control and do not want to run `install-webui.cmd`, `install-webui.ps1`, or `install-webui.sh`.

### Windows manual (PowerShell commands, no installer script)
1) Clone and enter repo:
```powershell
git clone https://github.com/sangoi-exe/stable-diffusion-webui-codex.git
cd .\stable-diffusion-webui-codex
$repoRoot = (Get-Location).Path
```

2) Prepare repo-local tool/cache paths:
```powershell
mkdir "$repoRoot\.uv\cache","$repoRoot\.uv\xdg-data","$repoRoot\.uv\xdg-cache","$repoRoot\.npm-cache" -Force | Out-Null
$env:UV_CACHE_DIR = "$repoRoot\.uv\cache"
$env:NPM_CONFIG_CACHE = "$repoRoot\.npm-cache"
$env:XDG_DATA_HOME = "$repoRoot\.uv\xdg-data"
$env:XDG_CACHE_HOME = "$repoRoot\.uv\xdg-cache"
$env:UV_PYTHON_INSTALL_DIR = "$repoRoot\.uv\python"
$env:UV_PYTHON_INSTALL_BIN = "0"
$env:UV_PYTHON_INSTALL_REGISTRY = "0"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_PYTHON_DOWNLOADS = "manual"
$env:UV_PROJECT_ENVIRONMENT = "$repoRoot\.venv"
$env:PYTHONPATH = "$repoRoot"
```

3) Install uv (choose one):
- With winget (recommended):
```powershell
winget install --id Astral-sh.uv -e
```
- Or manual binary download (no remote script execution):
```powershell
$uvVersion = "0.9.17"
$asset = "uv-x86_64-pc-windows-msvc.zip" # ARM64: uv-aarch64-pc-windows-msvc.zip
$uvDir = "$repoRoot\.uv\bin"
mkdir $uvDir -Force | Out-Null
$zip = Join-Path $uvDir $asset
irm "https://github.com/astral-sh/uv/releases/download/$uvVersion/$asset" -OutFile $zip
tar -xf $zip -C $uvDir
Remove-Item $zip -Force
$env:PATH = "$uvDir;$env:PATH"
```

4) Install managed Python + sync dependencies:
```powershell
uv python install 3.12.10
# Pick ONE backend extra: cpu | cu126 | cu128 | cu130
uv sync --locked --extra cu128
```

5) Provision FFmpeg + default RIFE model:
```powershell
& "$repoRoot\.venv\Scripts\python.exe" -c "import os; from apps.backend.video.runtime_dependencies import ensure_ffmpeg_binaries; p=ensure_ffmpeg_binaries(version=os.environ.get('CODEX_FFMPEG_VERSION','7.0.2')); print(p)"
& "$repoRoot\.venv\Scripts\python.exe" -c "from apps.backend.video.runtime_dependencies import ensure_rife_model_file; print(ensure_rife_model_file())"
```

6) Install repo-local Node.js + frontend deps:
```powershell
uv tool run --from nodeenv nodeenv -n 24.15.0 "$repoRoot\.nodeenv"
$npm = Join-Path $repoRoot ".nodeenv\Scripts\npm.cmd"
if (!(Test-Path $npm)) { $npm = Join-Path $repoRoot ".nodeenv\bin\npm.cmd" }
$interfaceDir = Join-Path $repoRoot "apps\interface"
$npmCache = Join-Path $repoRoot ".npm-cache"
Push-Location $interfaceDir
& $npm ci --cache $npmCache --no-audit --no-fund
Pop-Location
```

7) Run WebUI:
```powershell
& "$repoRoot\run-webui.bat"
```

## Safe updater contract (`update-webui.(bat|sh)`)
- Update scope is repo root only (no submodule/extension update automation).
- Dirty worktree check is fail-closed: tracked paths abort always; untracked paths abort unless `--force` is set (ignored paths do not).
- `--force` affects only the preflight dirty check; `git pull --ff-only` safety checks still apply.
- Abort diagnostics list explicit cause and offending files/directories when applicable.
- Non-destructive update path only: `git fetch --prune` + `git pull --ff-only`.
- Dependency verification (toolchain + torch backend resolution) runs on every update attempt after git safety checks.
- If repo-local npm is missing, updater auto-provisions `.nodeenv` via nodeenv (using `CODEX_NODE_VERSION`, default `24.15.0`).
- Environment refresh runs on every update attempt after dependency verification.
- Frontend refresh uses lock-preserving mode (`npm ci`), so `apps/interface/package-lock.json` is required.

## Node.js (frontend)
The installers provision a repo-local Node.js into `.nodeenv` via `nodeenv` (no system Node required).
The updater also provisions `.nodeenv` automatically if npm is missing.
Existing `.nodeenv` directories are not replaced automatically; if the managed Node.js version differs from `CODEX_NODE_VERSION`, install/update aborts so the operator can recreate `.nodeenv` or pin the existing version intentionally.
On Windows, updater/install scripts probe npm in both `.nodeenv\\Scripts\\npm.cmd` and `.nodeenv\\bin\\npm.cmd`.
Installer/updater frontend sync uses lock-preserving `npm ci` (requires `apps/interface/package-lock.json`).

Override:
- `CODEX_NODE_VERSION` (Node.js version pin for nodeenv; default: 24.15.0)

## PyTorch
This repo uses `uv.lock` to pin and lock dependency versions (including PyTorch variants). The installers choose **one** PyTorch backend via `uv` extras.

Default behavior:
- `CODEX_TORCH_MODE=auto` (default):
  - If AMD/ROCm is detected (Linux), use ROCm 6.4 wheels (`--extra rocm64`).
  - Else if `nvidia-smi` exists, prefer CUDA 12.8 wheels (`--extra cu128`), with fallback to `cu126` if the driver advertises CUDA 12.6 and `cu130` if CUDA 13 is advertised and the driver is new enough.
  - Otherwise CPU (`--extra cpu`).
- On macOS, the installers always use `cpu`.

Override:
- `CODEX_TORCH_MODE=cpu` (force CPU: `--extra cpu`)
- `CODEX_TORCH_MODE=cuda` (force CUDA: defaults to `--extra cu128`)
- `CODEX_TORCH_MODE=rocm` (Linux only: force ROCm 6.4 wheels: `--extra rocm64`)
- `CODEX_TORCH_MODE=skip` (skip torch/torchvision entirely; the WebUI will not run without PyTorch)
- `CODEX_TORCH_BACKEND=cpu|cu126|cu128|cu130|rocm64` (explicitly pick the PyTorch backend extra)
- `CODEX_CUDA_VARIANT=12.6|12.8|13|cu126|cu128|cu130` (aliases map to `cu126|cu128|cu130`; validated whenever set and used for backend selection when `CODEX_TORCH_BACKEND` is not set)
- `CODEX_INSTALL_CHECK=1` (check mode; verifies installer-managed dependencies only)
- `CODEX_REINSTALL_DEPS=1` (reinstall dependencies in-place without deleting environments)
- `CODEX_INSTALL_TRACE=1` (Linux/WSL installer: enable shell trace for debugging)
- `CODEX_FFMPEG_VERSION=<version>` (pin ffmpeg-downloader runtime build; default: `7.0.2`)

Windows-only custom PyTorch:
- `CODEX_TORCH_MODE=custom`
- `CODEX_CUSTOM_TORCH_SRC=<custom wheel/source URL or path>`
- `CODEX_CUSTOM_TORCH_SHA256=<sha256>` (optional; if set, installer verifies the wheel hash before install)
- `CODEX_PYTORCH_MANIFEST_URL=<remote json manifest url>` (optional override; defaults to this repo `main` branch)
- Menu options are loaded from remote JSON field `windows_custom_torch`.
- To publish a new custom build, update `pytorch_manifest.json` in the repo (installer keeps fetching it remotely; no installer code changes needed).
- Linux/WSL does **not** support `CODEX_TORCH_MODE=custom` in `install-webui.sh`.

If CUDA install fails, try a different backend:
- Example: `CODEX_TORCH_BACKEND=cu126 bash install-webui.sh`

## Troubleshooting

### Windows: `: was unexpected at this time` (cmd.exe parsing)
This is a **cmd.exe batch parsing** error (not a `uv`/Python error). It typically happens when a `.bat` script uses multi-line `(...)` blocks and cmd gets confused by special characters, causing labels like `:install_uv` to become “unexpected”.

Fix:
- `git pull` (the Windows installer routine was hardened to avoid this class of cmd parsing failure)
- Re-run `install-webui.cmd`

If it still happens, see the deep dive runbook:
- Open an issue with the full console output, plus:
  - your Windows version
  - whether you cloned via Git or downloaded a ZIP
  - whether the repo path contains non-ASCII characters

### `ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'`
Your `peft` and `transformers` are out of sync (common when extra packages pull older pins).

Fix: remove the venv and re-install with this repo’s locked dependencies:
- First try in-place reinstall:
  - Windows: `install-webui.cmd --reinstall-deps`
  - Linux/WSL: `bash install-webui.sh --reinstall-deps`
- If the environment is still irrecoverable, delete `.venv` and re-run `install-webui.(cmd|sh)`.

### Don’t mix “research deps” into the WebUI venv
Packages like `pyiqa`, `datasets`, `numba`, `opencv-python-headless` often pin conflicting `transformers`/`numpy`.

Keep them in a separate virtualenv.
