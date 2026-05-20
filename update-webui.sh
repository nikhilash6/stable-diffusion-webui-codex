#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
UV_BIN="${ROOT_DIR}/.uv/bin/uv"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
NODE_VERSION="${CODEX_NODE_VERSION:-24.15.0}"
NODEENV_DIR="${ROOT_DIR}/.nodeenv"
NODE_BIN="${NODEENV_DIR}/bin/node"
NPM_BIN="${NODEENV_DIR}/bin/npm"
INTERFACE_DIR="${ROOT_DIR}/apps/interface"
CODEX_FFMPEG_VERSION="${CODEX_FFMPEG_VERSION:-7.0.2}"
ALLOW_UNTRACKED=0

usage() {
  cat <<'EOF'
Usage: update-webui.sh [--force] [--help]

Safe updater for stable-diffusion-webui-codex.

Behavior:
- Fail-closed: aborts on unsafe git state (dirty tree, detached HEAD, no upstream, ahead/diverged, in-progress merge/rebase/cherry-pick/bisect).
- Never runs destructive commands (no git reset/clean/restore/delete operations).
- Updates only via: git fetch --prune + git pull --ff-only.
- Verifies dependency prerequisites on every run after git safety checks.
- Refreshes environment on every run after dependency verification.
- Auto-provisions repo-local Node.js/npm when missing via nodeenv.

Policy:
- Scope: repo root only (no submodule/extension updates).
- --force disables untracked-path preflight checks only; tracked changes still abort and git pull safety still applies.
- Ignored paths do not block update (gitignore entries are excluded from dirty checks).
- Frontend refresh uses lock-preserving mode: npm ci.

Environment overrides:
- CODEX_TORCH_MODE=custom (updater aborts before dependency sync to avoid overwriting custom PyTorch installs)
- CODEX_TORCH_BACKEND=cpu|cu126|cu128|cu130|rocm64
- CODEX_CUDA_VARIANT=12.6|12.8|13|cu126|cu128|cu130 (validated whenever set; used for backend selection when CODEX_TORCH_BACKEND is not set)
- CODEX_NODE_VERSION=<version> (default: 24.15.0; used for nodeenv auto-provisioning)
- CODEX_FFMPEG_VERSION=<version> (default: 7.0.2)
EOF
}

log() {
  printf '[update] %s\n' "$*"
}

abort() {
  local code="$1"
  shift
  printf '[update][%s] %s\n' "$code" "$*" >&2
  exit 1
}

normalize_path() {
  local path="$1"
  (cd "$path" && pwd -P)
}

require_command() {
  local name="$1"
  command -v "$name" >/dev/null 2>&1 || abort "E_TOOL_MISSING" "Required command '$name' not found."
}

print_paths_section() {
  local title="$1"
  shift
  printf '[update][E_WORKTREE_DIRTY] %s\n' "$title" >&2
  if (($# == 0)); then
    printf '  - (none)\n' >&2
    return
  fi
  local entry
  for entry in "$@"; do
    printf "  - '%s'\n" "$entry" >&2
  done
}

show_dirty_abort() {
  local status_output="$1"
  local ignore_untracked="${2:-0}"
  local tracked=()
  local untracked=()
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local code="${line:0:2}"
    local path="${line:3}"
    if [[ "$code" == "??" ]]; then
      untracked+=("$path")
    else
      tracked+=("[$code] $path")
    fi
  done <<< "$status_output"

  printf '[update][E_WORKTREE_DIRTY] Local changes detected; update aborted to protect your files.\n' >&2
  print_paths_section "Tracked changes:" "${tracked[@]}"
  if (( ignore_untracked == 1 )); then
    printf '[update][E_WORKTREE_DIRTY] Untracked-path preflight checks were disabled by --force.\n' >&2
    printf '[update][E_WORKTREE_DIRTY] Remediation: commit/stash tracked changes, then rerun.\n' >&2
  else
    print_paths_section "Untracked paths:" "${untracked[@]}"
    printf '[update][E_WORKTREE_DIRTY] Ignored paths are excluded by policy (P2=B).\n' >&2
    printf '[update][E_WORKTREE_DIRTY] Remediation: commit/stash tracked changes and move or remove untracked paths, then rerun.\n' >&2
  fi
  exit 1
}

validate_git_state() {
  require_command git

  local inside
  inside="$(git -C "$ROOT_DIR" rev-parse --is-inside-work-tree 2>/dev/null || true)"
  [[ "$inside" == "true" ]] || abort "E_NOT_GIT_REPO" "Script root '$ROOT_DIR' is not inside a git worktree."

  local top
  top="$(git -C "$ROOT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
  [[ -n "$top" ]] || abort "E_GIT_TOPLEVEL_UNRESOLVED" "Failed to resolve git top-level for '$ROOT_DIR'."
  [[ "$(normalize_path "$top")" == "$ROOT_DIR" ]] || abort "E_WRONG_REPO_ROOT" "Run updater from repository root '$ROOT_DIR' only."

  local branch
  branch="$(git -C "$ROOT_DIR" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
  [[ -n "$branch" ]] || abort "E_DETACHED_HEAD" "Detached HEAD detected. Checkout a branch and rerun."

  local upstream
  upstream="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  [[ -n "$upstream" ]] || abort "E_NO_UPSTREAM" "No upstream configured for branch '$branch'. Configure upstream first."

  local git_dir
  git_dir="$(git -C "$ROOT_DIR" rev-parse --git-dir 2>/dev/null || true)"
  [[ -n "$git_dir" ]] || abort "E_GIT_DIR_UNRESOLVED" "Failed to resolve git dir."
  if [[ "$git_dir" != /* ]]; then
    git_dir="${ROOT_DIR}/${git_dir}"
  fi

  [[ -f "$git_dir/MERGE_HEAD" ]] && abort "E_GIT_MERGE_IN_PROGRESS" "Merge in progress. Resolve or abort merge before updating."
  [[ -f "$git_dir/CHERRY_PICK_HEAD" ]] && abort "E_GIT_CHERRY_PICK_IN_PROGRESS" "Cherry-pick in progress. Resolve or abort before updating."
  [[ -f "$git_dir/REVERT_HEAD" ]] && abort "E_GIT_REVERT_IN_PROGRESS" "Revert in progress. Resolve or abort before updating."
  [[ -d "$git_dir/rebase-apply" || -d "$git_dir/rebase-merge" ]] && abort "E_GIT_REBASE_IN_PROGRESS" "Rebase in progress. Complete or abort rebase before updating."
  [[ -f "$git_dir/BISECT_LOG" ]] && abort "E_GIT_BISECT_IN_PROGRESS" "Bisect in progress. Finish bisect before updating."

  local status_output untracked_mode
  untracked_mode="all"
  if (( ALLOW_UNTRACKED == 1 )); then
    untracked_mode="no"
  fi
  status_output="$(git -C "$ROOT_DIR" status --porcelain=v1 --untracked-files="$untracked_mode")"
  [[ -z "$status_output" ]] || show_dirty_abort "$status_output" "$ALLOW_UNTRACKED"
}

ensure_nodeenv() {
  local nodeenv_args=()
  if [[ -e "$NODEENV_DIR" && ! -d "$NODEENV_DIR" ]]; then
    abort "E_NODEENV_PATH_INVALID" "Expected '$NODEENV_DIR' to be a directory."
  fi

  if [[ -x "$NODE_BIN" && -x "$NPM_BIN" ]]; then
    local existing
    existing="$("$NODE_BIN" -v 2>/dev/null | tr -d '\r\n')"
    existing="${existing#v}"
    [[ -n "$existing" ]] || abort "E_NODEENV_CORRUPT" "Found '$NODEENV_DIR', but node version could not be determined."
    if [[ "$existing" != "$NODE_VERSION" ]]; then
      abort "E_NODE_VERSION_MISMATCH" "'$NODEENV_DIR' contains Node.js $existing, but CODEX_NODE_VERSION=$NODE_VERSION. Set CODEX_NODE_VERSION=$existing or recreate '$NODEENV_DIR'."
    fi
    return 0
  fi

  if [[ -d "$NODEENV_DIR" ]]; then
    log "nodeenv appears incomplete under ${NODEENV_DIR}; attempting repair with Node.js ${NODE_VERSION} ..."
    nodeenv_args+=(--force)
  else
    log "npm missing; provisioning Node.js ${NODE_VERSION} into ${NODEENV_DIR} ..."
  fi

  "$UV_BIN" tool run --from nodeenv nodeenv "${nodeenv_args[@]}" -n "$NODE_VERSION" "$NODEENV_DIR" || abort "E_NODEENV_INSTALL_FAILED" "nodeenv install failed while preparing updater prerequisites."
  [[ -x "$NODE_BIN" ]] || abort "E_NODEENV_INCOMPLETE" "nodeenv completed, but node is missing at '$NODE_BIN'."
  [[ -x "$NPM_BIN" ]] || abort "E_NODEENV_INCOMPLETE" "nodeenv completed, but npm is missing at '$NPM_BIN'."

  local repaired
  repaired="$("$NODE_BIN" -v 2>/dev/null | tr -d '\r\n')"
  repaired="${repaired#v}"
  [[ -n "$repaired" ]] || abort "E_NODEENV_CORRUPT" "Found '$NODEENV_DIR', but node version could not be determined after repair."
  if [[ "$repaired" != "$NODE_VERSION" ]]; then
    abort "E_NODE_VERSION_MISMATCH" "'$NODEENV_DIR' contains Node.js $repaired, but CODEX_NODE_VERSION=$NODE_VERSION. Set CODEX_NODE_VERSION=$repaired or recreate '$NODEENV_DIR'."
  fi
}

resolve_cuda_variant_override() {
  local requested_variant="${CODEX_CUDA_VARIANT:-}"
  local requested_variant_lc
  requested_variant_lc="$(printf '%s' "$requested_variant" | tr '[:upper:]' '[:lower:]')"
  case "$requested_variant_lc" in
    "")
      ;;
    12.6|cu126)
      printf 'cu126\n'
      return 0
      ;;
    12.8|cu128)
      printf 'cu128\n'
      return 0
      ;;
    13|cu130)
      printf 'cu130\n'
      return 0
      ;;
    *)
      abort "E_INVALID_CUDA_VARIANT" "Invalid CODEX_CUDA_VARIANT='${requested_variant}'. Expected 12.6|12.8|13|cu126|cu128|cu130."
      ;;
  esac
}

validate_torch_mode_guard() {
  local mode_raw="${CODEX_TORCH_MODE:-auto}"
  local mode_lc
  mode_lc="$(printf '%s' "$mode_raw" | tr '[:upper:]' '[:lower:]')"
  if [[ "$mode_lc" == "custom" ]]; then
    abort "E_CUSTOM_TORCH_MODE_UNSUPPORTED" \
      "CODEX_TORCH_MODE=custom is not supported by update-webui.sh. Aborting before dependency sync to avoid overwriting a custom PyTorch install."
  fi
}

resolve_torch_backend() {
  local variant_backend
  variant_backend="$(resolve_cuda_variant_override)"

  local requested="${CODEX_TORCH_BACKEND:-}"
  if [[ -n "$requested" ]]; then
    case "$requested" in
      cpu|cu126|cu128|cu130|rocm64)
        printf '%s\n' "$requested"
        return 0
        ;;
      *)
        abort "E_INVALID_TORCH_BACKEND" "Invalid CODEX_TORCH_BACKEND='$requested'. Expected cpu|cu126|cu128|cu130|rocm64."
        ;;
    esac
  fi

  if [[ -n "$variant_backend" ]]; then
    printf '%s\n' "$variant_backend"
    return 0
  fi

  [[ -x "$PYTHON_BIN" ]] || abort "E_PYTHON_MISSING" "Python runtime missing at '$PYTHON_BIN'. Run install-webui.sh first."

  local detected=""
  set +e
  detected="$("$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("torch")
if spec is None:
    raise SystemExit(2)

import torch  # type: ignore

version = str(getattr(torch, "__version__", "")).lower()
if "+cu126" in version:
    print("cu126")
    raise SystemExit(0)
if "+cu128" in version:
    print("cu128")
    raise SystemExit(0)
if "+cu130" in version:
    print("cu130")
    raise SystemExit(0)
if "+rocm" in version:
    print("rocm64")
    raise SystemExit(0)
if "+cpu" in version:
    print("cpu")
    raise SystemExit(0)

torch_version = getattr(torch, "version", None)
cuda_version = getattr(torch_version, "cuda", None)
hip_version = getattr(torch_version, "hip", None)

if hip_version:
    print("rocm64")
    raise SystemExit(0)

if cuda_version:
    parts = [segment for segment in str(cuda_version).split(".") if segment.isdigit()]
    major = int(parts[0]) if parts else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    if major >= 13:
        print("cu130")
    elif major == 12 and minor <= 7:
        print("cu126")
    elif major == 12:
        print("cu128")
    else:
        print("cu128")
    raise SystemExit(0)

print("cpu")
PY
)"
  local detect_status=$?
  set -e

  if [[ "$detect_status" -eq 0 && -n "$detected" ]]; then
    printf '%s\n' "$detected"
    return 0
  fi

  local fallback_backend
  fallback_backend="$(resolve_torch_backend_from_system)"
  [[ -n "$fallback_backend" ]] || abort "E_TORCH_BACKEND_UNRESOLVED" "Could not determine torch backend extra. Set CODEX_TORCH_BACKEND explicitly."
  printf '%s\n' "$fallback_backend"
}

resolve_torch_backend_from_system() {
  if [[ -e /dev/kfd ]] && { command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; }; then
    printf 'rocm64\n'
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'cpu\n'
    return 0
  fi

  local cuda_version
  cuda_version="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  if [[ -z "$cuda_version" ]]; then
    printf 'cu128\n'
    return 0
  fi

  local driver_version driver_major
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  driver_major="${driver_version%%.*}"
  if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major < 525 )); then
    printf 'cpu\n'
    return 0
  fi

  local cuda_major cuda_minor
  IFS='.' read -r cuda_major cuda_minor <<< "$cuda_version"
  if [[ ! "$cuda_major" =~ ^[0-9]+$ ]]; then
    printf 'cu128\n'
    return 0
  fi
  [[ "$cuda_minor" =~ ^[0-9]+$ ]] || cuda_minor=0

  if (( cuda_major >= 13 )); then
    if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major >= 580 )); then
      printf 'cu130\n'
    else
      printf 'cu128\n'
    fi
  elif (( cuda_major == 12 && cuda_minor <= 7 )); then
    printf 'cu126\n'
  else
    printf 'cu128\n'
  fi
}

prepare_refresh_requirements() {
  [[ -x "$UV_BIN" ]] || abort "E_UV_MISSING" "uv not found at '$UV_BIN'. Run install-webui.sh first."
  [[ -x "$PYTHON_BIN" ]] || abort "E_PYTHON_MISSING" "Python runtime missing at '$PYTHON_BIN'. Run install-webui.sh first."
  ensure_nodeenv
  [[ -x "$NPM_BIN" ]] || abort "E_NPM_MISSING" "npm not found at '$NPM_BIN'."
  [[ -f "${INTERFACE_DIR}/package-lock.json" ]] || abort "E_NPM_LOCK_MISSING" "Lock-preserving update requires '${INTERFACE_DIR}/package-lock.json'."
}

refresh_environment() {
  local torch_backend="$1"
  export CODEX_ROOT="$ROOT_DIR"
  export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  export CODEX_FFMPEG_VERSION

  log "Refreshing Python dependencies with backend extra '${torch_backend}' ..."
  "$UV_BIN" sync --locked --extra "$torch_backend" || abort "E_UV_SYNC_FAILED" "uv sync failed."

  log "Refreshing runtime assets (ffmpeg/ffprobe + RIFE model) ..."
  "$PYTHON_BIN" - <<'PY' || abort "E_RUNTIME_PROVISION_FAILED" "Runtime dependency provisioning failed. Re-run install-webui.sh and retry."
import os
import sys

from apps.backend.video.runtime_dependencies import ensure_ffmpeg_binaries, ensure_rife_model_file

ffmpeg_version = os.environ.get("CODEX_FFMPEG_VERSION") or "7.0.2"
try:
    binaries = ensure_ffmpeg_binaries(version=ffmpeg_version, no_symlinks=True)
    model = ensure_rife_model_file()
except Exception as error:  # pragma: no cover
    print(f"[update][E_RUNTIME_PROVISION_FAILED] {error}", file=sys.stderr)
    raise SystemExit(1)
print(f"[update] ffmpeg: {binaries['ffmpeg']}")
print(f"[update] ffprobe: {binaries['ffprobe']}")
print(f"[update] RIFE model: {model}")
PY

  log "Refreshing frontend dependencies with npm ci ..."
  (cd "$INTERFACE_DIR" && "$NPM_BIN" ci --no-audit --no-fund) || abort "E_NPM_CI_FAILED" "npm ci failed."
}

main() {
  while (($# > 0)); do
    case "$1" in
      --help|-h)
        usage
        exit 0
        ;;
      --force|-f)
        ALLOW_UNTRACKED=1
        ;;
      *)
        abort "E_BAD_ARGS" "Unknown argument '$1'. Use --help."
        ;;
    esac
    shift
  done

  validate_git_state
  if (( ALLOW_UNTRACKED == 1 )); then
    log "Force mode enabled: untracked paths are ignored in dirty check."
  fi

  validate_torch_mode_guard

  local head_before
  head_before="$(git -C "$ROOT_DIR" rev-parse HEAD)"

  log "Fetching upstream refs ..."
  git -C "$ROOT_DIR" fetch --prune || abort "E_FETCH_FAILED" "git fetch --prune failed."

  local counts ahead behind
  counts="$(git -C "$ROOT_DIR" rev-list --left-right --count HEAD...@{u} 2>/dev/null || true)"
  [[ -n "$counts" ]] || abort "E_UPSTREAM_COUNT_FAILED" "Failed to compute ahead/behind status against upstream."
  read -r ahead behind <<< "$counts"
  [[ "$ahead" =~ ^[0-9]+$ && "$behind" =~ ^[0-9]+$ ]] || abort "E_UPSTREAM_COUNT_FAILED" "Invalid ahead/behind status from upstream: '$counts'."

  if (( ahead > 0 && behind > 0 )); then
    abort "E_DIVERGED" "Branch is diverged from upstream (ahead=${ahead}, behind=${behind}). Reconcile history manually first."
  fi
  if (( ahead > 0 )); then
    abort "E_AHEAD_OF_UPSTREAM" "Local branch is ahead of upstream by ${ahead} commit(s). Push/rebase before running updater."
  fi

  prepare_refresh_requirements
  local torch_backend
  torch_backend="$(resolve_torch_backend)"
  log "Resolved torch backend extra: ${torch_backend}"

  if (( behind == 0 )); then
    log "Already up to date. No commits pulled; running environment refresh."
  else
    log "Pulling updates (ff-only) ..."
    git -C "$ROOT_DIR" pull --ff-only || abort "E_PULL_FAILED" "git pull --ff-only failed."

    local head_after
    head_after="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    if [[ "$head_after" == "$head_before" ]]; then
      log "No commit change after pull. Running environment refresh anyway."
    else
      log "Pulled new commits from upstream."
    fi
  fi

  refresh_environment "$torch_backend"
  log "Update completed successfully."
}

main "$@"
