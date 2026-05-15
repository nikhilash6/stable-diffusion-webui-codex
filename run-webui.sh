#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UV_BIN="${ROOT_DIR}/.uv/bin/uv"
VENV_DIR="${CODEX_VENV_DIR:-${ROOT_DIR}/.venv}"
PY_BIN="${PYTHON:-${VENV_DIR}/bin/python}"
API_ENTRYPOINT="${ROOT_DIR}/apps/backend/interfaces/api/run_api.py"
UI_DIR="${ROOT_DIR}/apps/interface"

NODEENV_DIR="${ROOT_DIR}/.nodeenv"
NODEENV_BIN_DIR="${NODEENV_DIR}/bin"
NODEENV_NODE="${NODEENV_BIN_DIR}/node"
NODEENV_NPM="${NODEENV_BIN_DIR}/npm"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: ./run-webui.sh

Starts:
  - Backend API (FastAPI) from ${API_ENTRYPOINT}
  - Frontend UI (Vite) from ${UI_DIR}

Backend args:
  - Any extra args are forwarded to the backend entrypoint (e.g. '--lora-apply-mode online').

Launcher args:
  - '--pytorch-cuda-alloc-conf <value>': sets 'PYTORCH_CUDA_ALLOC_CONF' for the backend process (requires restart).
  - '--enable-default-pytorch-cuda-alloc-conf' / '--disable-default-pytorch-cuda-alloc-conf':
      toggles default allocator tuning when 'PYTORCH_CUDA_ALLOC_CONF' is unset.
  - '--enable-cuda-malloc' / '--disable-cuda-malloc':
      toggles backend '--cuda-malloc' forwarding and enforces allocator backend 'backend:cudaMallocAsync'.

Environment overrides:
  - CODEX_VENV_DIR   (default: \$CODEX_ROOT/.venv)
  - PYTHON           (default: \$CODEX_VENV_DIR/bin/python)
  - CODEX_MAIN_DEVICE (auto|cuda|cpu|mps|xpu|directml; global device authority mirrored to core/TE/VAE when explicit component values are unset)
  - CODEX_MOUNT_DEVICE (auto|cuda|cpu|mps|xpu|directml; model mount/load device authority; defaults to resolved main device)
  - CODEX_OFFLOAD_DEVICE (auto|cuda|cpu|mps|xpu|directml; model offload target authority; defaults to cpu)
  - CODEX_CORE_DEVICE (auto|cuda|cpu|mps|xpu|directml; required when no saved setting exists)
  - CODEX_TE_DEVICE (auto|cuda|cpu|mps|xpu|directml; required when no saved setting exists)
  - CODEX_VAE_DEVICE (auto|cuda|cpu|mps|xpu|directml; required when no saved setting exists)
  - CODEX_LORA_APPLY_MODE (default: online)
    Accepted values: 'merge' or 'online'
  - CODEX_LORA_ONLINE_MATH (weight_merge|activation; default: weight_merge)
  - CODEX_LORA_MERGE_MODE (fast|precise; default: fast)
  - CODEX_LORA_REFRESH_SIGNATURE (structural|content_sha256; default: content_sha256)
  - CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF (1|0; default: 1)
  - CODEX_CUDA_MALLOC (1|0; default: 0)
  - PYTORCH_CUDA_ALLOC_CONF (PyTorch allocator tuning; optional)
  - API_PORT_OVERRIDE / API_PORT / WEB_PORT (advanced; ports are auto-paired when unset)
EOF
  exit 0
fi

if [[ ! -x "${PY_BIN}" ]]; then
  echo "Error: expected Python at '${PY_BIN}'." >&2
  echo "Run: ./install-webui.sh" >&2
  if [[ -x "${UV_BIN}" ]]; then
    echo "Found uv at '${UV_BIN}', but the project environment is missing." >&2
  else
    echo "uv is missing at '${UV_BIN}'." >&2
  fi
  exit 1
fi

if [[ ! -f "${API_ENTRYPOINT}" ]]; then
  echo "Error: backend entrypoint not found: '${API_ENTRYPOINT}'." >&2
  exit 1
fi

if [[ ! -d "${UI_DIR}" ]]; then
  echo "Error: frontend directory not found: '${UI_DIR}'." >&2
  exit 1
fi

NPM_BIN="npm"
if [[ -e "${NODEENV_DIR}" && ! -d "${NODEENV_DIR}" ]]; then
  echo "Error: expected '${NODEENV_DIR}' to be a directory (nodeenv), but found a non-directory path." >&2
  exit 1
fi

if [[ -d "${NODEENV_DIR}" ]]; then
  if [[ ! -x "${NODEENV_NODE}" || ! -x "${NODEENV_NPM}" ]]; then
    echo "Error: '${NODEENV_DIR}' exists, but nodeenv is incomplete (missing node and/or npm)." >&2
    echo "Delete '${NODEENV_DIR}' and re-run: ./install-webui.sh" >&2
    exit 1
  fi
  export PATH="${NODEENV_BIN_DIR}:${PATH}"
  NPM_BIN="${NODEENV_NPM}"
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Error: missing Node.js ('node') and/or npm ('npm')." >&2
  echo "Run: ./install-webui.sh" >&2
  echo "Expected: '${NODEENV_DIR}' (repo-local Node.js via nodeenv)." >&2
  exit 1
fi

if [[ ! -d "${UI_DIR}/node_modules" ]]; then
  echo "Error: '${UI_DIR}/node_modules' missing." >&2
  echo "Run: ./install-webui.sh" >&2
  echo "Or: (cd '${UI_DIR}' && ${NPM_BIN} install)" >&2
  exit 1
fi

export CODEX_ROOT="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export FORCE_COLOR="${FORCE_COLOR:-1}"

DEFAULT_PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"

# Launcher-only arg parsing (strip args that are not backend CLI flags).
api_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pytorch-cuda-alloc-conf)
      if [[ $# -lt 2 ]]; then
        echo "Error: --pytorch-cuda-alloc-conf requires a value." >&2
        exit 2
      fi
      export PYTORCH_CUDA_ALLOC_CONF="$2"
      shift 2
      ;;
    --pytorch-cuda-alloc-conf=*)
      export PYTORCH_CUDA_ALLOC_CONF="${1#*=}"
      shift
      ;;
    --enable-default-pytorch-cuda-alloc-conf)
      export CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF="1"
      shift
      ;;
    --disable-default-pytorch-cuda-alloc-conf)
      export CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF="0"
      shift
      ;;
    --enable-cuda-malloc)
      export CODEX_CUDA_MALLOC="1"
      shift
      ;;
    --disable-cuda-malloc)
      export CODEX_CUDA_MALLOC="0"
      shift
      ;;
    *)
      api_args+=("$1")
      shift
      ;;
  esac
done
set -- "${api_args[@]}"

is_truthy() {
  local value
  value="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "${value}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

sanitize_allocator_env_contract() {
  local supported_toggle_key="CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF"
  local key
  local -a unsupported_keys=()
  while IFS='=' read -r key _; do
    if [[ "${key}" == PYTORCH_* && "${key}" == *_ALLOC_CONF && "${key}" != "PYTORCH_CUDA_ALLOC_CONF" ]]; then
      unsupported_keys+=("${key}")
      continue
    fi
    if [[ "${key}" == CODEX_ENABLE_DEFAULT_PYTORCH_* && "${key}" == *_ALLOC_CONF && "${key}" != "${supported_toggle_key}" ]]; then
      unsupported_keys+=("${key}")
    fi
  done < <(env)
  if (( ${#unsupported_keys[@]} > 0 )); then
    echo "[webui] Error: unsupported allocator env key(s): ${unsupported_keys[*]}." >&2
    echo "[webui] Error: supported allocator env keys are PYTORCH_CUDA_ALLOC_CONF and ${supported_toggle_key}." >&2
    exit 2
  fi
}

ensure_cuda_malloc_allocator_backend() {
  local target_backend="cudaMallocAsync"
  local target_backend_norm="cudamallocasync"
  local raw_conf="${PYTORCH_CUDA_ALLOC_CONF:-}"
  local trimmed_conf
  trimmed_conf="$(echo "${raw_conf}" | xargs)"
  if [[ -z "${trimmed_conf}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="backend:${target_backend}"
    return 0
  fi

  local has_backend=0
  local entry token key value value_norm
  local -a entries=()
  local -a normalized_entries=()
  IFS=',' read -r -a entries <<<"${trimmed_conf}"
  for entry in "${entries[@]}"; do
    token="$(echo "${entry}" | xargs)"
    if [[ -z "${token}" ]]; then
      continue
    fi
    if [[ "${token}" != *:* ]]; then
      echo "[webui] Error: invalid PYTORCH_CUDA_ALLOC_CONF entry '${token}' (expected key:value)." >&2
      exit 2
    fi
    key="${token%%:*}"
    value="${token#*:}"
    key="$(echo "${key}" | xargs)"
    value="$(echo "${value}" | xargs)"
    if [[ -z "${key}" || -z "${value}" ]]; then
      echo "[webui] Error: invalid PYTORCH_CUDA_ALLOC_CONF entry '${token}' (expected non-empty key:value)." >&2
      exit 2
    fi
    normalized_entries+=("${key}:${value}")
    if [[ "$(echo "${key}" | tr '[:upper:]' '[:lower:]')" == "backend" ]]; then
      if (( has_backend == 1 )); then
        echo "[webui] Error: invalid PYTORCH_CUDA_ALLOC_CONF (multiple backend entries)." >&2
        exit 2
      fi
      has_backend=1
      value_norm="$(echo "${value}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
      if [[ "${value_norm}" != "${target_backend_norm}" ]]; then
        echo "[webui] Error: CODEX_CUDA_MALLOC/--cuda-malloc requires backend:cudaMallocAsync in PYTORCH_CUDA_ALLOC_CONF, got backend:${value}." >&2
        exit 2
      fi
    fi
  done

  if (( has_backend == 0 )); then
    normalized_entries+=("backend:${target_backend}")
  fi
  export PYTORCH_CUDA_ALLOC_CONF="$(IFS=,; echo "${normalized_entries[*]}")"
}

normalize_device_choice() {
  local raw="${1:-}"
  raw="$(echo "${raw}" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "${raw}" in
    "") echo "" ;;
    auto) echo "auto" ;;
    cuda|gpu) echo "cuda" ;;
    cpu) echo "cpu" ;;
    mps) echo "mps" ;;
    xpu) echo "xpu" ;;
    directml|dml) echo "directml" ;;
    *) return 1 ;;
  esac
}

has_backend_flag() {
  local name="$1"
  shift || true
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${name}" || "${arg}" == "${name}="* ]]; then
      return 0
    fi
  done
  return 1
}

get_backend_flag_value() {
  local name="$1"
  shift || true
  while [[ $# -gt 0 ]]; do
    local arg="$1"
    shift
    if [[ "${arg}" == "${name}="* ]]; then
      echo "${arg#*=}"
      return 0
    fi
    if [[ "${arg}" == "${name}" ]]; then
      if [[ $# -lt 1 ]]; then
        return 1
      fi
      echo "$1"
      return 0
    fi
  done
  return 1
}

read_saved_device() {
  local key="$1"
  "${PY_BIN}" - "$ROOT_DIR/apps/settings_values.json" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"[webui] Error: invalid JSON in {path}: {exc}", file=sys.stderr)
    raise SystemExit(2)
if not isinstance(data, dict):
    print(f"[webui] Error: invalid settings file (expected object): {path}", file=sys.stderr)
    raise SystemExit(2)
value = data.get(key)
if value is None:
    raise SystemExit(0)
text = str(value).strip()
if text:
    print(text)
PY
}

prompt_device() {
  local label="$1"
  local choices=(auto cuda cpu mps xpu directml)
  echo "" >&2
  echo "[webui] Missing ${label}. Choose one:" >&2
  local i
  for i in "${!choices[@]}"; do
    printf "  %d) %s\n" "$((i + 1))" "${choices[i]}" >&2
  done
  while true; do
    local choice=""
    read -r -p "Select ${label} (1-${#choices[@]} or value, q to abort): " choice
    local lowered
    lowered="$(echo "${choice}" | tr '[:upper:]' '[:lower:]' | xargs)"
    if [[ "${lowered}" == "q" || "${lowered}" == "quit" || "${lowered}" == "exit" ]]; then
      echo "[webui] Aborted." >&2
      exit 130
    fi
    choice="$(normalize_device_choice "${choice}")" || choice=""
    if [[ -z "${choice}" ]]; then
      echo "Invalid choice. Allowed: ${choices[*]}" >&2
      continue
    fi
    echo "${choice}"
    return 0
  done
}

# Ensure backend device defaults are explicit (no silent fallbacks). We prefer:
# 1) explicit backend args
# 2) CODEX_*_DEVICE env vars
# 3) persisted values in apps/settings_values.json
# 4) interactive prompt (TTY only)
core_device=""
te_device=""
vae_device=""
main_device=""

if has_backend_flag "--main-device" "$@"; then
  raw_main_arg="$(get_backend_flag_value "--main-device" "$@" || true)"
  if [[ -z "${raw_main_arg}" ]]; then
    echo "[webui] Error: --main-device requires a value (auto|cuda|cpu|mps|xpu|directml)." >&2
    exit 2
  fi
  main_device="$(normalize_device_choice "${raw_main_arg}")" || {
    echo "[webui] Error: invalid --main-device value '${raw_main_arg}'." >&2
    echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
    exit 2
  }
else
  raw_main_env="$(echo "${CODEX_MAIN_DEVICE:-}" | xargs)"
  if [[ -n "${raw_main_env}" ]]; then
    main_device="$(normalize_device_choice "${raw_main_env}")" || {
      echo "[webui] Error: invalid CODEX_MAIN_DEVICE='${CODEX_MAIN_DEVICE}'." >&2
      echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
      exit 2
    }
  fi
  if [[ -z "${main_device}" ]]; then
    main_device="$(read_saved_device "codex_main_device")"
    if [[ -n "${main_device}" ]]; then
      main_device="$(normalize_device_choice "${main_device}")" || {
        echo "[webui] Error: invalid saved setting codex_main_device='${main_device}' in apps/settings_values.json." >&2
        echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
        exit 2
      }
    fi
  fi
fi

if ! has_backend_flag "--core-device" "$@"; then
  if [[ -n "${main_device}" ]]; then
    core_device="${main_device}"
  else
    raw_core_env="$(echo "${CODEX_CORE_DEVICE:-}" | xargs)"
    if [[ -n "${raw_core_env}" ]]; then
      core_device="$(normalize_device_choice "${raw_core_env}")" || {
        echo "[webui] Error: invalid CODEX_CORE_DEVICE='${CODEX_CORE_DEVICE}'." >&2
        echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
        exit 2
      }
    fi
    if [[ -z "${core_device}" ]]; then
      core_device="$(read_saved_device "codex_core_device")"
      if [[ -n "${core_device}" ]]; then
        core_device="$(normalize_device_choice "${core_device}")" || {
          echo "[webui] Error: invalid saved setting codex_core_device='${core_device}' in apps/settings_values.json." >&2
          echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
          exit 2
        }
      fi
    fi
  fi
fi

if ! has_backend_flag "--te-device" "$@"; then
  if [[ -n "${main_device}" ]]; then
    te_device="${main_device}"
  else
    raw_te_env="$(echo "${CODEX_TE_DEVICE:-}" | xargs)"
    if [[ -n "${raw_te_env}" ]]; then
      te_device="$(normalize_device_choice "${raw_te_env}")" || {
        echo "[webui] Error: invalid CODEX_TE_DEVICE='${CODEX_TE_DEVICE}'." >&2
        echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
        exit 2
      }
    fi
    if [[ -z "${te_device}" ]]; then
      te_device="$(read_saved_device "codex_te_device")"
      if [[ -n "${te_device}" ]]; then
        te_device="$(normalize_device_choice "${te_device}")" || {
          echo "[webui] Error: invalid saved setting codex_te_device='${te_device}' in apps/settings_values.json." >&2
          echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
          exit 2
        }
      fi
    fi
  fi
fi

if ! has_backend_flag "--vae-device" "$@"; then
  if [[ -n "${main_device}" ]]; then
    vae_device="${main_device}"
  else
    raw_vae_env="$(echo "${CODEX_VAE_DEVICE:-}" | xargs)"
    if [[ -n "${raw_vae_env}" ]]; then
      vae_device="$(normalize_device_choice "${raw_vae_env}")" || {
        echo "[webui] Error: invalid CODEX_VAE_DEVICE='${CODEX_VAE_DEVICE}'." >&2
        echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
        exit 2
      }
    fi
    if [[ -z "${vae_device}" ]]; then
      vae_device="$(read_saved_device "codex_vae_device")"
      if [[ -n "${vae_device}" ]]; then
        vae_device="$(normalize_device_choice "${vae_device}")" || {
          echo "[webui] Error: invalid saved setting codex_vae_device='${vae_device}' in apps/settings_values.json." >&2
          echo "[webui] Allowed: auto, cuda, cpu, mps, xpu, directml." >&2
          exit 2
        }
      fi
    fi
  fi
fi

need_prompt=0
if [[ -z "${core_device}" ]] && ! has_backend_flag "--core-device" "$@"; then
  need_prompt=1
fi
if [[ -z "${te_device}" ]] && ! has_backend_flag "--te-device" "$@"; then
  need_prompt=1
fi
if [[ -z "${vae_device}" ]] && ! has_backend_flag "--vae-device" "$@"; then
  need_prompt=1
fi

if (( need_prompt == 1 )); then
  if [[ ! -t 0 || ! -t 1 ]]; then
    echo "[webui] Error: backend device defaults are not configured and no TTY is available for prompting." >&2
    echo "[webui] Provide flags: --main-device/--core-device/--te-device/--vae-device, or set CODEX_MAIN_DEVICE/CODEX_CORE_DEVICE/CODEX_TE_DEVICE/CODEX_VAE_DEVICE." >&2
    exit 2
  fi
  if [[ -z "${core_device}" ]] && ! has_backend_flag "--core-device" "$@"; then
    core_device="$(prompt_device "CORE_DEVICE")"
  fi
  if [[ -z "${te_device}" ]] && ! has_backend_flag "--te-device" "$@"; then
    te_device="$(prompt_device "TE_DEVICE")"
  fi
  if [[ -z "${vae_device}" ]] && ! has_backend_flag "--vae-device" "$@"; then
    vae_device="$(prompt_device "VAE_DEVICE")"
  fi
fi

if [[ -n "${main_device}" ]] && ! has_backend_flag "--main-device" "$@"; then
  set -- "--main-device=${main_device}" "$@"
fi
if [[ -n "${core_device}" ]] && ! has_backend_flag "--core-device" "$@"; then
  set -- "--core-device=${core_device}" "$@"
fi
if [[ -n "${te_device}" ]] && ! has_backend_flag "--te-device" "$@"; then
  set -- "--te-device=${te_device}" "$@"
fi
if [[ -n "${vae_device}" ]] && ! has_backend_flag "--vae-device" "$@"; then
  set -- "--vae-device=${vae_device}" "$@"
fi

sanitize_allocator_env_contract
default_alloc_conf_enabled="${CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF:-1}"

if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && is_truthy "${default_alloc_conf_enabled}"; then
  export PYTORCH_CUDA_ALLOC_CONF="${DEFAULT_PYTORCH_CUDA_ALLOC_CONF}"
fi

cuda_malloc_requested=0
if has_backend_flag "--cuda-malloc" "$@"; then
  cuda_malloc_requested=1
fi
if is_truthy "${CODEX_CUDA_MALLOC:-0}"; then
  cuda_malloc_requested=1
  if ! has_backend_flag "--cuda-malloc" "$@"; then
    set -- "--cuda-malloc" "$@"
  fi
fi

if (( cuda_malloc_requested == 1 )); then
  ensure_cuda_malloc_allocator_backend
fi

is_uint() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

assert_port() {
  local label="$1"
  local value="$2"
  if ! is_uint "${value}"; then
    echo "Error: ${label} must be an integer; got '${value}'." >&2
    exit 1
  fi
  if (( value < 1 || value > 65535 )); then
    echo "Error: ${label} must be in 1..65535; got '${value}'." >&2
    exit 1
  fi
}

port_free() {
  local port="$1"
  "${PY_BIN}" - "$port" <<'PY'
import errno
import socket
import sys

port = int(sys.argv[1])
targets = [
    (socket.AF_INET, ("0.0.0.0", port)),
    (socket.AF_INET, ("127.0.0.1", port)),
    (socket.AF_INET6, ("::", port, 0, 0)),
    (socket.AF_INET6, ("::1", port, 0, 0)),
]
for family, addr in targets:
    try:
        s = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        continue
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(addr)
    except OSError as exc:
        if getattr(exc, "errno", None) in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
            continue
        raise SystemExit(1)
    finally:
        try:
            s.close()
        except Exception:
            pass
raise SystemExit(0)
PY
}

pick_ports() {
  local user_api="${API_PORT_OVERRIDE:-${API_PORT:-}}"
  local user_web="${WEB_PORT:-}"

  if [[ -n "${user_api}" && -n "${user_web}" ]]; then
    assert_port "API_PORT_OVERRIDE/API_PORT" "${user_api}"
    assert_port "WEB_PORT" "${user_web}"
    if ! port_free "${user_api}" || ! port_free "${user_web}"; then
      echo "Error: requested ports are busy: api=${user_api} web=${user_web}." >&2
      exit 1
    fi
    API_PORT_OVERRIDE="${user_api}"
    WEB_PORT="${user_web}"
    API_PORT="${user_api}"
    return 0
  fi

  if [[ -n "${user_api}" ]]; then
    assert_port "API_PORT_OVERRIDE/API_PORT" "${user_api}"
    local derived_web=$(( user_api + 10 ))
    assert_port "WEB_PORT (derived)" "${derived_web}"
    if ! port_free "${user_api}" || ! port_free "${derived_web}"; then
      echo "Error: requested ports are busy: api=${user_api} web=${derived_web}." >&2
      exit 1
    fi
    API_PORT_OVERRIDE="${user_api}"
    WEB_PORT="${derived_web}"
    API_PORT="${user_api}"
    return 0
  fi

  if [[ -n "${user_web}" ]]; then
    assert_port "WEB_PORT" "${user_web}"
    local derived_api=$(( user_web - 10 ))
    assert_port "API_PORT (derived)" "${derived_api}"
    if ! port_free "${derived_api}" || ! port_free "${user_web}"; then
      echo "Error: requested ports are busy: api=${derived_api} web=${user_web}." >&2
      exit 1
    fi
    API_PORT_OVERRIDE="${derived_api}"
    WEB_PORT="${user_web}"
    API_PORT="${derived_api}"
    return 0
  fi

  local candidates=(
    "7850 7860"
    "17850 17860"
    "27850 27860"
  )
  local api_port=""
  local web_port=""
  for pair in "${candidates[@]}"; do
    read -r api_port web_port <<<"${pair}"
    if port_free "${api_port}" && port_free "${web_port}"; then
      API_PORT_OVERRIDE="${api_port}"
      WEB_PORT="${web_port}"
      API_PORT="${api_port}"
      return 0
    fi
  done

  echo "Error: no free port pairs for API/UI." >&2
  echo "Tried: 7850/7860, 17850/17860, 27850/27860." >&2
  echo "Override via API_PORT_OVERRIDE and WEB_PORT." >&2
  exit 1
}

api_health_ok() {
  local port="$1"
  "${PY_BIN}" - "$port" <<'PY'
import json
import sys
from urllib.error import URLError
from urllib.request import urlopen

port = int(sys.argv[1])
try:
    with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.0) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except (URLError, TimeoutError, ValueError, OSError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("ok") is True else 1)
PY
}

wait_for_api() {
  local pid="$1"
  local port="$2"
  local attempts=60

  echo "[webui] Waiting for API /api/health on port ${port}..."
  for _ in $(seq 1 "${attempts}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Error: API process exited before becoming healthy." >&2
      wait "${pid}" || true
      exit 1
    fi
    if api_health_ok "${port}"; then
      echo "[webui] API is healthy."
      return 0
    fi
    sleep 1
  done

  echo "Error: API did not become healthy within ${attempts}s." >&2
  return 1
}

pick_ports
export API_PORT_OVERRIDE API_PORT WEB_PORT

echo "[webui] API: http://localhost:${API_PORT_OVERRIDE}"
echo "[webui]  UI: http://localhost:${WEB_PORT}"

api_pid=""
ui_pid=""

cleanup() {
  local code="${1:-0}"
  if [[ -n "${ui_pid}" ]]; then
    kill "${ui_pid}" 2>/dev/null || true
  fi
  if [[ -n "${api_pid}" ]]; then
    kill "${api_pid}" 2>/dev/null || true
  fi
  wait "${ui_pid}" 2>/dev/null || true
  wait "${api_pid}" 2>/dev/null || true
  exit "${code}"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM

(
  cd "${ROOT_DIR}"
  "${PY_BIN}" "${API_ENTRYPOINT}" "$@"
) &
api_pid="$!"

wait_for_api "${api_pid}" "${API_PORT_OVERRIDE}"

(
  cd "${UI_DIR}"
  "${NPM_BIN}" run dev -- --host
) &
ui_pid="$!"

set +e
wait -n "${api_pid}" "${ui_pid}"
status="$?"
set -e

echo "Error: API or UI exited (status=${status}). Shutting down..." >&2
cleanup "${status}"
