"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Manual CUDA smoke harness for the generic `attention_sram_v1` kernel tree.
Runs fail-loud preflight/build/warmup/parity/fallback checks so a CUDA-capable host can validate the
narrow SRAM attention slice from one command without guessing build or runtime steps.

Symbols (top-level; keep in sync; no ghosts):
- `PreflightReport` (dataclass): Toolchain/runtime discovery report for the local SRAM smoke environment.
- `BuildReport` (dataclass): Result summary for the in-place extension build step.
- `WarmupReport` (dataclass): Result summary for bridge warmup/readiness.
- `ParityReport` (dataclass): Result summary for the supported-tuple parity/layout smoke.
- `FallbackReport` (dataclass): Result summary for the unsupported-tuple fallback smoke.
- `main` (function): CLI entrypoint for `preflight|build|warmup|parity|fallback|full`.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import argparse
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

logger = get_backend_logger("runtime.kernels.attention_sram_v1.smoke")

_EXTENSION_NAME = "attention_sram_v1_cuda"
_EXPECTED_FALLBACK_CODE = "E_SRAM_ATTENTION_HEAD_DIM_UNSUPPORTED"
_DEFAULT_ATOL = 2e-2
_DEFAULT_RTOL = 2e-2
_DEFAULT_BATCH = 1
_DEFAULT_HEADS = 2
_DEFAULT_SEQ_LEN = 64
_DEFAULT_HEAD_DIM = 128
_DEFAULT_UNSUPPORTED_HEAD_DIM = 64
_DEFAULT_IS_CAUSAL = False
_JIT_ENV_KEY = "CODEX_ATTENTION_SRAM_JIT"


@dataclass(frozen=True, slots=True)
class PreflightReport:
    python: str
    executable: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_home: str | None
    nvcc_path: str | None
    kernel_dir: str
    setup_path: str
    built_modules: tuple[str, ...]
    codex_root: str | None
    pythonpath: str | None
    sram_mode_env: str | None
    sram_jit_env: str | None


@dataclass(frozen=True, slots=True)
class BuildReport:
    command: tuple[str, ...]
    kernel_dir: str
    built_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WarmupReport:
    mode: str
    build_enabled: bool
    attempted: bool
    loaded: bool
    ready: bool
    detail: str | None


@dataclass(frozen=True, slots=True)
class ParityReport:
    mode: str
    is_causal: bool
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    q_stride: tuple[int, ...]
    out_stride: tuple[int, ...]
    same_stride: bool
    allclose: bool
    max_abs_diff: float
    atol: float
    rtol: float


@dataclass(frozen=True, slots=True)
class FallbackReport:
    mode: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    reason_code: str | None
    reason_detail: str | None


def _kernel_dir() -> Path:
    return Path(__file__).resolve().parent


def _setup_path() -> Path:
    return _kernel_dir() / "setup.py"


def _built_modules() -> tuple[str, ...]:
    patterns = (
        f"{_EXTENSION_NAME}*.so",
        f"{_EXTENSION_NAME}*.pyd",
        f"{_EXTENSION_NAME}*.dll",
        f"{_EXTENSION_NAME}*.dylib",
    )
    hits: list[str] = []
    for pattern in patterns:
        hits.extend(str(path.name) for path in sorted(_kernel_dir().glob(pattern)))
    return tuple(hits)


def _cuda_home() -> str | None:
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return None
    if CUDA_HOME is None:
        return None
    return str(CUDA_HOME)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_dump(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _bridge_api():
    from apps.backend.runtime.attention.sram import (
        try_attention_pre_shaped,
        warmup_extension_for_load,
    )

    return try_attention_pre_shaped, warmup_extension_for_load


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def collect_preflight() -> PreflightReport:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_home = _cuda_home()
    nvcc_path = None
    if cuda_home:
        nvcc_executable_names = ("nvcc.exe", "nvcc") if os.name == "nt" else ("nvcc",)
        for nvcc_executable_name in nvcc_executable_names:
            nvcc_candidate = Path(cuda_home) / "bin" / nvcc_executable_name
            if nvcc_candidate.is_file():
                nvcc_path = str(nvcc_candidate)
                break
    return PreflightReport(
        python=sys.version.split()[0],
        executable=sys.executable,
        torch_version=str(torch.__version__),
        torch_cuda_version=None if torch.version.cuda is None else str(torch.version.cuda),
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_home=cuda_home,
        nvcc_path=nvcc_path,
        kernel_dir=str(_kernel_dir()),
        setup_path=str(_setup_path()),
        built_modules=_built_modules(),
        codex_root=os.environ.get("CODEX_ROOT"),
        pythonpath=os.environ.get("PYTHONPATH"),
        sram_mode_env=os.environ.get("CODEX_ATTENTION_SRAM_MODE"),
        sram_jit_env=os.environ.get("CODEX_ATTENTION_SRAM_JIT"),
    )


def emit_preflight() -> PreflightReport:
    report = collect_preflight()
    _json_dump(asdict(report))
    return report


def _require_cuda_runtime(report: PreflightReport) -> None:
    _require(report.cuda_available, "torch.cuda.is_available() is false in this environment.")
    _require(report.cuda_device_count > 0, "No CUDA devices are visible to PyTorch.")


def build_extension(report: PreflightReport) -> BuildReport:
    setup_path = _setup_path()
    _require(setup_path.is_file(), f"Missing setup.py at {setup_path}")
    command = (sys.executable, str(setup_path), "build_ext", "--inplace")
    logger.info("Building %s in %s", _EXTENSION_NAME, _kernel_dir())
    completed = subprocess.run(
        command,
        cwd=str(_kernel_dir()),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "attention_sram_v1 build failed.\n"
            f"command={' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    build_report = BuildReport(
        command=command,
        kernel_dir=str(_kernel_dir()),
        built_modules=_built_modules(),
    )
    _json_dump(asdict(build_report))
    return build_report


def run_warmup(mode: str) -> WarmupReport:
    report = collect_preflight()
    _require_cuda_runtime(report)
    _, warmup_extension_for_load = _bridge_api()
    with _temporary_env({_JIT_ENV_KEY: "0"}):
        status = warmup_extension_for_load(mode=mode)
    warmup_report = WarmupReport(
        mode=str(status.mode.value),
        build_enabled=bool(status.build_enabled),
        attempted=bool(status.attempted),
        loaded=bool(status.loaded),
        ready=bool(status.ready),
        detail=status.detail,
    )
    _json_dump(asdict(warmup_report))
    _require(warmup_report.loaded, f"SRAM warmup did not load the extension: {warmup_report.detail}")
    _require(warmup_report.ready, f"SRAM warmup did not reach ready state: {warmup_report.detail}")
    return warmup_report


def _build_supported_inputs(*, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_q = torch.randn((_DEFAULT_BATCH, _DEFAULT_SEQ_LEN, _DEFAULT_HEADS, _DEFAULT_HEAD_DIM), device=device, dtype=torch.float16)
    base_k = torch.randn((_DEFAULT_BATCH, _DEFAULT_SEQ_LEN, _DEFAULT_HEADS, _DEFAULT_HEAD_DIM), device=device, dtype=torch.float16)
    base_v = torch.randn((_DEFAULT_BATCH, _DEFAULT_SEQ_LEN, _DEFAULT_HEADS, _DEFAULT_HEAD_DIM), device=device, dtype=torch.float16)
    q = base_q.permute(0, 2, 1, 3)
    k = base_k.permute(0, 2, 1, 3)
    v = base_v.permute(0, 2, 1, 3)
    return q, k, v


def run_parity(
    *,
    mode: str,
    ensure_ready: bool = True,
) -> ParityReport:
    report = collect_preflight()
    _require_cuda_runtime(report)
    try_attention_pre_shaped, _ = _bridge_api()
    with _temporary_env({_JIT_ENV_KEY: "0"}):
        if ensure_ready:
            run_warmup(mode)
        device = torch.device("cuda", torch.cuda.current_device())
        q, k, v = _build_supported_inputs(device=device)
        result = try_attention_pre_shaped(mode=mode, q=q, k=k, v=v, is_causal=_DEFAULT_IS_CAUSAL)
        _require(
            result.output is not None,
            "SRAM parity smoke did not produce an output through the bridge-supported path: "
            f"reason_code={result.reason_code!r} reason_detail={result.reason_detail!r}",
        )
        _require(
            result.reason_code is None,
            "SRAM parity smoke unexpectedly reported a fallback reason on the supported tuple: "
            f"reason_code={result.reason_code!r} reason_detail={result.reason_detail!r}",
        )
        out = result.output
        ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=_DEFAULT_IS_CAUSAL)
        max_abs_diff = float((out - ref).abs().max().item())
        parity_report = ParityReport(
            mode=str(mode),
            is_causal=_DEFAULT_IS_CAUSAL,
            batch=_DEFAULT_BATCH,
            heads=_DEFAULT_HEADS,
            seq_len=_DEFAULT_SEQ_LEN,
            head_dim=_DEFAULT_HEAD_DIM,
            q_stride=tuple(int(value) for value in q.stride()),
            out_stride=tuple(int(value) for value in out.stride()),
            same_stride=tuple(int(value) for value in out.stride()) == tuple(int(value) for value in q.stride()),
            allclose=bool(torch.allclose(out, ref, atol=_DEFAULT_ATOL, rtol=_DEFAULT_RTOL)),
            max_abs_diff=max_abs_diff,
            atol=_DEFAULT_ATOL,
            rtol=_DEFAULT_RTOL,
        )
    _json_dump(asdict(parity_report))
    _require(parity_report.same_stride, "SRAM parity smoke returned an output with a different stride/layout than q.")
    _require(parity_report.allclose, f"SRAM parity smoke diverged from PyTorch SDPA (max_abs_diff={max_abs_diff}).")
    return parity_report


def run_fallback(
    *,
    mode: str,
    ensure_ready: bool = True,
) -> FallbackReport:
    report = collect_preflight()
    _require_cuda_runtime(report)
    try_attention_pre_shaped, _ = _bridge_api()
    with _temporary_env({_JIT_ENV_KEY: "0"}):
        if ensure_ready:
            run_warmup(mode)
        device = torch.device("cuda", torch.cuda.current_device())
        bad = torch.randn(
            (_DEFAULT_BATCH, _DEFAULT_HEADS, _DEFAULT_SEQ_LEN, _DEFAULT_UNSUPPORTED_HEAD_DIM),
            device=device,
            dtype=torch.float16,
        )
        result = try_attention_pre_shaped(mode=mode, q=bad, k=bad, v=bad, is_causal=False)
        fallback_report = FallbackReport(
            mode=str(mode),
            batch=_DEFAULT_BATCH,
            heads=_DEFAULT_HEADS,
            seq_len=_DEFAULT_SEQ_LEN,
            head_dim=_DEFAULT_UNSUPPORTED_HEAD_DIM,
            reason_code=result.reason_code,
            reason_detail=result.reason_detail,
        )
    _json_dump(asdict(fallback_report))
    _require(result.output is None, "Fallback smoke unexpectedly returned an output for an unsupported tuple.")
    _require(
        result.reason_code == _EXPECTED_FALLBACK_CODE,
        "Fallback smoke returned the wrong reason code: "
        f"expected={_EXPECTED_FALLBACK_CODE!r} got={result.reason_code!r}",
    )
    return fallback_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual CUDA smoke harness for the generic attention_sram_v1 kernel tree.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="Print the current toolchain/runtime preflight report.")
    subparsers.add_parser("build", help="Run `setup.py build_ext --inplace` for attention_sram_v1_cuda.")

    warmup_parser = subparsers.add_parser("warmup", help="Run bridge warmup and require loaded+ready.")
    warmup_parser.add_argument("--mode", default="auto", choices=("auto", "force"))

    parity_parser = subparsers.add_parser(
        "parity",
        help="Run the locked supported-tuple parity and stride smoke against PyTorch SDPA.",
    )
    parity_parser.add_argument("--mode", default="auto", choices=("auto", "force"))

    fallback_parser = subparsers.add_parser(
        "fallback",
        help="Run the locked unsupported-tuple fallback smoke through the bridge.",
    )
    fallback_parser.add_argument("--mode", default="auto", choices=("auto", "force"))

    full_parser = subparsers.add_parser(
        "full",
        help="Run preflight, build, warmup, parity, and fallback in sequence on the locked tuples.",
    )
    full_parser.add_argument("--mode", default="auto", choices=("auto", "force"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="[%(levelname)s] %(message)s")

    try:
        if args.command == "preflight":
            emit_preflight()
            return 0
        if args.command == "build":
            build_extension(collect_preflight())
            return 0
        if args.command == "warmup":
            run_warmup(args.mode)
            return 0
        if args.command == "parity":
            run_parity(mode=args.mode)
            return 0
        if args.command == "fallback":
            run_fallback(mode=args.mode)
            return 0
        if args.command == "full":
            report = emit_preflight()
            build_extension(report)
            run_warmup(args.mode)
            run_parity(mode=args.mode, ensure_ready=False)
            run_fallback(mode=args.mode, ensure_ready=False)
            return 0
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
