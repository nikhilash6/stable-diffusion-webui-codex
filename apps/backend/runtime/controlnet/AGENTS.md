# apps/backend/runtime/controlnet Overview
Date: 2025-10-30
Last Review: 2026-05-11
Status: Active

## Purpose
- Provide typed configuration, graph orchestration, and runtime helpers for ControlNet integration.

## Notes
- `config.py` defines `ControlRequest`, `ControlNode`, and related dataclasses; enforce validation before wiring nodes into the graph.
- `runtime.py` exposes `ControlComposite` / `build_composite`, bridging the new graph model with legacy interfaces (`get_control`, `cleanup`, etc.).
- Patchers should call `UnetPatcher.add_control_node` with a prepared `ControlNode`; sampling activates the composite via `UnetPatcher.activate_control()`.
- `preprocessors/` contains Codex-native preprocessing model modules (edge detectors, etc.); it does not expose a package-level registry or slug resolver.
- `__init__.py` is a package marker (no re-exports); import ControlNet types/helpers from their owning modules.
- 2026-01-20: Removed unused `converters.py` helper (no call sites).
- 2026-02-23: `config.py::ControlNode.prepare(...)` now resolves the no-parameter fallback device from memory-manager mount-device authority instead of constructing a local CPU fallback literal.
