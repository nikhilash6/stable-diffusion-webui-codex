# apps/backend/runtime/sampling Overview
<!-- tags: runtime, sampling, sigma, scheduler -->
Last Review: 2026-03-28
Status: Active

## Purpose
- Native sampling primitives for Codex engines: sigma schedule construction, sampling context assembly, and native sampler execution.

## Key Components
- `sigma_schedules.py`
  - `SchedulerName` enum: canonical scheduler identifiers for sigma-ladder construction.
  - `build_sigma_schedule(...)`: builds sigma ladders for all exposed schedulers.
  - Predictor-backed schedulers (`simple`, `uniform`, `normal`, `beta`, `ddim`, `sgm_uniform`, `turbo`) require predictor data.
  - `beta` uses a predictor-ladder contract: Beta inverse-CDF over timestep probabilities, rounded ladder indices (duplicates preserved), exact requested non-terminal step count, and one terminal zero.
- `flow_shift_resolver.py`
  - `resolve_flow_shift_for_sampling(...)`: resolves effective flow shift for flow-match predictors from `scheduler_config.json` sources.
- `context.py`
  - `SamplingContext`: immutable per-run sampler/scheduler/noise setup.
  - `build_sampling_context(...)`: resolves sigma bounds + flow shift and builds the active sigma schedule.
- `deis.py`
  - `build_deis_coefficients(...)`: DEIS coefficient table builder for the native DEIS lane.
- `log_snr.py`
  - Shared half-logSNR helpers used by native stochastic logSNR samplers.
- `sa_solver.py`
  - SA-Solver-specific coefficient and tau helpers used by native `sa-solver` / `sa-solver pece` lanes.
- `registry.py`
  - `SamplerSpec` dataclass and `get_sampler_spec(name)` for canonical sampler/scheduler compatibility resolution.
- `driver.py`
  - `CodexSampler` native runtime driver.
  - Runs native lanes including `ddpm` and `dpm++ sde` with deterministic driver-owned `ImageRNG` stochastic draws.
  - Handles progress/cancellation/preview hooks under the driver contract.
- `block_progress.py`
  - `RichBlockProgressController` owns the console-only block-progress bar contract.
  - The controller must materialize its Rich task on the first real callback update, not with a fake `total=1` placeholder, so fast transformer lanes like Anima start with truthful block totals.
- `interval_noise.py`
  - Nested-interval normalized-noise helper used by base `dpm++ sde` to reproduce Brownian-style interval covariance from driver-owned iid draws.
- `__init__.py`
  - Import-light public surface for sampler/scheduler catalog exports.

## Current Contracts
- Scheduler vs sampler split is strict:
  - Scheduler controls only the sigma ladder.
  - Sampler integrator is selected by `SamplerKind` in `driver.py`.
- Flow/const img2img partial denoise:
  - `driver.py` trims the already-built sigma ladder with diffusers-style flow semantics (`t_start = int(steps - min(steps * strength, steps))`) instead of rebuilding a longer ladder.
  - For `discard_penultimate` samplers, flow partial denoise trims the already-discarded ladder so low-strength runs keep the last real denoise step instead of collapsing to zero steps.
  - Init-latent startup for `img2img` now goes through predictor-owned `noise_scaling(...)`; raw `init_latent + sigma * noise` is invalid for flow/const predictors.
- Base non-flow img2img partial denoise:
  - `driver.py` now matches the Forge default owner split: effective steps are proportional to denoise (`t_enc = int(min(denoise, 0.999) * steps)`, `effective_steps = t_enc + 1`) instead of rebuilding a longer ladder to preserve the requested count.
  - Internal fixed-step continuation is still available, but only through explicit callers such as hires second passes that pass `img2img_fix_steps=True`.
- Exact top-level swap_model resume:
  - `driver.py` now owns an opaque `SamplingBoundaryState` seam for exact same-latent mid-schedule resume.
  - Boundary capture/resume is same-family only, uses primary-family proof from the captured boundary-state engine id, and skips img2img `predictor.noise_scaling(...)`; unsupported samplers (`dpm fast`, `dpm adaptive`, `restart`, and `dpm++ 2m sde*`) fail loud instead of approximating the handoff.
  - Exact resume rejects fresh initial-noise overrides and explicit top-level `swap_model.seed` overrides; the seam inherits base RNG continuity from the captured run instead of pretending a new stage seed/noise can steer the resumed trajectory.
  - `SamplingBoundaryState` now carries APG momentum state when present, so APG-enabled exact resume keeps guidance continuity instead of silently zeroing that buffer at the boundary.
  - Supported exact-resume samplers are the driver-owned allow-list implemented in `driver.py`; changing that list is a runtime contract change, not a caller preference.
- Canonical names are strict:
  - No alias mapping.
  - Empty or unknown sampler/scheduler names fail fast.
- Sigma ladder precision is fp32:
  - Schedules are built and consumed in fp32 to protect timestep/sigma mapping stability.
- `ddpm + beta` seam:
  - `ddpm` executes natively in `driver.py`.
  - `beta` is predictor-ladder based and no longer uses continuous sigma interpolation.
  - Base non-flow img2img follows the same proportional effective-step contract as the rest of the driver; only explicit fixed-step callers preserve requested-count semantics.
- `dpm++ 2m cfg++` seam:
  - `dpm++ 2m cfg++` executes natively in `driver.py`.
  - The driver captures `uncond_denoised` through the sampler post-CFG hook and uses the dedicated CFG++ recurrence instead of aliasing plain `dpm++ 2m`.
  - Public scheduler exposure is intentionally narrowed to `karras` until broader parity is proven.
- `dpm++ 2m sde` seam:
  - Base `dpm++ 2m sde` executes natively in `driver.py` with prediction-type-aware half-logSNR recurrence.
  - `dpm++ 2m sde heun` reuses the same native runtime state with the upstream Heun correction term instead of the midpoint correction.
  - `dpm++ 2m sde gpu` and `dpm++ 2m sde heun gpu` are explicit native labels backed by the same deterministic interval-noise core as their non-GPU counterparts; this runtime does not expose a separate BrownianTree CPU/GPU split.
  - Stochastic renoise is driver-owned and deterministic (`ImageRNG` seeded-step draws); ambient randomness is not used.
  - Partial-denoise resume burn is deterministic and consumes one draw per skipped positive interval (none for terminal `sigma_next == 0`).
  - Public scheduler exposure is intentionally narrowed to `exponential`.
- `dpm++ 3m sde` seam:
  - `dpm++ 3m sde` executes natively in `driver.py` as its own three-history SDE branch.
  - The lane reuses the model-sampling-aware half-logSNR seam and driver-owned deterministic `ImageRNG` noise; it is not an alias of `dpm++ 2m sde`.
  - Public scheduler exposure is intentionally narrowed to `exponential`.
- `dpm++ sde` seam:
  - Base `dpm++ sde` executes natively in `driver.py`.
  - The lane uses prediction-type-aware half-logSNR drift math plus deterministic driver-owned nested interval noise derived from `ImageRNG` draws instead of `torchsde`.
  - Each positive step consumes two deterministic draws to compose the `[sigma_start, sigma_mid]` and `[sigma_start, sigma_end]` normalized interval noises; partial-run skip-burn consumes the same draw count.
  - Terminal `sigma_next == 0` follows the Comfy parity target and returns `x = denoised`.
  - Public scheduler exposure is intentionally narrowed to `karras`.
- `euler cfg++` / `euler a cfg++` seam:
  - `euler cfg++` and `euler a cfg++` execute natively in `driver.py`.
  - Both lanes use the driver post-CFG hook to capture `uncond_denoised`; `euler a cfg++` also uses driver-owned deterministic `ImageRNG` noise.
  - Public scheduler exposure is intentionally narrowed to `euler_discrete` until broader parity is proven.
- `dpm++ 2s ancestral cfg++` seam:
  - `dpm++ 2s ancestral cfg++` executes natively in `driver.py`.
  - The lane captures `uncond_denoised` through the sampler post-CFG hook and uses a dedicated CFG++ ancestral recurrence instead of aliasing plain `dpm++ 2s ancestral`.
  - Stochastic renoise uses driver-owned deterministic `ImageRNG`; RF/CONST midpoint projection uses the shared half-logSNR helper seam.
  - Public scheduler exposure is intentionally narrowed to `karras` until broader parity is proven.
- `dpm adaptive` seam:
  - `dpm adaptive` executes natively in `driver.py`.
  - The lane reports open-ended progress truthfully: adaptive sampling does not publish a fake fixed total-step count, and image progress surfaces emit `percent=None` / `total_steps=None` when the accepted-step count is still unbounded.
  - Guidance `cfg_trunc_ratio` is explicitly unsupported for `dpm adaptive` because it requires an honest total-step contract; stochastic adaptive (`eta != 0`) also remains fail-loud pending a deterministic noise contract.
  - Public scheduler exposure is intentionally narrowed to `karras`.
- Sampler inventory surface:
  - Raw `/api/samplers` inventory is current-only and lists executable rows from the runtime-backed sampler catalog.
  - Rows absent from `SamplerKind` and the sampler catalog remain non-executable and fail at parse/spec resolution instead of reaching the native driver.
- Driver-owned stochastic determinism:
  - Native stochastic lanes use seeded `ImageRNG` through the driver.
  - No ambient randomness in native stochastic update paths.
- Runtime boundaries:
  - No imports from archived upstream snapshots into active runtime code.
  - If a required runtime invariant is missing (predictor data, malformed schedule, invalid options), fail loud.

## Risks / Invariants
- `steps >= 1`; sigma ladders must include a terminal zero.
- Predictors must expose finite sigma bounds (`sigma_min`, `sigma_max`).
- Predictor-backed ladder schedules require valid predictor ladder data (`predictor.sigmas` and related helpers when used).
- Sampling cancellation is honored by raising `RuntimeError("cancelled")` from the driver.

## Logging
- `CODEX_LOG_SAMPLER=1` enables sampler setup/step diagnostics.
- `CODEX_LOG_SIGMAS=1` logs compact sigma-ladder summaries.
- Sampler diagnostics do not alter routing; they are observability only.
