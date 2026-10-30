# Changelog of DAMASK-Metal

Versions are named `<DAMASK version>-metal.<n>`.

## Unreleased

### Added
- `tdt`: optional parent slip hardening by the total twin fraction, with the keys and law of
  `phenopowerlaw` (`c_1`, `c_2`, `f_sat_sl-tw`; defaults off), on the CPU and the GPU.
- `tdt`: optional solute pinning of parent slip (`xi_pin_sl` per slip family, `gamma_pin`; default off).
  The flow rule uses xi_eff = xi + xi_pin exp(-gamma/gamma_pin) per system, with gamma the accumulated
  shear of that system: an excess resistance that delays the onset of slip and is released by the
  system's own slip, which gives a yield point and a Lueders-type plateau. Hardening acts on xi unchanged.
  CPU and GPU. With pinning off, results are bit-identical to the previous build. CPU-GPU agreement
  is 4e-5 relative for gamma_pin = 0.01; a very sharp release (gamma_pin = 0.001) reaches 1e-3,
  limited by the state tolerance, so keep gamma_pin >= ~0.003.

## 3.1.0-metal.1: first public version

Based on DAMASK 3.1.0.

### Added
- **Apple-GPU (Metal) backend for the grid solver:**
  - Gamma convolution on the GPU (FP32 FFTs via MPSGraph);
  - constitutive update and tangent on the GPU for `phenopowerlaw` (with and without twinning),
    `kinehardening`, `dislotungsten`, `tdt` and `twintrip`;
  - `fast`, `mixed` and `df64` precision modes;
  - an FP64 certificate of every GPU result, with fallback to the CPU;
  - an optional concurrent CPU/GPU split.
- **New plasticity models (experimental, not yet calibrated against experiments):**
  - `tdt`: twinning, detwinning and retwinning with slip inside twins (hcp);
  - `twintrip`: dislocation-density slip in fcc austenite with twinning and γ→ε→α′ martensite.
- **Tests and tools:**
  - Metal tests in CTest (`ctest -L cpu`, `ctest -L gpu`);
  - repeated-run and fault-injection checks;
  - a benchmark script with results for an M6 Mac mini.

### Changed
- **macOS build fixes:** no GCC LTO with Apple's linker; `ISO_Fortran_binding.h` is found for
  Apple clang; link settings for the C++ unit tests.
