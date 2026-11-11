# Changelog of DAMASK-Metal

Versions are named `<DAMASK version>-metal.<n>`.

## Unreleased

### Added
- `kocksmecking` (experimental): new plasticity model, slip only. Power-law kinetics as `phenopowerlaw`,
  slip resistance g = g_0 + chi mu b sqrt(sum a_ab rho_b) and Kocks-Mecking evolution of the dislocation
  density per slip system, dot_rho = (sqrt(sum a_ab rho_b)/K - 2 y_c rho) |dot_gamma| / b (equations of
  Charpagne et al., Acta Mater. 215:117037, 2021). Keys `N_sl`, `dot_gamma_0_sl`, `n_sl`, `g_0_sl`, `b_sl`,
  `rho_0`, `K`, `y_c` (per slip family), `chi` (default 1), `mu`, `h_sl-sl` (a_ab), `atol_rho`, `atol_gamma`;
  outputs `rho_sl`, `g_sl`, `gamma_sl`. CPU and GPU (policy `cp_model_km`); non-Schmid law allowed.
  Example: `examples/config/phase/mechanical/plastic/kocksmecking_IN718-aged.yaml` (aged Inconel 718,
  Table 1 of that paper). Stock models are bit-identical to the previous build (pheno, phenotw and dtung
  at 16³, CPU and GPU).
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
