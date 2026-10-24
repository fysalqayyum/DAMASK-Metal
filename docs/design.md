# Design of the Metal backend

This document is for developers. It covers how the GPU code is organised, why it is
numerically safe without double-precision hardware, and how to add a plasticity model.

## Architecture

```
Fortran (DAMASK)                         Objective-C++ (host)                 Metal (GPU)
--------------------------------------   ----------------------------------   -------------------------
grid/spectral_utilities.f90  -- Gamma -> C_Metal_gamma_convolution  -------> MPSGraph FFT + spectral.metal
homogenization.f90: eligible cells
phase_mechanical.f90: GPU_setup,         C_Metal_cp_add_phase_*  (per phase)
  constitutive_GPU (pack, submit)  ----> C_Metal_cp_submit -> cp_run: pack -> cp.metal: cp_integrate<M>,
  constitutive_GPU_finish (unpack) <---- C_Metal_cp_wait      certificate        cp_tangent<M>
phase_mechanical_plastic_<model>.f90:    (cp_verify, FP64)    <- unpack  <--- (cp_core.h + policy M)
  <model>_GPUadd (parameters)
src/Metal.f90: Fortran interfaces (bind(C))      src/grid/metal/DAMASK_Metal.{h,mm}
```

- **Batching:**
  - `homogenization.f90` collects the cells that are eligible for the GPU and sends them as one batch
    per phase.
  - The CPU integrates all other cells while the GPU runs (`C_Metal_cp_submit` returns at once).
  - Afterwards the GPU points are unpacked. Points the GPU rejected go through DAMASK's normal CPU
    path, `cell_response`.
- **Shader compilation:**
  - The Metal sources are embedded into the executable at build time (`cmake/EmbedMetal.cmake`).
  - They are compiled at run time with IEEE semantics (`MTLMathModeSafe`), one library per model
    type and size (template instantiation through `CP_MODEL_TYPE`), and cached.
  - Neither Xcode nor the offline `metal` compiler is needed.
- **Data layout:**
  - Inputs and outputs are component-major (structure of arrays) df64 arrays. Their layout is given
    by `cp_layout<M>` in `cp_core.h`.
  - Batches are processed in chunks of 16384 points, double-buffered so that host packing and the
    certificate overlap with the GPU.

## Precision model

Apple GPUs have no FP64. The kernels use three kinds of arithmetic:

- **FP32**: kinetics (shear rates, hardening), the Newton iteration on the stress and the Jacobian.
  The Newton iteration corrects its own rounding errors as long as the residual is accurate.
- **df64**, i.e. double-single: an unevaluated sum of two floats with about 48 significant bits
  (`df64.h`, after Thall 2006 and Dekker/Knuth). It is used where cancellation or accumulation
  needs more than FP32:
  - the elastic strain (FᵀF − I)/2;
  - the state accumulation X = X₀ + Δt·Ẋ;
  - the inverse of the plastic deformation gradient;
  - all outputs.
- **Constants** are uploaded as float pairs (`cp_pair`, `cp_to_pair`) and promoted to df64 where
  needed. Physical constants such as k_B and π must be stored this way: float literals caused 10⁻⁶
  errors.
- **FP contraction is off** everywhere (`-ffp-contract=off`, `MTLMathModeSafe`), so that host
  double and GPU results are reproducible and comparable.

`precision: mixed` and `df64` run the whole stress solve in df64 and agree with the FP64 CPU path
to about 10⁻¹¹. `fast` reaches the accuracy of DAMASK's tolerances (stress about 10⁻⁶ relative)
in about half the time.

## Certificate

`cp_verify` (in `DAMASK_Metal.mm`) checks every GPU result on the CPU before it is accepted:

1. All outputs are finite.
2. **Kinematics, in FP64:**
   - invFp = invFp₀ (I − Δt Lp);
   - Fe = F invFp / j (j from the returned state for `FI_STATE` models);
   - P = F invFp S invFpᵀ.
3. **Physics, in FP32 with loose tolerances**, evaluated at the returned state:
   - the stress residual;
   - the flow rule Lp(S);
   - the state update X − X₀ − Δt Ẋ, within 100× DAMASK's state tolerance;
   - the delta state, in FP64;
   - the stiffness of the returned state (`STATE_C` models).

A rate may switch on in the last state update; for example, slip inside a twin starts once the twin
fraction exceeds `f_min`. The state check then accepts either side of the switch. Models declare
such entries through `switch_entry`.

Rejected points are recomputed on the CPU. The certificate exists because silent corruption of whole
SIMD groups was observed during development. It was fixed by moving the Fp inverse and normalisation
to the host, in FP64.

Two QA tests keep the certificate effective:
- `tests/metal/qa/fault_injection.sh` corrupts every 64th GPU result in 7 different ways and
  requires every corrupted point to be rejected.
- `tests/metal/qa/gpu_loop.sh` repeats the GPU tests and a captured replay many times, and requires
  bit-identical, never wrongly accepted results.

## Constitutive core and model policies

`cp_core.h` holds the model-independent algorithm. It follows DAMASK's `integrateStress` and
`integrateStateFPI`:
- an elastic stress residual free of cancellation;
- a Newton iteration on the second Piola–Kirchhoff stress S, with trust region and backtracking;
- a warm start;
- fixed-point state iteration with DAMASK's damper;
- the delta state after convergence;
- DAMASK's analytic tangent (`phase_mechanical_dPdF`).

A **model policy** `M` supplies only the physics: the kinetics or the flow rule, the state rates, the
dependent state, tolerances and stress scales. It also declares traits that select variants of the
core:

| Trait | Meaning | Used by |
|---|---|---|
| `FLOW` | the policy returns Lp and dLp/dS itself (instead of per-system shear rates) | tdt, twintrip |
| `STATE_C` | the stiffness depends on the state (computed in `dependent()`) | tdt, twintrip |
| `FLAT` | the state is passed as DAMASK stores it (no padding per block) | twintrip |
| `DT_STATE` | the state rate depends on Δt (closed-form implicit update of stiff state) | tdt, twintrip |
| `PRECOND` | diagonally preconditioned state iteration X −= r/(1 − Δt J_d) | tdt, twintrip |
| `FI_STATE` | isotropic eigenstrain of the state, Fi = j(X) I | twintrip (α′) |
| `ACCEL` | Anderson(1) acceleration of the state iteration instead of the damper | twintrip |

Policies: `cp_model_pheno`, `cp_model_kineh`, `cp_model_dtung` and `cp_model_phenotw` are in
`cp_core.h`; `cp_model_tdt` and `cp_model_twintrip` are in `cp_models_composite.h`. Slip counts are
padded to 12/24/30/48; padded systems are inert.

## Adding a plasticity model

1. **Policy:** write a policy struct (see `cp_model_dtung` for a slip model with dependent state, or
   `cp_model_tdt` for a composite model). Port the Fortran rates exactly, in the same order of
   operations.
2. **Golden vectors:** let DAMASK write golden vectors (Lp, dLp/dMp, rates and dependent state) for
   synthetic states. Then test the policy in double on the host against them, to about 10⁻¹² (see
   `DAMASK_TWINTRIP_GOLDEN` and `tests/metal/test_twintrip_policy.cpp`).
3. **Host entry point:** add a `C_Metal_cp_add_phase_*` entry point or reuse `C_Metal_cp_add_phase_model`.
   - Add the policy to the dispatch in `C_Metal_cp_integrate`.
   - Add its parameter struct to the size check in `cp.metal`/`cp_init`.
4. **Fortran:** add `plastic_<model>_GPUadd` (packs the parameters), the eligibility conditions in
   `GPU_setup`, and, if needed, `plastic_<model>_GPUdependent`.
5. **Validation:**
   - replay a captured batch (`bench_cp_replay`) in `fast` and `mixed` precision;
   - compare full CPU and GPU runs (`benchmarks/compare_runs.py`);
   - run `fault_injection.sh` and `gpu_loop.sh` with a capture of the new model.

## Limitations

- **One MPI rank:** the GPU is used only with a single MPI rank (grid solver, OpenMP for the CPU
  part).
- **Not supported on the GPU:** eigenstrain kinematics other than the isotropic, state-defined Fi of
  `twintrip`; damage; multi-constituent homogenization.
- **Gamma convolution:** it runs in FP32 on the GPU. The spectral solver's equilibrium tolerance is
  far above its rounding level.
