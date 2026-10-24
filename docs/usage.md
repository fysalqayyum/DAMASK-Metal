# Using DAMASK-Metal

DAMASK-Metal is used exactly like DAMASK 3.1.0's grid solver (`DAMASK_grid -g … -l … -m … [-n …]`).
The GPU is used by default when it is available; nothing has to be changed in existing inputs.

## Configuration: `numerics.yaml`

All keys are optional and go into the `GPU` section of the grid solver:

```yaml
solver:
  grid:
    GPU:
      Gamma: true            # spectral Gamma convolution on the GPU (FP32 FFTs)
      constitutive: true     # constitutive update of eligible phases on the GPU
      tangent: true          # tangent dP/dF of GPU-integrated points on the GPU
      precision: fast        # fast | mixed | df64 (see below)
      split: 1.0             # share of eligible points on the GPU, 0..1, or auto
      N_iter_state_max: -1   # GPU state-iteration cap; -1: per model (see below)
      eps_rel_S: 1.0e-10     # relative stress tolerance of the GPU stress solve
      verify: false          # also run the Gamma convolution on the CPU and report the difference
```

- **`precision`**:
  - `fast` (default): FP32 kinetics and stress solve; df64 (double-single, ~48-bit) for the
    elastic strain, state accumulation and kinematics.
  - `mixed`: FP32 Newton pre-solve, then a df64 polish.
  - `df64`: df64 throughout.

  All three reach DAMASK's accuracy. `fast` is the quickest; `mixed` and `df64` agree with the CPU
  to about 10⁻¹¹ per point.
- **`split`**: the fraction of eligible points sent to the GPU. The rest run concurrently on the
  CPU. `auto` balances the two by measured time. On the machines tested, `1.0` (all on the GPU)
  was fastest.
- **`N_iter_state_max`**: the GPU state-iteration cap. Points that need more iterations are
  recomputed on the CPU.
  - The default `-1` uses min(DAMASK's `N_iter_state_max`, 14).
  - `tdt` and `twintrip` use DAMASK's own value, because twin nucleation needs more iterations.
- The Gamma convolution on the GPU and the constitutive update are independent: either can be
  switched off.

## Which points run on the GPU

A phase is integrated on the GPU when all of these hold:

- its plasticity model has a GPU implementation: `phenopowerlaw`, `kinehardening`,
  `dislotungsten`, `tdt`, `twintrip`;
- it has no damage and no eigenstrain kinematics (e.g. thermal expansion);
- its homogenization has one constituent and mechanical homogenization `type: pass`;
- DAMASK runs as a single MPI rank (`mpiexec -n 1`, or without `mpiexec`).

Even within such a phase, a point stays on the CPU if its state is not representable on the GPU.
For example, its stiffness may differ from the phase stiffness, or its eigenstrain Fi may not be
the identity (for `twintrip`: not isotropic). The log reports per phase whether it runs on the GPU,
and if not, why.

### CPU fallback and the result certificate

Every GPU result is checked on the CPU, in double precision, before it is accepted:

- kinematics: Fp, Fe and P are consistent;
- the stress residual;
- the flow rule;
- the state update;
- the delta state;
- the stiffness of the returned state.

Points the GPU did not converge, and points that fail this check, are recomputed on the CPU with
DAMASK's own code. In normal runs 0–2 % of points go back to the CPU; `dislotungsten` can reach
about 11 %. The check also guards against silent GPU errors (see [design.md](design.md#certificate)).

## Diagnostics

`DAMASK_METAL_VERBOSE=1` prints, per GPU call:

- the number of points;
- the CPU fallbacks and certificate rejections;
- the GPU and host times;
- a breakdown of failure causes (`stress`, `nan`, `state`, `singular`).

The benchmark script uses it. The remaining environment variables are developer tools:

| Variable | Effect |
|---|---|
| `DAMASK_METAL_PRECISION=fast\|mixed\|df64` | override `precision` |
| `DAMASK_METAL_VERIFY_TANGENT=1` | also compute the tangent on the CPU and print the largest difference |
| `DAMASK_METAL_VERIFY_DEBUG=1` | print which certificate check failed for which point |
| `DAMASK_METAL_DUMP=<base>` | write the inputs of failed points and the phase parameters (tdt) |
| `DAMASK_METAL_CAPTURE=<base>`, `DAMASK_METAL_CAPTURE_CALL=<k>` | write GPU call k (inputs and phase) for replay with `bench_cp_replay`, then exit |
| `DAMASK_METAL_INJECT=<0..6>` | QA only: corrupt every 64th GPU result to test the certificate |
| `DAMASK_METAL_NO_VERIFY=1` | diagnostics only: skip the certificate (**never for production runs**) |
| `DAMASK_METAL_CP_TG=<n>` | threads per threadgroup of the constitutive kernels (default 32) |
| `DAMASK_METAL_CP_CHUNK=<n>` | points per GPU command buffer (default 16384; larger values gave no end-to-end gain) |
| `DAMASK_METAL_NO_MIXED=1` | same as `precision: df64` |
| `DAMASK_TWINTRIP_GOLDEN=<file>` | write twintrip golden vectors for `test_twintrip_policy`, then exit |

## Quality checks

After building the tests (`cmake --build build --target metal-tests`):

```sh
ctest --test-dir build -L metal                      # unit and GPU tests
tests/metal/qa/make_capture.sh tdt 16                # capture a tdt batch for replay (writes ./capture)
tests/metal/qa/gpu_loop.sh 60 capture/tdt16          # 60 repeated GPU runs: no wrong result may be accepted
tests/metal/qa/fault_injection.sh capture/tdt16      # every injected corruption must be rejected
```

The repeated-run loop exists because silent corruption of whole SIMD groups was observed during
development (fixed). Run it after changes to the GPU kernels or after macOS updates.
