# DAMASK-Metal

[![build](https://github.com/fysalqayyum/DAMASK-Metal/actions/workflows/build.yml/badge.svg?branch=metal)](https://github.com/fysalqayyum/DAMASK-Metal/actions/workflows/build.yml)

**DAMASK 3.1.0 with an Apple-GPU (Metal) backend for the spectral grid solver.**

DAMASK-Metal runs the constitutive update (stress and state integration, and the tangent) and the
spectral Gamma convolution of [DAMASK](https://damask-multiphysics.org)'s grid solver on the GPU of
Apple-silicon Macs. Every GPU result is checked in double precision on the CPU before it is used;
points that fail the check are recomputed on the CPU, so results stay at DAMASK's accuracy.

It is an independent extension of DAMASK, not an official DAMASK release. For DAMASK itself see
[README-DAMASK.md](README-DAMASK.md) and https://damask-multiphysics.org.

## Status

| Plasticity model | GPU | Notes |
|---|---|---|
| `phenopowerlaw` (slip) | ✓ | up to 48 slip systems |
| `phenopowerlaw` (slip + twinning) | ✓ | |
| `kinehardening` | ✓ | |
| `dislotungsten` | ✓ | slow-converging points fall back to the CPU (~1–11 %) |
| `tdt` (new: twinning, detwinning, slip in twins; hcp) | ✓ | **experimental**: not yet calibrated against experiments |
| `twintrip` (new: TWIP/TRIP, γ→ε→α′; fcc) | ✓ | **experimental**: not yet calibrated against experiments |
| all other models, damage, eigenstrain kinematics, multi-constituent homogenization | CPU | unchanged DAMASK behaviour |

GPU requirements: grid solver, one MPI rank, single-constituent homogenization. Everything else
runs on the CPU exactly as in DAMASK 3.1.0; CPU and GPU phases can be mixed in one simulation.

## Performance

Apple M6 Mac mini (12 CPU cores, 12 GPU cores, 16 GB), wall time of complete simulations, CPU =
DAMASK 3.1.0 on all 12 cores, GPU = DAMASK-Metal (default settings). 64³: first 10 increments
except pheno. Details and reproduction: [docs/benchmarks.md](docs/benchmarks.md).

| Case | 16³ | 32³ | 64³ |
|---|---|---|---|
| Al, phenopowerlaw | 2.1× | 3.5× | 3.2× |
| Mg, phenopowerlaw + twins | 2.2× | 4.2× | 3.4× |
| W, dislotungsten | 1.8× | 2.2× | 2.0× |
| Mg, tdt (twinning/detwinning) | 1.9× | 3.2× | 4.1× |
| fcc, twintrip (twinning) | 3.5× | 4.4× | 4.1× |
| fcc, twintrip (ε + α′ martensite) | 3.2× | 3.9× | – |

Volume-averaged stresses of CPU and GPU runs agree to 10⁻⁷–10⁻⁵ relative (twintrip ε + α′: 2·10⁻⁴, a
marginally converging case on the CPU as well).

## Quick start

Requires macOS on Apple silicon with the Xcode **Command Line Tools** (Xcode itself is not needed),
Homebrew, and PETSc 3.19–3.25. Full instructions: [docs/install-macos.md](docs/install-macos.md).

```sh
git clone https://github.com/fysalqayyum/DAMASK-Metal.git
cd DAMASK-Metal
source env/DAMASK-Metal.sh                    # PETSC_DIR/PETSC_ARCH and Apple-silicon MPI settings
cmake -S . -B build -DGRID=ON -DMESH=OFF -DCMAKE_BUILD_TYPE=PERFORMANCE -DCMAKE_INSTALL_PREFIX=$PWD/build/install
cmake --build build --parallel --target install
cmake --build build --target metal-tests && ctest --test-dir build -L metal
build/install/bin/DAMASK_grid -g examples/grid/20grains16x16x16.vti -l examples/grid/tensionX.yaml \
                              -m examples/grid/material.yaml
```

The GPU is used automatically; see [docs/usage.md](docs/usage.md) for the `GPU` section of
`numerics.yaml` (precision, CPU/GPU split, switches) and the diagnostics.

## Documentation

- [docs/install-macos.md](docs/install-macos.md): installation on Apple silicon
- [docs/usage.md](docs/usage.md): configuration, eligibility, fallbacks, diagnostics
- [docs/design.md](docs/design.md): architecture, precision model, result certificate, adding a model
- [docs/benchmarks.md](docs/benchmarks.md): benchmark cases, results, reproduction
- [NOTICE.md](NOTICE.md): changes relative to DAMASK 3.1.0

## How this was developed

The DAMASK-Metal code was written with the AI coding assistant Claude (Anthropic) under the
direction of Faisal Qayyum, who defined the goals and the validation criteria and reviewed and
validated the results. Commits authored this way carry a `Co-Authored-By: Claude` trailer.
Validation: unit tests against independent ports of DAMASK's Fortran, golden vectors written by
DAMASK itself, CPU-vs-GPU comparisons of full simulations, repeated-run and fault-injection tests
of the result certificate (`tests/metal`).

## Citation and licence

If you use DAMASK-Metal, cite DAMASK (F. Roters et al., *Comput. Mater. Sci.* 158 (2019) 420–478,
doi:[10.1016/j.commatsci.2018.04.030](https://doi.org/10.1016/j.commatsci.2018.04.030)) and this
repository (https://github.com/fysalqayyum/DAMASK-Metal). `CITATION.cff` is DAMASK's own.

DAMASK-Metal is licensed, like DAMASK, under the GNU Affero General Public License v3.0 or later
([LICENSE](LICENSE)). Copyright of DAMASK: Max-Planck-Institut für Nachhaltige Materialien GmbH
and Max-Planck-Institut für Eisenforschung GmbH; of the additions: Faisal Qayyum.
Issues and questions about the GPU backend: this repository's GitHub issues (not the DAMASK helpdesk).
