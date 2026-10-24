# Notice of modifications

DAMASK-Metal is a modified version of DAMASK 3.1.0. The base is tag `v3.1.0`, commit
`e19217224`, of https://github.com/damask-multiphysics/DAMASK, copyright Max-Planck-Institut für
Nachhaltige Materialien GmbH and Max-Planck-Institut für Eisenforschung GmbH. This notice is given
under section 5(a) of the GNU Affero General Public License v3.0. The modifications were made by
Faisal Qayyum, starting October 2026; the complete history is in this repository
(`git log v3.1.0..`).

## New files

- Apple-GPU (Metal) backend: `src/grid/metal/*`, `src/Metal.f90`, `cmake/EmbedMetal.cmake`
- Plasticity models `tdt` and `twintrip`: `src/phase_mechanical_plastic_tdt.f90`,
  `src/phase_mechanical_plastic_twintrip.f90`, and example configurations in
  `examples/config/phase/mechanical/plastic/`
- Tests and quality checks: `tests/metal/`
- Benchmarks: `benchmarks/`
- Documentation and project files: `README.md` (the original is kept as `README-DAMASK.md`),
  `docs/`, `NOTICE.md`, `CHANGELOG-Metal.md`, `CONTRIBUTING.md`, `env/DAMASK-Metal.sh`

## Modified DAMASK files

| File | Change |
|---|---|
| `CMakeLists.txt`, `src/CMakeLists.txt` | `METAL` option, embedding of the shaders, Metal tests, ISO_Fortran_binding.h for Apple clang |
| `cmake/Compiler-GNU.cmake` | no `-flto` on macOS (GCC LTO is incompatible with Apple's linker) |
| `tests/unit_cpp/CMakeLists.txt` | link directories and Boost for the C++ unit tests on macOS |
| `src/grid/spectral_utilities.f90` | `GPU` numerics, Gamma convolution on the GPU |
| `src/homogenization.f90`, `src/homogenization_mechanical.f90` | batching of eligible cells for the GPU |
| `src/phase.f90`, `src/phase_mechanical.f90` | GPU constitutive update (setup, submit, unpack) |
| `src/phase_mechanical_plastic.f90` | registration of tdt and twintrip, shared helpers |
| `src/phase_mechanical_plastic_{phenopowerlaw,kinehardening,dislotungsten}.f90` | parameter upload to the GPU (`*_GPUadd`) |
| `src/phase_mechanical_elastic.f90`, `src/phase_mechanical_eigen.f90` | stiffness and eigenstrain hooks for tdt and twintrip |
| `src/crystal.f90` | lattice rotations of twins and martensites (`crystal_rotation_twin` etc.) |
| `.gitattributes`, `.gitignore` | binary test data, build outputs |

With `-DMETAL=OFF`, or when no GPU is available, the behaviour of the stock DAMASK models is
unchanged.
