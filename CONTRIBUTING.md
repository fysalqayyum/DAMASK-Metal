# Contributing to DAMASK-Metal

Issues and pull requests are welcome. Include in every report the Mac model, the macOS version and
the output of a failing run with `DAMASK_METAL_VERBOSE=1`.

## Scope

Changes to DAMASK itself, i.e. code that is not specific to the GPU backend or the `tdt` and
`twintrip` models, belong upstream: https://damask-multiphysics.org.

## Rules for changes

- **Results:**
  - A change must not alter the results of existing models unless that is its purpose.
  - Check this by comparing runs before and after the change with `benchmarks/compare_runs.py`;
    the differences must be exactly zero for pure refactoring.
  - Any change to GPU kernels (`src/grid/metal/*.h`, `*.metal`) must also pass:
    ```sh
    cmake --build build --target metal-tests && ctest --test-dir build -L metal
    tests/metal/qa/make_capture.sh tdt 16
    tests/metal/qa/gpu_loop.sh 60 capture/tdt16
    tests/metal/qa/fault_injection.sh capture/tdt16
    ```
- **Numerical conventions (see [docs/design.md](docs/design.md)):**
  - FP contraction stays off.
  - Constants for the GPU are float pairs (`cp_to_pair`).
  - Ports of DAMASK's Fortran keep its order of operations.
- **Style:**
  - Follow the style of the surrounding code: DAMASK's Fortran conventions (two-space indentation,
    `pREAL`, `!<` comments) and the existing C++ style in `src/grid/metal`.
  - Every new file starts with `SPDX-License-Identifier: AGPL-3.0-or-later`.
- **Commits:**
  - One topic per commit, with a message that explains why.
  - If you use AI assistance, say so in the commit message (e.g. a `Co-Authored-By` trailer).

By contributing you agree that your contribution is licensed under the AGPL-3.0-or-later, the
licence of this repository.
