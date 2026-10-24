# Installing DAMASK-Metal on Apple silicon

Tested on macOS 27.0 (Apple M6) with GCC 16.2, Open MPI 5.0.11, CMake 4.4 and PETSc 3.25.6.
Any Apple-silicon Mac (M1 or newer) with a recent macOS should work; the GPU code needs no Xcode,
only the Command Line Tools (the Metal shaders are compiled at run time by the system).

## 1. Command Line Tools and Homebrew packages

```sh
xcode-select --install                       # Command Line Tools (skip if already installed)
brew install gcc open-mpi cmake pkgconf libfyaml boost
```

`gcc` provides `gfortran`; Open MPI's `mpif90` wraps it. C/C++/Objective-C++ are compiled by Apple
clang. `boost` is only needed for the optional C++ unit tests (`-DTEST=ON`, also needs `googletest`).

## 2. PETSc

DAMASK needs PETSc 3.19–3.25 with FFTW and HDF5 (Fortran bindings). Build it once:

```sh
mkdir -p ~/Software && cd ~/Software
curl -LO https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.25.6.tar.gz
tar xzf petsc-3.25.6.tar.gz && cd petsc-3.25.6
export PETSC_DIR=$PWD PETSC_ARCH=arch-damask-opt
./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 \
            --with-debugging=0 --with-blaslapack-lib="-framework Accelerate" \
            --download-fftw --download-hdf5 --with-hdf5-fortran-bindings=1 --download-zlib \
            --with-ssl=0 --with-x=0
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
```

If you use another location or architecture name, set `PETSC_DIR` and `PETSC_ARCH` before sourcing
the environment script below.

## 3. DAMASK-Metal

```sh
git clone https://github.com/fysalqayyum/DAMASK-Metal.git
cd DAMASK-Metal
source env/DAMASK-Metal.sh
cmake -S . -B build -DGRID=ON -DMESH=OFF -DCMAKE_BUILD_TYPE=PERFORMANCE \
      -DCMAKE_INSTALL_PREFIX=$PWD/build/install
cmake --build build --parallel --target install
```

This installs `build/install/bin/DAMASK_grid`. The Metal backend is enabled by default on macOS
(`-DMETAL=OFF` builds plain DAMASK 3.1.0 behaviour). `env/DAMASK-Metal.sh` also works around an
Open MPI/hwloc hang on chips with three CPU core types; source it in every shell that runs DAMASK.

## 4. Tests

```sh
cmake --build build --target metal-tests
ctest --test-dir build -L cpu                # host tests, no GPU needed (seconds)
ctest --test-dir build -L gpu                # GPU tests (about a minute)
```

All tests must pass. More thorough GPU checks (repeated runs, fault injection) are described in
[usage.md](usage.md#quality-checks).

## 5. First run

```sh
cd examples/grid
../../build/install/bin/DAMASK_grid -g 20grains16x16x16.vti -l tensionX.yaml -m material.yaml
```

The start of the log shows `Metal GPU: <device>` and, per phase, whether it runs on the GPU
(`phase 1: phenopowerlaw constitutive update on GPU`) or on the CPU, with the reason.

## Troubleshooting

- **`PETSc location (PETSC_DIR) is not defined`**: source `env/DAMASK-Metal.sh` (or set the
  variables) before running CMake.
- **`ISO_Fortran_binding.h not found`**: CMake takes this header from `gfortran`; make sure the
  Homebrew `gcc` is the compiler behind `mpif90` (`mpif90 --version`).
- **`mpiexec`/`DAMASK_grid` hangs at start on M-series chips with three core types**: the hwloc
  settings in `env/DAMASK-Metal.sh` are missing.
- **`Metal GPU not available, running on CPU`**: no Metal device (e.g. some virtual machines);
  DAMASK-Metal then behaves like DAMASK 3.1.0.
