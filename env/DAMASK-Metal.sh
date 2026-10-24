# SPDX-License-Identifier: AGPL-3.0-or-later
# Build and run environment for DAMASK-Metal on Apple silicon (bash or zsh).
# usage: source env/DAMASK-Metal.sh   (set PETSC_DIR/PETSC_ARCH beforehand to override the defaults)

# PETSc built as described in docs/install-macos.md
export PETSC_DIR=${PETSC_DIR:-$HOME/Software/petsc-3.25.6}
export PETSC_ARCH=${PETSC_ARCH:-arch-damask-opt}

# Open MPI's hwloc 2.15 hangs on chips with three CPU performance levels (e.g. M6; fixed in
# hwloc 2.16, open-mpi/hwloc PR #839): skip hwloc's darwin backend and map by hardware thread.
# Harmless on other Macs; remove once Homebrew ships hwloc >= 2.16.
export HWLOC_COMPONENTS=-darwin
export PRTE_MCA_rmaps_default_mapping_policy=hwthread:oversubscribe
export PRTE_MCA_hwloc_default_binding_policy=none

# OpenMP threads for the CPU part (default: all cores)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(sysctl -n hw.ncpu)}

[ -d "$PETSC_DIR/$PETSC_ARCH/lib" ] || echo "DAMASK-Metal: PETSc not found in $PETSC_DIR/$PETSC_ARCH (see docs/install-macos.md)" >&2
