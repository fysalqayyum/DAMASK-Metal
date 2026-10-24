#!/bin/zsh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Repeated-run check for silent GPU corruption (see docs/design.md, "Certificate"): runs the GPU tests
# N times in fast and mixed precision and, optionally, replays a captured batch, whose difference to
# its reference (bench_cp_replay vs <base>.ref) must be the same in every run, i.e. deterministic.
#
# usage: tests/metal/qa/gpu_loop.sh [N=60] [capture base, e.g. tdt16 (needs <base>.ref, see make_capture.sh)]
# environment: BUILD (default: build) — CMake build directory with 'metal-tests' built
# pass: no 'accepted but wrong' > 0, no FAIL, one distinct replay result
set -u
root=${0:A:h:h:h:h}
bin=${BUILD:-$root/build}/tests/metal
src=$root/src/grid/metal
n=${1:-60}; cap=${2:-}
[[ -x $bin/test_cp_gpu ]] || { print -u2 "build the tests first: cmake --build build --target metal-tests"; exit 2; }
log=${LOG:-$PWD/gpu_loop.log}; : > $log
to() { perl -e 'alarm shift; exec @ARGV' "$@"; }                                                    # per-run timeout (no GNU timeout on macOS)
for r in $(seq 1 $n); do
  print "=== run $r" >> $log
  DAMASK_METAL_PRECISION=fast  to 60 $bin/test_cp_gpu $src 2>&1 | grep -E "accepted|OK|FAIL" | sed 's/^/pheno-fast /' >> $log
  DAMASK_METAL_PRECISION=mixed to 60 $bin/test_cp_gpu $src 2>&1 | grep -E "accepted|OK|FAIL" | sed 's/^/pheno-mixed /' >> $log
  DAMASK_METAL_PRECISION=fast  to 60 $bin/test_cp_gpu_models $src 2>&1 | sed 's/^/models-fast /' >> $log
  [[ -n $cap ]] && DAMASK_METAL_PRECISION=fast to 60 $bin/bench_cp_replay $src $cap 1 2>&1 | grep 'vs reference' | sed 's/^/replay /' >> $log
  sleep 1                                                                                           # let the GPU idle between runs
done
wrong=$(grep -c 'accepted but wrong: [1-9]' $log); fails=$(grep -c 'FAIL' $log)
distinct=$(grep '^replay' $log | sort -u | wc -l)
print "runs $n: accepted-but-wrong lines $wrong, FAIL lines $fails, distinct replay results $distinct (log: $log)"
(( wrong + fails == 0 && distinct <= 1 ))
