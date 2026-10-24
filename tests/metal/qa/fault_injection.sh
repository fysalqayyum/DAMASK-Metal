#!/bin/zsh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Fault injection: DAMASK_METAL_INJECT=<kind> corrupts every 64th GPU result before the certificate
# (kinds 0-6: stress, first state block, last state block, Lp, P, Fe, Fp inverse; see cp_run in
# DAMASK_Metal.mm). The certificate must reject every corrupted point, so each run reports about n/64
# rejections more than without injection, and 'accepted but wrong' stays 0. Kinds that leave a result
# unchanged (e.g. Lp already zero) can be accepted legitimately and show slightly fewer rejections.
#
# usage: tests/metal/qa/fault_injection.sh [capture base for the replay, optional]
# environment: BUILD (default: build)
# The test programs report FAIL under injection because they expect no rejections; read the counts.
set -u
root=${0:A:h:h:h:h}
bin=${BUILD:-$root/build}/tests/metal
src=$root/src/grid/metal
cap=${1:-}
[[ -x $bin/test_cp_gpu ]] || { print -u2 "build the tests first: cmake --build build --target metal-tests"; exit 2; }
log=${LOG:-$PWD/fault_injection.log}; : > $log
for prec in fast mixed; do
  for k in -1 0 1 2 3 4 5 6; do                                                                     # -1: no injection (baseline)
    print "=== $prec inject $k" >> $log
    DAMASK_METAL_PRECISION=$prec DAMASK_METAL_INJECT=$k $bin/test_cp_gpu $src 2>&1 | grep -E "accepted|not converged" | sed 's/^/pheno /' >> $log
    DAMASK_METAL_PRECISION=$prec DAMASK_METAL_INJECT=$k $bin/test_cp_gpu_models $src 2>&1 | grep -o '^[A-C]\..*(certificate rejected [0-9]*)' | sed 's/^/models /' >> $log
    [[ -n $cap ]] && DAMASK_METAL_PRECISION=$prec DAMASK_METAL_INJECT=$k $bin/bench_cp_replay $src $cap 1 2>&1 | grep -o 'CPU fallback.*)' | sed 's/^/replay /' >> $log
    sleep 1
  done
done
wrong=$(grep -c 'accepted but wrong: [1-9]' $log)
print "accepted-but-wrong lines: $wrong (log: $log)"
(( wrong == 0 ))
