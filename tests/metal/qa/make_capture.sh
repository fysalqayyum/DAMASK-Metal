#!/bin/zsh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Capture one GPU batch of a benchmark case for replay (bench_cp_replay, gpu_loop.sh,
# fault_injection.sh): runs DAMASK_grid with DAMASK_METAL_CAPTURE until GPU call <call>, which writes
# <out>/<case><grid>.{phase|ttphase,batch} and exits, then saves the replay output as reference (.ref).
#
# usage: tests/metal/qa/make_capture.sh <case: tdt|twin> [grid=16] [call=100] [out=./capture]
# environment: DAMASK_GRID (default: build/install/bin/DAMASK_grid), BUILD (default: build)
set -eu
root=${0:A:h:h:h:h}
B=${DAMASK_GRID:-$root/build/install/bin/DAMASK_grid}
bin=${BUILD:-$root/build}/tests/metal
c=${1:?case}; g=${2:-16}; call=${3:-100}; out=${4:-$PWD/capture}
[[ -x $bin/bench_cp_replay ]] || { print -u2 "build the tests first: cmake --build build --target metal-tests"; exit 2; }
mkdir -p $out; out=${out:A}
w=$(mktemp -d); trap 'rm -rf $w' EXIT
BENCH_SETUP_ONLY=1 BENCH_OUT=$w $root/benchmarks/run_benchmarks.sh $g $c > /dev/null                # inputs only
cd $w/${c}_${g}_gpu
DAMASK_METAL_CAPTURE=$out/$c$g DAMASK_METAL_CAPTURE_CALL=$call \
  $B -g geom.vti -l load.yaml -m material.yaml -n numerics.yaml > /dev/null 2> capture.err || true
grep 'captured call' capture.err || { print -u2 "no batch captured (fewer than $call GPU calls?)"; exit 1; }
$bin/bench_cp_replay $root/src/grid/metal $out/$c$g 1 --save-ref | tail -1
print "replay with: $bin/bench_cp_replay $root/src/grid/metal $out/$c$g"
