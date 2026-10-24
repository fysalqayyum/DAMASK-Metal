#!/bin/zsh
# SPDX-License-Identifier: AGPL-3.0-or-later
# CPU (FP64, FFTW) vs GPU (Metal, fast precision) benchmark of DAMASK_grid, one run at a time.
#
# usage: benchmarks/run_benchmarks.sh <grid: 16|32|64> [case ...]
#   cases: pheno phenotw dtung tdt twin trip   (default: all)
# environment:
#   DAMASK_GRID   DAMASK_grid executable   (default: build/install/bin/DAMASK_grid of this repository)
#   BENCH_OUT     output directory          (default: ./bench-out)
#   OMP_NUM_THREADS                         (default: all cores)
#   BENCH_SETUP_ONLY=1                      write the run directories without running DAMASK
#
# Each case uses the first load step of benchmarks/cases/<case>/load.yaml (pheno: examples/grid) with
# t and N replaced as listed below, identical for CPU and GPU. At 64^3 all cases except pheno run only
# their first 10 increments. One line per run is appended to $BENCH_OUT/results.txt:
#   case grid variant wall_s increments newton_its gpu_points to_cpu_pct rejected max_rss_MB exit
# Compare CPU and GPU results of a case with benchmarks/compare_runs.py (needs h5py).
set -u
root=${0:A:h:h}
B=${${DAMASK_GRID:-$root/build/install/bin/DAMASK_grid}:A}
OUT=${${BENCH_OUT:-$PWD/bench-out}:A}                                                             # absolute: the loop changes directory
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(sysctl -n hw.ncpu)}
[[ -x $B ]] || { print -u2 "DAMASK_grid not found: $B (set DAMASK_GRID)"; exit 2; }
g=${1:?usage: run_benchmarks.sh <16|32|64> [case ...]}; shift
[[ $g == (16|32|64) ]] || { print -u2 "grid must be 16, 32 or 64"; exit 2; }
if (( $# )); then cases=($@); else cases=(pheno phenotw dtung tdt twin trip); fi
for c in $cases; do                                                                                 # before anything is deleted below
  [[ $c == (pheno|phenotw|dtung|tdt|twin|trip) ]] || { print -u2 "unknown case $c"; exit 2; }
done
mkdir -p $OUT
[[ -f $OUT/results.txt ]] || print "case grid variant wall_s increments newton_its gpu_points to_cpu_pct rejected max_rss_MB exit" > $OUT/results.txt

# first load step of <file> with t and N replaced; output every N increments (first and last)
mkload() {
  python3 - "$1" "$2" "$3" > load.yaml <<'PY'
import sys, re
src, t, N = sys.argv[1], sys.argv[2], sys.argv[3]
head, steps = open(src).read().split('loadstep:')
first = re.split(r'\n  - ', '\n' + steps.strip('\n'))[1]
first = re.sub(r't:\s*[0-9.e+-]+', 't: ' + t, first, count=1)
first = re.sub(r'N:\s*[0-9]+', 'N: ' + N, first, count=1)
first = re.sub(r'f_out:\s*[0-9]+', 'f_out: ' + N, first)
print(head + 'loadstep:\n  - ' + first.rstrip())
PY
}

for c in $cases; do
  for v in cpu gpu; do
    d=$OUT/${c}_${g}_$v
    rm -rf $d; mkdir -p $d; cd $d || exit 2
    cp $root/examples/grid/20grains${g}x${g}x${g}.vti geom.vti
    C=$root/benchmarks/cases/$c
    cut=3
    case $c in                                                                                      # load step: t N (64^3: first 10 increments)
      pheno)   cp $root/examples/grid/material.yaml .; mkload $root/examples/grid/tensionX.yaml 10 40 ;;
      phenotw) cp $C/material.yaml .; if [[ $g == 64 ]]; then mkload $C/load.yaml 2.5 10; else mkload $C/load.yaml 10 40; fi ;;
      dtung)   cp $C/material.yaml .; if [[ $g == 64 ]]; then mkload $C/load.yaml 2.5 10; else mkload $C/load.yaml 5 20; fi ;;
      tdt)     cp $C/material.yaml .; if [[ $g == 64 ]]; then mkload $C/load.yaml 2.5 10; else mkload $C/load.yaml 10 40; fi ;;
      twin)    cp $C/material.yaml .; if [[ $g == 64 ]]; then mkload $C/load.yaml 25 10;  else mkload $C/load.yaml 150 60; fi ;;
      trip)    cp $C/material.yaml .; cut=6; mkload $C/load.yaml 150 60 ;;                       # marginal on the CPU too: more cutbacks
      *) print -u2 "unknown case $c"; exit 2 ;;
    esac
    if [[ $v == cpu ]]; then gpu=false; else gpu=true; fi
    print "solver:\n  grid:\n    N_cutback_max: $cut\n    mechanical:\n      N_iter_min: 4\n      N_iter_max: 40\n    GPU:\n      Gamma: $gpu\n      constitutive: $gpu\n      tangent: $gpu" > numerics.yaml
    [[ -n ${BENCH_SETUP_ONLY:-} ]] && continue
    DAMASK_METAL_VERBOSE=1 /usr/bin/time -lp $B -g geom.vti -l load.yaml -m material.yaml -n numerics.yaml > run.log 2> err.log
    rc=$?
    wall=$(awk '/^real/{print $2}' err.log)
    rss=$(awk '/maximum resident set size/{printf "%.0f", $1/1048576}' err.log)
    incs=$(awk 'NR>1{print $1}' *.sta | tail -1)
    its=$(awk 'NR>1{s+=$5} END{print s}' *.sta)
    stats=$(grep 'DAMASK_Metal: cp ' err.log | sed -E 's/.*cp ([0-9]+) points, ([0-9]+) to CPU \(([0-9]+) rejected.*/\1 \2 \3/' \
            | awk '{p+=$1; n+=$2; r+=$3} END{if (p > 0) printf "%d %.3f %d", p, 100*n/p, r; else printf "0 - -"}')
    print "$c $g $v $wall $incs $its $stats $rss $rc" | tee -a $OUT/results.txt
  done
done
