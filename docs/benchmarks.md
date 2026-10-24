# Benchmarks

The benchmark compares DAMASK 3.1.0's CPU path (FP64 constitutive update, FFTW, all cores, OpenMP)
with the GPU path of DAMASK-Metal (default settings: `precision: fast`, `split: 1.0`). Both run in
the same executable; the `GPU` switches in `numerics.yaml` select the path.

## Cases

All cases use the 20-grain geometries `examples/grid/20grains{16,32,64}³.vti` and the first load
step of the listed load file.

| Case | Material (`benchmarks/cases`) | Load | Increments (16³/32³) |
|---|---|---|---|
| `pheno` | Al, phenopowerlaw (`examples/grid/material.yaml`) | tension, 10 s | 40 |
| `phenotw` | Mg, phenopowerlaw with twinning | compression, 10 s | 40 |
| `dtung` | W, dislotungsten | tension, 5 s | 20 |
| `tdt` | Mg, tdt (twinning, detwinning; parameters not calibrated) | compression, 10 s | 40 |
| `twin` | fcc, twintrip with twinning (parameters not calibrated) | tension, 150 s | 60 |
| `trip` | fcc, twintrip with ε and α′ martensite (not calibrated) | tension, 150 s, up to 6 cutbacks | 60 |

At 64³, `pheno` runs in full. The other cases run only their first 10 increments (the time scaled
accordingly), and `trip` is not run.

## Results: Apple M6 Mac mini

Hardware: 12 CPU cores (2 super, 4 performance, 6 efficiency), 12 GPU cores, 16 GB memory. Software:
macOS 27.0, GCC 16.2, PETSc 3.25.6. Raw data, including Newton iterations, CPU fallbacks,
certificate rejections and peak memory, is in
[`benchmarks/results/apple-m6-mac-mini_2026-10-24.txt`](../benchmarks/results/apple-m6-mac-mini_2026-10-24.txt).

Wall time in seconds, CPU / GPU (speed-up):

| Case | 16³ | 32³ | 64³ |
|---|---|---|---|
| pheno | 2.50 / 1.20 (2.1×) | 20.9 / 5.93 (3.5×) | 162 / 50.7 (3.2×) |
| phenotw | 5.02 / 2.25 (2.2×) | 41.5 / 9.80 (4.2×) | 101 / 29.3 (3.4×) |
| dtung | 6.29 / 3.47 (1.8×) | 56.3 / 26.0 (2.2×) | 44.7 / 22.3 (2.0×) |
| tdt | 92.7 / 48.5 (1.9×) | 1119 / 348 (3.2×) | 476 / 115 (4.1×) |
| twin | 191 / 54.8 (3.5×) | 1684 / 382 (4.4×) | 1002 / 244 (4.1×) |
| trip | 385 / 119 (3.2×) | 3308 / 842 (3.9×) | – |

- **CPU fallback:** the share of GPU points recomputed on the CPU is 0 % for pheno and phenotw,
  under 0.4 % for twintrip, 0.6–1.1 % for tdt, and 11 % for dislotungsten at 16³ and 32³.
- **Iterations:** twintrip needs about 18 % more Newton iterations on the GPU than on the CPU. It is
  nevertheless the largest gain.
- **Peak memory:** at 64³, 2.2–3.0 GB on the CPU and 2.7–4.3 GB on the GPU.
- **Accuracy:** volume-averaged stresses agree to 10⁻⁷–10⁻⁵ relative between CPU and GPU (TRIP at
  16³: 2·10⁻⁴). Point-wise differences up to about 10⁻³ (a few % for twintrip and tdt at single
  points) are path effects of the solver tolerances: a GPU run in `mixed` precision (agreeing with
  the CPU to 10⁻¹¹ per point on identical inputs) differs from the CPU run by the same amount.

## Reproducing

```sh
source env/DAMASK-Metal.sh
benchmarks/run_benchmarks.sh 16                    # all cases at 16^3, CPU then GPU, one run at a time
benchmarks/run_benchmarks.sh 32 pheno tdt          # selected cases
column -t bench-out/results.txt
python3 benchmarks/compare_runs.py bench-out/tdt_16_cpu/*.hdf5 bench-out/tdt_16_gpu/*.hdf5   # needs h5py
```

Approximate total run times on the M6: 16³ about 20 min; 32³ about 2.5 h; 64³ about 1 h (mostly the
CPU runs). The GPU runs need at most a few GB of memory and do not block the display.
