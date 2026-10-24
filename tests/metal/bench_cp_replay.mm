// Replay benchmark of the GPU constitutive update: loads a batch captured from a DAMASK run
// (DAMASK_METAL_CAPTURE=<base>, DAMASK_METAL_CAPTURE_CALL=<k>: <base>.phase (tdt) or .ttphase (twintrip),
// <base>.batch), integrates it
// R times through the public API (incl. certificate) and reports time per point, CPU fallbacks and
// the difference to a reference output (<base>.ref, written with --save-ref).
// usage: bench_cp_replay <src/grid/metal> <base> [R] [--save-ref]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../../src/grid/metal/DAMASK_Metal.h"
#include "metal_test_util.h"


int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s src/grid/metal base [R] [--save-ref]\n", argv[0]); return 2; }
  const std::string d = argv[1], base = argv[2];
  const int R = argc > 3 ? std::max(1, atoi(argv[3])) : 5;
  const bool save = argc > 4 && std::string(argv[4]) == "--save-ref";

  FILE* fp = std::fopen((base + ".phase").c_str(), "rb");                                         // tdt
  const bool tt = !fp && (fp = std::fopen((base + ".ttphase").c_str(), "rb"));                    // twintrip
  FILE* fb = std::fopen((base + ".batch").c_str(), "rb");
  if (!fp || !fb) { std::fprintf(stderr, "missing %s.phase/.ttphase/.batch\n", base.c_str()); return 1; }
  const int kMaxBlob = 65536, kMaxState = 1024, kMaxPoints = 1 << 24;                              // sanity limits for file-supplied sizes
  int dims[11], nb = -1, model = CP_MODEL_TDT;
  double C66[36], atols[3], numerics[7];
  std::vector<double> atolv;
  bool okr = (!tt || std::fread(&model, sizeof(int), 1, fp) == 1)
          && std::fread(dims, sizeof(int), tt ? 11 : 4, fp) == size_t(tt ? 11 : 4) && std::fread(&nb, sizeof(int), 1, fp) == 1;
  if (!okr || nb < 0 || nb > kMaxBlob || (tt && (dims[10] < 0 || dims[10] > kMaxState))) {
    std::fprintf(stderr, "invalid phase file\n"); return 1;
  }
  std::vector<double> blob(nb);
  okr = okr && std::fread(C66, 8, 36, fp) == 36 && std::fread(blob.data(), 8, nb, fp) == size_t(nb);
  if (tt) { atolv.resize(dims[10]); okr = okr && std::fread(atolv.data(), 8, dims[10], fp) == size_t(dims[10]); }
  else okr = okr && std::fread(atols, 8, 3, fp) == 3;
  okr = okr && std::fread(numerics, 8, 7, fp) == 7;
  int n = 0, NS = 0; double dt;
  okr = okr && std::fread(&n, 4, 1, fb) == 1 && std::fread(&NS, 4, 1, fb) == 1 && std::fread(&dt, 8, 1, fb) == 1;
  if (!okr || n < 1 || n > kMaxPoints || NS < 1 || NS > kMaxState) { std::fprintf(stderr, "invalid batch file\n"); return 1; }
  std::vector<double> F(9*size_t(n)), Fp0(9*size_t(n)), S0(9*size_t(n)), Lpg(9*size_t(n)), st0(size_t(NS)*n), T(n);
  for (auto* a : {&F, &Fp0, &S0, &Lpg}) okr = okr && std::fread(a->data(), 8, 9*size_t(n), fb) == 9*size_t(n);
  okr = okr && std::fread(st0.data(), 8, st0.size(), fb) == st0.size() && std::fread(T.data(), 8, n, fb) == size_t(n);
  if (!okr) { std::fprintf(stderr, "read error\n"); return 1; }
  std::fclose(fp); std::fclose(fb);
  if (const char* sub = getenv("BENCH_N")) n = std::min(n, std::max(1, atoi(sub)));                 // first BENCH_N points only (arrays keep their stride)
  if (const char* rep = getenv("BENCH_REPLICATE")) {                                                // more points: copies of the batch
    const int k = std::min(std::max(1, atoi(rep)), kMaxPoints/n);
    auto grow = [&](std::vector<double>& a) { size_t m = a.size(); a.resize(m*k); for (int c = 1; c < k; ++c) std::copy(a.begin(), a.begin()+m, a.begin()+c*m); };
    for (auto* a : {&F, &Fp0, &S0, &Lpg, &st0, &T}) grow(*a);
    n *= k;
  }

  std::string src = std::string(getenv("BENCH_DEFINES") ? getenv("BENCH_DEFINES") : "") + "\n" + cp_shader_source(d);
  void* ctx = C_Metal_new(8, 8, 8, slurp(d + "/spectral.metal").c_str());
  if (!ctx || C_Metal_cp_set_source(ctx, src.c_str())) { std::printf("Metal init failed\n"); return 1; }
  if (getenv("BENCH_NSTATE")) numerics[6] = atof(getenv("BENCH_NSTATE"));                          // state iteration cap
  if (getenv("BENCH_NSTRESS")) numerics[5] = atof(getenv("BENCH_NSTRESS"));
  int ph = tt ? C_Metal_cp_add_phase_twintrip(ctx, dims, nb, C66, blob.data(), atolv.data(), numerics)
              : C_Metal_cp_add_phase_tdt(ctx, dims, nb, C66, blob.data(), atols, numerics);
  if (ph < 0) { std::printf("add_phase failed (%d)\n", ph); return 1; }

  std::vector<double> P(9*size_t(n)), S(9*size_t(n)), Fp(9*size_t(n)), Fe(9*size_t(n)), Lp(9*size_t(n)), st(size_t(NS)*n);
  std::vector<float> dPdF(81*size_t(n));
  std::vector<int> stat(n), it(n);
  double best = 1e30, sum = 0;
  for (int r = 0; r <= R; ++r) {                                                                    // r = 0: warm-up (shader compilation)
    auto t0 = std::chrono::steady_clock::now();
    if (C_Metal_cp_integrate(ctx, ph, n, NS, dt, F.data(), Fp0.data(), S0.data(), Lpg.data(), st0.data(), T.data(),
                             P.data(), S.data(), Fp.data(), Fe.data(), Lp.data(), st.data(), getenv("BENCH_NO_TANGENT") ? nullptr : dPdF.data(), stat.data(), it.data())) {
      std::printf("integrate failed\n"); return 1;
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (r > 0) { best = std::min(best, t); sum += t; }
  }
  if (getenv("BENCH_ITER_HIST")) {                                                                  // stress iterations per point: distribution, failed points separately
    std::vector<int> ok_i, bad_i;
    for (int i = 0; i < n; ++i) (stat[i] ? bad_i : ok_i).push_back(it[i]);
    std::sort(ok_i.begin(), ok_i.end()); std::sort(bad_i.begin(), bad_i.end());
    auto q = [](const std::vector<int>& v, double f) { return v.empty() ? -1 : v[std::min(v.size()-1, size_t(f*v.size()))]; };
    std::printf("iterations converged: median %d p90 %d p99 %d max %d | failed (%zu): median %d max %d\n",
                q(ok_i,.5), q(ok_i,.9), q(ok_i,.99), q(ok_i,1.0), bad_i.size(), q(bad_i,.5), q(bad_i,1.0));
    for (int i = 0; i < n; ++i) if (stat[i]) std::printf(" %d:%d", i, stat[i]);
    std::printf("\n");
  }
  int fails = 0, rejected = 0; long its = 0;
  for (int i = 0; i < n; ++i) { fails += stat[i] != 0; rejected += stat[i] == 5 /* CP_FAIL_VERIFY */; its += it[i]; }
  if (getenv("BENCH_DEFINES") && strstr(getenv("BENCH_DEFINES"), "CP_ITER_SPLIT")) {
    long a = 0, b = 0, c = 0; int m = 0;
    for (int i = 0; i < n; ++i) if (!stat[i]) { ++m; a += it[i]/1000000; b += (it[i]%1000000)/1000; c += it[i]%1000; }
    std::printf("per point: state iterations %.2f, float presolve iterations %.2f, df64 stress iterations %.2f\n", double(a)/m, double(b)/m, double(c)/m);
  }
  std::printf("points %d  dt %.3g  best %.2f ms  mean %.2f ms  -> %.2f us/point  CPU fallback %d (certificate rejected %d)  mean stress iterations %.1f\n",
              n, dt, 1e3*best, 1e3*sum/R, 1e6*best/n, fails, rejected, double(its)/n);
  const std::string ref = base + ".ref";
  if (save) {
    FILE* fo = std::fopen(ref.c_str(), "wb");
    std::fwrite(stat.data(), 4, n, fo); std::fwrite(P.data(), 8, 9*size_t(n), fo); std::fwrite(st.data(), 8, st.size(), fo);
    std::fclose(fo); std::printf("reference saved\n");
  } else if (FILE* fi = std::fopen(ref.c_str(), "rb")) {
    std::vector<int> s2(n); std::vector<double> P2(9*n), st2(st.size());
    bool ok2 = std::fread(s2.data(), 4, n, fi) == size_t(n) && std::fread(P2.data(), 8, P2.size(), fi) == P2.size()
            && std::fread(st2.data(), 8, st2.size(), fi) == st2.size();
    std::fclose(fi);
    if (!ok2) { std::printf("bad reference\n"); return 1; }
    double eP = 0, eX = 0, wa = 0, wb = 0; int both = 0, wq = -1, wi = -1;
    for (int i = 0; i < n; ++i) {
      if (stat[i] || s2[i]) continue;
      ++both;
      double d = 0, s = 0;
      for (int k = 0; k < 9; ++k) { d += (P[9*i+k]-P2[9*i+k])*(P[9*i+k]-P2[9*i+k]); s += P2[9*i+k]*P2[9*i+k]; }
      eP = std::max(eP, std::sqrt(d/std::max(s, 1e-300)));
      for (int q = 0; q < NS; ++q) {
        double a = st[size_t(NS)*i+q], b = st2[size_t(NS)*i+q];
        double e = std::fabs(a-b)/std::max(std::fabs(b), 1e-3);
        if (e > eX) { eX = e; wq = q; wa = a; wb = b; wi = i; }
      }
    }
    std::printf("vs reference (%d points ok in both): max rel |dP| %.2e  max state diff %.2e (point %d, state entry %d: %.10g vs %.10g)\n", both, eP, eX, wi, wq, wa, wb);
    if (getenv("BENCH_STATE_HIST")) {                                                               // worst difference per state entry
      for (int q = 0; q < NS; ++q) { double m = 0, v = 0;
        for (int i = 0; i < n; ++i) { if (stat[i] || s2[i]) continue; double a = st[size_t(NS)*i+q], b = st2[size_t(NS)*i+q];
          double e = std::fabs(a-b)/std::max(std::fabs(b), 1e-3); if (e > m) { m = e; v = b; } }
        std::printf("  entry %3d  max rel diff %.2e  (value %.4g)\n", q, m, v); }
    }
  }
  C_Metal_delete(ctx);
  return 0;
}
