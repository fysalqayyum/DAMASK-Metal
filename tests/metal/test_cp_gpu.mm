// Step B: constitutive update on the GPU (df64 residuals, float Jacobian; src/grid/metal/cp.metal via
// DAMASK_Metal.mm) vs. the same algorithm in double on the CPU (tests/metal/ref_pheno_kernel.h).
// Inputs are realistic: 40 grains x 120 increments of tension (examples/grid material, Al, 12 slip
// systems) recorded from the CPU double run; all steps are then integrated as one GPU batch.
// usage: test_cp_gpu <src/grid/metal directory> [n_points_for_timing]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "../../src/grid/metal/DAMASK_Metal.h"
#include "ref_pheno_kernel.h"
#include "metal_test_util.h"

static const int N = 12;


int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s src/grid/metal [n_timing]\n", argv[0]); return 2; }
  std::string dir = argv[1];
  std::string src = cp_shader_source(dir);
  const int n_timing = argc > 2 ? std::atoi(argv[2]) : 262144;

  // material: examples/grid/material.yaml (Al, cF, phenopowerlaw), double precision
  double C66[36] = {0}, P_sl[9*N], P_pos[9*N], P_neg[9*N], h_ss[N*N], g0[N], nn[N], aa[N], h0[N], xinf[N];
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C66[a+6*b] = a == b ? 106.75e9 : 60.41e9;
  for (int a = 3; a < 6; ++a) C66[a+6*a] = 28.34e9;
  const int nrm[4][3] = {{1,1,1},{-1,-1,1},{1,-1,-1},{-1,1,-1}};
  const int dir_[12][3] = {{0,1,-1},{-1,0,1},{1,-1,0}, {0,-1,-1},{1,0,1},{-1,1,0},
                           {0,-1,1},{-1,0,-1},{1,1,0}, {0,1,1},{1,0,-1},{-1,-1,0}};
  for (int i = 0; i < N; ++i) {
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
      double v = dir_[i][a]*nrm[i/3][b] / (std::sqrt(2.0)*std::sqrt(3.0));
      P_sl[9*i+CP_IDX(a,b)] = v; P_pos[9*i+CP_IDX(a,b)] = v; P_neg[9*i+CP_IDX(a,b)] = -v;
    }
    g0[i] = 0.001; nn[i] = 20; aa[i] = 2.25; h0[i] = 75e6; xinf[i] = 63e6;
    for (int j = 0; j < N; ++j) h_ss[i+N*j] = i == j ? 1.0 : 1.4;
  }
  const double numerics[7] = {1e-10, 1e-6, 1e-6, 0.5, 0.5, 40, 20};

  // CPU double reference (same struct layout as the GPU, H = double)
  auto pr = [](double x) { cp_pair p; p.hi = float(x); p.lo = float(x - double(p.hi)); return p; };
  cp_pheno_param prm; std::memset(&prm, 0, sizeof(prm));
  for (int k = 0; k < 36; ++k) prm.C66[k] = pr(C66[k]);
  for (int k = 0; k < 9*N; ++k) { prm.P_sl[k] = pr(P_sl[k]); prm.P_nS_pos[k] = pr(P_pos[k]); prm.P_nS_neg[k] = pr(P_neg[k]); }
  for (int k = 0; k < N*N; ++k) prm.h_sl_sl[k] = pr(h_ss[k]);
  for (int k = 0; k < N; ++k) { prm.dot_gamma_0[k] = pr(g0[k]); prm.n[k] = pr(nn[k]); prm.a[k] = pr(aa[k]); prm.h_0[k] = pr(h0[k]); prm.xi_inf[k] = pr(xinf[k]); }
  prm.atol_xi = 1.0f; prm.atol_gamma = 1e-6f; prm.N_sl = N;
  cp_numerics num; num.rtol_S = 1e-12f; num.atol_S = 1e-6f; num.rtol_state = 1e-6f; num.stepsize_ls = 0.5f;
  num.trust_S = 0.5f; num.n_stress = 40; num.n_state = 20;

  // record histories: inputs of every step + double results
  struct In { double F[9], Fp0[9], S0[9], Lp[9], st0[2*N]; };
  struct Out { double P[9], S[9], Fp[9], Fe[9], Lp[9], st[2*N]; float dPdF[81]; };
  std::vector<In> in; std::vector<Out> ref;
  std::mt19937_64 g(7); std::uniform_real_distribution<double> u(-1, 1); std::normal_distribution<double> nd;
  const double dt = 1.0;
  for (int gr = 0; gr < 40; ++gr) {
    double q[4], s = 0; for (double& x : q) { x = nd(g); s += x*x; } s = std::sqrt(s); for (double& x : q) x /= s;
    double a=q[0],b=q[1],c=q[2],d=q[3];
    double R[9] = {a*a+b*b-c*c-d*d, 2*(b*c+a*d), 2*(b*d-a*c), 2*(b*c-a*d), a*a-b*b+c*c-d*d, 2*(c*d+a*b), 2*(b*d+a*c), 2*(c*d-a*b), a*a-b*b-c*c+d*d};
    double Fdot[9] = {1e-3, 0, 0, 0, -5e-4, 0, 0, 0, -5e-4};
    for (double& x : Fdot) x += 2e-4*u(g);
    double Fp[9], S3[9] = {0}, Lp[9] = {0}, st[2*N];
    std::memcpy(Fp, R, sizeof(Fp));
    for (int i = 0; i < N; ++i) { st[i] = 31e6; st[N+i] = 0; }
    for (int inc = 0; inc < 120; ++inc) {
      In x; for (int k = 0; k < 9; ++k) x.F[k] = (k%4==0) + Fdot[k]*dt*(inc+1);
      std::memcpy(x.Fp0, Fp, sizeof(Fp)); std::memcpy(x.S0, S3, sizeof(S3)); std::memcpy(x.Lp, Lp, sizeof(Lp)); std::memcpy(x.st0, st, sizeof(st));
      cp_point<double,N> pt;
      for (int k = 0; k < 9; ++k) { pt.F[k] = x.F[k]; pt.Fp0[k] = Fp[k]; pt.Lp_guess[k] = Lp[k]; }
      pt.S0[0] = S3[0]; pt.S0[1] = S3[4]; pt.S0[2] = S3[8]; pt.S0[3] = S3[7]; pt.S0[4] = S3[6]; pt.S0[5] = S3[3];
      for (int i = 0; i < N; ++i) { pt.xi0[i] = st[i]; pt.gamma0[i] = st[N+i]; }
      cp_integrate<double,N>(prm, num, dt, pt);
      if (pt.status != CP_OK) { std::printf("CPU reference failed (grain %d inc %d)\n", gr, inc); return 1; }
      Out y;
      for (int k = 0; k < 9; ++k) { y.P[k] = pt.P[k]; y.Fp[k] = pt.Fp[k]; y.Fe[k] = pt.Fe[k]; y.Lp[k] = pt.Lp[k]; }
      double Sv[9] = {pt.S[0], pt.S[5], pt.S[4], pt.S[5], pt.S[1], pt.S[3], pt.S[4], pt.S[3], pt.S[2]};
      std::memcpy(y.S, Sv, sizeof(Sv));
      for (int i = 0; i < N; ++i) { y.st[i] = pt.xi[i]; y.st[N+i] = pt.gamma[i]; }
      in.push_back(x); ref.push_back(y);
      std::memcpy(Fp, y.Fp, sizeof(Fp)); std::memcpy(S3, y.S, sizeof(S3)); std::memcpy(Lp, y.Lp, sizeof(Lp)); std::memcpy(st, y.st, sizeof(st));
    }
  }

  void* ctx = C_Metal_new(8, 8, 8, slurp(dir + "/spectral.metal").c_str());                         // spectral part unused here
  if (!ctx) { std::printf("no Metal device\n"); return 1; }
  if (C_Metal_cp_set_source(ctx, src.c_str())) { std::printf("constitutive shaders failed to build\n"); return 1; }
  int ph = C_Metal_cp_add_phase(ctx, N, C66, P_sl, P_pos, P_neg, h_ss, g0, nn, aa, h0, xinf, 1.0, 1e-6, numerics);
  if (ph < 0) { std::printf("add_phase failed (%d)\n", ph); return 1; }

  auto run = [&](int n, const std::vector<In>& src_in, std::vector<Out>& out, std::vector<int>& status, std::vector<int>& its) {
    std::vector<double> F(9*n), Fp0(9*n), S0(9*n), Lp(9*n), st0(2*N*n);
    for (int i = 0; i < n; ++i) {
      const In& x = src_in[i % src_in.size()];
      std::memcpy(&F[9*i], x.F, 72); std::memcpy(&Fp0[9*i], x.Fp0, 72); std::memcpy(&S0[9*i], x.S0, 72);
      std::memcpy(&Lp[9*i], x.Lp, 72); std::memcpy(&st0[2*N*i], x.st0, 16*N);
    }
    std::vector<double> P(9*n), S(9*n), Fp(9*n), Fe(9*n), Lpo(9*n), st(2*N*n); std::vector<float> T(81*size_t(n));
    status.assign(n, -1); its.assign(n, 0);
    auto t0 = std::chrono::steady_clock::now();
    int rc = C_Metal_cp_integrate(ctx, ph, n, 2*N, dt, F.data(), Fp0.data(), S0.data(), Lp.data(), st0.data(), nullptr,
                                  P.data(), S.data(), Fp.data(), Fe.data(), Lpo.data(), st.data(), T.data(), status.data(), its.data());
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    out.resize(n);
    for (int i = 0; i < n; ++i) {
      std::memcpy(out[i].P, &P[9*i], 72); std::memcpy(out[i].S, &S[9*i], 72); std::memcpy(out[i].Fp, &Fp[9*i], 72);
      std::memcpy(out[i].Fe, &Fe[9*i], 72); std::memcpy(out[i].Lp, &Lpo[9*i], 72); std::memcpy(out[i].st, &st[2*N*i], 16*N);
      std::memcpy(out[i].dPdF, &T[81*size_t(i)], 81*sizeof(float));
    }
    return rc ? -1.0 : t;
  };
  auto rel = [](const double* a, const double* b, int n) {
    double d = 0, s = 0; for (int k = 0; k < n; ++k) { d += (a[k]-b[k])*(a[k]-b[k]); s += b[k]*b[k]; }
    return std::sqrt(d/std::max(s, 1e-300));
  };

  std::vector<Out> gpu; std::vector<int> status, its;
  int n = int(in.size());
  if (run(n, in, gpu, status, its) < 0) { std::printf("GPU integration failed\n"); return 1; }
  double eP = 0, eS = 0, eFp = 0, ext = 0; int fail = 0; long it_sum = 0;
  for (int i = 0; i < n; ++i) {
    if (status[i] != CP_OK) { ++fail; continue; }
    eP = std::max(eP, rel(gpu[i].P, ref[i].P, 9)); eS = std::max(eS, rel(gpu[i].S, ref[i].S, 9));
    eFp = std::max(eFp, rel(gpu[i].Fp, ref[i].Fp, 9)); ext = std::max(ext, rel(gpu[i].st, ref[i].st, 2*N));
    it_sum += its[i];
  }
  // GPU tangent (float) vs CPU tangent (double) at the CPU reference state
  double eT = 0;
  for (int i = 0; i < n; ++i) {
    if (status[i] != CP_OK) continue;
    double Sv[6] = {ref[i].S[0], ref[i].S[4], ref[i].S[8], ref[i].S[7], ref[i].S[6], ref[i].S[3]}, td[81], tf[81];
    double iFp[9]; { double Fe_[9]; std::memcpy(Fe_, ref[i].Fe, 72); double iF_[9]; cp_inv33<double>(in[i].F, iF_); cp_mul33<double>(iF_, Fe_, iFp); }  // invFp_new = F^-1 Fe
    if (!cp_tangent<double,N>(prm, dt, in[i].F, in[i].Fp0, iFp, ref[i].Fe, Sv, ref[i].st, td)) continue;
    for (int k = 0; k < 81; ++k) tf[k] = gpu[i].dPdF[k];
    eT = std::max(eT, rel(tf, td, 81));
  }
  int wrong = 0, rejected = 0;
  for (int i = 0; i < n; ++i) { if (status[i] == CP_OK && rel(gpu[i].Fp, ref[i].Fp, 9) > 1e-6) ++wrong; if (status[i] == CP_FAIL_VERIFY) ++rejected; }
  std::printf("GPU (df64) vs CPU (double), %d steps: max rel err P %.2e  S %.2e  Fp %.2e  state %.2e  tangent(float) %.2e\n", n, eP, eS, eFp, ext, eT);
  std::printf("  accepted but wrong: %d, rejected by certificate: %d\n", wrong, rejected);
  std::printf("  not converged on GPU: %d (%.2f %%) -> CPU fallback; mean Newton its %.2f\n", fail, 100.0*fail/n, double(it_sum)/std::max(1, n-fail));

  double t = run(n_timing, in, gpu, status, its);
  double tg, th; C_Metal_last_timing(ctx, &tg, &th);
  int fail_t = 0; for (int s : status) fail_t += s != CP_OK;
  std::printf("throughput: %d points in %.3f s (GPU %.3f s, host pack/unpack %.3f s) = %.2f Mpoints/s, failures %d\n",
              n_timing, t, tg, th, n_timing/t/1e6, fail_t);
  C_Metal_delete(ctx);
  const bool fast = getenv("DAMASK_METAL_PRECISION") && !std::strcmp(getenv("DAMASK_METAL_PRECISION"), "fast");  // DAMASK-level accuracy
  bool ok = eP < (fast ? 2e-6 : 1e-9) && fail <= n/100 && wrong == 0 && eT < 1e-4;
  std::printf("%s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
