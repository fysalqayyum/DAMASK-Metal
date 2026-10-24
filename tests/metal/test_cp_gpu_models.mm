// GPU constitutive update through the model-independent core (cp_core.h) vs the same core in FP64 on
// the CPU, via the public API (C_Metal_cp_add_phase_model / C_Metal_cp_integrate, incl. certificate):
//  A. kinehardening, 12 slip systems, tension -> compression (back-stress flips)
//  B. phenopowerlaw with 9 slip systems: runs in the padded 12-system kernel; CPU reference unpadded
// usage: test_cp_gpu_models <src/grid/metal directory>
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
#include "../../src/grid/metal/cp_core.h"

static std::string slurp(const std::string& f) { std::ifstream s(f); std::stringstream b; b << s.rdbuf(); return b.str(); }
static std::string strip(std::string s) { size_t p; while ((p = s.find("#include \"")) != std::string::npos) s.erase(p, s.find('\n', p) - p); return s; }
static cp_pair pr(double x) { cp_pair p; p.hi = float(x); p.lo = float(x - double(p.hi)); return p; }
static double rel(const double* a, const double* b, int n) {
  double d = 0, s = 0; for (int k = 0; k < n; ++k) { d += (a[k]-b[k])*(a[k]-b[k]); s += b[k]*b[k]; }
  return std::sqrt(d/std::max(s, 1e-300));
}

static void cF(int N, double* C66, double* P, double* Pp, double* Pn) {
  const double C11 = 198e9, C12 = 125e9, C44 = 122e9;
  for (int k = 0; k < 36; ++k) C66[k] = 0;
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C66[a+6*b] = a == b ? C11 : C12;
  for (int a = 3; a < 6; ++a) C66[a+6*a] = C44;
  const int nrm[4][3] = {{1,1,1},{-1,-1,1},{1,-1,-1},{-1,1,-1}};
  const int dir[12][3] = {{0,1,-1},{-1,0,1},{1,-1,0}, {0,-1,-1},{1,0,1},{-1,1,0},
                          {0,-1,1},{-1,0,-1},{1,1,0}, {0,1,1},{1,0,-1},{-1,-1,0}};
  for (int i = 0; i < N; ++i) for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
    double v = dir[i][a]*nrm[i/3][b] / (std::sqrt(2.0)*std::sqrt(3.0));
    P[9*i+CP_IDX(a,b)] = v; Pp[9*i+CP_IDX(a,b)] = v; Pn[9*i+CP_IDX(a,b)] = -v;
  }
}

// run one model: record CPU-core (double) histories, then integrate all steps as one GPU batch
template<class M, typename Fill, typename Init>
static int run(void* ctx, const char* name, int model, int N, const std::vector<double>& vecs, const std::vector<double>& scal_in,
               Fill fill, Init init, const double* Fdot0, double dt, int n_grains, int n_inc, bool reverse, double Temp,
               const double* Cel = nullptr) {
  typedef cp_layout<M> L;
  const int NS = (M::NBX + M::NBY + M::NBD) * N;
  typename M::param prm; std::memset(&prm, 0, sizeof(prm));
  double C66[36]; std::vector<double> P(9*N), Pp(9*N), Pn(9*N), h(N*N, 1.0);
  cF(N, C66, P.data(), Pp.data(), Pn.data());
  if (Cel) for (int k = 0; k < 36; ++k) C66[k] = 0;
  if (Cel) { for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C66[a+6*b] = a == b ? Cel[0] : Cel[1]; for (int a = 3; a < 6; ++a) C66[a+6*a] = Cel[2]; }
  for (int k = 0; k < 36; ++k) prm.C66[k] = pr(C66[k]);
  for (int k = 0; k < 9*N; ++k) { prm.P_sl[k] = pr(P[k]); prm.P_nS_pos[k] = pr(Pp[k]); prm.P_nS_neg[k] = pr(Pn[k]); }
  for (int k = 0; k < N*N; ++k) prm.h_sl_sl[k] = pr(h[k]);
  fill(prm, h);
  prm.atol_xi = 1.0f; prm.atol_gamma = 1e-6f; prm.N_sl = N;
  cp_numerics num; num.rtol_S = 1e-10f; num.atol_S = 1e-6f; num.rtol_state = 1e-6f; num.stepsize_ls = 0.5f;
  num.trust_S = 0.5f; num.n_stress = 40; num.n_state = 20;
  const double numerics[7] = {1e-10, 1e-6, 1e-6, 0.5, 0.5, 40, 20};
  int ph = C_Metal_cp_add_phase_model(ctx, model, N, C66, P.data(), Pp.data(), Pn.data(), h.data(), vecs.data(), scal_in.data(), numerics);
  if (ph < 0) { std::printf("%s: add_phase failed (%d)\n", name, ph); return 1; }

  std::mt19937_64 g(5); std::uniform_real_distribution<double> u(-1, 1); std::normal_distribution<double> nd;
  std::vector<double> iF, iFp0, iS0, iLp, ist, rP, rst;
  for (int gr = 0; gr < n_grains; ++gr) {
    double q[4], s = 0; for (double& x : q) { x = nd(g); s += x*x; } s = std::sqrt(s); for (double& x : q) x /= s;
    double a=q[0],b=q[1],c=q[2],d=q[3];
    double R[9] = {a*a+b*b-c*c-d*d, 2*(b*c+a*d), 2*(b*d-a*c), 2*(b*c-a*d), a*a-b*b+c*c-d*d, 2*(c*d+a*b), 2*(b*d+a*c), 2*(c*d-a*b), a*a-b*b-c*c+d*d};
    double Fdot[9]; for (int k = 0; k < 9; ++k) Fdot[k] = Fdot0[k] + 0.2*std::fabs(Fdot0[0])*u(g);
    double Fp[9], S3[9] = {0}, Lp[9] = {0}; std::memcpy(Fp, R, sizeof(Fp));
    std::vector<double> st(NS, 0.0); init(st);
    if (M::NBD) {                                                                                   // crystallite_init: dependent state of the initial state
      std::vector<double> Xp(M::ND, 0.0), Dp(M::NDEP > 0 ? M::NDEP : 1);
      for (int blk = 0; blk < M::NBX; ++blk) for (int i = 0; i < N; ++i) Xp[blk*M::N+i] = st[blk*N+i];
      M::template dependent<double>(prm, Xp.data(), nullptr, Temp, Dp.data());
      for (int blk = 0; blk < M::NBD; ++blk) for (int i = 0; i < N; ++i) st[(M::NBX+M::NBY+blk)*N+i] = Dp[blk*M::N+i];
    }
    for (int inc = 0; inc < n_inc; ++inc) {
      double t = (inc+1)*dt, sc = (reverse && inc >= n_inc/2) ? (n_inc/2)*dt - (t - (n_inc/2)*dt) : t;
      double F[9]; for (int k = 0; k < 9; ++k) F[k] = (k%4==0) + Fdot[k]*sc;
      std::vector<double> in(L::n_in, 0.0), out(L::n_out, 0.0);
      for (int k = 0; k < 9; ++k) { in[L::F+k] = F[k]; in[L::Fp0+k] = Fp[k]; in[L::Lp0+k] = Lp[k]; }
      const double sv[6] = {S3[0], S3[4], S3[8], S3[7], S3[6], S3[3]};
      for (int k = 0; k < 6; ++k) in[L::S0+k] = sv[k];
      for (int blk = 0; blk < M::NBX + M::NBY + M::NBD; ++blk) for (int i = 0; i < N; ++i)
        in[(blk < M::NBX ? L::X0 + blk*N : blk < M::NBX + M::NBY ? L::Y0 + (blk-M::NBX)*N : L::D0 + (blk-M::NBX-M::NBY)*N) + i] = st[blk*N + i];
      in[L::T] = Temp;
      int its; int status = cp_core_integrate<double,M>(prm, num, dt, {in.data(),1,0}, {out.data(),1,0}, its);
      if (status != CP_OK) { std::printf("%s: CPU reference failed g%d i%d (%d)\n", name, gr, inc, status); break; }
      iF.insert(iF.end(), F, F+9); iFp0.insert(iFp0.end(), Fp, Fp+9); iS0.insert(iS0.end(), S3, S3+9); iLp.insert(iLp.end(), Lp, Lp+9);
      ist.insert(ist.end(), st.begin(), st.end());
      double iFpn[9]; for (int k = 0; k < 9; ++k) iFpn[k] = out[L::iFp+k];
      double dd = cp_inv33<double>(iFpn, Fp), scl = std::cbrt(1.0/dd); for (double& x : Fp) x /= scl;
      double S6[6]; for (int k = 0; k < 6; ++k) S6[k] = out[L::S+k];
      double Sm[9] = {S6[0], S6[5], S6[4], S6[5], S6[1], S6[3], S6[4], S6[3], S6[2]}; std::memcpy(S3, Sm, sizeof(S3));
      for (int k = 0; k < 9; ++k) Lp[k] = out[L::Lp+k];
      for (int blk = 0; blk < M::NBX + M::NBY + M::NBD; ++blk) for (int i = 0; i < N; ++i)
        st[blk*N + i] = out[(blk < M::NBX ? L::X + blk*N : blk < M::NBX + M::NBY ? L::Y + (blk-M::NBX)*N : L::D + (blk-M::NBX-M::NBY)*N) + i];
      rP.insert(rP.end(), &out[L::P], &out[L::P]+9); rst.insert(rst.end(), st.begin(), st.end());
    }
  }
  const int n = int(iF.size()/9);
  std::vector<double> gP(9*n), gS(9*n), gFp(9*n), gFe(9*n), gLp(9*n), gst(NS*size_t(n)); std::vector<float> T(81*size_t(n));
  std::vector<int> status(n), its(n);
  std::vector<double> Tv(n, Temp);
  if (C_Metal_cp_integrate(ctx, ph, n, dt, iF.data(), iFp0.data(), iS0.data(), iLp.data(), ist.data(), Tv.data(),
                           gP.data(), gS.data(), gFp.data(), gFe.data(), gLp.data(), gst.data(), T.data(), status.data(), its.data())) {
    std::printf("%s: GPU call failed\n", name); return 1;
  }
  double eP = 0, eS = 0; int fail = 0, rej = 0, flips = 0;
  for (int i = 0; i < n; ++i) {
    if (status[i] != CP_OK) { ++fail; rej += status[i] == CP_FAIL_VERIFY; continue; }
    eP = std::max(eP, rel(&gP[9*i], &rP[9*i], 9));
    eS = std::max(eS, rel(&gst[NS*size_t(i)], &rst[NS*size_t(i)], NS));
    if (M::NBY) for (int q = 0; q < N; ++q) flips += std::fabs(rst[NS*size_t(i) + M::NBX*N + q] - ist[NS*size_t(i) + M::NBX*N + q]) > 0.1;
  }
  const bool fast = getenv("DAMASK_METAL_PRECISION") && !std::strcmp(getenv("DAMASK_METAL_PRECISION"), "fast");  // DAMASK-level accuracy
  bool ok = n == n_grains*n_inc && eP < (fast ? 2e-6 : 1e-8) && eS < (fast ? 1e-5 : 1e-6) && fail <= n/1000;
  std::printf("%s: %d steps%s, GPU vs CPU core: max rel err P %.2e  state %.2e; to CPU %d (certificate rejected %d)  %s\n",
              name, n, M::NBY ? (" (" + std::to_string(flips) + " back-stress flips)").c_str() : "", eP, eS, fail, rej, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s src/grid/metal\n", argv[0]); return 2; }
  std::string d = argv[1];
  std::string src = strip(slurp(d + "/cp_gpu_config.h") + slurp(d + "/df64.h") + slurp(d + "/cp_kernel.h")
                        + slurp(d + "/cp_slim.h") + slurp(d + "/cp_core.h") + slurp(d + "/cp_models_composite.h") + slurp(d + "/cp.metal"));
  void* ctx = C_Metal_new(8, 8, 8, slurp(d + "/spectral.metal").c_str());
  if (!ctx || C_Metal_cp_set_source(ctx, src.c_str())) { std::printf("Metal init failed\n"); return 1; }
  int fails = 0;
  {
    const int N = 12;
    std::vector<double> v(8*N);
    for (int i = 0; i < N; ++i) { v[i] = 1e-4; v[N+i] = 20; v[2*N+i] = 0.065e9; v[3*N+i] = 55e9; v[4*N+i] = 0.045e9;
                                  v[5*N+i] = 1.3e9; v[6*N+i] = 0.015e9; v[7*N+i] = 0.027e9; }
    auto fill = [&](cp_kineh_param& p, std::vector<double>&) {
      for (int i = 0; i < N; ++i) { p.dot_gamma_0[i] = pr(1e-4); p.n[i] = pr(20); p.h_0_xi[i] = pr(0.065e9); p.h_0_chi[i] = pr(55e9);
                                    p.h_inf_xi[i] = pr(0.045e9); p.h_inf_chi[i] = pr(1.3e9); p.xi_inf[i] = pr(0.015e9); p.chi_inf[i] = pr(0.027e9); }
    };
    const double Fdot[9] = {4e-4, 0, 0, 0, -2e-4, 0, 0, 0, -2e-4};
    auto init = [&](std::vector<double>& st) { for (int i = 0; i < N; ++i) st[i] = 0.070e9; };
    fails += run<cp_model_kineh<12>>(ctx, "A. kinehardening (12)", 2, N, v, {1.0, 1e-6}, fill, init, Fdot, 0.5, 30, 120, true, 0.0);
  }
  {
    const int N = 9;
    std::vector<double> v(5*N);
    for (int i = 0; i < N; ++i) { v[i] = 1e-3; v[N+i] = 20; v[2*N+i] = 2.25; v[3*N+i] = 75e6; v[4*N+i] = 63e6; }
    auto fill = [&](cp_pheno_param& p, std::vector<double>&) {
      for (int i = 0; i < N; ++i) { p.dot_gamma_0[i] = pr(1e-3); p.n[i] = pr(20); p.a[i] = pr(2.25); p.h_0[i] = pr(75e6); p.xi_inf[i] = pr(63e6); }
    };
    const double Fdot[9] = {1e-3, 0, 0, 0, -5e-4, 0, 0, 0, -5e-4};
    auto init = [&](std::vector<double>& st) { for (int i = 0; i < N; ++i) st[i] = 31e6; };
    fails += run<cp_model_pheno<9>>(ctx, "B. phenopowerlaw (9, padded to 12)", 1, N, v, {1.0, 1e-6}, fill, init, Fdot, 1.0, 40, 120, false, 0.0);
  }
  {
    const int N = 12; const double b = 2.72e-10;
    const double vals[12] = {b, 2.61154e-19, 1.0, 2.03e9, 0.86, 1.69, 2.566e-10, 2.992e-9, 9.1e11, 8.3e-5, b, b*b*b};
    std::vector<double> v(12*N + N*N), fp(N*N);
    const int nrm[4][3] = {{1,1,1},{-1,-1,1},{1,-1,-1},{-1,1,-1}};
    const int dir[12][3] = {{0,1,-1},{-1,0,1},{1,-1,0}, {0,-1,-1},{1,0,1},{-1,1,0},
                            {0,-1,1},{-1,0,-1},{1,1,0}, {0,1,1},{1,0,-1},{-1,-1,0}};
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
      double t[3] = {double(dir[j][1]*nrm[j/3][2]-dir[j][2]*nrm[j/3][1]), double(dir[j][2]*nrm[j/3][0]-dir[j][0]*nrm[j/3][2]),
                     double(dir[j][0]*nrm[j/3][1]-dir[j][1]*nrm[j/3][0])};
      fp[i+N*j] = std::fabs(nrm[i/3][0]*t[0]+nrm[i/3][1]*t[1]+nrm[i/3][2]*t[2])/(std::sqrt(3.0)*std::sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]));
    }
    for (int k = 0; k < 12; ++k) for (int i = 0; i < N; ++i) v[k*N+i] = vals[k];
    for (int k = 0; k < N*N; ++k) v[12*N+k] = fp[k];
    const std::vector<double> sc = {1.0, 1e-6, 1e20, 0.0, 1.0, 160e9, 0.2796, 1.380649e-23, 1.0};
    auto fill = [&](cp_dtung_param& p, std::vector<double>& h) {
      for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) { h[i+N*j] = i == j ? 0.009 : (i/3 == j/3 ? 0.72 : 0.05); p.h_sl_sl[i+N*j] = pr(h[i+N*j]); p.forest[i+N*j] = pr(fp[i+N*j]); }
      cp_pair* f[12] = {p.b_sl, p.Q_s, p.i_sl, p.tau_P, p.p, p.q, p.h, p.w, p.omega, p.B, p.d_caron, p.f_at};
      for (int k = 0; k < 12; ++k) for (int i = 0; i < N; ++i) f[k][i] = pr(vals[k]);
      p.D = pr(sc[2]); p.D_0 = pr(sc[3]); p.Q_cl = pr(sc[4]); p.mu = pr(sc[5]); p.nu = pr(sc[6]); p.k_B = pr(sc[7]); p.dipoleformation = 1;
    };
    auto init = [&](std::vector<double>& st) { for (int i = 0; i < N; ++i) { st[i] = 1e12; st[N+i] = 1.0; } };
    const double Fdot[9] = {2e-4, 0, 0, 0, -1e-4, 0, 0, 0, -1e-4}, Cel[3] = {523e9, 203e9, 160e9};
    fails += run<cp_model_dtung<12>>(ctx, "C. dislotungsten (12, T = 300 K)", 3, N, v, sc, fill, init, Fdot, 0.5, 30, 120, false, 300.0, Cel);
  }
  C_Metal_delete(ctx);
  return fails;
}
