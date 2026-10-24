// CPU validation of src/grid/metal/cp_kernel.h (S-formulation, mixed precision) against an
// independent C++ port of DAMASK's integrateStress/integrateStateFPI (Lp-formulation, double).
// Material: examples/grid/material.yaml (Al, cF, phenopowerlaw, 12 slip systems).
// build: clang++ -std=c++17 -O2 -ffp-contract=off test_cp_kernel.cpp
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include "../../src/grid/metal/cp_kernel.h"

static const int N = 12;
typedef double M33[9];                                                                              // column-major

static cp_pair pr(double x) { cp_pair p; p.hi = float(x); p.lo = float(x - double(p.hi)); return p; }
static double  dv(cp_pair p) { return double(p.hi) + double(p.lo); }

//--------------------------------------------------------------------------------------------------
// material (examples/grid/material.yaml)
static void setup(cp_pheno_param& p, cp_numerics& num, double tight) {
  std::memset(&p, 0, sizeof(p));
  const double C11 = 106.75e9, C12 = 60.41e9, C44 = 28.34e9;
  double C[36] = {0};
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C[a+6*b] = a == b ? C11 : C12;
  for (int a = 3; a < 6; ++a) C[a+6*a] = C44;
  for (int k = 0; k < 36; ++k) p.C66[k] = pr(C[k]);
  const int nrm[4][3] = {{1,1,1},{-1,-1,1},{1,-1,-1},{-1,1,-1}};
  const int dir[12][3] = {{0,1,-1},{-1,0,1},{1,-1,0}, {0,-1,-1},{1,0,1},{-1,1,0},
                          {0,-1,1},{-1,0,-1},{1,1,0}, {0,1,1},{1,0,-1},{-1,-1,0}};
  for (int i = 0; i < N; ++i) {
    const int* n = nrm[i/3]; const int* d = dir[i];
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
      double v = d[a]*n[b] / (std::sqrt(2.0)*std::sqrt(3.0));
      p.P_sl[9*i+CP_IDX(a,b)] = pr(v); p.P_nS_pos[9*i+CP_IDX(a,b)] = pr(v); p.P_nS_neg[9*i+CP_IDX(a,b)] = pr(-v);
    }
    p.dot_gamma_0[i] = pr(0.001); p.n[i] = pr(20); p.a[i] = pr(2.25);
    p.h_0[i] = pr(75e6); p.xi_inf[i] = pr(63e6);
    for (int j = 0; j < N; ++j) p.h_sl_sl[i+N*j] = pr(i == j ? 1.0 : 1.4);
  }
  p.atol_xi = 1.0f; p.atol_gamma = 1e-6f; p.N_sl = N;
  num.rtol_S = float(tight); num.atol_S = 1e-6f; num.rtol_state = 1e-6f;
  num.stepsize_ls = 0.5f; num.trust_S = 0.5f; num.n_stress = 40; num.n_state = 20;
}

//--------------------------------------------------------------------------------------------------
// reference: port of DAMASK integrateStress (Lp residual) + integrateStateFPI, Fi = I, double
struct Ref { double S[6], xi[N], gamma[N], P[9], Fp[9], Lp[9]; int it_stress; };

static void ref_kin(const cp_pheno_param& p, const double* Mp, const double* xi, double* gd, double* dgd, bool* pos) {
  for (int i = 0; i < N; ++i) {
    double tp = 0, tn = 0;
    for (int m = 0; m < 9; ++m) { tp += Mp[m]*dv(p.P_nS_pos[9*i+m]); tn += Mp[m]*dv(p.P_nS_neg[9*i+m]); }
    pos[i] = tp > tn; double tm = std::max(tp, tn);
    gd[i] = (pos[i] ? 1 : -1) * dv(p.dot_gamma_0[i]) * std::pow(tm/xi[i], dv(p.n[i]));
    dgd[i] = gd[i] != 0 ? gd[i]*dv(p.n[i])/tm : 0;
  }
}
static void ref_dotState(const cp_pheno_param& p, const double* S6, const double* xi, double* dxi, double* dg) {
  double Mp[9] = {S6[0],S6[5],S6[4], S6[5],S6[1],S6[3], S6[4],S6[3],S6[2]};
  double gd[N], dgd[N]; bool pos[N];
  ref_kin(p, Mp, xi, gd, dgd, pos);
  for (int i = 0; i < N; ++i) dg[i] = std::fabs(gd[i]);
  for (int i = 0; i < N; ++i) {
    double x = 1 - xi[i]/dv(p.xi_inf[i]);
    double h = 0; for (int j = 0; j < N; ++j) h += dv(p.h_sl_sl[i+N*j])*dg[j];
    dxi[i] = dv(p.h_0[i]) * std::copysign(std::pow(std::fabs(x), dv(p.a[i])), x) * h;
  }
}
static bool solve9(double* A, double* b) {                                                         // A(r,c) at r+9c
  for (int c = 0; c < 9; ++c) {
    int pv = c; for (int r = c+1; r < 9; ++r) if (std::fabs(A[r+9*c]) > std::fabs(A[pv+9*c])) pv = r;
    if (A[pv+9*c] == 0) return false;
    for (int k = 0; k < 9; ++k) std::swap(A[c+9*k], A[pv+9*k]); std::swap(b[c], b[pv]);
    for (int r = c+1; r < 9; ++r) { double l = A[r+9*c]/A[c+9*c]; for (int k = c; k < 9; ++k) A[r+9*k] -= l*A[c+9*k]; b[r] -= l*b[c]; }
  }
  for (int r = 8; r >= 0; --r) { double s = b[r]; for (int k = r+1; k < 9; ++k) s -= A[r+9*k]*b[k]; b[r] = s/A[r+9*r]; }
  return true;
}
// one integrateStress; state xi given; Lp guess in/out
static bool ref_stress(const cp_pheno_param& p, double rtol, const double* F, const double* Fp0, const double* xi,
                       double dt, double* Lp, double* S6, double* P, double* Fp, int& its) {
  double C3[81];                                                                                     // C_ijkl at i+3j+9k+27l
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k) for (int l=0;l<3;++l) {
    auto v = [](int a,int b){ return a==b ? a : 6-a-b-0*0 + (0); };
    int I = i==j ? i : (3 + (3 - i - j) - 0) ; I = i==j ? i : (i+j==3 ? 3 : (i+j==2 ? 4 : 5)); (void)v;
    int K = k==l ? k : (k+l==3 ? 3 : (k+l==2 ? 4 : 5));
    C3[i+3*j+9*k+27*l] = dv(p.C66[I+6*K]);
  }
  double iFp0[9]; cp_inv33<double>(Fp0, iFp0);
  double A[9]; cp_mul33<double>(F, iFp0, A);
  double Lpg[9], Lpold[9], dLp[9] = {0}, step = 1, rold = INFINITY;
  std::memcpy(Lpg, Lp, sizeof(Lpg)); std::memcpy(Lpold, Lp, sizeof(Lpg));
  double B[9], Fe[9], S[9];
  for (int it = 0; it < 40; ++it) {
    ++its;
    for (int k = 0; k < 9; ++k) B[k] = (k%4==0) - dt*Lpg[k];
    cp_mul33<double>(A, B, Fe);
    double FtF[9]; cp_mulT33<double>(Fe, Fe, FtF);
    double E[9]; for (int k = 0; k < 9; ++k) E[k] = 0.5*(FtF[k] - (k%4==0));
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) { double s=0; for (int k=0;k<3;++k) for (int l=0;l<3;++l) s += C3[i+3*j+9*k+27*l]*E[k+3*l]; S[i+3*j]=s; }
    double gd[N], dgd[N]; bool pos[N]; ref_kin(p, S, xi, gd, dgd, pos);
    double Lc[9] = {0}; for (int i = 0; i < N; ++i) for (int m = 0; m < 9; ++m) Lc[m] += gd[i]*dv(p.P_sl[9*i+m]);
    double nL=0,nC=0,nR=0, R[9];
    for (int m = 0; m < 9; ++m) { R[m] = Lpg[m]-Lc[m]; nL += Lpg[m]*Lpg[m]; nC += Lc[m]*Lc[m]; nR += R[m]*R[m]; }
    nL=std::sqrt(nL); nC=std::sqrt(nC); nR=std::sqrt(nR);
    if (!std::isfinite(nR)) return false;
    if (nR < std::max(rtol*std::max(nL,nC), 1e-8*rtol/1e-6)) break;
    if (nR < rold) { rold = nR; std::memcpy(Lpold, Lpg, sizeof(Lpg)); step = 1; }
    else { step *= 0.5; for (int m = 0; m < 9; ++m) Lpg[m] = Lpold[m] + dLp[m]*step; continue; }
    // dRLp/dLp = I - dLp_dS : dS_dFe : dFe_dLp
    double dLdS[81] = {0};                                                                           // (ij,kl)
    for (int i = 0; i < N; ++i) { const cp_pair* Pn = pos[i] ? p.P_nS_pos : p.P_nS_neg;
      for (int a = 0; a < 9; ++a) for (int b = 0; b < 9; ++b) dLdS[a+9*b] += dgd[i]*dv(p.P_sl[9*i+a])*dv(Pn[9*i+b]); }
    double dSdLp[81] = {0};                                                                          // dS_mn/dLp_cd
    for (int c=0;c<3;++c) for (int d=0;d<3;++d) {
      double dFe[9]; for (int o=0;o<3;++o) for (int q=0;q<3;++q) dFe[o+3*q] = -dt*A[o+3*c]*(q==d);   // dFe_oq/dLp_cd
      double dE[9]; double t1[9]; cp_mulT33<double>(dFe, Fe, t1);
      for (int o=0;o<3;++o) for (int q=0;q<3;++q) dE[o+3*q] = 0.5*(t1[o+3*q] + t1[q+3*o]);
      for (int m=0;m<3;++m) for (int n=0;n<3;++n) { double s=0; for (int k=0;k<3;++k) for (int l=0;l<3;++l) s += C3[m+3*n+9*k+27*l]*dE[k+3*l]; dSdLp[(m+3*n)+9*(c+3*d)] = s; }
    }
    double J9[81];
    for (int a = 0; a < 9; ++a) for (int b = 0; b < 9; ++b) { double s = 0; for (int k = 0; k < 9; ++k) s += dLdS[a+9*k]*dSdLp[k+9*b]; J9[a+9*b] = (a==b) - s; }
    double rhs[9]; for (int m = 0; m < 9; ++m) rhs[m] = R[m];
    if (!solve9(J9, rhs)) return false;
    for (int m = 0; m < 9; ++m) { dLp[m] = -rhs[m]; Lpg[m] += dLp[m]*step; }
  }
  std::memcpy(Lp, Lpg, sizeof(Lpg));
  S6[0]=S[0]; S6[1]=S[4]; S6[2]=S[8]; S6[3]=S[CP_IDX(1,2)]; S6[4]=S[CP_IDX(0,2)]; S6[5]=S[CP_IDX(0,1)];
  double iFpn[9]; cp_mul33<double>(iFp0, B, iFpn);
  double t1[9], t2[9], t3[9]; cp_mul33<double>(F, iFpn, t1); cp_mul33<double>(t1, S, t2); cp_mul3T3<double>(t2, iFpn, t3);
  std::memcpy(P, t3, sizeof(t3));
  double Fpn[9]; cp_inv33<double>(iFpn, Fpn); double s = std::cbrt(cp_det33<double>(Fpn));
  for (int k = 0; k < 9; ++k) Fp[k] = Fpn[k]/s;
  return true;
}
static bool ref_integrate(const cp_pheno_param& p, double rtol_Lp, const double* F, const double* Fp0,
                          const double* S0, const double* xi0, const double* g0, double dt, double* Lp, Ref& out) {
  double dxi[N], dg[N], l1x[N], l1g[N], l2x[N], l2g[N];
  ref_dotState(p, S0, xi0, dxi, dg);
  for (int i = 0; i < N; ++i) { out.xi[i] = xi0[i]+dxi[i]*dt; out.gamma[i] = g0[i]+dg[i]*dt; l1x[i]=l1g[i]=0; }
  out.it_stress = 0;
  for (int it = 0; it < 20; ++it) {
    for (int i = 0; i < N; ++i) { l2x[i] = it>0 ? l1x[i] : 0; l2g[i] = it>0 ? l1g[i] : 0; l1x[i]=dxi[i]; l1g[i]=dg[i]; }
    if (!ref_stress(p, rtol_Lp, F, Fp0, out.xi, dt, Lp, out.S, out.P, out.Fp, out.it_stress)) return false;
    ref_dotState(p, out.S, out.xi, dxi, dg);
    double d01=0,d12=0,d22=0;
    for (int i = 0; i < N; ++i) { d01 += dxi[i]*l1x[i]+dg[i]*l1g[i];
      d12 += (dxi[i]-l1x[i])*(l1x[i]-l2x[i]) + (dg[i]-l1g[i])*(l1g[i]-l2g[i]);
      d22 += (l1x[i]-l2x[i])*(l1x[i]-l2x[i]) + (l1g[i]-l2g[i])*(l1g[i]-l2g[i]); }
    double z = (std::min(d01,d12) < 0 && d22 > 0) ? 0.75+0.25*std::tanh(2+4*d12/d22) : 1;
    bool conv = true;
    for (int i = 0; i < N; ++i) {
      dxi[i] = dxi[i]*z + l1x[i]*(1-z); dg[i] = dg[i]*z + l1g[i]*(1-z);
      double rx = out.xi[i]-xi0[i]-dxi[i]*dt, rg = out.gamma[i]-g0[i]-dg[i]*dt;
      out.xi[i] -= rx; out.gamma[i] -= rg;
      conv = conv && std::fabs(rx) <= std::max(1.0, 1e-6*std::fabs(out.xi[i])) && std::fabs(rg) <= std::max(1e-6, 1e-6*std::fabs(out.gamma[i]));
    }
    if (conv) { std::memcpy(out.Lp, Lp, 9*sizeof(double)); return true; }
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
template<typename H> static H hc(double x);
template<> double hc<double>(double x) { return x; }
template<> float  hc<float>(double x)  { return float(x); }
template<> df64   hc<df64>(double x)   { cp_pair q = pr(x); return df_make(q.hi, q.lo); }
static double hd(double x) { return x; }
static double hd(float x)  { return x; }
static double hd(df64 x)   { return double(x.hi) + double(x.lo); }

struct State { double F[9], Fp[9], S[6], Lp[9], xi[N], gamma[N], P[9]; };

template<typename H>
static int kernel_step(const cp_pheno_param& p, const cp_numerics& num, const double* F, State& s, double dt,
                       double* dPdF = nullptr, int* its = nullptr) {
  cp_point<H,N> pt;
  for (int k = 0; k < 9; ++k) { pt.F[k] = hc<H>(F[k]); pt.Fp0[k] = hc<H>(s.Fp[k]); pt.Lp_guess[k] = hc<H>(s.Lp[k]); }
  for (int k = 0; k < 6; ++k) pt.S0[k] = hc<H>(s.S[k]);
  for (int i = 0; i < N; ++i) { pt.xi0[i] = hc<H>(s.xi[i]); pt.gamma0[i] = hc<H>(s.gamma[i]); }
  cp_integrate<H,N>(p, num, hc<H>(dt), pt);
  if (pt.status != CP_OK) return pt.status;
  for (int k = 0; k < 9; ++k) { s.Fp[k] = hd(pt.Fp[k]); s.Lp[k] = hd(pt.Lp[k]); s.P[k] = hd(pt.P[k]); }
  for (int k = 0; k < 6; ++k) s.S[k] = hd(pt.S[k]);
  for (int i = 0; i < N; ++i) { s.xi[i] = hd(pt.xi[i]); s.gamma[i] = hd(pt.gamma[i]); }
  if (dPdF) for (int k = 0; k < 81; ++k) dPdF[k] = double(pt.dPdF[k]);
  if (its) *its += pt.n_iter_stress;
  return CP_OK;
}

static void rand_rot(std::mt19937_64& g, double* R) {
  std::normal_distribution<double> n;
  double q[4]; double s = 0; for (double& x : q) { x = n(g); s += x*x; } s = std::sqrt(s); for (double& x : q) x /= s;
  double a=q[0],b=q[1],c=q[2],d=q[3];
  double M[9] = {a*a+b*b-c*c-d*d, 2*(b*c+a*d), 2*(b*d-a*c),  2*(b*c-a*d), a*a-b*b+c*c-d*d, 2*(c*d+a*b),
                 2*(b*d+a*c), 2*(c*d-a*b), a*a-b*b-c*c+d*d};
  std::memcpy(R, M, sizeof(M));
}
static double rel(const double* a, const double* b, int n) {
  double d = 0, s = 0; for (int k = 0; k < n; ++k) { d += (a[k]-b[k])*(a[k]-b[k]); s += b[k]*b[k]; }
  return std::sqrt(d/std::max(s, 1e-300));
}

int main() {
  std::mt19937_64 g(7);
  std::uniform_real_distribution<double> u(-1, 1);
  const int n_grains = 40, n_inc = 120;
  const double dt = 1.0;                                                                             // tensionX-like: 1e-3/s, 120 s => 12 %
  cp_pheno_param p; cp_numerics num_tight, num_d, num_df, num_f;
  setup(p, num_tight, 1e-12);
  setup(p, num_d,  1e-12);
  setup(p, num_df, 1e-11);
  setup(p, num_f,  2e-6);

  double err_d = 0, err_df = 0, err_f = 0, err_xi_df = 0;
  long it_ref = 0, it_d = 0, it_df = 0, it_f = 0; int fail_d = 0, fail_df = 0, fail_f = 0, fail_ref = 0;
  double err_tan = 0;
  for (int gr = 0; gr < n_grains; ++gr) {
    double R[9]; rand_rot(g, R);
    double Fdot[9] = {1e-3, 0, 0, 0, -5e-4, 0, 0, 0, -5e-4};
    for (int k = 0; k < 9; ++k) Fdot[k] += 2e-4*u(g);
    State sr, sd, sdf, sf;
    for (State* s : {&sr, &sd, &sdf, &sf}) {
      for (int k = 0; k < 9; ++k) { s->F[k] = (k%4==0); s->Fp[k] = R[k]; s->Lp[k] = 0; s->P[k] = 0; }
      for (int k = 0; k < 6; ++k) s->S[k] = 0;
      for (int i = 0; i < N; ++i) { s->xi[i] = 31e6; s->gamma[i] = 0; }
    }
    bool ok_d = true, ok_df = true, ok_f = true;                                                     // each variant fails independently
    for (int inc = 0; inc < n_inc; ++inc) {
      double F[9]; for (int k = 0; k < 9; ++k) F[k] = (k%4==0) + Fdot[k]*dt*(inc+1);
      Ref out; int its = 0;
      if (!ref_integrate(p, 1e-12, F, sr.Fp, sr.S, sr.xi, sr.gamma, dt, sr.Lp, out)) { ++fail_ref; printf("ref fail gr %d inc %d\n", gr, inc); break; }
      std::memcpy(sr.S, out.S, sizeof(out.S)); std::memcpy(sr.xi, out.xi, sizeof(out.xi)); std::memcpy(sr.gamma, out.gamma, sizeof(out.gamma));
      std::memcpy(sr.Fp, out.Fp, sizeof(out.Fp)); std::memcpy(sr.P, out.P, sizeof(out.P)); it_ref += out.it_stress;
      int a;
      if (ok_d)  { a = kernel_step<double>(p, num_d,  F, sd,  dt, nullptr, &its); it_d  += its; its = 0;
                   if (a) { ok_d  = false; ++fail_d;  printf("double fail %d gr %d inc %d\n", a, gr, inc); }
                   else err_d  = std::max(err_d,  rel(sd.P,  sr.P, 9)); }
      if (ok_df) { a = kernel_step<df64>  (p, num_df, F, sdf, dt, nullptr, &its); it_df += its; its = 0;
                   if (a) { ok_df = false; ++fail_df; printf("df64 fail %d gr %d inc %d\n", a, gr, inc); }
                   else { err_df = std::max(err_df, rel(sdf.P, sr.P, 9)); err_xi_df = std::max(err_xi_df, rel(sdf.xi, sr.xi, N)); } }
      if (ok_f)  { a = kernel_step<float> (p, num_f,  F, sf,  dt, nullptr, &its); it_f  += its; its = 0;
                   if (a) { ok_f  = false; ++fail_f;  printf("float fail %d gr %d inc %d\n", a, gr, inc); }
                   else err_f  = std::max(err_f,  rel(sf.P,  sr.P, 9)); }
    }
  }
  // tangent check: fresh point, plastic regime
  {
    double R[9]; rand_rot(g, R);
    State s0; for (int k = 0; k < 9; ++k) { s0.Fp[k] = R[k]; s0.Lp[k] = 0; } for (int k = 0; k < 6; ++k) s0.S[k] = 0;
    for (int i = 0; i < N; ++i) { s0.xi[i] = 31e6; s0.gamma[i] = 0; }
    double F[9] = {1.004, 0.0003, 0, 0.0001, 0.998, 0.0002, 0, 0, 0.998};
    State s = s0; double dPdF[81]; int bad = kernel_step<double>(p, num_d, F, s, 1.0, dPdF);
    double fd[81]; const double h = 1e-8;
    for (int c = 0; c < 9; ++c) {
      double Fp_[9], Fm_[9]; std::memcpy(Fp_, F, sizeof(F)); std::memcpy(Fm_, F, sizeof(F)); Fp_[c] += h; Fm_[c] -= h;
      // tangent holds state fixed: freeze state by using n_state=1 is not identical; compare with full FD and report
      State a = s0, b = s0; bad |= kernel_step<double>(p, num_d, Fp_, a, 1.0) | kernel_step<double>(p, num_d, Fm_, b, 1.0);
      for (int m = 0; m < 9; ++m) fd[m+9*c] = (a.P[m]-b.P[m])/(2*h);
    }
    err_tan = bad ? NAN : rel(dPdF, fd, 81);
    if (bad) printf("tangent check: kernel step failed (status %d)\n", bad);
  }
  printf("grains %d x increments %d (12%% strain), relative error of P vs DAMASK-port (tight tol):\n", n_grains, n_inc);
  printf("  S-formulation double : max %.2e   Newton its %ld (ref Lp-formulation: %ld)\n", err_d, it_d, it_ref);
  printf("  S-formulation df64   : max %.2e   Newton its %ld   (xi err %.2e)\n", err_df, it_df, err_xi_df);
  printf("  S-formulation float  : max %.2e   Newton its %ld\n", err_f, it_f);
  printf("  failures: ref %d double %d df64 %d float %d\n", fail_ref, fail_d, fail_df, fail_f);
  printf("  tangent dP/dF vs finite differences (incl. state change): rel %.2e\n", err_tan);
  // float is reported only (the GPU uses df64 where accuracy demands it)
  const bool ok = err_d < 1e-10 && err_df < 1e-9 && fail_ref + fail_d + fail_df == 0 && err_tan < 1e-2;
  printf("%s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
