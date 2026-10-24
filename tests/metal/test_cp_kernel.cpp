// CPU validation of the reference kernel cp_integrate (ref_pheno_kernel.h; S-formulation, mixed precision) against an
// independent C++ port of DAMASK's integrateStress/integrateStateFPI (Lp-formulation, double).
// Material: examples/grid/material.yaml (Al, cF, phenopowerlaw, 12 slip systems).
// build: clang++ -std=c++17 -O2 -ffp-contract=off test_cp_kernel.cpp
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include "ref_pheno_port.h"


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
