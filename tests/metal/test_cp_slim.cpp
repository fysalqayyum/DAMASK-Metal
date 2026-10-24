// CPU validation of the slim reference kernel cp_integrate_slim (ref_pheno_kernel.h; setup as test_cp_kernel.cpp) against an
// independent C++ port of DAMASK's integrateStress/integrateStateFPI (Lp-formulation, double).
// Material: examples/grid/material.yaml (Al, cF, phenopowerlaw, 12 slip systems).
// build: clang++ -std=c++17 -O2 -ffp-contract=off test_cp_slim.cpp
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include "ref_pheno_port.h"


//--------------------------------------------------------------------------------------------------
// QA of the slim formulation (cp_integrate_slim) on the CPU:
//  1. cp_integrate_slim<double> vs cp_integrate<double> (identical algorithm, must agree to rounding)
//  2. cp_tangent<double> vs central finite differences of the DAMASK integrateStress port (state fixed)
//  3. cp_tangent<float> vs cp_tangent<double>
static double rel(const double* a, const double* b, int n) {
  double d = 0, s = 0; for (int k = 0; k < n; ++k) { d += (a[k]-b[k])*(a[k]-b[k]); s += b[k]*b[k]; }
  return std::sqrt(d/std::max(s, 1e-300));
}

int main() {
  cp_pheno_param p; cp_numerics num;
  setup(p, num, 1e-12);
  num.trust_S = 0.5f;
  typedef cp_slim_layout<N> L;
  std::mt19937_64 g(7); std::uniform_real_distribution<double> u(-1, 1); std::normal_distribution<double> nd;
  const double dt = 1.0;
  double e_slim = 0; int n_cmp = 0, mism_status = 0;
  double e_tan = 0, e_tan_f = 0; int n_tan = 0;
  for (int gr = 0; gr < 40; ++gr) {
    double q[4], s = 0; for (double& x : q) { x = nd(g); s += x*x; } s = std::sqrt(s); for (double& x : q) x /= s;
    double a=q[0],b=q[1],c=q[2],d=q[3];
    double R[9] = {a*a+b*b-c*c-d*d, 2*(b*c+a*d), 2*(b*d-a*c), 2*(b*c-a*d), a*a-b*b+c*c-d*d, 2*(c*d+a*b), 2*(b*d+a*c), 2*(c*d-a*b), a*a-b*b-c*c+d*d};
    double Fdot[9] = {1e-3, 0, 0, 0, -5e-4, 0, 0, 0, -5e-4};
    for (double& x : Fdot) x += 2e-4*u(g);
    cp_point<double,N> pt;
    double Fp[9], S6[6] = {0}, Lp[9] = {0}, xi[N], ga[N];
    std::memcpy(Fp, R, sizeof(Fp));
    for (int i = 0; i < N; ++i) { xi[i] = 31e6; ga[i] = 0; }
    for (int inc = 0; inc < 120; ++inc) {
      double F[9]; for (int k = 0; k < 9; ++k) F[k] = (k%4==0) + Fdot[k]*dt*(inc+1);
      for (int k = 0; k < 9; ++k) { pt.F[k] = F[k]; pt.Fp0[k] = Fp[k]; pt.Lp_guess[k] = Lp[k]; }
      for (int k = 0; k < 6; ++k) pt.S0[k] = S6[k];
      for (int i = 0; i < N; ++i) { pt.xi0[i] = xi[i]; pt.gamma0[i] = ga[i]; }
      cp_integrate<double,N>(p, num, dt, pt);

      std::vector<double> in(L::n_in), out(L::n_out, 0.0);                                          // n = 1: component c at c
      for (int k = 0; k < 9; ++k) { in[L::F+k] = F[k]; in[L::Fp0+k] = Fp[k]; in[L::Lp0+k] = Lp[k]; }
      for (int k = 0; k < 6; ++k) in[L::S0+k] = S6[k];
      for (int i = 0; i < N; ++i) { in[L::xi0+i] = xi[i]; in[L::gamma0+i] = ga[i]; }
      cp_soa_const<double> vi = {in.data(), 1, 0}; cp_soa<double> vo = {out.data(), 1, 0};
      int its; int st = cp_integrate_slim<double,N>(p, num, dt, vi, vo, its);
      if (st != pt.status) { ++mism_status; break; }
      if (st != CP_OK) break;
      e_slim = std::max({e_slim, rel(&out[L::P], pt.P, 9), [&]{ double iF[9], Fq[9]; std::memcpy(iF, &out[L::iFp], sizeof(iF)); double d = cp_inv33<double>(iF, Fq); double s = std::cbrt(1.0/d); for (double& x : Fq) x /= s; return rel(Fq, pt.Fp, 9); }(), rel(&out[L::xi], pt.xi, N),
                         rel(&out[L::S], pt.S, 6), rel(&out[L::Fe], pt.Fe, 9)});
      ++n_cmp;

      // tangent check every 10 increments: FD of the DAMASK integrateStress port with the final state fixed
      if (inc % 10 == 9) {
        double xif[N]; std::memcpy(xif, &out[L::xi], sizeof(xif));
        double Lpg[9], S0f[6], P0[9], Fp0n[9]; int itx = 0;
        std::memcpy(Lpg, &out[L::Lp], sizeof(Lpg));
        if (ref_stress(p, 1e-13, F, Fp, xif, dt, Lpg, S0f, P0, Fp0n, itx)) {
          double Fe0[9], invFpu[9], B[9], invFp0[9];                                                // stored Fe uses the unnormalized invFp_new
          cp_inv33<double>(Fp, invFp0);
          for (int k = 0; k < 9; ++k) B[k] = (k%4==0) - dt*Lpg[k];
          cp_mul33<double>(invFp0, B, invFpu); cp_mul33<double>(F, invFpu, Fe0);
          double tan_d[81]; float tan_f[81];
          bool ok = cp_tangent<double,N>(p, dt, F, Fp, invFpu, Fe0, S0f, xif, tan_d);
          float Ff[9], Fpf[9], Fpnf[9], Fef[9], Sf[6], xf[N];
          for (int k = 0; k < 9; ++k) { Ff[k] = float(F[k]); Fpf[k] = float(Fp[k]); Fpnf[k] = float(invFpu[k]); Fef[k] = float(Fe0[k]); }
          for (int k = 0; k < 6; ++k) Sf[k] = float(S0f[k]);
          for (int k = 0; k < N; ++k) xf[k] = float(xif[k]);
          ok = ok && cp_tangent<float,N>(p, float(dt), Ff, Fpf, Fpnf, Fef, Sf, xf, tan_f);
          double fd[81]; const double h = 1e-7;
          for (int c = 0; c < 9 && ok; ++c) {
            double Fa[9], Fb[9], La[9], Lb[9], Sa[6], Sb[6], Pa[9], Pb[9], Fpa[9], Fpb[9]; int i1 = 0;
            std::memcpy(Fa, F, 72); std::memcpy(Fb, F, 72); Fa[c] += h; Fb[c] -= h;
            std::memcpy(La, Lpg, 72); std::memcpy(Lb, Lpg, 72);
            ok = ref_stress(p, 1e-13, Fa, Fp, xif, dt, La, Sa, Pa, Fpa, i1) && ref_stress(p, 1e-13, Fb, Fp, xif, dt, Lb, Sb, Pb, Fpb, i1);
            for (int m = 0; m < 9; ++m) fd[m + 9*c] = (Pa[m]-Pb[m])/(2*h);
          }
          if (ok) {
            double tf[81]; for (int k = 0; k < 81; ++k) tf[k] = tan_f[k];
            e_tan = std::max(e_tan, rel(tan_d, fd, 81)); e_tan_f = std::max(e_tan_f, rel(tf, tan_d, 81)); ++n_tan;
          }
        }
      }
      { double iF[9]; std::memcpy(iF, &out[L::iFp], sizeof(iF)); double d = cp_inv33<double>(iF, Fp); double s = std::cbrt(1.0/d); for (double& x : Fp) x /= s; } std::memcpy(S6, &out[L::S], sizeof(S6)); std::memcpy(Lp, &out[L::Lp], sizeof(Lp));
      std::memcpy(xi, &out[L::xi], sizeof(xi)); std::memcpy(ga, &out[L::gamma], sizeof(ga));
    }
  }
  std::printf("1. slim vs reference integrator: %d steps, status mismatches %d, max rel diff %.2e\n", n_cmp, mism_status, e_slim);
  std::printf("2. tangent (double) vs FD of DAMASK integrateStress port: %d points, max rel err %.2e\n", n_tan, e_tan);
  std::printf("3. tangent float vs double: max rel diff %.2e\n", e_tan_f);
  bool ok = mism_status == 0 && n_cmp == 4800 && e_slim < 1e-13 && n_tan > 0 && e_tan < 1e-5 && e_tan_f < 1e-4;
  std::printf("%s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
