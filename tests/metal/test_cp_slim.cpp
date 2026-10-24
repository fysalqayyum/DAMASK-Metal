// (shared setup copied from test_cp_kernel.cpp) CPU validation of src/grid/metal/cp_kernel.h (S-formulation, mixed precision) against an
// independent C++ port of DAMASK's integrateStress/integrateStateFPI (Lp-formulation, double).
// Material: examples/grid/material.yaml (Al, cF, phenopowerlaw, 12 slip systems).
// build: clang++ -std=c++17 -O2 -ffp-contract=off test_cp_kernel.cpp
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include "../../src/grid/metal/cp_slim.h"

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
// QA of the slim formulation (cp_slim.h) on the CPU, before any GPU run:
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
