// CPU validation of the model-independent core (src/grid/metal/cp_core.h):
//  1. phenopowerlaw through the core vs the slim reference kernel (ref_pheno_kernel.h)
//  2. kinehardening through the core vs an independent port of DAMASK's Fortran
//     (integrateStress Lp-formulation, integrateStateFPI with damper, plastic_kinehardening_deltaState),
//     tension -> compression to exercise stress reversals (back-stress flips)
//  3. kinehardening tangent (cp_core_tangent) vs finite differences of the reference (state fixed)
// build: clang++ -std=c++17 -O2 -ffp-contract=off test_cp_core.cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
#include "../../src/grid/metal/cp_models_composite.h"
#include "ref_pheno_kernel.h"
#ifndef MG_GEOMETRY
#define MG_GEOMETRY "data/Mg_geometry.txt"
#endif

static const int N = 12;
static cp_pair pr(double x) { cp_pair p; p.hi = float(x); p.lo = float(x - double(p.hi)); return p; }
static double  dv(cp_pair p) { return double(p.hi) + double(p.lo); }
static double rel(const double* a, const double* b, int n) {
  double d = 0, s = 0; for (int k = 0; k < n; ++k) { d += (a[k]-b[k])*(a[k]-b[k]); s += b[k]*b[k]; }
  return std::sqrt(d/std::max(s, 1e-300));
}

// cF {111}<110> Schmid tensors and cubic stiffness into any param struct with C66/P_* members
template<typename P>
static void geometry(P& p, double C11, double C12, double C44) {
  double C[36] = {0};
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C[a+6*b] = a == b ? C11 : C12;
  for (int a = 3; a < 6; ++a) C[a+6*a] = C44;
  for (int k = 0; k < 36; ++k) p.C66[k] = pr(C[k]);
  const int nrm[4][3] = {{1,1,1},{-1,-1,1},{1,-1,-1},{-1,1,-1}};
  const int dir[12][3] = {{0,1,-1},{-1,0,1},{1,-1,0}, {0,-1,-1},{1,0,1},{-1,1,0},
                          {0,-1,1},{-1,0,-1},{1,1,0}, {0,1,1},{1,0,-1},{-1,-1,0}};
  for (int i = 0; i < N; ++i) for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
    double v = dir[i][a]*nrm[i/3][b] / (std::sqrt(2.0)*std::sqrt(3.0));
    p.P_sl[9*i+CP_IDX(a,b)] = pr(v); p.P_nS_pos[9*i+CP_IDX(a,b)] = pr(v); p.P_nS_neg[9*i+CP_IDX(a,b)] = pr(-v);
  }
}

//--------------------------------------------------------------------------------------------------
// reference: port of DAMASK integrateStress (Lp residual, dgesv) with a model kinetics callback
// kin(Mp, gd, dgd, pos) as in kinetics()/LpAndItsTangent
typedef std::function<void(const double*, double*, double*, bool*)> Kin;
static bool solve9(double* A, double* b) {
  for (int c = 0; c < 9; ++c) {
    int pv = c; for (int r = c+1; r < 9; ++r) if (std::fabs(A[r+9*c]) > std::fabs(A[pv+9*c])) pv = r;
    if (A[pv+9*c] == 0) return false;
    for (int k = 0; k < 9; ++k) std::swap(A[c+9*k], A[pv+9*k]); std::swap(b[c], b[pv]);
    for (int r = c+1; r < 9; ++r) { double l = A[r+9*c]/A[c+9*c]; for (int k = c; k < 9; ++k) A[r+9*k] -= l*A[c+9*k]; b[r] -= l*b[c]; }
  }
  for (int r = 8; r >= 0; --r) { double s = b[r]; for (int k = r+1; k < 9; ++k) s -= A[r+9*k]*b[k]; b[r] = s/A[r+9*r]; }
  return true;
}
template<typename P>
static bool ref_stress(const P& p, const Kin& kin, const double* F, const double* Fp0, double dt,
                       double* Lp, double* S6, double* P_, double* Fp, double* Fe, double* invFpu) {
  double C3[81];
  auto v = [](int i, int j) { return i == j ? i : 6 - i - j; };
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k) for (int l=0;l<3;++l)
    C3[i+3*j+9*k+27*l] = dv(p.C66[v(i,j)+6*v(k,l)]);
  double iFp0[9]; cp_inv33<double>(Fp0, iFp0);
  double A[9]; cp_mul33<double>(F, iFp0, A);
  double Lpg[9], Lpold[9], dLp[9] = {0}, step = 1, rold = INFINITY, B[9], Fe_[9], S[9];
  std::memcpy(Lpg, Lp, sizeof(Lpg)); std::memcpy(Lpold, Lp, sizeof(Lpg));
  for (int it = 0; ; ++it) {
    if (it >= 40) return false;
    for (int k = 0; k < 9; ++k) B[k] = (k%4==0) - dt*Lpg[k];
    cp_mul33<double>(A, B, Fe_);
    double FtF[9]; cp_mulT33<double>(Fe_, Fe_, FtF);
    double E[9]; for (int k = 0; k < 9; ++k) E[k] = 0.5*(FtF[k] - (k%4==0));
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) { double s=0; for (int k=0;k<3;++k) for (int l=0;l<3;++l) s += C3[i+3*j+9*k+27*l]*E[k+3*l]; S[i+3*j]=s; }
    double gd[N], dgd[N]; bool pos[N]; kin(S, gd, dgd, pos);
    double Lc[9] = {0}; for (int i = 0; i < N; ++i) for (int m = 0; m < 9; ++m) Lc[m] += gd[i]*dv(p.P_sl[9*i+m]);
    double nL=0,nC=0,nR=0, R[9];
    for (int m = 0; m < 9; ++m) { R[m] = Lpg[m]-Lc[m]; nL += Lpg[m]*Lpg[m]; nC += Lc[m]*Lc[m]; nR += R[m]*R[m]; }
    nL=std::sqrt(nL); nC=std::sqrt(nC); nR=std::sqrt(nR);
    if (!std::isfinite(nR)) return false;
    if (nR < std::max(1e-11*std::max(nL,nC), 1e-14)) break;                                     // 1e5 x tighter than DAMASK
    if (nR < rold) { rold = nR; std::memcpy(Lpold, Lpg, sizeof(Lpg)); step = 1; }
    else { step *= 0.5; for (int m = 0; m < 9; ++m) Lpg[m] = Lpold[m] + dLp[m]*step; continue; }
    double dLdS[81] = {0};
    for (int i = 0; i < N; ++i) { const cp_pair* Pn = pos[i] ? p.P_nS_pos : p.P_nS_neg;
      for (int a = 0; a < 9; ++a) for (int b = 0; b < 9; ++b) dLdS[a+9*b] += dgd[i]*dv(p.P_sl[9*i+a])*dv(Pn[9*i+b]); }
    double dSdLp[81] = {0};
    for (int c=0;c<3;++c) for (int d=0;d<3;++d) {
      double dFe[9]; for (int o=0;o<3;++o) for (int q=0;q<3;++q) dFe[o+3*q] = -dt*A[o+3*c]*(q==d);
      double t1[9]; cp_mulT33<double>(dFe, Fe_, t1); double dE[9];
      for (int o=0;o<3;++o) for (int q=0;q<3;++q) dE[o+3*q] = 0.5*(t1[o+3*q] + t1[q+3*o]);
      for (int m=0;m<3;++m) for (int n=0;n<3;++n) { double s=0; for (int k=0;k<3;++k) for (int l=0;l<3;++l) s += C3[m+3*n+9*k+27*l]*dE[k+3*l]; dSdLp[(m+3*n)+9*(c+3*d)] = s; }
    }
    double J9[81];
    for (int a = 0; a < 9; ++a) for (int b = 0; b < 9; ++b) { double s = 0; for (int k = 0; k < 9; ++k) s += dLdS[a+9*k]*dSdLp[k+9*b]; J9[a+9*b] = (a==b) - s; }
    double rhs[9]; std::memcpy(rhs, R, sizeof(rhs));
    if (!solve9(J9, rhs)) return false;
    for (int m = 0; m < 9; ++m) { dLp[m] = -rhs[m]; Lpg[m] += dLp[m]*step; }
  }
  std::memcpy(Lp, Lpg, sizeof(Lpg));
  S6[0]=S[0]; S6[1]=S[4]; S6[2]=S[8]; S6[3]=S[CP_IDX(1,2)]; S6[4]=S[CP_IDX(0,2)]; S6[5]=S[CP_IDX(0,1)];
  cp_mul33<double>(iFp0, B, invFpu);
  double t1[9]; cp_mul33<double>(F, invFpu, Fe); cp_mul33<double>(Fe, S, t1); cp_mul3T3<double>(t1, invFpu, P_);
  double Fpn[9]; cp_inv33<double>(invFpu, Fpn); double s = std::cbrt(cp_det33<double>(Fpn));
  for (int k = 0; k < 9; ++k) Fp[k] = Fpn[k]/s;
  return true;
}

//--------------------------------------------------------------------------------------------------
// kinehardening reference (phase_mechanical_plastic_kinehardening.f90), state = [xi, chi, gamma | sgn, chi_flip, gamma_flip]
struct KH { cp_kineh_param p; double atol_xi = 1.0, atol_gamma = 1e-6; };
static void kh_kin(const KH& m, const double* st, const double* Mp, double* gd, double* dgd, bool* pos) {
  for (int i = 0; i < N; ++i) {
    double tp = -st[N+i], tn = st[N+i];
    for (int k = 0; k < 9; ++k) { tp += Mp[k]*dv(m.p.P_nS_pos[9*i+k]); tn += Mp[k]*dv(m.p.P_nS_neg[9*i+k]); }
    pos[i] = tp > tn; double tm = std::max(tp, tn);
    gd[i] = (pos[i] ? 1 : -1) * dv(m.p.dot_gamma_0[i]) * std::pow(tm/st[i], dv(m.p.n[i]));
    dgd[i] = gd[i] != 0 ? gd[i]*dv(m.p.n[i])/tm : 0;
  }
}
static void kh_dot(const KH& m, const double* st, const double* S6, double* d) {
  double Mp[9] = {S6[0],S6[5],S6[4], S6[5],S6[1],S6[3], S6[4],S6[3],S6[2]}, gd[N], dgd[N]; bool pos[N];
  kh_kin(m, st, Mp, gd, dgd, pos);
  double sumG = 0; for (int i = 0; i < N; ++i) sumG += st[2*N+i];
  for (int i = 0; i < N; ++i) d[2*N+i] = std::fabs(gd[i]);
  for (int i = 0; i < N; ++i) {
    double h = 0; for (int j = 0; j < N; ++j) h += dv(m.p.h_sl_sl[i+N*j])*d[2*N+j];
    double hx0 = dv(m.p.h_0_xi[i]), hxi = dv(m.p.h_inf_xi[i]), xinf = dv(m.p.xi_inf[i]);
    d[i] = h*(hxi + (hx0 - hxi*(1 - sumG*hx0/xinf))*std::exp(-sumG*hx0/xinf));
    double hc0 = dv(m.p.h_0_chi[i]), hci = dv(m.p.h_inf_chi[i]), cinf = dv(m.p.chi_inf[i]);
    double y = (st[2*N+i]-st[5*N+i])*hc0/(cinf+st[4*N+i]);
    d[N+i] = st[3*N+i]*d[2*N+i]*(hci + (hc0 - hci*(1 - y))*std::exp(-y));
  }
}
static void kh_delta(const KH& m, double* st, const double* S6) {
  double Mp[9] = {S6[0],S6[5],S6[4], S6[5],S6[1],S6[3], S6[4],S6[3],S6[2]}, gd[N], dgd[N]; bool pos[N];
  kh_kin(m, st, Mp, gd, dgd, pos);
  for (int i = 0; i < N; ++i) {
    double sgn = std::fabs(gd[i]) <= 1e-10 ? st[3*N+i] : std::copysign(1.0, gd[i]);
    if (std::fabs(sgn - st[3*N+i]) > 0.1) {
      double d_sgn = sgn - st[3*N+i], d_cf = std::fabs(st[N+i]) - st[4*N+i], d_gf = st[2*N+i] - st[5*N+i];
      st[3*N+i] += d_sgn; st[4*N+i] += d_cf; st[5*N+i] += d_gf;
    }
  }
}
// integrateStateFPI + deltaState
static bool kh_ref(const KH& m, const double* F, const double* Fp0, const double* S0, const double* st0, double dt,
                   double* Lp, double* st, double* S6, double* P, double* Fp, double* Fe, double* iFpu) {
  const int ND = 3*N;
  std::memcpy(st, st0, 6*N*sizeof(double));
  double dot[3*N], l1[3*N], l2[3*N];
  kh_dot(m, st0, S0, dot);
  for (int j = 0; j < ND; ++j) { st[j] = st0[j] + dot[j]*dt; l1[j] = 0; }
  for (int it = 0; it < 20; ++it) {
    for (int j = 0; j < ND; ++j) { l2[j] = it > 0 ? l1[j] : 0; l1[j] = dot[j]; }
    Kin kin = [&](const double* Mp, double* gd, double* dgd, bool* pos) { kh_kin(m, st, Mp, gd, dgd, pos); };
    if (!ref_stress(m.p, kin, F, Fp0, dt, Lp, S6, P, Fp, Fe, iFpu)) return false;
    kh_dot(m, st, S6, dot);
    double d01=0,d12=0,d22=0;
    for (int j = 0; j < ND; ++j) { d01 += dot[j]*l1[j]; d12 += (dot[j]-l1[j])*(l1[j]-l2[j]); d22 += (l1[j]-l2[j])*(l1[j]-l2[j]); }
    double z = (std::min(d01,d12) < 0 && d22 > 0) ? 0.75+0.25*std::tanh(2+4*d12/d22) : 1;
    bool conv = true;
    for (int j = 0; j < ND; ++j) {
      dot[j] = dot[j]*z + l1[j]*(1-z);
      double r = st[j]-st0[j]-dot[j]*dt; st[j] -= r;
      double at = j < 2*N ? m.atol_xi : m.atol_gamma;
      conv = conv && std::fabs(r) <= std::max(at, 1e-6*std::fabs(st[j]));
    }
    if (conv) { kh_delta(m, st, S6); return true; }
  }
  return false;
}


//--------------------------------------------------------------------------------------------------
// dislotungsten reference (phase_mechanical_plastic_dislotungsten.f90), X = [rho_mob, rho_dip, gamma], D = [tau_pass, Lambda]
struct DT { cp_dtung_param p; double T = 300.0; };
static void dt_dep(const DT& m, const double* X, double* D) {
  for (int i = 0; i < N; ++i) {
    double s = 0, f = 0;
    for (int j = 0; j < N; ++j) { s += dv(m.p.h_sl_sl[i+N*j])*(X[j]+X[N+j]); f += dv(m.p.forest[i+N*j])*(X[j]+X[N+j]); }
    D[i] = dv(m.p.mu)*dv(m.p.b_sl[i])*std::sqrt(s);
    D[N+i] = std::pow(1.0/dv(m.p.D) + std::sqrt(f)/dv(m.p.i_sl[i]), -1.0);
  }
}
static void dt_kin(const DT& m, const double* X, const double* D, const double* Mp, double* gd, double* dgd, bool* pos, double* te) {
  const double kB = dv(m.p.k_B);
  for (int i = 0; i < N; ++i) {
    double tp = 0, tn = 0;
    for (int k = 0; k < 9; ++k) { tp += Mp[k]*dv(m.p.P_nS_pos[9*i+k]); tn += Mp[k]*dv(m.p.P_nS_neg[9*i+k]); }
    pos[i] = tp > tn;
    double teff = std::max(std::max(tp,tn) - D[i], 0.0); te[i] = teff;
    double BR = dv(m.p.Q_s[i])/(kB*m.T), b = dv(m.p.b_sl[i]), effL = D[N+i] - dv(m.p.w[i]);
    double tauP = dv(m.p.tau_P[i]), p = dv(m.p.p[i]), q = dv(m.p.q[i]);
    if (teff > 1e-8) {
      double SR = teff/tauP, SRp = std::pow(SR,p), SRpm1 = std::pow(SR,p-1.0);
      double t_n = b*std::exp(BR*std::pow(1.0-SRp,q))/(dv(m.p.omega[i])*effL);
      double t_k = effL*dv(m.p.B[i])/(2.0*b*teff);
      gd[i] = X[i]*b*dv(m.p.h[i])/(t_n+t_k)*(tp > tn ? 1.0 : -1.0);
      double dtn = -1.0*t_n*BR*p*q*std::pow(1.0-SRp,q-1.0)*SRpm1/tauP, dtk = -1.0*t_k/teff;
      dgd[i] = -1.0*gd[i]*(dtn+dtk)/(t_n+t_k);
    } else { gd[i] = 0; dgd[i] = 0; }
  }
}
static void dt_dot(const DT& m, const double* X, const double* D, const double* S6, double* d) {
  double Mp[9] = {S6[0],S6[5],S6[4], S6[5],S6[1],S6[3], S6[4],S6[3],S6[2]}, gd[N], dgd[N], te[N]; bool pos[N];
  dt_kin(m, X, D, Mp, gd, dgd, pos, te);
  const double mu = dv(m.p.mu), nu = dv(m.p.nu), kB = dv(m.p.k_B), PI = 3.14159265358979323846;
  for (int i = 0; i < N; ++i) {
    double ag = std::fabs(gd[i]), b = dv(m.p.b_sl[i]), dc = dv(m.p.d_caron[i]), Lam = D[N+i], d_hat, form, climb;
    if (ag == 0) { d_hat = Lam; form = 0; }
    else {
      d_hat = mu*b/(8.0*PI*(1.0-nu)*te[i]);
      d_hat = std::min(Lam, d_hat); d_hat = std::max(dc, d_hat);
      form = m.p.dipoleformation ? ag*2.0*(d_hat-dc)/b*X[i] : 0.0;
    }
    if (d_hat - dc == 0) climb = 0;
    else {
      double v_cl = (3.0*mu*dv(m.p.D_0)*std::exp(-dv(m.p.Q_cl)/(kB*m.T))*dv(m.p.f_at[i])/(2.0*PI*kB*m.T))*(1.0/(d_hat+dc));
      climb = (4.0*v_cl*X[N+i])/(d_hat-dc);
    }
    d[i]     = ag/(b*Lam) - form - ag*2.0*dc/b*X[i];
    d[N+i]   = form - climb - ag*2.0*dc/b*X[N+i];
    d[2*N+i] = ag;
  }
}
// integrateStateFPI with dependent state: D_in as stored (first predictor), recomputed at each integrateStress
static bool dt_ref(const DT& m, const double* F, const double* Fp0, const double* S0, const double* st0, const double* D_in,
                   double dt, double* Lp, double* st, double* D, double* S6, double* P, double* Fp, double* Fe, double* iFpu) {
  const int ND = 3*N;
  double dot[3*N], l1[3*N], l2[3*N];
  dt_dot(m, st0, D_in, S0, dot);
  for (int j = 0; j < ND; ++j) { st[j] = st0[j] + dot[j]*dt; l1[j] = 0; }
  for (int it = 0; it < 20; ++it) {
    for (int j = 0; j < ND; ++j) { l2[j] = it > 0 ? l1[j] : 0; l1[j] = dot[j]; }
    dt_dep(m, st, D);
    Kin kin = [&](const double* Mp, double* gd, double* dgd, bool* pos) { double te[N]; dt_kin(m, st, D, Mp, gd, dgd, pos, te); };
    if (!ref_stress(m.p, kin, F, Fp0, dt, Lp, S6, P, Fp, Fe, iFpu)) { if (getenv("DT_DEBUG")) std::printf("   dt_ref: stress failed at FPI it %d\n", it); return false; }
    dt_dot(m, st, D, S6, dot);
    double d01=0,d12=0,d22=0;
    for (int j = 0; j < ND; ++j) { d01 += dot[j]*l1[j]; d12 += (dot[j]-l1[j])*(l1[j]-l2[j]); d22 += (l1[j]-l2[j])*(l1[j]-l2[j]); }
    double z = (std::min(d01,d12) < 0 && d22 > 0) ? 0.75+0.25*std::tanh(2+4*d12/d22) : 1;
    bool conv = true;
    for (int j = 0; j < ND; ++j) {
      dot[j] = dot[j]*z + l1[j]*(1-z);
      double r = st[j]-st0[j]-dot[j]*dt; st[j] -= r;
      double at = j < 2*N ? 1.0 : 1e-6;
      conv = conv && std::fabs(r) <= std::max(at, 1e-6*std::fabs(st[j]));
    }
    if (conv) return true;
  }
  if (getenv("DT_DEBUG")) std::printf("   dt_ref: FPI not converged\n");
  return false;
}


//--------------------------------------------------------------------------------------------------
// phenopowerlaw with twins reference (phase_mechanical_plastic_phenopowerlaw.f90): systems [slip(ns), twin(nt)],
// state [xi_sl, xi_tw, gamma_sl, gamma_tw]. Real Mg geometry from damask-python (data/Mg_geometry.txt).
static const int NT_MAX = 48;
struct PT {
  int ns, nt, n;
  double P[NT_MAX][9], Ppos[NT_MAX][9], Pneg[NT_MAX][9], gch[NT_MAX];
  double g0[NT_MAX], nn[NT_MAX], a[NT_MAX], h0[NT_MAX], xinf[NT_MAX], c1[NT_MAX], c2[NT_MAX], fsat[NT_MAX];
  double h0ts[NT_MAX], h0tt[NT_MAX], c3[NT_MAX], c4[NT_MAX], H[NT_MAX*NT_MAX];
  double C66[36];
};
static double powF(double x, double c) { return std::pow(x, c); }                                   // Fortran real**real (0**0 = 1)
static void pt_kin(const PT& m, const double* st, const double* Mp, double* gd, double* dgd, bool* pos) {
  double sumF = 0; for (int k = 0; k < m.nt; ++k) sumF += st[m.n + m.ns + k]/m.gch[m.ns+k];
  for (int i = 0; i < m.n; ++i) {
    double tp = 0, tn = 0;
    for (int k = 0; k < 9; ++k) { tp += Mp[k]*m.Ppos[i][k]; tn += Mp[k]*m.Pneg[i][k]; }
    if (i < m.ns) {
      pos[i] = tp > tn;
      gd[i] = (pos[i] ? 1 : -1)*m.g0[i]*std::pow(std::max(tp,tn)/st[i], m.nn[i]);
      dgd[i] = gd[i] != 0 ? gd[i]*m.nn[i]/std::max(tp,tn) : 0;
    } else {
      pos[i] = true;
      gd[i] = tp > 0 ? (1.0 - sumF)*m.g0[i]*std::pow(tp/st[i], m.nn[i]) : 0;
      dgd[i] = gd[i] != 0 ? gd[i]*m.nn[i]/tp : 0;
    }
  }
}
static void pt_dot(const PT& m, const double* st, const double* S6, double* d) {
  double Mp[9] = {S6[0],S6[5],S6[4], S6[5],S6[1],S6[3], S6[4],S6[3],S6[2]}, gd[NT_MAX], dgd[NT_MAX]; bool pos[NT_MAX];
  pt_kin(m, st, Mp, gd, dgd, pos);
  const int ns = m.ns, nt = m.nt, n = m.n;
  double sumF = 0, sumG = 0;
  for (int k = 0; k < nt; ++k) sumF += st[n+ns+k]/m.gch[ns+k];
  for (int i = 0; i < ns; ++i) sumG += st[n+i];
  for (int i = 0; i < n; ++i) d[n+i] = i < ns ? std::fabs(gd[i]) : gd[i];
  for (int i = 0; i < ns; ++i) {
    double x = 1 - st[i]/(m.xinf[i] + m.fsat[i]*std::sqrt(sumF));
    double left = std::copysign(std::pow(std::fabs(x), m.a[i]), x), hs = 0, ht = 0;
    for (int j = 0; j < ns; ++j) hs += m.H[i+n*j]*d[n+j];
    for (int k = ns; k < n; ++k) ht += m.H[i+n*k]*d[n+k];
    d[i] = m.h0[i]*(1 + m.c1[i]*powF(sumF, m.c2[i]))*left*hs + ht;
  }
  for (int k = ns; k < n; ++k) {
    double hs = 0, ht = 0;
    for (int j = 0; j < ns; ++j) hs += m.H[k+n*j]*d[n+j];
    for (int l = ns; l < n; ++l) ht += m.H[k+n*l]*d[n+l];
    d[k] = m.h0ts[k]*powF(sumG, m.c3[k])*hs + m.h0tt[k]*powF(sumF, m.c4[k])*ht;
  }
}
// integrateStress (Lp formulation) generic in the number of systems, then FPI
static bool pt_stress(const PT& m, const double* st, const double* F, const double* Fp0, double dt,
                      double* Lp, double* S6, double* Pout, double* Fp, double* Fe, double* iFpu) {
  double C3[81];
  auto v = [](int i, int j) { return i == j ? i : 6 - i - j; };
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k) for (int l=0;l<3;++l) C3[i+3*j+9*k+27*l] = m.C66[v(i,j)+6*v(k,l)];
  double iFp0[9]; cp_inv33<double>(Fp0, iFp0);
  double A[9]; cp_mul33<double>(F, iFp0, A);
  double Lpg[9], Lpold[9], dLp[9] = {0}, step = 1, rold = INFINITY, B[9], Fe_[9], S[9];
  std::memcpy(Lpg, Lp, 72); std::memcpy(Lpold, Lp, 72);
  for (int it = 0; ; ++it) {
    if (it >= 40) return false;
    for (int k = 0; k < 9; ++k) B[k] = (k%4==0) - dt*Lpg[k];
    cp_mul33<double>(A, B, Fe_);
    double FtF[9]; cp_mulT33<double>(Fe_, Fe_, FtF);
    double E[9]; for (int k = 0; k < 9; ++k) E[k] = 0.5*(FtF[k] - (k%4==0));
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) { double s=0; for (int k=0;k<3;++k) for (int l=0;l<3;++l) s += C3[i+3*j+9*k+27*l]*E[k+3*l]; S[i+3*j]=s; }
    double gd[NT_MAX], dgd[NT_MAX]; bool pos[NT_MAX]; pt_kin(m, st, S, gd, dgd, pos);
    double Lc[9] = {0}; for (int i = 0; i < m.n; ++i) for (int q = 0; q < 9; ++q) Lc[q] += gd[i]*m.P[i][q];
    double nL=0,nC=0,nR=0, R[9];
    for (int q = 0; q < 9; ++q) { R[q] = Lpg[q]-Lc[q]; nL += Lpg[q]*Lpg[q]; nC += Lc[q]*Lc[q]; nR += R[q]*R[q]; }
    nL=std::sqrt(nL); nC=std::sqrt(nC); nR=std::sqrt(nR);
    if (!std::isfinite(nR)) return false;
    if (nR < std::max(1e-11*std::max(nL,nC), 1e-14)) break;
    if (nR < rold) { rold = nR; std::memcpy(Lpold, Lpg, 72); step = 1; }
    else { step *= 0.5; for (int q = 0; q < 9; ++q) Lpg[q] = Lpold[q] + dLp[q]*step; continue; }
    double dLdS[81] = {0};
    for (int i = 0; i < m.n; ++i) { const double* Pn = pos[i] ? m.Ppos[i] : m.Pneg[i];
      for (int a = 0; a < 9; ++a) for (int b = 0; b < 9; ++b) dLdS[a+9*b] += dgd[i]*m.P[i][a]*Pn[b]; }
    double dSdLp[81] = {0};
    for (int c=0;c<3;++c) for (int d=0;d<3;++d) {
      double dFe[9]; for (int o=0;o<3;++o) for (int q=0;q<3;++q) dFe[o+3*q] = -dt*A[o+3*c]*(q==d);
      double t1[9]; cp_mulT33<double>(dFe, Fe_, t1); double dE[9];
      for (int o=0;o<3;++o) for (int q=0;q<3;++q) dE[o+3*q] = 0.5*(t1[o+3*q] + t1[q+3*o]);
      for (int a=0;a<3;++a) for (int b=0;b<3;++b) { double s=0; for (int k=0;k<3;++k) for (int l=0;l<3;++l) s += C3[a+3*b+9*k+27*l]*dE[k+3*l]; dSdLp[(a+3*b)+9*(c+3*d)] = s; }
    }
    double J9[81];
    for (int a = 0; a < 9; ++a) for (int b = 0; b < 9; ++b) { double s = 0; for (int k = 0; k < 9; ++k) s += dLdS[a+9*k]*dSdLp[k+9*b]; J9[a+9*b] = (a==b) - s; }
    double rhs[9]; std::memcpy(rhs, R, 72);
    if (!solve9(J9, rhs)) return false;
    for (int q = 0; q < 9; ++q) { dLp[q] = -rhs[q]; Lpg[q] += dLp[q]*step; }
  }
  std::memcpy(Lp, Lpg, 72);
  S6[0]=S[0]; S6[1]=S[4]; S6[2]=S[8]; S6[3]=S[CP_IDX(1,2)]; S6[4]=S[CP_IDX(0,2)]; S6[5]=S[CP_IDX(0,1)];
  cp_mul33<double>(iFp0, B, iFpu);
  double t1[9]; cp_mul33<double>(F, iFpu, Fe); cp_mul33<double>(Fe, S, t1); cp_mul3T3<double>(t1, iFpu, Pout);
  double Fpn[9]; cp_inv33<double>(iFpu, Fpn); double sc = std::cbrt(cp_det33<double>(Fpn));
  for (int k = 0; k < 9; ++k) Fp[k] = Fpn[k]/sc;
  return true;
}
static bool pt_ref(const PT& m, const double* F, const double* Fp0, const double* S0, const double* st0, double dt,
                   double* Lp, double* st, double* S6, double* P, double* Fp, double* Fe, double* iFpu) {
  const int ND = 2*m.n;
  std::vector<double> dot(ND), l1(ND), l2(ND);
  pt_dot(m, st0, S0, dot.data());
  for (int j = 0; j < ND; ++j) { st[j] = st0[j] + dot[j]*dt; l1[j] = 0; }
  for (int it = 0; it < 20; ++it) {
    for (int j = 0; j < ND; ++j) { l2[j] = it > 0 ? l1[j] : 0; l1[j] = dot[j]; }
    if (!pt_stress(m, st, F, Fp0, dt, Lp, S6, P, Fp, Fe, iFpu)) return false;
    pt_dot(m, st, S6, dot.data());
    double d01=0,d12=0,d22=0;
    for (int j = 0; j < ND; ++j) { d01 += dot[j]*l1[j]; d12 += (dot[j]-l1[j])*(l1[j]-l2[j]); d22 += (l1[j]-l2[j])*(l1[j]-l2[j]); }
    double z = (std::min(d01,d12) < 0 && d22 > 0) ? 0.75+0.25*std::tanh(2+4*d12/d22) : 1;
    bool conv = true;
    for (int j = 0; j < ND; ++j) {
      dot[j] = dot[j]*z + l1[j]*(1-z);
      double r = st[j]-st0[j]-dot[j]*dt; st[j] -= r;
      conv = conv && std::fabs(r) <= std::max(j < m.n ? 1.0 : 1e-6, 1e-6*std::fabs(st[j]));
    }
    if (conv) return true;
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
// test-only policy: phenopowerlaw through the FLOW (Lp and its derivative from the policy), STATE_C
// (stiffness from the dependent state) and FLAT (unpadded state) paths of the core
template<int NSL>
struct cp_model_pheno_flow {
  enum { N = NSL, ND = 2*NSL, NY = 0, NDEP = 36, NBX = 2, NBY = 0, NBD = 0, FLOW = 1, STATE_C = 1, FLAT = 1, DC = 0, DT_STATE = 0, PRECOND = 0, FI_STATE = 0, ACCEL = 0 };
  typedef cp_pheno_param param;
  typedef cp_model_pheno<NSL> B;
  template<typename H>
  static void flow(const param& p, const H* S, const H* X, const H*, const H*, H, H, H* Lp, typename cp_ops<H>::J* K, int mode) {
    typedef cp_ops<H> O; typedef typename O::J J;
    H gd[N]; J dgd[N]; bool pos[N];
    cp_kinetics<H,N>(p, S, X, gd, dgd, pos);
    for (int k = 0; k < 9; ++k) {
      H s = O::c(0.0f);
      for (int i = 0; i < N; ++i) s = s + gd[i] * O::ld(p.P_sl[9*i+k]);
      Lp[k] = s;
    }
    if (mode == 0) return;
    const int nc = mode == 1 ? 6 : 9;
    for (int q = 0; q < 9*nc; ++q) K[q] = J(0);
    for (int i = 0; i < N; ++i) {
      const cp_pair* Pns = pos[i] ? p.P_nS_pos : p.P_nS_neg;
      for (int k = 0; k < nc; ++k) {
        int a = cp_voigt_i(k), b = cp_voigt_j(k);
        J c = mode == 2 ? cp_ldJ<J>(Pns[9*i+k])
                        : k < 3 ? cp_ldJ<J>(Pns[9*i+CP_IDX(a,a)]) : cp_ldJ<J>(Pns[9*i+CP_IDX(a,b)]) + cp_ldJ<J>(Pns[9*i+CP_IDX(b,a)]);
        for (int m = 0; m < 9; ++m) K[m+9*k] += dgd[i] * c * cp_ldJ<J>(p.P_sl[9*i+m]);
      }
    }
  }
  template<typename H>
  static void dot_state(const param& p, const H* S, const H* X, const H* Y, const H* D, H T, H* dX) { B::template dot_state<H>(p, S, X, Y, D, T, dX); }
  template<typename H>
  static void delta_state(const param&, const H*, const H*, H*, const H*, H) {}
  template<typename H>
  static void dependent(const param& p, const H*, const H*, H, H* D) { for (int k = 0; k < 36; ++k) D[k] = cp_ops<H>::ld(p.C66[k]); }
  static float atol(const param& p, int j) { return B::atol(p, j); }
  template<typename H>
  static float stress_scale(const param& p, const H* X) { return B::template stress_scale<H>(p, X); }
  template<typename H>
  static float cap_ratio(const param& p, const H* S, const H* X, const H* Y, const H* D, H T, float dt, float kappa) {
    return B::template cap_ratio<H>(p, S, X, Y, D, T, dt, kappa);
  }
};

//--------------------------------------------------------------------------------------------------
int main() {
  int fails = 0;
  std::mt19937_64 g(11); std::uniform_real_distribution<double> u(-1, 1); std::normal_distribution<double> nd;
  auto rot = [&](double* R) {
    double q[4], s = 0; for (double& x : q) { x = nd(g); s += x*x; } s = std::sqrt(s); for (double& x : q) x /= s;
    double a=q[0],b=q[1],c=q[2],d=q[3];
    double M[9] = {a*a+b*b-c*c-d*d, 2*(b*c+a*d), 2*(b*d-a*c), 2*(b*c-a*d), a*a-b*b+c*c-d*d, 2*(c*d+a*b), 2*(b*d+a*c), 2*(c*d-a*b), a*a-b*b-c*c+d*d};
    std::memcpy(R, M, sizeof(M));
  };
  cp_numerics num; num.rtol_S = 1e-12f; num.atol_S = 1e-6f; num.rtol_state = 1e-6f; num.stepsize_ls = 0.5f;
  num.trust_S = 0.5f; num.n_stress = 40; num.n_state = 20;

  //------------------------------------------------------------------------------------------------
  // 1. phenopowerlaw: core vs slim
  {
    cp_pheno_param p; std::memset(&p, 0, sizeof(p));
    geometry(p, 106.75e9, 60.41e9, 28.34e9);
    for (int i = 0; i < N; ++i) {
      p.dot_gamma_0[i] = pr(0.001); p.n[i] = pr(20); p.a[i] = pr(2.25); p.h_0[i] = pr(75e6); p.xi_inf[i] = pr(63e6);
      for (int j = 0; j < N; ++j) p.h_sl_sl[i+N*j] = pr(i == j ? 1.0 : 1.4);
    }
    p.atol_xi = 1.0f; p.atol_gamma = 1e-6f; p.N_sl = N;
    typedef cp_model_pheno<N> Mo; typedef cp_layout<Mo> L; typedef cp_slim_layout<N> LS;
    double e = 0; int n = 0, mism = 0;
    for (int gr = 0; gr < 40; ++gr) {
      double R[9]; rot(R);
      double Fdot[9] = {1e-3, 0, 0, 0, -5e-4, 0, 0, 0, -5e-4}; for (double& x : Fdot) x += 2e-4*u(g);
      std::vector<double> st(2*N); for (int i = 0; i < N; ++i) { st[i] = 31e6; st[N+i] = 0; }
      double Fp[9], S6[6] = {0}, Lp[9] = {0}; std::memcpy(Fp, R, sizeof(Fp));
      for (int inc = 0; inc < 120; ++inc) {
        double F[9]; for (int k = 0; k < 9; ++k) F[k] = (k%4==0) + Fdot[k]*(inc+1);
        std::vector<double> ia(LS::n_in), oa(LS::n_out, 0.0), ib(L::n_in, 0.0), ob(L::n_out, 0.0);
        for (int k = 0; k < 9; ++k) { ia[LS::F+k] = ib[L::F+k] = F[k]; ia[LS::Fp0+k] = ib[L::Fp0+k] = Fp[k]; ia[LS::Lp0+k] = ib[L::Lp0+k] = Lp[k]; }
        for (int k = 0; k < 6; ++k) ia[LS::S0+k] = ib[L::S0+k] = S6[k];
        for (int i = 0; i < N; ++i) { ia[LS::xi0+i] = ib[L::X0+i] = st[i]; ia[LS::gamma0+i] = ib[L::X0+N+i] = st[N+i]; }
        int i1, i2;
        int sa = cp_integrate_slim<double,N>(p, num, 1.0, {ia.data(),1,0}, {oa.data(),1,0}, i1);
        int sb = cp_core_integrate<double,Mo>(p, num, 1.0, {ib.data(),1,0}, {ob.data(),1,0}, i2);
        if (sa != sb) { ++mism; break; }
        if (sa != CP_OK) break;
        e = std::max({e, rel(&ob[L::P], &oa[LS::P], 9), rel(&ob[L::S], &oa[LS::S], 6), rel(&ob[L::X], &oa[LS::xi], 2*N),
                      rel(&ob[L::iFp], &oa[LS::iFp], 9)});
        ++n;
        { double iF[9]; std::memcpy(iF, &ob[L::iFp], sizeof(iF)); double dd = cp_inv33<double>(iF, Fp); double s = std::cbrt(1.0/dd); for (double& x : Fp) x /= s; }
        std::memcpy(S6, &ob[L::S], sizeof(S6)); std::memcpy(Lp, &ob[L::Lp], sizeof(Lp)); std::memcpy(st.data(), &ob[L::X], 2*N*sizeof(double));
      }
    }
    bool ok = mism == 0 && n == 4800 && e < 1e-14;
    fails += !ok;
    std::printf("1. phenopowerlaw core vs slim: %d steps, status mismatches %d, max rel diff %.2e  %s\n", n, mism, e, ok ? "OK" : "FAIL");
  }

  //------------------------------------------------------------------------------------------------
  // 2./3. kinehardening: core vs DAMASK port, tension -> compression
  {
    KH m; std::memset(&m.p, 0, sizeof(m.p));
    geometry(m.p, 198e9, 125e9, 122e9);
    for (int i = 0; i < N; ++i) {
      m.p.dot_gamma_0[i] = pr(1e-4); m.p.n[i] = pr(20);
      m.p.xi_inf[i] = pr(0.015e9); m.p.h_0_xi[i] = pr(0.065e9); m.p.h_inf_xi[i] = pr(0.045e9);
      m.p.chi_inf[i] = pr(0.027e9); m.p.h_0_chi[i] = pr(55e9); m.p.h_inf_chi[i] = pr(1.3e9);
      for (int j = 0; j < N; ++j) m.p.h_sl_sl[i+N*j] = pr(1.0);
    }
    m.p.atol_xi = 1.0f; m.p.atol_gamma = 1e-6f; m.p.N_sl = N;
    typedef cp_model_kineh<N> Mo; typedef cp_layout<Mo> L;
    double eP = 0, eX = 0, eY = 0, eT = 0; int n = 0, mism = 0, flips = 0, n_tan = 0;
    const double dt = 0.5;
    for (int gr = 0; gr < 30; ++gr) {
      double R[9]; rot(R);
      double Fdot[9] = {4e-4, 0, 0, 0, -2e-4, 0, 0, 0, -2e-4}; for (double& x : Fdot) x += 8e-5*u(g);   // DAMASK converges without cutback
      std::vector<double> st(6*N, 0.0); for (int i = 0; i < N; ++i) st[i] = 0.070e9;
      double Fp[9], S6[6] = {0}, Lp[9] = {0}; std::memcpy(Fp, R, sizeof(Fp));
      double Fr[9], Lpr[9] = {0}, S6r[6] = {0}; std::memcpy(Fr, R, sizeof(Fr));
      std::vector<double> str = st;
      double t = 0;
      for (int inc = 0; inc < 120; ++inc) {
        t += dt; double s = inc < 60 ? t : 60*dt - (t - 60*dt);                                     // load, then reverse into compression
        double F[9]; for (int k = 0; k < 9; ++k) F[k] = (k%4==0) + Fdot[k]*s;
        // reference
        double Pr[9], Fpr[9], Fer[9], iFr[9]; std::vector<double> st_new(6*N);
        if (!kh_ref(m, F, Fr, S6r, str.data(), dt, Lpr, st_new.data(), S6r, Pr, Fpr, Fer, iFr)) { std::printf("   reference failed g%d i%d\n", gr, inc); break; }
        // core
        std::vector<double> ib(L::n_in, 0.0), ob(L::n_out, 0.0);
        for (int k = 0; k < 9; ++k) { ib[L::F+k] = F[k]; ib[L::Fp0+k] = Fp[k]; ib[L::Lp0+k] = Lp[k]; }
        for (int k = 0; k < 6; ++k) ib[L::S0+k] = S6[k];
        for (int j = 0; j < 6*N; ++j) ib[L::X0+j] = st[j];                                           // X0 and Y0 are contiguous
        int its; int sb = cp_core_integrate<double,Mo>(m.p, num, dt, {ib.data(),1,0}, {ob.data(),1,0}, its);
        if (sb != CP_OK) { ++mism; std::printf("   core status %d g%d i%d\n", sb, gr, inc); break; }
        eP = std::max(eP, rel(&ob[L::P], Pr, 9));
        eX = std::max(eX, rel(&ob[L::X], st_new.data(), 3*N));
        for (int i = 0; i < N; ++i) {                                                                // delta state: discrete values
          eY = std::max(eY, std::fabs(ob[L::Y+i] - st_new[3*N+i]));
          eY = std::max({eY, std::fabs(ob[L::Y+N+i]-st_new[4*N+i])/0.027e9, std::fabs(ob[L::Y+2*N+i]-st_new[5*N+i])/std::max(1e-3, std::fabs(st_new[5*N+i]))});
          flips += std::fabs(st_new[3*N+i] - str[3*N+i]) > 0.1;
        }
        ++n;
        // tangent check (core double) vs FD of the reference stress update with final state fixed
        if (inc % 15 == 14) {
          double tan_d[81], fd[81]; const double h = 1e-7;
          Kin kin = [&](const double* Mp, double* gd, double* dgd, bool* pos) { kh_kin(m, st_new.data(), Mp, gd, dgd, pos); };
          double Lq[9], Sq[6], Pq[9], Fpq[9], Feq[9], iFq[9];
          std::memcpy(Lq, Lpr, sizeof(Lq));
          bool ok = ref_stress(m.p, kin, F, Fr, dt, Lq, Sq, Pq, Fpq, Feq, iFq)
                 && cp_core_tangent<double,Mo>(m.p, dt, F, Fr, iFq, Feq, Sq, st_new.data(), st_new.data()+3*N, nullptr, 0.0, tan_d);
          for (int c = 0; c < 9 && ok; ++c) {
            double Fa[9], Fb[9], La[9], Lb[9], Sa[6], Sb[6], Pa[9], Pb[9], X1[9], X2[9], X3[9];
            std::memcpy(Fa, F, 72); std::memcpy(Fb, F, 72); Fa[c] += h; Fb[c] -= h;
            std::memcpy(La, Lq, 72); std::memcpy(Lb, Lq, 72);
            ok = ref_stress(m.p, kin, Fa, Fr, dt, La, Sa, Pa, X1, X2, X3) && ref_stress(m.p, kin, Fb, Fr, dt, Lb, Sb, Pb, X1, X2, X3);
            for (int q = 0; q < 9; ++q) fd[q + 9*c] = (Pa[q]-Pb[q])/(2*h);
          }
          if (ok) { eT = std::max(eT, rel(tan_d, fd, 81)); ++n_tan; }
        }
        // advance both
        { double iF[9]; std::memcpy(iF, &ob[L::iFp], sizeof(iF)); double dd = cp_inv33<double>(iF, Fp); double sc = std::cbrt(1.0/dd); for (double& x : Fp) x /= sc; }
        std::memcpy(S6, &ob[L::S], sizeof(S6)); std::memcpy(Lp, &ob[L::Lp], sizeof(Lp));
        for (int j = 0; j < 6*N; ++j) st[j] = ob[L::X+j];
        std::memcpy(Fr, Fpr, sizeof(Fr)); str = st_new;
      }
    }
    // both sides stop the FPI at DAMASK's state tolerance (eps_rel_state = 1e-6); with the stiff back-stress
    // hardening they may stop at different iterations, so agreement is bounded by that tolerance, not by rounding
    bool ok = mism == 0 && n == 3600 && eP < 1e-6 && eX < 1e-6 && eY < 1e-6 && flips > 0 && n_tan > 0 && eT < 1e-5;
    fails += !ok;
    std::printf("2. kinehardening core vs DAMASK port: %d steps (%d stress reversals), status failures %d\n"
                "   max rel err P %.2e  dot state %.2e  delta state %.2e\n", n, flips, mism, eP, eX, eY);
    std::printf("3. kinehardening tangent vs FD (%d points): max rel err %.2e   %s\n", n_tan, eT, ok ? "OK" : "FAIL");
  }
  //------------------------------------------------------------------------------------------------
  // 4. dislotungsten: core vs DAMASK port (W parameters, T = 300 K, Schmid geometry of cF for the test)
  {
    DT m; std::memset(&m.p, 0, sizeof(m.p));
    geometry(m.p, 523e9, 203e9, 160e9);
    const int nrm[4][3] = {{1,1,1},{-1,-1,1},{1,-1,-1},{-1,1,-1}};
    const int dir[12][3] = {{0,1,-1},{-1,0,1},{1,-1,0}, {0,-1,-1},{1,0,1},{-1,1,0},
                            {0,-1,1},{-1,0,-1},{1,1,0}, {0,1,1},{1,0,-1},{-1,-1,0}};
    const double b = 2.72e-10;
    for (int i = 0; i < N; ++i) {
      m.p.b_sl[i] = pr(b); m.p.Q_s[i] = pr(2.61154e-19); m.p.i_sl[i] = pr(1.0); m.p.tau_P[i] = pr(2.03e9);
      m.p.p[i] = pr(0.86); m.p.q[i] = pr(1.69); m.p.h[i] = pr(2.566e-10); m.p.w[i] = pr(2.992e-9);
      m.p.omega[i] = pr(9.1e11); m.p.B[i] = pr(8.3e-5); m.p.d_caron[i] = pr(1.0*b); m.p.f_at[i] = pr(b*b*b);
      for (int j = 0; j < N; ++j) {
        m.p.h_sl_sl[i+N*j] = pr(i == j ? 0.009 : (i/3 == j/3 ? 0.72 : 0.05));
        double t[3] = {double(dir[j][1]*nrm[j/3][2]-dir[j][2]*nrm[j/3][1]), double(dir[j][2]*nrm[j/3][0]-dir[j][0]*nrm[j/3][2]),
                       double(dir[j][0]*nrm[j/3][1]-dir[j][1]*nrm[j/3][0])};
        double tn = std::sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]);
        m.p.forest[i+N*j] = pr(std::fabs(nrm[i/3][0]*t[0]+nrm[i/3][1]*t[1]+nrm[i/3][2]*t[2])/(std::sqrt(3.0)*tn));
      }
    }
    m.p.D = pr(1e20); m.p.D_0 = pr(0.0); m.p.Q_cl = pr(1.0); m.p.mu = pr(160e9); m.p.nu = pr(0.2796); m.p.k_B = pr(1.380649e-23);
    m.p.dipoleformation = 1; m.p.atol_xi = 1.0f; m.p.atol_gamma = 1e-6f; m.p.N_sl = N;
    typedef cp_model_dtung<N> Mo; typedef cp_layout<Mo> L;
    double eP = 0, eX = 0, eD = 0, eT = 0; int n = 0, mism = 0, n_tan = 0, plastic = 0;
    const double dt = 0.5;
    for (int gr = 0; gr < 30; ++gr) {
      double R[9]; rot(R);
      double Fdot[9] = {2e-4, 0, 0, 0, -1e-4, 0, 0, 0, -1e-4}; for (double& x : Fdot) x += 4e-5*u(g);   // 1e-4 per step
      std::vector<double> st(3*N), Dst(2*N); for (int i = 0; i < N; ++i) { st[i] = 1.0e12; st[N+i] = 1.0; st[2*N+i] = 0; }  // annealed metal
      dt_dep(m, st.data(), Dst.data());                                                             // crystallite_init
      double Fp[9], S6[6] = {0}, Lp[9] = {0}; std::memcpy(Fp, R, sizeof(Fp));
      double Fr[9], Lpr[9] = {0}, S6r[6] = {0}; std::memcpy(Fr, R, sizeof(Fr));
      std::vector<double> str = st, Dr = Dst;
      for (int inc = 0; inc < 120; ++inc) {
        double F[9]; for (int k = 0; k < 9; ++k) F[k] = (k%4==0) + Fdot[k]*dt*(inc+1);
        double Pr[9], Fpr[9], Fer[9], iFr[9]; std::vector<double> st_new(3*N), D_new(2*N);
        if (!dt_ref(m, F, Fr, S6r, str.data(), Dr.data(), dt, Lpr, st_new.data(), D_new.data(), S6r, Pr, Fpr, Fer, iFr)) {
          std::printf("   dislotungsten reference failed g%d i%d\n", gr, inc); break; }
        std::vector<double> ib(L::n_in, 0.0), ob(L::n_out, 0.0);
        for (int k = 0; k < 9; ++k) { ib[L::F+k] = F[k]; ib[L::Fp0+k] = Fp[k]; ib[L::Lp0+k] = Lp[k]; }
        for (int k = 0; k < 6; ++k) ib[L::S0+k] = S6[k];
        for (int j = 0; j < 3*N; ++j) ib[L::X0+j] = st[j];
        for (int j = 0; j < 2*N; ++j) ib[L::D0+j] = Dst[j];
        ib[L::T] = m.T;
        int its; int sb = cp_core_integrate<double,Mo>(m.p, num, dt, {ib.data(),1,0}, {ob.data(),1,0}, its);
        if (sb != CP_OK) { ++mism; std::printf("   dislotungsten core status %d g%d i%d\n", sb, gr, inc); break; }
        eP = std::max(eP, rel(&ob[L::P], Pr, 9)); eX = std::max(eX, rel(&ob[L::X], st_new.data(), 3*N));
        eD = std::max(eD, rel(&ob[L::D], D_new.data(), 2*N));
        plastic += st_new[2*N] > 1e-6;
        ++n;
        if (inc % 15 == 14) {                                                                       // tangent vs FD, state and dependent state fixed
          double tan_d[81], fd[81]; const double h = 1e-7;
          Kin kin = [&](const double* Mp, double* gd, double* dgd, bool* pos) { double te[N]; dt_kin(m, st_new.data(), D_new.data(), Mp, gd, dgd, pos, te); };
          double Lq[9], Sq[6], Pq[9], Fpq[9], Feq[9], iFq[9];
          std::memcpy(Lq, Lpr, sizeof(Lq));
          bool ok = ref_stress(m.p, kin, F, Fr, dt, Lq, Sq, Pq, Fpq, Feq, iFq)
                 && cp_core_tangent<double,Mo>(m.p, dt, F, Fr, iFq, Feq, Sq, st_new.data(), nullptr, D_new.data(), m.T, tan_d);
          for (int c = 0; c < 9 && ok; ++c) {
            double Fa[9], Fb[9], La[9], Lb[9], Sa[6], Sb[6], Pa[9], Pb[9], X1[9], X2[9], X3[9];
            std::memcpy(Fa, F, 72); std::memcpy(Fb, F, 72); Fa[c] += h; Fb[c] -= h;
            std::memcpy(La, Lq, 72); std::memcpy(Lb, Lq, 72);
            ok = ref_stress(m.p, kin, Fa, Fr, dt, La, Sa, Pa, X1, X2, X3) && ref_stress(m.p, kin, Fb, Fr, dt, Lb, Sb, Pb, X1, X2, X3);
            for (int q = 0; q < 9; ++q) fd[q + 9*c] = (Pa[q]-Pb[q])/(2*h);
          }
          if (ok) { eT = std::max(eT, rel(tan_d, fd, 81)); ++n_tan; }
        }
        { double iF[9]; std::memcpy(iF, &ob[L::iFp], sizeof(iF)); double dd = cp_inv33<double>(iF, Fp); double sc = std::cbrt(1.0/dd); for (double& x : Fp) x /= sc; }
        std::memcpy(S6, &ob[L::S], sizeof(S6)); std::memcpy(Lp, &ob[L::Lp], sizeof(Lp));
        for (int j = 0; j < 3*N; ++j) st[j] = ob[L::X+j];
        for (int j = 0; j < 2*N; ++j) Dst[j] = ob[L::D+j];
        std::memcpy(Fr, Fpr, sizeof(Fr)); str = st_new; Dr = D_new;
      }
    }
    bool ok = mism == 0 && n == 3600 && plastic > 0 && eP < 1e-6 && eX < 1e-6 && eD < 1e-6 && n_tan > 0 && eT < 1e-5;
    fails += !ok;
    std::printf("4. dislotungsten core vs DAMASK port: %d steps (%d plastic), status failures %d\n"
                "   max rel err P %.2e  dot state %.2e  dependent state %.2e; tangent vs FD (%d) %.2e   %s\n",
                n, plastic, mism, eP, eX, eD, n_tan, eT, ok ? "OK" : "FAIL");
  }
  //------------------------------------------------------------------------------------------------
  // 5. phenopowerlaw WITH twins (Mg, 18 slip + 12 twins = 30): core policy vs DAMASK port
  {
    PT m; std::memset(&m, 0, sizeof(m));
    FILE* fg = std::fopen(MG_GEOMETRY, "r");
    if (!fg || std::fscanf(fg, "%d %d", &m.ns, &m.nt) != 2) { std::printf("5. Mg geometry not found\n"); return fails + 1; }
    m.n = m.ns + m.nt;
    for (int i = 0; i < m.n; ++i) for (int k = 0; k < 9; ++k) if (std::fscanf(fg, "%lf", &m.P[i][k]) != 1) return fails + 1;
    for (int k = 0; k < m.nt; ++k) if (std::fscanf(fg, "%lf", &m.gch[m.ns+k]) != 1) return fails + 1;
    std::fclose(fg);
    for (int i = 0; i < m.n; ++i) for (int k = 0; k < 9; ++k) { m.Ppos[i][k] = m.P[i][k]; m.Pneg[i][k] = i < m.ns ? -m.P[i][k] : m.P[i][k]; }
    const double xi0[5] = {10e6, 55e6, 60e6, 0, 60e6}, xinf[5] = {40e6, 135e6, 150e6, 0, 150e6};
    const int fam[18] = {0,0,0, 1,1,1, 2,2,2,2,2,2, 4,4,4,4,4,4};
    for (int i = 0; i < m.ns; ++i) {
      m.g0[i] = 1e-3; m.nn[i] = 20; m.a[i] = 2.25; m.h0[i] = 0.5e9; m.xinf[i] = xinf[fam[i]];
      m.c1[i] = 2.0; m.c2[i] = 1.5; m.fsat[i] = 10.0;                                               // twin-induced slip hardening exercised
    }
    for (int k = m.ns; k < m.n; ++k) { m.g0[k] = 1e-3; m.nn[k] = 20; m.h0ts[k] = 0.15e9; m.h0tt[k] = 50e6; m.c3[k] = 0.8; m.c4[k] = 0.0; m.gch[k] = m.gch[k]; }
    for (int i = 0; i < m.n; ++i) for (int j = 0; j < m.n; ++j) {
      bool si = i < m.ns, sj = j < m.ns;
      m.H[i+m.n*j] = si && sj ? (i == j ? 1.0 : 1.2) : si && !sj ? 2.0e6 : !si && sj ? 1.0 : (i == j ? 1.0 : 3.0);
    }
    const double C11 = 59.5e9, C12 = 25.6e9, C13 = 21.5e9, C33 = 61.7e9, C44 = 16.5e9;
    double C[36] = {0};
    C[0] = C[7] = C11; C[14] = C33; C[6] = C[1] = C12; C[12] = C[2] = C[13] = C[8] = C13; C[21] = C[28] = C44; C[35] = 0.5*(C11-C12);
    std::memcpy(m.C66, C, sizeof(C));

    const int NP = 30;
    typedef cp_model_phenotw<NP> Mo; typedef cp_layout<Mo> L;
    static cp_phenotw_param p; std::memset(&p, 0, sizeof(p));
    for (int k = 0; k < 36; ++k) p.C66[k] = pr(C[k]);
    for (int i = 0; i < m.n; ++i) for (int k = 0; k < 9; ++k) { p.P_sl[9*i+k] = pr(m.P[i][k]); p.P_nS_pos[9*i+k] = pr(m.Ppos[i][k]); p.P_nS_neg[9*i+k] = pr(m.Pneg[i][k]); }
    for (int i = 0; i < m.n; ++i) for (int j = 0; j < m.n; ++j) p.h_sl_sl[i+NP*j] = pr(m.H[i+m.n*j]);
    for (int i = 0; i < NP; ++i) {
      p.dot_gamma_0[i] = pr(m.g0[i]); p.n[i] = pr(i < m.n ? m.nn[i] : 1.0); p.a[i] = pr(i < m.n ? m.a[i] : 1.0); p.h_0[i] = pr(m.h0[i]);
      p.xi_inf[i] = pr(i < m.ns ? m.xinf[i] : 1.0); p.c_1[i] = pr(m.c1[i]); p.c_2[i] = pr(i < m.ns ? m.c2[i] : 1.0); p.f_sat[i] = pr(m.fsat[i]);
      p.h_0_tw_sl[i] = pr(m.h0ts[i]); p.h_0_tw_tw[i] = pr(m.h0tt[i]); p.c_3[i] = pr(i >= m.ns && i < m.n ? m.c3[i] : 1.0); p.c_4[i] = pr(m.c4[i]);
      p.gamma_char[i] = pr(i >= m.ns && i < m.n ? m.gch[i] : 1.0);
    }
    p.atol_xi = 1.0f; p.atol_gamma = 1e-6f; p.N_sl = m.n; p.N_slip = m.ns; p.N_twin = m.nt;

    double eP = 0, eX = 0; int n = 0, mism = 0, twinning = 0;
    const double dt = 0.5;
    for (int gr = 0; gr < 30; ++gr) {
      double R[9]; rot(R);
      double Fdot[9] = {1e-3, 0, 0, 0, -5e-4, 0, 0, 0, -5e-4}; for (double& x : Fdot) x += 2e-4*u(g);
      if (gr % 2) for (double& x : Fdot) x = -x;                                                    // compression half: other twin sense
      std::vector<double> st(2*m.n, 0.0), stp(2*NP, 0.0);
      for (int i = 0; i < m.ns; ++i) st[i] = xi0[fam[i]];
      for (int k = m.ns; k < m.n; ++k) st[k] = k < m.ns + 6 ? 40e6 : 60e6;
      double Fp[9], S6[6] = {0}, Lp[9] = {0}; std::memcpy(Fp, R, sizeof(Fp));
      double Fr[9], Lpr[9] = {0}, S6r[6] = {0}; std::memcpy(Fr, R, sizeof(Fr));
      std::vector<double> str = st;
      for (int inc = 0; inc < 60; ++inc) {
        double F[9]; for (int k = 0; k < 9; ++k) F[k] = (k%4==0) + Fdot[k]*dt*(inc+1);
        double Pr[9], Fpr[9], Fer[9], iFr[9]; std::vector<double> st_new(2*m.n);
        if (!pt_ref(m, F, Fr, S6r, str.data(), dt, Lpr, st_new.data(), S6r, Pr, Fpr, Fer, iFr)) { std::printf("   pheno+twin reference failed g%d i%d\n", gr, inc); break; }
        std::vector<double> ib(L::n_in, 0.0), ob(L::n_out, 0.0);
        for (int k = 0; k < 9; ++k) { ib[L::F+k] = F[k]; ib[L::Fp0+k] = Fp[k]; ib[L::Lp0+k] = Lp[k]; }
        for (int k = 0; k < 6; ++k) ib[L::S0+k] = S6[k];
        for (int blk = 0; blk < 2; ++blk) for (int i = 0; i < NP; ++i) ib[L::X0+blk*NP+i] = i < m.n ? st[blk*m.n+i] : (blk == 0 ? 1.0 : 0.0);
        int its; int sb = cp_core_integrate<double,Mo>(p, num, dt, {ib.data(),1,0}, {ob.data(),1,0}, its);
        if (sb != CP_OK) { ++mism; std::printf("   pheno+twin core status %d g%d i%d\n", sb, gr, inc); break; }
        std::vector<double> stc(2*m.n);
        for (int blk = 0; blk < 2; ++blk) for (int i = 0; i < m.n; ++i) stc[blk*m.n+i] = ob[L::X+blk*NP+i];
        eP = std::max(eP, rel(&ob[L::P], Pr, 9)); eX = std::max(eX, rel(stc.data(), st_new.data(), 2*m.n));
        for (int k = m.ns; k < m.n; ++k) twinning += st_new[m.n+k] > 1e-5;
        ++n;
        { double iF[9]; std::memcpy(iF, &ob[L::iFp], sizeof(iF)); double dd = cp_inv33<double>(iF, Fp); double sc = std::cbrt(1.0/dd); for (double& x : Fp) x /= sc; }
        std::memcpy(S6, &ob[L::S], sizeof(S6)); std::memcpy(Lp, &ob[L::Lp], sizeof(Lp)); st = stc;
        std::memcpy(Fr, Fpr, sizeof(Fr)); str = st_new;
      }
    }
    bool ok = mism == 0 && n == 1800 && twinning > 0 && eP < 1e-6 && eX < 1e-6;
    fails += !ok;
    std::printf("5. phenopowerlaw+twins (Mg 18+12) core vs DAMASK port: %d steps (%d twinning system-steps), failures %d\n"
                "   max rel err P %.2e  state %.2e   %s\n", n, twinning, mism, eP, eX, ok ? "OK" : "FAIL");
  }

  //------------------------------------------------------------------------------------------------
  // 6. FLOW/STATE_C/FLAT core paths: phenopowerlaw routed through them vs the rank-1 policy
  {
    cp_pheno_param p; std::memset(&p, 0, sizeof(p));
    geometry(p, 106.75e9, 60.41e9, 28.34e9);
    for (int i = 0; i < N; ++i) {
      p.dot_gamma_0[i] = pr(0.001); p.n[i] = pr(20); p.a[i] = pr(2.25); p.h_0[i] = pr(75e6); p.xi_inf[i] = pr(63e6);
      for (int j = 0; j < N; ++j) p.h_sl_sl[i+N*j] = pr(i == j ? 1.0 : 1.4);
    }
    p.atol_xi = 1.0f; p.atol_gamma = 1e-6f; p.N_sl = N;
    typedef cp_model_pheno<N> Ma; typedef cp_model_pheno_flow<N> Mb; typedef cp_layout<Ma> La; typedef cp_layout<Mb> Lb;
    double e = 0, eT = 0; int n = 0, mism = 0;
    for (int gr = 0; gr < 40; ++gr) {
      double R[9]; rot(R);
      double Fdot[9] = {1e-3, 0, 0, 0, -5e-4, 0, 0, 0, -5e-4}; for (double& x : Fdot) x += 2e-4*u(g);
      std::vector<double> st(2*N); for (int i = 0; i < N; ++i) { st[i] = 31e6; st[N+i] = 0; }
      double Fp[9], S6[6] = {0}, Lp[9] = {0}; std::memcpy(Fp, R, sizeof(Fp));
      for (int inc = 0; inc < 120; ++inc) {
        double F[9]; for (int k = 0; k < 9; ++k) F[k] = (k%4==0) + Fdot[k]*(inc+1);
        std::vector<double> ia(La::n_in, 0.0), oa(La::n_out, 0.0), ib(Lb::n_in, 0.0), ob(Lb::n_out, 0.0);
        for (int k = 0; k < 9; ++k) { ia[La::F+k] = ib[Lb::F+k] = F[k]; ia[La::Fp0+k] = ib[Lb::Fp0+k] = Fp[k]; ia[La::Lp0+k] = ib[Lb::Lp0+k] = Lp[k]; }
        for (int k = 0; k < 6; ++k) ia[La::S0+k] = ib[Lb::S0+k] = S6[k];
        for (int i = 0; i < 2*N; ++i) ia[La::X0+i] = ib[Lb::X0+i] = st[i];
        for (int k = 0; k < 36; ++k) ib[Lb::D0+k] = dv(p.C66[k]);
        int i1, i2;
        int sa = cp_core_integrate<double,Ma>(p, num, 1.0, {ia.data(),1,0}, {oa.data(),1,0}, i1);
        int sb = cp_core_integrate<double,Mb>(p, num, 1.0, {ib.data(),1,0}, {ob.data(),1,0}, i2);
        if (sa != sb) { ++mism; break; }
        if (sa != CP_OK) break;
        e = std::max({e, rel(&ob[Lb::P], &oa[La::P], 9), rel(&ob[Lb::S], &oa[La::S], 6), rel(&ob[Lb::X], &oa[La::X], 2*N),
                      rel(&ob[Lb::iFp], &oa[La::iFp], 9)});
        double ta[81], tb[81];
        bool okt = cp_core_tangent<double,Ma>(p, 1.0, F, Fp, &oa[La::iFp], &oa[La::Fe], &oa[La::S], &oa[La::X], nullptr, nullptr, 0.0, ta)
                && cp_core_tangent<double,Mb>(p, 1.0, F, Fp, &ob[Lb::iFp], &ob[Lb::Fe], &ob[Lb::S], &ob[Lb::X], nullptr, &ob[Lb::D], 0.0, tb);
        eT = okt ? std::max(eT, rel(tb, ta, 81)) : 1.0;
        ++n;
        { double iF[9]; std::memcpy(iF, &oa[La::iFp], sizeof(iF)); double dd = cp_inv33<double>(iF, Fp); double s = std::cbrt(1.0/dd); for (double& x : Fp) x /= s; }
        std::memcpy(S6, &oa[La::S], sizeof(S6)); std::memcpy(Lp, &oa[La::Lp], sizeof(Lp)); std::memcpy(st.data(), &oa[La::X], 2*N*sizeof(double));
      }
    }
    bool ok = mism == 0 && n == 4800 && e < 1e-12 && eT < 1e-12;
    fails += !ok;
    std::printf("6. FLOW/STATE_C/FLAT paths (pheno_flow) vs rank-1 pheno: %d steps, status mismatches %d, max rel diff %.2e, tangent %.2e  %s\n",
                n, mism, e, eT, ok ? "OK" : "FAIL");
  }


  //------------------------------------------------------------------------------------------------
  // 7. tdt policy (Mg: 18 slip in 4 families, 6 tension twins): derivatives vs finite differences,
  //    Reuss stiffness vs an independent computation, c-axis tension -> compression (twin, detwin)
  {
    int ns, nt_all;
    FILE* fg = std::fopen(MG_GEOMETRY, "r");
    if (!fg || std::fscanf(fg, "%d %d", &ns, &nt_all) != 2) { std::printf("7. Mg geometry not found\n"); return fails + 1; }
    std::vector<double> Pall(9*(ns+nt_all)), gch_all(nt_all);
    for (double& x : Pall) if (std::fscanf(fg, "%lf", &x) != 1) return fails + 1;
    for (double& x : gch_all) if (std::fscanf(fg, "%lf", &x) != 1) return fails + 1;
    std::fclose(fg);
    const int nt = 6, nf = 4;
    const int famv[18] = {1,1,1, 2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4};
    std::vector<double> Psl(Pall.begin(), Pall.begin()+9*ns), Ptw(Pall.begin()+9*ns, Pall.begin()+9*(ns+nt)), Q(9*nt), Pst(9*ns*nt);
    for (int k = 0; k < nt; ++k) {                                                                  // Q = 2 n n - I, n n = P^T P
      const double* P = &Ptw[9*k]; double nn[9] = {0}, n[3];
      for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) for (int c = 0; c < 3; ++c) nn[CP_IDX(a,b)] += P[CP_IDX(c,a)]*P[CP_IDX(c,b)];
      int c = 0; for (int a = 1; a < 3; ++a) if (nn[CP_IDX(a,a)] > nn[CP_IDX(c,c)]) c = a;
      double s = std::sqrt(nn[CP_IDX(c,c)]); for (int a = 0; a < 3; ++a) n[a] = nn[CP_IDX(a,c)]/s;
      for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) Q[9*k+CP_IDX(a,b)] = 2*n[a]*n[b] - (a == b);
      for (int i = 0; i < ns; ++i) {
        double T1[9], T2[9]; cp_mul33<double>(&Q[9*k], &Psl[9*i], T1); cp_mul3T3<double>(T1, &Q[9*k], T2);
        std::memcpy(&Pst[9*(i+ns*k)], T2, sizeof(T2));
      }
    }
    const double C11 = 59.5e9, C12 = 25.6e9, C13 = 21.5e9, C33 = 61.7e9, C44 = 16.5e9;
    double C[36] = {0};
    C[0] = C[7] = C11; C[14] = C33; C[6] = C[1] = C12; C[12] = C[2] = C[13] = C[8] = C13; C[21] = C[28] = C44; C[35] = 0.5*(C11-C12);
    // rotated stiffness (independent: full 4th-order rotation) and its inverse
    auto rotC = [&](const double* Qm, double* Cr) {
      auto c4 = [&](int i, int j, int k, int l) { return C[cp_voigt(i,j) + 6*cp_voigt(k,l)]; };
      for (int I = 0; I < 6; ++I) for (int J = 0; J < 6; ++J) {
        int i = cp_voigt_i(I), j = cp_voigt_j(I), k = cp_voigt_i(J), l = cp_voigt_j(J); double s = 0;
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) for (int c = 0; c < 3; ++c) for (int d = 0; d < 3; ++d)
          s += Qm[CP_IDX(i,a)]*Qm[CP_IDX(j,b)]*Qm[CP_IDX(k,c)]*Qm[CP_IDX(l,d)]*c4(a,b,c,d);
        Cr[I+6*J] = s;
      }
    };
    std::vector<double> Mel(36*(1+nt));
    cp_inv66<double>(C, &Mel[0]);
    for (int k = 0; k < nt; ++k) { double Cr[36]; rotC(&Q[9*k], Cr); cp_inv66<double>(Cr, &Mel[36*(k+1)]); }
    // blob as plastic_tdt_GPUadd
    const double xi0[4] = {10e6, 55e6, 60e6, 60e6}, xinf[4] = {40e6, 135e6, 150e6, 150e6};
    std::vector<double> blob;
    auto app = [&](const std::vector<double>& v) { blob.insert(blob.end(), v.begin(), v.end()); };
    app(Psl); app(Pst); app(Ptw); app(Mel);
    std::vector<double> v(ns);
    for (int i = 0; i < ns; ++i) v[i] = 1e-3;              app(v);
    for (int i = 0; i < ns; ++i) v[i] = 20;                app(v);
    for (int i = 0; i < ns; ++i) v[i] = 2.25;              app(v);
    for (int i = 0; i < ns; ++i) v[i] = 0.5e9;             app(v);
    for (int i = 0; i < ns; ++i) v[i] = xinf[famv[i]-1];   app(v);
    for (int i = 0; i < ns; ++i) v[i] = famv[i];           app(v);
    app({60e6, 150e6, 160e6, 160e6});                                                               // xi_inf_sl_tw
    app({0.3e9, 0.5e9, 0.5e9, 0.5e9});                                                              // h_0_sl_tw (family)
    app({2.25, 2.25, 2.25, 2.25});                                                                  // a_sl_tw
    std::vector<double> gch(nt); for (int k = 0; k < nt; ++k) gch[k] = std::fabs(gch_all[k]);
    app(std::vector<double>(nt, 1e-3)); app(std::vector<double>(nt, 20)); app(gch);
    app(std::vector<double>(nt, 10e6));                                                             // dxi_nuc
    app(std::vector<double>(nt, 15e6));                                                             // dxi_retw
    app(std::vector<double>(nt, 20e6));                                                             // H_chi
    app(std::vector<double>(nt, 0.1e9));                                                            // h_0_dtw
    std::vector<double> hss(ns*ns), hst(ns*nt), hts(nt*ns), htt(nt*nt);
    for (int i = 0; i < ns; ++i) for (int j = 0; j < ns; ++j) hss[i+ns*j] = i == j ? 1.0 : 1.2;
    for (int i = 0; i < ns; ++i) for (int k = 0; k < nt; ++k) hst[i+ns*k] = 50e6*gch[k];
    for (int k = 0; k < nt; ++k) for (int j = 0; j < ns; ++j) hts[k+nt*j] = 0.1e9;
    for (int k = 0; k < nt; ++k) for (int l = 0; l < nt; ++l) htt[k+nt*l] = 50e6*(k == l ? 1.0 : 3.0)*gch[l];
    app(hss); app(hst); app(hts); app(htt);
    app({0.01, 1.0, 1e-12});
    app(std::vector<double>(ns, 0.5)); app(std::vector<double>(ns, 1.5)); app(std::vector<double>(ns, 20e6)); // c_1, c_2, f_sat_sl-tw
    app(std::vector<double>(ns, 8e6)); app(std::vector<double>(1, 0.02));                           // xi_pin_sl, gamma_pin (pinning exercised)
    const int dims[4] = {ns, nt, nf, 1};
    const double atols[3] = {1.0, 1e-6, 1e-6};
    typedef cp_model_tdt<18,6,6> Mo; typedef cp_layout<Mo> L;
    static cp_tdt_param p; std::vector<int> smap; std::vector<double> sinit;
    const int used = cp_tdt_pack(dims, C, blob.data(), atols, 18, 6, 6, p, smap, sinit);
    const int NS = int(smap.size()) - 36;                                                           // DAMASK state incl. f0
    bool ok = NS == 2*ns + (5+nf)*nt && used == int(blob.size());

    // (a) derivatives at a state with twins present
    std::vector<double> X(sinit.begin(), sinit.begin()+Mo::ND+Mo::NY), Xd(NS, 0.0);
    for (int i = 0; i < ns; ++i) Xd[i] = xi0[famv[i]-1];
    const double fr[6] = {0.2, 0.05, 0.0, 0.0, 0.1, 0.0};
    for (int k = 0; k < nt; ++k) { Xd[2*ns+k] = fr[k]; Xd[2*ns+nt+k] = 30e6; Xd[2*ns+2*nt+k] = 20e6; Xd[NS-nt+k] = 0.9*fr[k] + 0.01; }
    for (int k = 0; k < nt; ++k) for (int f = 0; f < nf; ++f) Xd[2*ns+3*nt+nf*k+f] = 1.3*xi0[f];
    for (int q = 0; q < NS; ++q) X[smap[q]] = Xd[q];
    double eG = 0, eK = 0;
    for (int trial = 0; trial < 50; ++trial) {
      double S[6]; for (double& x : S) x = 40e6*u(g);
      double Lp[9], G[54], K[81];
      Mo::flow<double>(p, S, X.data(), X.data()+Mo::ND, nullptr, 0.0, 0.5, Lp, G, 1);
      Mo::flow<double>(p, S, X.data(), X.data()+Mo::ND, nullptr, 0.0, 0.5, Lp, K, 2);
      double gn = 0; for (double x : G) gn = std::max(gn, std::fabs(x));
      for (int k = 0; k < 6; ++k) {
        double h = 1e-6*40e6, Sp[6], Sm[6], Lpp[9], Lpm[9];
        std::memcpy(Sp, S, sizeof(S)); std::memcpy(Sm, S, sizeof(S)); Sp[k] += h; Sm[k] -= h;
        Mo::flow<double>(p, Sp, X.data(), X.data()+Mo::ND, nullptr, 0.0, 0.5, Lpp, nullptr, 0);
        Mo::flow<double>(p, Sm, X.data(), X.data()+Mo::ND, nullptr, 0.0, 0.5, Lpm, nullptr, 0);
        int a = cp_voigt_i(k), b = cp_voigt_j(k);
        for (int m = 0; m < 9; ++m) {
          double fd = (Lpp[m]-Lpm[m])/(2*h);
          eG = std::max(eG, std::fabs(G[m+9*k]-fd)/std::max(gn, 1e-300));
          double kk = k < 3 ? K[m+9*CP_IDX(a,a)] : K[m+9*CP_IDX(a,b)] + K[m+9*CP_IDX(b,a)];
          eK = std::max(eK, std::fabs(kk-G[m+9*k])/std::max(gn, 1e-300));
        }
      }
    }
    // (b) Reuss stiffness: independent (rotated C, inverted compliance mixture) vs dependent()
    double D[36], Sb[36] = {0}, Cref[36], w0 = 1;
    for (int k = 0; k < nt; ++k) w0 -= fr[k];
    { double S0[36]; cp_inv66<double>(C, S0); for (int q = 0; q < 36; ++q) Sb[q] = w0*S0[q]; }
    for (int k = 0; k < nt; ++k) { double Cr[36], Sr[36]; rotC(&Q[9*k], Cr); cp_inv66<double>(Cr, Sr); for (int q = 0; q < 36; ++q) Sb[q] += fr[k]*Sr[q]; }
    cp_inv66<double>(Sb, Cref);
    Mo::dependent<double>(p, X.data(), nullptr, 0.0, D);
    double eC = rel(D, Cref, 36);
    ok = ok && eG < 1e-6 && eK < 1e-12 && eC < 1e-12;

    // (c) single crystal, c-axis (z) tension to 4 %, then compression to -2 %: twins grow, then shrink
    std::vector<double> st(Mo::ND+Mo::NY); std::copy(sinit.begin(), sinit.begin()+Mo::ND+Mo::NY, st.begin());
    for (int i = 0; i < ns; ++i) st[i] = xi0[famv[i]-1];
    for (int k = 0; k < nt; ++k) { st[Mo::iXT+k] = 30e6; st[Mo::iXD+k] = 20e6; for (int f = 0; f < nf; ++f) st[Mo::iXR+6*k+f] = 1.3*xi0[f]; }
    double Fp[9] = {1,0,0, 0,1,0, 0,0,1}, S6[6] = {0}, Lp[9] = {0}, D0[36];
    Mo::dependent<double>(p, st.data(), nullptr, 0.0, D0);
    const double dt = 0.5, de = 5e-4;                                                               // strain rate 1e-3/s
    double ez = 0, f_max = 0, f_end = 0, f_min = 1, sz_tw = 0; int n = 0, bad = 0, its_tot = 0;
    for (int inc = 0; inc < 200; ++inc) {
      ez += inc < 80 ? de : -de;
      double lt = 1.0/std::sqrt(1.0+ez), F[9] = {lt,0,0, 0,lt,0, 0,0,1+ez};                       // isochoric
      std::vector<double> ib(L::n_in, 0.0), ob(L::n_out, 0.0);
      for (int k = 0; k < 9; ++k) { ib[L::F+k] = F[k]; ib[L::Fp0+k] = Fp[k]; ib[L::Lp0+k] = Lp[k]; }
      for (int k = 0; k < 6; ++k) ib[L::S0+k] = S6[k];
      for (int j = 0; j < Mo::ND+Mo::NY; ++j) ib[L::X0+j] = st[j];
      for (int k = 0; k < 36; ++k) ib[L::D0+k] = D0[k];
      int its; int sb = cp_core_integrate<double,Mo>(p, num, dt, {ib.data(),1,0}, {ob.data(),1,0}, its);
      if (sb != CP_OK) { ++bad; std::printf("   tdt status %d at increment %d (e_z %.4f)\n", sb, inc, ez); break; }
      its_tot += its; ++n;
      { double iF[9]; std::memcpy(iF, &ob[L::iFp], sizeof(iF)); double dd = cp_inv33<double>(iF, Fp); double s = std::cbrt(1.0/dd); for (double& x : Fp) x /= s; }
      std::memcpy(S6, &ob[L::S], sizeof(S6)); std::memcpy(Lp, &ob[L::Lp], sizeof(Lp)); std::memcpy(st.data(), &ob[L::X], (Mo::ND+Mo::NY)*sizeof(double));
      std::memcpy(D0, &ob[L::D], sizeof(D0));
      double ft = 0; for (int k = 0; k < nt; ++k) { ft += st[Mo::iF+k]; f_min = std::min(f_min, st[Mo::iF+k]); }
      if (inc == 79) { f_max = ft; sz_tw = S6[2]; }
      f_end = ft;
      if (inc % 20 == 19) std::printf("   inc %3d  e_z %+.4f  S_zz %8.2f MPa  sum f %.4f\n", inc+1, ez, S6[2]/1e6, ft);
    }
    ok = ok && bad == 0 && n == 200 && f_max > 0.2 && f_end < 0.5*f_max && f_min >= 0.0;
    fails += !ok;
    std::printf("7. tdt (Mg 18 slip, 6 twins): dLp/dS vs FD %.2e, K vs G %.2e, Reuss C %.2e; c-axis cycle: %d incs (%.1f its/inc), "
                "sum f %.3f after tension (S_zz %.0f MPa), %.3f after reversal, min f %.1e  %s\n",
                eG, eK, eC, n, n ? double(its_tot)/n : 0.0, f_max, sz_tw/1e6, f_end, f_min, ok ? "OK" : "FAIL");
  }

  return fails;
}
