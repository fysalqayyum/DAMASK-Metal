// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Single-source (Metal Shading Language / C++) constitutive update for one material point:
//> Hooke elasticity + phenopowerlaw slip, no eigenstrain (Fi = I), FPI state integration.
//> @details Solves the same discrete equations as DAMASK's integrateStress/integrateStateFPI
//> (phase_mechanical.f90), but restructured for SIMD hardware:
//>  - unknown is the symmetric 2nd PK stress S (6 Voigt components) instead of Lp (9), i.e.
//>    R(S) = S - C:E(Fe(Lp(S))), Fe = A(I - dt Lp), A = F Fp0^-1
//>  - residual/state in precision H (df64 on GPU, double on CPU), Jacobian and 6x6 LU in J
//>    (float on GPU) => mixed-precision iterative refinement, converges to H accuracy
//>  - compile-time slip system count N (fully unrolled), no runtime model dispatch
//>  - bounded iteration budgets; cells exceeding them report status != CP_OK and are retried
//>    by the host (larger budget / sub-stepping / CPU fallback) after stream compaction
//> Memory layout of all 3x3 tensors is Fortran column-major: (i,j) at i+3*j (zero-based).
//> Voigt order as in DAMASK: 11,22,33,23,13,12; strain shear components are engineering (2E_ij).
//--------------------------------------------------------------------------------------------------
#ifndef DAMASK_CP_KERNEL_H
#define DAMASK_CP_KERNEL_H

#include "df64.h"

#ifdef __METAL_VERSION__
  #define CP_THREAD   thread
  #define CP_CONSTANT constant                                                                    // uniformly accessed parameters: constant address space
  #define CP_FMIN(a,b) metal::fmin(a,b)
  #define CP_FMAX(a,b) metal::fmax(a,b)
  #define CP_TANH(a)   metal::precise::tanh(a)
  #define CP_FABS(a)   metal::fabs(a)
#else
  #include <cmath>
  #define CP_THREAD
  #define CP_CONSTANT const
  #define CP_FMIN(a,b) std::fmin(a,b)
  #define CP_FMAX(a,b) std::fmax(a,b)
  #define CP_TANH(a)   std::tanh(a)
  #define CP_FABS(a)   std::fabs(a)
#endif

#define CP_MAX_SL 48
#define CP_IDX(i,j) ((i)+3*(j))

struct cp_pair { float hi; float lo; };                                                             // x = hi + lo

enum {
  CP_OK            = 0,
  CP_FAIL_STRESS   = 1,                                                                             // stress Newton did not converge within budget
  CP_FAIL_NAN      = 2,
  CP_FAIL_STATE    = 3,                                                                             // state FPI did not converge within budget
  CP_FAIL_SINGULAR = 4,
  CP_FAIL_VERIFY   = 5                                                                              // rejected by the host-side FP64 certificate
};

struct cp_numerics {
  float rtol_S;                                                                                     // |R| <= rtol_S*|S| + atol_S
  float atol_S;
  float rtol_state;                                                                                 // eps_rel_state
  float stepsize_ls;                                                                                // r_linesearch_Lp
  float trust_S;                                                                                    // max |dS|/|S| per Newton step
  int   n_stress;                                                                                   // N_iter_Lp_max
  int   n_state;                                                                                    // N_iter_state_max
  int   mixed = 0;                                                                                  // cp_core: FP32 Newton pre-solve before the H iterations
};

struct cp_pheno_param {
  cp_pair C66[36];                                                                                  // (a,b) at a+6*b
  cp_pair P_sl    [CP_MAX_SL*9];
  cp_pair P_nS_pos[CP_MAX_SL*9];
  cp_pair P_nS_neg[CP_MAX_SL*9];
  cp_pair h_sl_sl [CP_MAX_SL*CP_MAX_SL];                                                            // (i,j) at i+N*j
  cp_pair dot_gamma_0[CP_MAX_SL];
  cp_pair n          [CP_MAX_SL];
  cp_pair a          [CP_MAX_SL];
  cp_pair h_0        [CP_MAX_SL];
  cp_pair xi_inf     [CP_MAX_SL];
  float   atol_xi;
  float   atol_gamma;
  int     N_sl;
  int     pad;
};


//--------------------------------------------------------------------------------------------------
// scalar traits: H = residual precision, J = Jacobian precision
template<typename H> struct cp_ops;

template<> struct cp_ops<float> {
  typedef float J;
  static float c(float x)      { return x; }
  static float fj(float x)     { return x; }
  static float ld(cp_pair p)   { return p.hi + p.lo; }
  static float f(float x)      { return x; }
  static float ex(float x)     { return DF_EXP(x); }
  static float lg(float x)     { return DF_LOG(x); }
  static float sq(float x)     { return DF_SQRT(x); }
  static float ab(float x)     { return x < 0.0f ? -x : x; }
  static bool  finite(float x) { return DF_ISFINITE(x); }
};

template<> struct cp_ops<df64> {
  typedef float J;
  static df64  c(float x)      { return df_from(x); }
  static df64  fj(float x)     { return df_from(x); }
  static df64  ld(cp_pair p)   { return df_make(p.hi, p.lo); }
  static float f(df64 x)       { return df_to_float(x); }
  static df64  ex(df64 x)      { return df_exp(x); }
  static df64  lg(df64 x)      { return df_log(x); }
  static df64  sq(df64 x)      { return df_sqrt(x); }
  static df64  ab(df64 x)      { return df_abs(x); }
  static bool  finite(df64 x)  { return df_isfinite(x); }
};

#ifndef __METAL_VERSION__
template<> struct cp_ops<double> {
  typedef double J;
  static double c(float x)      { return double(x); }
  static double fj(double x)    { return x; }
  static double ld(cp_pair p)   { return double(p.hi) + double(p.lo); }
  static double f(double x)     { return x; }
  static double ex(double x)    { return std::exp(x); }
  static double lg(double x)    { return std::log(x); }
  static double sq(double x)    { return std::sqrt(x); }
  static double ab(double x)    { return std::fabs(x); }
  static bool   finite(double x){ return std::isfinite(x); }
};

//> host: double as an unevaluated float pair hi + lo (constants for the GPU)
inline cp_pair cp_to_pair(double x) { cp_pair p; p.hi = float(x); p.lo = float(x - double(p.hi)); return p; }
#endif


template<typename J> inline J cp_ldJ(cp_pair p) { return J(p.hi) + J(p.lo); }

//--------------------------------------------------------------------------------------------------
// 3x3 helpers (column-major), generic in T
template<typename T>
inline void cp_mul33(CP_THREAD const T* a, CP_THREAD const T* b, CP_THREAD T* c) {                 // c = a b
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i)
      c[CP_IDX(i,j)] = a[CP_IDX(i,0)]*b[CP_IDX(0,j)] + a[CP_IDX(i,1)]*b[CP_IDX(1,j)] + a[CP_IDX(i,2)]*b[CP_IDX(2,j)];
}

template<typename T>
inline void cp_mulT33(CP_THREAD const T* a, CP_THREAD const T* b, CP_THREAD T* c) {                // c = a^T b
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i)
      c[CP_IDX(i,j)] = a[CP_IDX(0,i)]*b[CP_IDX(0,j)] + a[CP_IDX(1,i)]*b[CP_IDX(1,j)] + a[CP_IDX(2,i)]*b[CP_IDX(2,j)];
}

template<typename T>
inline void cp_mul3T3(CP_THREAD const T* a, CP_THREAD const T* b, CP_THREAD T* c) {                // c = a b^T
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i)
      c[CP_IDX(i,j)] = a[CP_IDX(i,0)]*b[CP_IDX(j,0)] + a[CP_IDX(i,1)]*b[CP_IDX(j,1)] + a[CP_IDX(i,2)]*b[CP_IDX(j,2)];
}

template<typename T>
inline T cp_det33(CP_THREAD const T* a) {
  return a[CP_IDX(0,0)]*(a[CP_IDX(1,1)]*a[CP_IDX(2,2)] - a[CP_IDX(1,2)]*a[CP_IDX(2,1)])
       - a[CP_IDX(0,1)]*(a[CP_IDX(1,0)]*a[CP_IDX(2,2)] - a[CP_IDX(1,2)]*a[CP_IDX(2,0)])
       + a[CP_IDX(0,2)]*(a[CP_IDX(1,0)]*a[CP_IDX(2,1)] - a[CP_IDX(1,1)]*a[CP_IDX(2,0)]);
}

//> inverse via adjugate; returns determinant (caller checks for zero)
template<typename T>
inline T cp_inv33(CP_THREAD const T* a, CP_THREAD T* b) {
  b[CP_IDX(0,0)] =  a[CP_IDX(1,1)]*a[CP_IDX(2,2)] - a[CP_IDX(1,2)]*a[CP_IDX(2,1)];
  b[CP_IDX(1,0)] = -(a[CP_IDX(1,0)]*a[CP_IDX(2,2)] - a[CP_IDX(1,2)]*a[CP_IDX(2,0)]);
  b[CP_IDX(2,0)] =  a[CP_IDX(1,0)]*a[CP_IDX(2,1)] - a[CP_IDX(1,1)]*a[CP_IDX(2,0)];
  b[CP_IDX(0,1)] = -(a[CP_IDX(0,1)]*a[CP_IDX(2,2)] - a[CP_IDX(0,2)]*a[CP_IDX(2,1)]);
  b[CP_IDX(1,1)] =  a[CP_IDX(0,0)]*a[CP_IDX(2,2)] - a[CP_IDX(0,2)]*a[CP_IDX(2,0)];
  b[CP_IDX(2,1)] = -(a[CP_IDX(0,0)]*a[CP_IDX(2,1)] - a[CP_IDX(0,1)]*a[CP_IDX(2,0)]);
  b[CP_IDX(0,2)] =  a[CP_IDX(0,1)]*a[CP_IDX(1,2)] - a[CP_IDX(0,2)]*a[CP_IDX(1,1)];
  b[CP_IDX(1,2)] = -(a[CP_IDX(0,0)]*a[CP_IDX(1,2)] - a[CP_IDX(0,2)]*a[CP_IDX(1,0)]);
  b[CP_IDX(2,2)] =  a[CP_IDX(0,0)]*a[CP_IDX(1,1)] - a[CP_IDX(0,1)]*a[CP_IDX(1,0)];
  T det = a[CP_IDX(0,0)]*b[CP_IDX(0,0)] + a[CP_IDX(0,1)]*b[CP_IDX(1,0)] + a[CP_IDX(0,2)]*b[CP_IDX(2,0)];
  for (int k = 0; k < 9; ++k) b[k] = b[k] / det;
  return det;
}

//> contraction of a (possibly unsymmetric) 3x3 tensor with the symmetric Voigt basis tensor k
inline int cp_voigt_i(int k) { return k < 3 ? k : (k == 3 ? 1 : 0); }
inline int cp_voigt_j(int k) { return k < 3 ? k : (k == 5 ? 1 : 2); }

//> 6x6 LU with partial pivoting, solves A x = b in place (x returned in b); false if singular
template<typename T>
inline bool cp_solve66(CP_THREAD T* A, CP_THREAD T* b) {                                            // A(r,c) at r+6*c
  for (int c = 0; c < 6; ++c) {
    int p = c;
    T amax = A[c+6*c] < T(0) ? -A[c+6*c] : A[c+6*c];
    for (int r = c+1; r < 6; ++r) {
      T v = A[r+6*c] < T(0) ? -A[r+6*c] : A[r+6*c];
      if (v > amax) { amax = v; p = r; }
    }
    if (!(amax > T(0))) return false;
    if (p != c) {
      for (int k = 0; k < 6; ++k) { T t = A[c+6*k]; A[c+6*k] = A[p+6*k]; A[p+6*k] = t; }
      T t = b[c]; b[c] = b[p]; b[p] = t;
    }
    T inv = T(1) / A[c+6*c];
    for (int r = c+1; r < 6; ++r) {
      T l = A[r+6*c] * inv;
      A[r+6*c] = l;
      for (int k = c+1; k < 6; ++k) A[r+6*k] -= l * A[c+6*k];
      b[r] -= l * b[c];
    }
  }
  for (int r = 5; r >= 0; --r) {
    T s = b[r];
    for (int k = r+1; k < 6; ++k) s -= A[r+6*k] * b[k];
    b[r] = s / A[r+6*r];
  }
  return true;
}


//--------------------------------------------------------------------------------------------------
//> per material point data
template<typename H, int N>
struct cp_point {
  // in
  H F[9];                                                                                           // deformation gradient at end of step
  H Fp0[9];                                                                                         // plastic deformation gradient at start of step
  H S0[6];                                                                                          // stress at start of step (initial dotState)
  H Lp_guess[9];                                                                                    // initial guess (defines initial S)
  H xi0[N], gamma0[N];                                                                              // state at start of step
  // out
  H S[6];
  H xi[N], gamma[N];
  H P[9], Fp[9], Lp[9], Fe[9];
  typename cp_ops<H>::J dPdF[81];                                                                   // dP_ij/dF_kl at i+3j+9k+27l
  int status;
  int n_iter_stress;                                                                                // total Newton iterations (statistics)
  int n_iter_state;
};


//--------------------------------------------------------------------------------------------------
//> resolved shear, shear rates, and their derivative w.r.t. the active resolved stress
template<typename H, int N>
inline void cp_kinetics(CP_CONSTANT cp_pheno_param& p,
                        CP_THREAD const H* S, CP_THREAD const H* xi,
                        CP_THREAD H* gd, CP_THREAD typename cp_ops<H>::J* dgd, CP_THREAD bool* pos) {
  typedef cp_ops<H> O;
  for (int i = 0; i < N; ++i) {
    H tp = O::c(0.0f), tn = O::c(0.0f);
    for (int k = 0; k < 6; ++k) {
      int a = cp_voigt_i(k), b = cp_voigt_j(k);
      H cp_ = k < 3 ? O::ld(p.P_nS_pos[9*i+CP_IDX(a,a)])
                    : O::ld(p.P_nS_pos[9*i+CP_IDX(a,b)]) + O::ld(p.P_nS_pos[9*i+CP_IDX(b,a)]);
      H cn_ = k < 3 ? O::ld(p.P_nS_neg[9*i+CP_IDX(a,a)])
                    : O::ld(p.P_nS_neg[9*i+CP_IDX(a,b)]) + O::ld(p.P_nS_neg[9*i+CP_IDX(b,a)]);
      tp = tp + S[k]*cp_;
      tn = tn + S[k]*cn_;
    }
    pos[i] = tp > tn;
    H tm = pos[i] ? tp : tn;
    H r  = tm / xi[i];
    bool active = O::f(r) > 0.0f;
    H mag = active ? O::ld(p.dot_gamma_0[i]) * O::ex(O::ld(p.n[i]) * O::lg(r)) : O::c(0.0f);
    gd[i]  = pos[i] ? mag : -mag;
    dgd[i] = active ? O::f(gd[i]) * O::f(O::ld(p.n[i])) / O::f(tm) : 0.0f;
  }
}

//> Green-Lagrange strain of Fe = A B in Voigt notation (engineering shear), cancellation free:
//> with D = B - I and Ea = (A^T A - I)/2:  E = Ea + sym(M D) + D^T M D / 2
template<typename H>
inline void cp_strain(CP_THREAD const H* M, CP_THREAD const H* Ea, CP_THREAD const H* D,
                      CP_THREAD H* MB, CP_THREAD H* e) {
  H MD[9], DtMD[9];
  cp_mul33(M, D, MD);
  cp_mulT33(D, MD, DtMD);
  for (int k = 0; k < 9; ++k) MB[k] = M[k] + MD[k];
  for (int k = 0; k < 6; ++k) {
    int a = cp_voigt_i(k), b = cp_voigt_j(k);
    e[k] = k < 3 ? Ea[CP_IDX(a,a)] + MD[CP_IDX(a,a)] + DtMD[CP_IDX(a,a)] * 0.5f
                 : Ea[CP_IDX(a,b)] * 2.0f + MD[CP_IDX(a,b)] + MD[CP_IDX(b,a)] + DtMD[CP_IDX(a,b)];
  }
}

//> elastic stress residual R(S) = S - C:E, E = E(A B), B = I - dt Lp(S)
template<typename H, int N>
inline void cp_residual(CP_CONSTANT cp_pheno_param& p, CP_THREAD const H* M, CP_THREAD const H* Ea, H dt,
                        CP_THREAD const H* S, CP_THREAD const H* xi,
                        CP_THREAD H* R, CP_THREAD H* Lp, CP_THREAD H* B, CP_THREAD H* MB,
                        CP_THREAD H* gd, CP_THREAD typename cp_ops<H>::J* dgd, CP_THREAD bool* pos) {
  typedef cp_ops<H> O;
  cp_kinetics<H,N>(p, S, xi, gd, dgd, pos);
  H D[9];
  for (int k = 0; k < 9; ++k) {
    H s = O::c(0.0f);
    for (int i = 0; i < N; ++i) s = s + gd[i] * O::ld(p.P_sl[9*i+k]);
    Lp[k] = s;
    D[k]  = -(dt * s);
    B[k]  = (k % 4 == 0 ? O::c(1.0f) : O::c(0.0f)) + D[k];
  }
  H e[6];
  cp_strain(M, Ea, D, MB, e);
  for (int r = 0; r < 6; ++r) {
    H s = O::c(0.0f);
    for (int c = 0; c < 6; ++c) s = s + O::ld(p.C66[r+6*c]) * e[c];
    R[r] = S[r] - s;
  }
}

//> dR/dS (6x6) in precision J
template<typename H, int N>
inline void cp_jacobian(CP_CONSTANT cp_pheno_param& p, CP_THREAD const H* MB, H dt,
                        CP_THREAD const typename cp_ops<H>::J* dgd, CP_THREAD const bool* pos,
                        CP_THREAD typename cp_ops<H>::J* Jac) {
  typedef cp_ops<H> O;
  typedef typename O::J J;
  J mb[9];
  for (int k = 0; k < 9; ++k) mb[k] = J(O::f(MB[k]));
  J dtj = J(O::f(dt));
  J C[36];
  for (int k = 0; k < 36; ++k) C[k] = cp_ldJ<J>(p.C66[k]);
  for (int k = 0; k < 6; ++k) {
    int a = cp_voigt_i(k), b = cp_voigt_j(k);
    J dB[9];
    for (int m = 0; m < 9; ++m) dB[m] = J(0);
    for (int i = 0; i < N; ++i) {
      CP_CONSTANT cp_pair* Pns = pos[i] ? p.P_nS_pos : p.P_nS_neg;
      J c = k < 3 ? cp_ldJ<J>(Pns[9*i+CP_IDX(a,a)])
                  : cp_ldJ<J>(Pns[9*i+CP_IDX(a,b)]) + cp_ldJ<J>(Pns[9*i+CP_IDX(b,a)]);
      J w = -dtj * dgd[i] * c;
      for (int m = 0; m < 9; ++m) dB[m] += w * cp_ldJ<J>(p.P_sl[9*i+m]);
    }
    J dE[9];                                                                                        // dB^T MB (symmetrised below)
    cp_mulT33(dB, mb, dE);
    J de[6];
    for (int q = 0; q < 6; ++q) {
      int u = cp_voigt_i(q), v = cp_voigt_j(q);
      de[q] = q < 3 ? dE[CP_IDX(u,u)] : dE[CP_IDX(u,v)] + dE[CP_IDX(v,u)];                          // d(2E) shear, d(E) normal
    }
    for (int r = 0; r < 6; ++r) {
      J s = J(0);
      for (int c = 0; c < 6; ++c) s += C[r+6*c] * de[c];
      Jac[r+6*k] = (r == k ? J(1) : J(0)) - s;
    }
  }
}

template<typename H>
inline float cp_norm6(CP_THREAD const H* v) {
  typedef cp_ops<H> O;
  float s = 0.0f;
  for (int k = 0; k < 6; ++k) { float x = O::f(v[k]); s += x*x; }
  return DF_SQRT(s);
}

//> scale S such that no system slips more than kappa within dt: dt gd0 (tau/xi)^n <= kappa
//> (plastic slip in one step cannot exceed the available elastic trial strain)
template<typename H, int N>
inline void cp_cap_stress(CP_CONSTANT cp_pheno_param& p, float dt, float kappa,
                          CP_THREAD const H* xi, CP_THREAD H* S) {
  typedef cp_ops<H> O;
  float q = 0.0f;                                                                                   // max over systems of (tau/xi) / (tau/xi)_cap
  for (int i = 0; i < N; ++i) {
    float t = 0.0f;
    for (int k = 0; k < 6; ++k) {
      int a = cp_voigt_i(k), b = cp_voigt_j(k);
      float c = k < 3 ? cp_ldJ<float>(p.P_sl[9*i+CP_IDX(a,a)])
                      : cp_ldJ<float>(p.P_sl[9*i+CP_IDX(a,b)]) + cp_ldJ<float>(p.P_sl[9*i+CP_IDX(b,a)]);
      t += O::f(S[k]) * c;
    }
    float r_cap = DF_EXP(DF_LOG(kappa / (dt * cp_ldJ<float>(p.dot_gamma_0[i]))) / cp_ldJ<float>(p.n[i]));
    q = CP_FMAX(q, (t < 0.0f ? -t : t) / O::f(xi[i]) / r_cap);
  }
  float s = q > 1.0f ? 1.0f / q : 1.0f;
  for (int k = 0; k < 6; ++k) S[k] = S[k] * s;
}

//> Newton iteration on S with backtracking (as DAMASK's Lp loop) and a trust region on |dS|;
//> a non-finite residual counts as "not improved" (backtrack); returns CP_OK on convergence
template<typename H, int N>
inline int cp_solve_stress(CP_CONSTANT cp_pheno_param& p, CP_CONSTANT cp_numerics& num,
                           CP_THREAD const H* M, CP_THREAD const H* Ea, H dt, CP_THREAD const H* xi, CP_THREAD H* S,
                           CP_THREAD H* Lp, CP_THREAD H* B, CP_THREAD H* MB,
                           CP_THREAD typename cp_ops<H>::J* Jac, CP_THREAD typename cp_ops<H>::J* dgd,
                           CP_THREAD bool* pos, CP_THREAD int& iters) {
  typedef cp_ops<H> O;
  typedef typename O::J J;
  H R[6], S_old[6], gd[N];
  J dS[6];
  float r_old = INFINITY, step = 1.0f;
  for (int k = 0; k < 6; ++k) { S_old[k] = S[k]; dS[k] = J(0); }

  for (int it = 0; it < num.n_stress; ++it) {
    ++iters;
    cp_residual<H,N>(p, M, Ea, dt, S, xi, R, Lp, B, MB, gd, dgd, pos);
    float rn = cp_norm6(R);
    if (!DF_ISFINITE(rn) && it == 0) return CP_FAIL_NAN;
    if (rn <= num.rtol_S * cp_norm6(S) + num.atol_S) {
      cp_jacobian<H,N>(p, MB, dt, dgd, pos, Jac);                                                   // keep consistent Jacobian for tangent
      return CP_OK;
    }
    if (rn < r_old) {                                                                               // improved: new Newton direction
      r_old = rn;
      step  = 1.0f;
      for (int k = 0; k < 6; ++k) S_old[k] = S[k];
      cp_jacobian<H,N>(p, MB, dt, dgd, pos, Jac);
      J b[6];
      for (int k = 0; k < 6; ++k) b[k] = -J(O::f(R[k]));
      J A[36];
      for (int k = 0; k < 36; ++k) A[k] = Jac[k];
      if (!cp_solve66(A, b)) return CP_FAIL_SINGULAR;
      float nd = 0.0f;
      for (int k = 0; k < 6; ++k) { float x = float(b[k]); nd += x*x; }
      nd = DF_SQRT(nd);
      float lim = num.trust_S * CP_FMAX(cp_norm6(S), O::f(xi[0]));                                  // xi: floor for S ~ 0
      J scale = J(nd > lim ? lim / nd : 1.0f);
      for (int k = 0; k < 6; ++k) dS[k] = b[k] * scale;
    } else {                                                                                        // not improved: shorten step
      step *= num.stepsize_ls;
    }
    for (int k = 0; k < 6; ++k) S[k] = S_old[k] + O::fj(dS[k] * J(step));
  }
  return CP_FAIL_STRESS;
}

//> phenopowerlaw rate of state (slip only)
template<typename H, int N>
inline void cp_dotState(CP_CONSTANT cp_pheno_param& p, CP_THREAD const H* S, CP_THREAD const H* xi,
                        CP_THREAD H* dxi, CP_THREAD H* dgamma) {
  typedef cp_ops<H> O;
  H gd[N];
  typename O::J dgd[N];
  bool pos[N];
  cp_kinetics<H,N>(p, S, xi, gd, dgd, pos);
  for (int i = 0; i < N; ++i) dgamma[i] = O::ab(gd[i]);
  for (int i = 0; i < N; ++i) {
    H x  = O::c(1.0f) - xi[i] / O::ld(p.xi_inf[i]);
    H ax = O::ab(x);
    H left = O::f(ax) > 0.0f ? O::ex(O::ld(p.a[i]) * O::lg(ax)) : O::c(0.0f);
    if (O::f(x) < 0.0f) left = -left;
    H h = O::c(0.0f);
    for (int j = 0; j < N; ++j) h = h + O::ld(p.h_sl_sl[i+N*j]) * dgamma[j];
    dxi[i] = O::ld(p.h_0[i]) * left * h;
  }
}

//--------------------------------------------------------------------------------------------------
//> integrate stress and state of one material point over dt (full step, no sub-stepping)
template<typename H, int N>
inline void cp_integrate(CP_CONSTANT cp_pheno_param& p, CP_CONSTANT cp_numerics& num, H dt,
                         CP_THREAD cp_point<H,N>& pt) {
  typedef cp_ops<H> O;
  typedef typename O::J J;
  pt.n_iter_stress = 0;
  pt.n_iter_state  = 0;

  H invFp0[9], A[9], M[9], Ea[9];
  H detFp0 = cp_inv33(pt.Fp0, invFp0);
  if (!(O::f(O::ab(detFp0)) > 0.0f)) { pt.status = CP_FAIL_SINGULAR; return; }
  cp_mul33(pt.F, invFp0, A);
  cp_mulT33(A, A, M);
  for (int k = 0; k < 9; ++k) Ea[k] = (M[k] - (k % 4 == 0 ? O::c(1.0f) : O::c(0.0f))) * 0.5f;     // only cancellation, once per point in H

  // initial stress: elastic response to the Lp guess (same starting point as DAMASK's Lp loop)
  {
    H D[9], MB[9], e[6];
    for (int k = 0; k < 9; ++k) D[k] = -(dt * pt.Lp_guess[k]);
    cp_strain(M, Ea, D, MB, e);
    for (int r = 0; r < 6; ++r) {
      H s = O::c(0.0f);
      for (int c = 0; c < 6; ++c) s = s + O::ld(p.C66[r+6*c]) * e[c];
      pt.S[r] = s;
    }
  }

  // fixed point iteration on state (integrateStateFPI)
  H dxi[N], dgam[N], dl1_xi[N], dl1_ga[N], dl2_xi[N], dl2_ga[N];
  cp_dotState<H,N>(p, pt.S0, pt.xi0, dxi, dgam);
  for (int i = 0; i < N; ++i) {
    if (!O::finite(dxi[i])) { pt.status = CP_FAIL_NAN; return; }
    pt.xi[i]    = pt.xi0[i]    + dxi[i]  * dt;
    pt.gamma[i] = pt.gamma0[i] + dgam[i] * dt;
    dl1_xi[i] = O::c(0.0f); dl1_ga[i] = O::c(0.0f);
  }

  H Lp[9], B[9], MB[9];
  J Jac[36], dgd[N];
  bool pos[N];

  // warm start: better of {elastic response to Lp guess, previous stress}, each capped
  {
    float kappa = 0.0f;
    for (int k = 0; k < 9; ++k) { float x = O::f(Ea[k]); kappa += x*x; }
    kappa = DF_SQRT(kappa);
    H S1[6], R0[6], R1[6], gd[N];
    for (int k = 0; k < 6; ++k) S1[k] = pt.S0[k];
    cp_cap_stress<H,N>(p, O::f(dt), kappa, pt.xi, pt.S);
    cp_cap_stress<H,N>(p, O::f(dt), kappa, pt.xi, S1);
    cp_residual<H,N>(p, M, Ea, dt, pt.S, pt.xi, R0, Lp, B, MB, gd, dgd, pos);
    cp_residual<H,N>(p, M, Ea, dt, S1,   pt.xi, R1, Lp, B, MB, gd, dgd, pos);
    float r0 = cp_norm6(R0), r1 = cp_norm6(R1);
    if (r1 < r0 || !DF_ISFINITE(r0)) for (int k = 0; k < 6; ++k) pt.S[k] = S1[k];
  }

  pt.status = CP_FAIL_STATE;
  for (int it = 0; it < num.n_state; ++it) {
    ++pt.n_iter_state;
    for (int i = 0; i < N; ++i) {
      dl2_xi[i] = it > 0 ? dl1_xi[i] : O::c(0.0f);
      dl2_ga[i] = it > 0 ? dl1_ga[i] : O::c(0.0f);
      dl1_xi[i] = dxi[i];
      dl1_ga[i] = dgam[i];
    }

    int st = cp_solve_stress<H,N>(p, num, M, Ea, dt, pt.xi, pt.S, Lp, B, MB, Jac, dgd, pos, pt.n_iter_stress);
    if (st != CP_OK) { pt.status = st; return; }

    cp_dotState<H,N>(p, pt.S, pt.xi, dxi, dgam);

    // damper (float precision is sufficient: affects the iterates, not the fixed point)
    float d01 = 0.0f, d12 = 0.0f, d22 = 0.0f;
    for (int i = 0; i < N; ++i) {
      float o0x = O::f(dxi[i]),    o1x = O::f(dl1_xi[i]), o2x = O::f(dl2_xi[i]);
      float o0g = O::f(dgam[i]),   o1g = O::f(dl1_ga[i]), o2g = O::f(dl2_ga[i]);
      d01 += o0x*o1x + o0g*o1g;
      d12 += (o0x-o1x)*(o1x-o2x) + (o0g-o1g)*(o1g-o2g);
      d22 += (o1x-o2x)*(o1x-o2x) + (o1g-o2g)*(o1g-o2g);
    }
    float zeta = (CP_FMIN(d01, d12) < 0.0f && d22 > 0.0f) ? 0.75f + 0.25f * CP_TANH(2.0f + 4.0f*d12/d22) : 1.0f;

    bool conv = true;
    for (int i = 0; i < N; ++i) {
      dxi[i]  = dxi[i]  * zeta + dl1_xi[i] * (1.0f - zeta);
      dgam[i] = dgam[i] * zeta + dl1_ga[i] * (1.0f - zeta);
      H rx = pt.xi[i]    - pt.xi0[i]    - dxi[i]  * dt;
      H rg = pt.gamma[i] - pt.gamma0[i] - dgam[i] * dt;
      if (!O::finite(rx) || !O::finite(rg)) { pt.status = CP_FAIL_NAN; return; }
      pt.xi[i]    = pt.xi[i]    - rx;
      pt.gamma[i] = pt.gamma[i] - rg;
      float ax = O::f(O::ab(rx)), ag = O::f(O::ab(rg));
      conv = conv && ax <= CP_FMAX(p.atol_xi,    num.rtol_state * O::f(O::ab(pt.xi[i])))
                  && ag <= CP_FMAX(p.atol_gamma, num.rtol_state * O::f(O::ab(pt.gamma[i])));
    }
    if (conv) { pt.status = CP_OK; break; }
  }
  if (pt.status != CP_OK) return;

  // outputs (last stress solve; as in DAMASK, P etc. belong to the state before the final update)
  H invFp_new[9], Fp_new[9], T1[9], T2[9];
  cp_mul33(invFp0, B, invFp_new);
  H det_inv = cp_inv33(invFp_new, Fp_new);
  if (!(O::f(O::ab(det_inv)) > 0.0f)) { pt.status = CP_FAIL_SINGULAR; return; }
  H detFp = O::c(1.0f) / det_inv;
  H scale = O::ex(O::lg(O::ab(detFp)) / O::c(3.0f));                                                // det(Fp)^(1/3), det > 0
  H Sm[9];
  for (int k = 0; k < 9; ++k) Sm[k] = O::c(0.0f);
  Sm[CP_IDX(0,0)] = pt.S[0]; Sm[CP_IDX(1,1)] = pt.S[1]; Sm[CP_IDX(2,2)] = pt.S[2];
  Sm[CP_IDX(1,2)] = Sm[CP_IDX(2,1)] = pt.S[3];
  Sm[CP_IDX(0,2)] = Sm[CP_IDX(2,0)] = pt.S[4];
  Sm[CP_IDX(0,1)] = Sm[CP_IDX(1,0)] = pt.S[5];
  H FiFp[9];
  cp_mul33(pt.F, invFp_new, FiFp);                                                                  // F Fp^-1 (= Fe for Fi = I)
  cp_mul33(FiFp, Sm, T1);
  cp_mul3T3(T1, invFp_new, T2);
  for (int k = 0; k < 9; ++k) {
    pt.P[k]  = T2[k];
    pt.Fe[k] = FiFp[k];
    pt.Lp[k] = Lp[k];
    pt.Fp[k] = Fp_new[k] / scale;
  }

  // consistent tangent dP/dF (state held fixed, as in DAMASK), precision J
#ifndef CP_NO_TANGENT
  {
    J LU[36];
    for (int k = 0; k < 36; ++k) LU[k] = Jac[k];
    J C[36];
    for (int k = 0; k < 36; ++k) C[k] = cp_ldJ<J>(p.C66[k]);
    J iFp0[9], Bj[9], Aj[9], Fj[9], iFpn[9], Sj[9];
    for (int k = 0; k < 9; ++k) {
      iFp0[k] = J(O::f(invFp0[k])); Bj[k] = J(O::f(B[k])); Aj[k] = J(O::f(A[k]));
      Fj[k] = J(O::f(pt.F[k])); iFpn[k] = J(O::f(invFp_new[k])); Sj[k] = J(O::f(Sm[k]));
    }
    J AB[9];
    cp_mul33(Aj, Bj, AB);
    for (int l = 0; l < 3; ++l) for (int kk = 0; kk < 3; ++kk) {                                   // dF = e_kk (x) e_l
      J dF[9];
      for (int m = 0; m < 9; ++m) dF[m] = J(0);
      dF[CP_IDX(kk,l)] = J(1);
      J dA[9], T[9], dEf[9];
      cp_mul33(dF, iFp0, dA);
      cp_mulT33(AB, dA, T);                                                                         // (AB)^T dA
      cp_mul33(T, Bj, dEf);                                                                         // (AB)^T dA B, dE = sym(.)
      J de[6];
      for (int q = 0; q < 6; ++q) {
        int u = cp_voigt_i(q), v = cp_voigt_j(q);
        de[q] = q < 3 ? dEf[CP_IDX(u,u)] : dEf[CP_IDX(u,v)] + dEf[CP_IDX(v,u)];
      }
      J dS6[6];
      for (int r = 0; r < 6; ++r) {                                                                 // -dR/dF = C:dE  =>  J dS = C:dE
        J s = J(0);
        for (int c = 0; c < 6; ++c) s += C[r+6*c] * de[c];
        dS6[r] = s;
      }
      J A6[36];
      for (int m = 0; m < 36; ++m) A6[m] = LU[m];
      cp_solve66(A6, dS6);
      J dSm[9];
      dSm[CP_IDX(0,0)] = dS6[0]; dSm[CP_IDX(1,1)] = dS6[1]; dSm[CP_IDX(2,2)] = dS6[2];
      dSm[CP_IDX(1,2)] = dSm[CP_IDX(2,1)] = dS6[3];
      dSm[CP_IDX(0,2)] = dSm[CP_IDX(2,0)] = dS6[4];
      dSm[CP_IDX(0,1)] = dSm[CP_IDX(1,0)] = dS6[5];
      J dB[9];                                                                                      // dB = -dt dLp, dLp = sum dgd (P_nS:dS) P_sl
      for (int m = 0; m < 9; ++m) dB[m] = J(0);
      for (int i = 0; i < N; ++i) {
        CP_CONSTANT cp_pair* Pns = pos[i] ? p.P_nS_pos : p.P_nS_neg;
        J t = J(0);
        for (int m = 0; m < 9; ++m) t += cp_ldJ<J>(Pns[9*i+m]) * dSm[m];
        J w = -J(O::f(dt)) * dgd[i] * t;
        for (int m = 0; m < 9; ++m) dB[m] += w * cp_ldJ<J>(p.P_sl[9*i+m]);
      }
      J diFp[9];
      cp_mul33(iFp0, dB, diFp);
      // dP = dF iFpn S iFpn^T + F diFp S iFpn^T + F iFpn dS iFpn^T + F iFpn S diFp^T
      J X1[9], X2[9], Y[9], dP[9], FiF[9];
      for (int m = 0; m < 9; ++m) dP[m] = J(0);
      cp_mul33(dF, iFpn, X1); cp_mul33(X1, Sj, X2); cp_mul3T3(X2, iFpn, Y);  for (int m = 0; m < 9; ++m) dP[m] += Y[m];
      cp_mul33(Fj, diFp, X1); cp_mul33(X1, Sj, X2); cp_mul3T3(X2, iFpn, Y);  for (int m = 0; m < 9; ++m) dP[m] += Y[m];
      cp_mul33(Fj, iFpn, FiF);
      cp_mul33(FiF, dSm, X2); cp_mul3T3(X2, iFpn, Y);                        for (int m = 0; m < 9; ++m) dP[m] += Y[m];
      cp_mul33(FiF, Sj, X2);  cp_mul3T3(X2, diFp, Y);                        for (int m = 0; m < 9; ++m) dP[m] += Y[m];
      for (int m = 0; m < 9; ++m) pt.dPdF[m + 9*CP_IDX(kk,l)] = dP[m];
    }
  }
#endif
}

#endif
