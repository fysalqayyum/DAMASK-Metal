// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Single-source (Metal Shading Language / C++) building blocks of the constitutive update:
//> constants (cp_pair), status codes, numerics, phenopowerlaw parameters and kinetics, precision
//> traits (cp_ops), small dense linear algebra. Used by cp_core.h; the complete reference pipeline
//> built from them is in tests/metal/ref_pheno_kernel.h.
//> @details Formulation (as in cp_core.h), the same discrete equations as DAMASK's
//> integrateStress/integrateStateFPI (phase_mechanical.f90), restructured for SIMD hardware:
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

template<typename H>
inline float cp_norm6(CP_THREAD const H* v) {
  typedef cp_ops<H> O;
  float s = 0.0f;
  for (int k = 0; k < 6; ++k) { float x = O::f(v[k]); s += x*x; }
  return DF_SQRT(s);
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


#endif
