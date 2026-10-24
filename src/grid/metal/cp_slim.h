// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Slim GPU formulation of the constitutive update (same discrete equations as cp_integrate in
//> cp_kernel.h) plus DAMASK's analytic tangent (phase_mechanical_dPdF for Fi = I).
//> @details Designed for a small per-thread footprint (registers instead of spilled stack):
//>  - no per-point struct in private memory: inputs/outputs are component-major ("SoA") device
//>    arrays, element (c,i) at c*n + i (coalesced across a SIMD group)
//>  - inputs are re-read from device memory when needed instead of being kept alive
//>  - outputs are written directly when they become available
//>  - the tangent is a separate kernel (float), evaluated at the converged state
//> Component layout of the input/output arrays: see cp_slim_layout.
//--------------------------------------------------------------------------------------------------
#ifndef DAMASK_CP_SLIM_H
#define DAMASK_CP_SLIM_H

#include "cp_kernel.h"

#ifdef __METAL_VERSION__
  #define CP_DEVICE device
  #define CP_UNROLL _Pragma("unroll")
  #define CP_POW(a,b) metal::precise::pow(a,b)
#else
  #include <cmath>
  #define CP_DEVICE
  #define CP_UNROLL
  #define CP_POW(a,b) std::pow(a,b)
#endif

template<int N>
struct cp_slim_layout {                                                                             // component offsets
  enum : int {
    // inputs
    F      = 0,                                                                                     // 9
    Fp0    = 9,                                                                                     // 9
    S0     = 18,                                                                                    // 6 (Voigt)
    Lp0    = 24,                                                                                    // 9 (initial guess)
    xi0    = 33,                                                                                    // N
    gamma0 = 33 + N,                                                                                // N
    n_in   = 33 + 2*N,
    // outputs
    S      = 0,                                                                                     // 6 (Voigt)
    xi     = 6,                                                                                     // N
    gamma  = 6 + N,                                                                                 // N
    P      = 6 + 2*N,                                                                               // 9
    iFp    = 15 + 2*N,                                                                              // 9: invFp_new (unnormalized); Fp is formed on the host in FP64
    Fe     = 24 + 2*N,                                                                              // 9
    Lp     = 33 + 2*N,                                                                              // 9
    n_out  = 42 + 2*N
  };
};

//> view of point i in component-major arrays of length n
template<typename H>
struct cp_soa {
  CP_DEVICE H* d;
  uint32_t n, i;
  H    get(int c) const       { return d[uint32_t(c)*n + i]; }
  void set(int c, H v) const  { d[uint32_t(c)*n + i] = v; }
};

template<typename H>
struct cp_soa_const {
  CP_DEVICE const H* d;
  uint32_t n, i;
  H get(int c) const { return d[uint32_t(c)*n + i]; }
};


//--------------------------------------------------------------------------------------------------
//> integrate stress and state of one material point over dt; identical algorithm to cp_integrate
//> (warm start, trust region, FPI with damper), outputs written to out on success only.
//> Returns the status (CP_OK, CP_FAIL_*); iters: total stress Newton iterations.
template<typename H, int N>
inline int cp_integrate_slim(CP_CONSTANT cp_pheno_param& p, CP_CONSTANT cp_numerics& num, H dt,
                             cp_soa_const<H> in, cp_soa<H> out, CP_THREAD int& iters) {
  typedef cp_ops<H> O;
  typedef typename O::J J;
  typedef cp_slim_layout<N> L;
  iters = 0;

  H M[9], Ea[9];
  {
    H Fp0[9], invFp0[9], F[9], A[9];
    CP_UNROLL for (int k = 0; k < 9; ++k) { Fp0[k] = in.get(L::Fp0+k); F[k] = in.get(L::F+k); }
    H detFp0 = cp_inv33(Fp0, invFp0);
    if (!(O::f(O::ab(detFp0)) > 0.0f)) return CP_FAIL_SINGULAR;
    cp_mul33(F, invFp0, A);
    cp_mulT33(A, A, M);
    CP_UNROLL for (int k = 0; k < 9; ++k) Ea[k] = (M[k] - (k % 4 == 0 ? O::c(1.0f) : O::c(0.0f))) * 0.5f;
  }

  H S[6], xi[N], gamma[N];
  {                                                                                                 // initial stress: elastic response to the Lp guess
    H D[9], MB[9], e[6];
    CP_UNROLL for (int k = 0; k < 9; ++k) D[k] = -(dt * in.get(L::Lp0+k));
    cp_strain(M, Ea, D, MB, e);
    CP_UNROLL for (int r = 0; r < 6; ++r) {
      H s = O::c(0.0f);
      CP_UNROLL for (int c = 0; c < 6; ++c) s = s + O::ld(p.C66[r+6*c]) * e[c];
      S[r] = s;
    }
  }

  H dxi[N], dgam[N], dl1_xi[N], dl1_ga[N];
  float dl2_xi[N], dl2_ga[N];                                                                       // enter float arithmetic (damper) only
  {
    H S0[6], xi0[N];
    CP_UNROLL for (int k = 0; k < 6; ++k) S0[k] = in.get(L::S0+k);
    CP_UNROLL for (int k = 0; k < N; ++k) xi0[k] = in.get(L::xi0+k);
    cp_dotState<H,N>(p, S0, xi0, dxi, dgam);
    CP_UNROLL for (int i = 0; i < N; ++i) {
      if (!O::finite(dxi[i])) return CP_FAIL_NAN;
      xi[i]    = xi0[i] + dxi[i] * dt;
      gamma[i] = in.get(L::gamma0+i) + dgam[i] * dt;
      dl1_xi[i] = O::c(0.0f); dl1_ga[i] = O::c(0.0f);
    }

    // warm start: better of {elastic response to Lp guess, previous stress}, each capped
    float kappa = 0.0f;
    CP_UNROLL for (int k = 0; k < 9; ++k) { float x = O::f(Ea[k]); kappa += x*x; }
    kappa = DF_SQRT(kappa);
    H R0[6], R1[6], gd[N], Lp_[9], B_[9], MB_[9];
    J dgd_[N];
    bool pos_[N];
    cp_cap_stress<H,N>(p, O::f(dt), kappa, xi, S);
    cp_cap_stress<H,N>(p, O::f(dt), kappa, xi, S0);
    cp_residual<H,N>(p, M, Ea, dt, S,  xi, R0, Lp_, B_, MB_, gd, dgd_, pos_);
    cp_residual<H,N>(p, M, Ea, dt, S0, xi, R1, Lp_, B_, MB_, gd, dgd_, pos_);
    float r0 = cp_norm6(R0), r1 = cp_norm6(R1);
    if (r1 < r0 || !DF_ISFINITE(r0)) CP_UNROLL for (int k = 0; k < 6; ++k) S[k] = S0[k];
  }

  H Lp[9], B[9], MB[9];
  J Jac[36], dgd[N];
  bool pos[N];
  int status = CP_FAIL_STATE;
  for (int it = 0; it < num.n_state; ++it) {
    CP_UNROLL for (int i = 0; i < N; ++i) {
      dl2_xi[i] = it > 0 ? O::f(dl1_xi[i]) : 0.0f;
      dl2_ga[i] = it > 0 ? O::f(dl1_ga[i]) : 0.0f;
      dl1_xi[i] = dxi[i];
      dl1_ga[i] = dgam[i];
    }

    int st = cp_solve_stress<H,N>(p, num, M, Ea, dt, xi, S, Lp, B, MB, Jac, dgd, pos, iters);
    if (st != CP_OK) return st;

    cp_dotState<H,N>(p, S, xi, dxi, dgam);

    float d01 = 0.0f, d12 = 0.0f, d22 = 0.0f;
    CP_UNROLL for (int i = 0; i < N; ++i) {
      float o0x = O::f(dxi[i]),  o1x = O::f(dl1_xi[i]), o2x = dl2_xi[i];
      float o0g = O::f(dgam[i]), o1g = O::f(dl1_ga[i]), o2g = dl2_ga[i];
      d01 += o0x*o1x + o0g*o1g;
      d12 += (o0x-o1x)*(o1x-o2x) + (o0g-o1g)*(o1g-o2g);
      d22 += (o1x-o2x)*(o1x-o2x) + (o1g-o2g)*(o1g-o2g);
    }
    float zeta = (CP_FMIN(d01, d12) < 0.0f && d22 > 0.0f) ? 0.75f + 0.25f * CP_TANH(2.0f + 4.0f*d12/d22) : 1.0f;

    bool conv = true;
    CP_UNROLL for (int i = 0; i < N; ++i) {
      dxi[i]  = dxi[i]  * zeta + dl1_xi[i] * (1.0f - zeta);
      dgam[i] = dgam[i] * zeta + dl1_ga[i] * (1.0f - zeta);
      H rx = xi[i]    - in.get(L::xi0+i)    - dxi[i]  * dt;
      H rg = gamma[i] - in.get(L::gamma0+i) - dgam[i] * dt;
      if (!O::finite(rx) || !O::finite(rg)) return CP_FAIL_NAN;
      xi[i]    = xi[i]    - rx;
      gamma[i] = gamma[i] - rg;
      float ax = O::f(O::ab(rx)), ag = O::f(O::ab(rg));
      conv = conv && ax <= CP_FMAX(p.atol_xi,    num.rtol_state * O::f(O::ab(xi[i])))
                  && ag <= CP_FMAX(p.atol_gamma, num.rtol_state * O::f(O::ab(gamma[i])));
    }
    if (conv) { status = CP_OK; break; }
  }
  if (status != CP_OK) return status;

  // outputs (last stress solve; as in DAMASK, P etc. belong to the state before the final update)
  H invFp_new[9];
  {
    H Fp0[9], invFp0[9];
    CP_UNROLL for (int k = 0; k < 9; ++k) Fp0[k] = in.get(L::Fp0+k);
    cp_inv33(Fp0, invFp0);
    cp_mul33(invFp0, B, invFp_new);
  }
  // Fp = inv(invFp_new)/det^(1/3) is formed on the host in FP64: on the GPU this step intermittently
  // returned garbage for whole SIMD groups (all other outputs correct), see DAMASK_Metal.mm
  if (!(O::f(cp_det33(invFp_new)) > 0.0f)) return CP_FAIL_SINGULAR;
  {
    H F[9], Fe[9], Sm[9], T1[9], P[9];
    CP_UNROLL for (int k = 0; k < 9; ++k) F[k] = in.get(L::F+k);
    cp_mul33(F, invFp_new, Fe);
    CP_UNROLL for (int k = 0; k < 9; ++k) Sm[k] = O::c(0.0f);
    Sm[CP_IDX(0,0)] = S[0]; Sm[CP_IDX(1,1)] = S[1]; Sm[CP_IDX(2,2)] = S[2];
    Sm[CP_IDX(1,2)] = Sm[CP_IDX(2,1)] = S[3];
    Sm[CP_IDX(0,2)] = Sm[CP_IDX(2,0)] = S[4];
    Sm[CP_IDX(0,1)] = Sm[CP_IDX(1,0)] = S[5];
    cp_mul33(Fe, Sm, T1);
    cp_mul3T3(T1, invFp_new, P);
    CP_UNROLL for (int k = 0; k < 9; ++k) {
      out.set(L::P+k,  P[k]);
      out.set(L::Fe+k, Fe[k]);
      out.set(L::iFp+k, invFp_new[k]);
      out.set(L::Lp+k, Lp[k]);
    }
  }
  CP_UNROLL for (int k = 0; k < 6; ++k) out.set(L::S+k, S[k]);
  CP_UNROLL for (int i = 0; i < N; ++i) { out.set(L::xi+i, xi[i]); out.set(L::gamma+i, gamma[i]); }
  return CP_OK;
}


//--------------------------------------------------------------------------------------------------
//> 9x9 LU with partial pivoting, solves A X = B for 9 right hand sides in place (X in B)
template<typename T>
inline bool cp_solve99(CP_THREAD T* A, CP_THREAD T* B) {                                            // (r,c) at r+9c
  for (int c = 0; c < 9; ++c) {
    int pv = c;
    T amax = A[c+9*c] < T(0) ? -A[c+9*c] : A[c+9*c];
    for (int r = c+1; r < 9; ++r) {
      T v = A[r+9*c] < T(0) ? -A[r+9*c] : A[r+9*c];
      if (v > amax) { amax = v; pv = r; }
    }
    if (!(amax > T(0))) return false;
    if (pv != c) for (int k = 0; k < 9; ++k) {
      T t = A[c+9*k]; A[c+9*k] = A[pv+9*k]; A[pv+9*k] = t;
      t = B[c+9*k]; B[c+9*k] = B[pv+9*k]; B[pv+9*k] = t;
    }
    T inv = T(1) / A[c+9*c];
    for (int r = c+1; r < 9; ++r) {
      T l = A[r+9*c] * inv;
      for (int k = c+1; k < 9; ++k) A[r+9*k] -= l * A[c+9*k];
      for (int k = 0; k < 9; ++k)   B[r+9*k] -= l * B[c+9*k];
    }
  }
  for (int k = 0; k < 9; ++k)
    for (int r = 8; r >= 0; --r) {
      T s = B[r+9*k];
      for (int c = r+1; c < 9; ++c) s -= A[r+9*c] * B[c+9*k];
      B[r+9*k] = s / A[r+9*r];
    }
  return true;
}

inline int cp_voigt(int i, int j) { return i == j ? i : 6 - i - j; }                                // 11,22,33,23,13,12 -> 0..5

//--------------------------------------------------------------------------------------------------
//> DAMASK's analytic tangent (phase_mechanical_dPdF) for Fi = I, Li = 0, in precision T.
//> Inputs as stored by DAMASK after integrateStress: F, Fp0, invFp_new (unnormalized; DAMASK's Fp is its
//> normalized inverse), Fe (= F invFp_new), S (Voigt), xi (final state); kinetics re-evaluated at (S, xi).
//> dPdF(i,j,k,l) at i+3j+9k+27l. Returns false if singular.
template<typename T, int N>
inline bool cp_tangent(CP_CONSTANT cp_pheno_param& p, T dt,
                       CP_THREAD const T* F, CP_THREAD const T* Fp0, CP_THREAD const T* invFp_new,
                       CP_THREAD const T* Fe, CP_THREAD const T* Sv, CP_THREAD const T* xi,
                       CP_THREAD T* dPdF) {
  T invFp[9], invFp0[9], Sm[9];
  {                                                                                                 // inverse of the normalized Fp: invFp_new / det(invFp_new)^(1/3)
    T d = cp_det33(invFp_new);
    if (!(d > T(0))) return false;
    T s = CP_POW(d, T(1)/T(3));
    for (int k = 0; k < 9; ++k) invFp[k] = invFp_new[k] / s;
  }
  cp_inv33(Fp0, invFp0);
  for (int k = 0; k < 9; ++k) Sm[k] = T(0);
  Sm[0] = Sv[0]; Sm[4] = Sv[1]; Sm[8] = Sv[2];
  Sm[CP_IDX(1,2)] = Sm[CP_IDX(2,1)] = Sv[3];
  Sm[CP_IDX(0,2)] = Sm[CP_IDX(2,0)] = Sv[4];
  Sm[CP_IDX(0,1)] = Sm[CP_IDX(1,0)] = Sv[5];

  T FiFp[9], T2[9];
  cp_mul33(F, invFp, FiFp);                                                                         // F invFp (normalized Fp), assembly
  cp_mul33(F, invFp0, T2);                                                                          // temp_33_2 = F invSubFp0

  // shear rate derivatives at (S, xi) as plastic_LpAndItsTangents (Mp = S for Fi = I)
  T dgd[N];
  bool pos[N];
  for (int i = 0; i < N; ++i) {
    T tp = T(0), tn = T(0);
    for (int k = 0; k < 9; ++k) { tp += Sm[k]*cp_ldJ<T>(p.P_nS_pos[9*i+k]); tn += Sm[k]*cp_ldJ<T>(p.P_nS_neg[9*i+k]); }
    T tm = tp > tn ? tp : tn;
    T r  = tm / xi[i];
    T gd = r > T(0) ? cp_ldJ<T>(p.dot_gamma_0[i]) * CP_POW(r, cp_ldJ<T>(p.n[i])) : T(0);          // |dot_gamma|
    pos[i] = tp > tn;                                                                               // sign of dot_gamma, selects P_nS (merge(...,dot_gamma>0))
    dgd[i] = gd != T(0) ? (pos[i] ? gd : -gd) * cp_ldJ<T>(p.n[i]) / tm : T(0);                      // ddot_gamma_dtau = dot_gamma n / tau
  }

  // L = dSdFe : (T2 . dLpdS) * dt  (9x9), rhs = dSdFe . invFp^T  (9x9)
  T A99[81], R99[81];
  for (int q = 0; q < 81; ++q) { A99[q] = T(0); R99[q] = T(0); }
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    int a = i + 3*j;
    // dSdFe(i,j,k,l) = sum_m Fe(k,m) C(i,j,m,l)
    T dS[9];
    for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
      T s = T(0);
      for (int m = 0; m < 3; ++m) s += Fe[CP_IDX(k,m)] * cp_ldJ<T>(p.C66[cp_voigt(i,j) + 6*cp_voigt(m,l)]);
      dS[CP_IDX(k,l)] = s;
    }
    for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {                                      // rhs(i,j,:,:) = dS . invFp^T
      T s = T(0);
      for (int m = 0; m < 3; ++m) s += dS[CP_IDX(k,m)] * invFp[CP_IDX(l,m)];
      R99[a + 9*(k+3*l)] = s;
    }
    // lhs(i,j,k,l) = dt sum_mn dS(m,n) temp(m,n,k,l), temp(:,:,k,l) = T2 dLpdS(:,:,k,l),
    // dLpdS(a,b,k,l) = sum_s dgd_s P_sl(a,b,s) P_nS(k,l,s)  =>  lhs = dt sum_s dgd_s (dS:T2.P_sl_s) P_nS_s(k,l)
    for (int s = 0; s < N; ++s) {
      if (dgd[s] == T(0)) continue;
      T c = T(0);
      for (int m = 0; m < 3; ++m) for (int n = 0; n < 3; ++n) {
        T tp = T(0);
        for (int o = 0; o < 3; ++o) tp += T2[CP_IDX(m,o)] * cp_ldJ<T>(p.P_sl[9*s+CP_IDX(o,n)]);
        c += dS[CP_IDX(m,n)] * tp;
      }
      CP_CONSTANT cp_pair* Pns = pos[s] ? p.P_nS_pos : p.P_nS_neg;
      T w = dt * dgd[s] * c;
      for (int k = 0; k < 9; ++k) A99[a + 9*k] += w * cp_ldJ<T>(Pns[9*s+k]);
    }
  }
  for (int q = 0; q < 9; ++q) A99[q + 9*q] += T(1);
  if (!cp_solve99(A99, R99)) return false;                                                          // R99 <- dSdF

  // dFpinvdF(:,:,k,l) = -dt invFp0 . (dLpdS : dSdF)(:,:,k,l)
  // dPdF = delta_ik (invFp S invFp^T)_lj^T ... (see phase_mechanical_dPdF)
  T X[9], Y[9], T1[9], T3[9], W[9];
  cp_mul3T3(Sm, invFp, T1);                                                                         // temp_33_1 = S invFp^T
  cp_mul33(FiFp, Sm, T3);                                                                           // temp_33_3 = F invFp S
  cp_mul33(invFp, T1, W);                                                                           // invFp S invFp^T
  for (int q = 0; q < 81; ++q) dPdF[q] = T(0);
  for (int pp = 0; pp < 3; ++pp)                                                                    // dPdF(p,:,p,:) = transpose(invFp temp_33_1)
    for (int j = 0; j < 3; ++j) for (int l = 0; l < 3; ++l)
      dPdF[pp + 3*j + 9*pp + 27*l] = W[CP_IDX(l,j)];
  for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
    int b = k + 3*l;
    T dSdF[9], tLS[9], dFpi[9];
    for (int q = 0; q < 9; ++q) dSdF[q] = R99[q + 9*b];
    for (int q = 0; q < 9; ++q) tLS[q] = T(0);                                                      // (dLpdS : dSdF)(:,:,k,l)
    for (int s = 0; s < N; ++s) {
      if (dgd[s] == T(0)) continue;
      CP_CONSTANT cp_pair* Pns = pos[s] ? p.P_nS_pos : p.P_nS_neg;
      T c = T(0);
      for (int q = 0; q < 9; ++q) c += cp_ldJ<T>(Pns[9*s+q]) * dSdF[q];
      for (int q = 0; q < 9; ++q) tLS[q] += dgd[s] * c * cp_ldJ<T>(p.P_sl[9*s+q]);
    }
    cp_mul33(invFp0, tLS, dFpi);
    for (int q = 0; q < 9; ++q) dFpi[q] = -dt * dFpi[q];
    cp_mul33(F, dFpi, X);  cp_mul33(X, T1, Y);                                                       // F dFpinvdF temp_33_1
    for (int q = 0; q < 9; ++q) dPdF[q + 9*b] += Y[q];
    cp_mul33(FiFp, dSdF, X); cp_mul3T3(X, invFp, Y);                                                // temp_33_2 dSdF invFp^T
    for (int q = 0; q < 9; ++q) dPdF[q + 9*b] += Y[q];
    cp_mul3T3(T3, dFpi, Y);                                                                         // temp_33_3 dFpinvdF^T
    for (int q = 0; q < 9; ++q) dPdF[q + 9*b] += Y[q];
  }
  return true;
}

#endif
