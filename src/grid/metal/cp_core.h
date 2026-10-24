// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Model-independent core of the GPU constitutive update for slip-based plasticity models
//> (Hooke elasticity; Fi = I, or Fi = j(X) I with FI_STATE), and the model policies.
//> @details The core does: elastic stress residual (cancellation free), Jacobian, Newton solve on S
//> with trust region and backtracking, warm start, FPI state integration with DAMASK's damper,
//> delta state after convergence, outputs, and DAMASK's analytic tangent (phase_mechanical_dPdF).
//> A model policy M provides only the physics:
//>   enum { N, ND, NY, NDEP }  number of (padded) slip systems, dot/delta/dependent-state sizes
//>   dependent<H>(p,X,Y,T, D)            dependent state from the state (plastic_dependentState); evaluated
//>                                       at the start of every stress solve, D0 (input) is DAMASK's stored
//>                                       value used by the first predictor, D (output) that of the last solve
//>   typedef ... param         device parameter struct with members C66, P_sl, P_nS_pos, P_nS_neg
//>   kinetics<H>(p,S,X,Y,D,T, gd,dgd,pos)  signed shear rates, d(gd)/d(tau), branch (tau_pos > tau_neg)
//>   dot_state<H>(p,S,X,Y,D,T, dX)         rate of the dot state
//>   delta_state<H>(p,S,X,Y,D,T)         instantaneous update of Y after convergence (in place)
//>   atol(p,j)                           absolute tolerance of dot-state component j
//>   cap_ratio(p,S,X,Y,D,T,dt,kappa)       max over systems of the factor by which S must shrink so that
//>                                       no system slips more than kappa within dt (warm start)
//>   stress_scale(p,X)                   stress magnitude used as trust-region floor for S ~ 0
//>   enum { NBX, NBY, NBD }              X, Y, D consist of NBX/NBY/NBD blocks of N per-system values
//>   pad_value(group,block)              value of padded systems (group 0: X, 1: Y, 2: D) for host packing
//> State layout as DAMASK's plasticState%state: X = dot state, followed by Y = delta state.
//> Padded systems (i >= actual count) must be inert: P = 0, gd = 0, unit resistances.
//--------------------------------------------------------------------------------------------------
#ifndef DAMASK_CP_CORE_H
#define DAMASK_CP_CORE_H

#include "cp_slim.h"

//--------------------------------------------------------------------------------------------------
//> component offsets of the component-major input/output arrays of model M
template<class M>
struct cp_layout {
  enum : int {
    // inputs
    F   = 0,                                                                                        // 9
    Fp0 = 9,                                                                                        // 9
    S0  = 18,                                                                                       // 6 (Voigt)
    Lp0 = 24,                                                                                       // 9 (initial guess)
    X0  = 33,                                                                                       // ND
    Y0  = 33 + M::ND,                                                                               // NY
    D0  = 33 + M::ND + M::NY,                                                                       // NDEP
    T   = 33 + M::ND + M::NY + M::NDEP,                                                             // 1 (temperature)
    n_in = 34 + M::ND + M::NY + M::NDEP,
    // outputs
    S   = 0,                                                                                        // 6 (Voigt)
    X   = 6,                                                                                        // ND
    Y   = 6 + M::ND,                                                                                // NY
    D   = 6 + M::ND + M::NY,                                                                        // NDEP
    P   = 6 + M::ND + M::NY + M::NDEP,                                                              // 9
    iFp = 15 + M::ND + M::NY + M::NDEP,                                                             // 9: invFp_new (unnormalized)
    Fe  = 24 + M::ND + M::NY + M::NDEP,                                                             // 9
    Lp  = 33 + M::ND + M::NY + M::NDEP,                                                             // 9
    n_out = 42 + M::ND + M::NY + M::NDEP
  };
};


//--------------------------------------------------------------------------------------------------
//> Policy traits (enum members of every policy):
//>   FLOW    = 0: rank-1 kinetics (per-system shear rates, Lp = sum gd P_sl), see above
//>           = 1: the policy returns Lp and its derivative itself (no per-system arrays in the core):
//>                flow<H>(p,S,X,Y,D,T, Lp, K, mode)  mode 0: Lp only; 1: also G = dLp/dS_Voigt (9x6,
//>                K[m+9*k], shear columns k >= 3 = d/dS_ab + d/dS_ba); 2: also dLp/dMp (9x9, K[q+9*r])
//>   STATE_C = 1: stiffness depends on the state; dependent() writes C66 (column-major) to D[DC..DC+35]
//>                and the core takes C from there (the last stress solve's C is output with D)
//>   FLAT    = 1: X, Y, D are passed to the GPU as DAMASK stores them (no per-block padding)
//>   DT_STATE = 1: the rate depends on the time step (closed-form implicit update of stiff state, see
//>                cp_model_tdt): dot_state_dt<H>(p,S,X,Y,D,T,dt, dX) instead of dot_state
//>                Jd (may be null): diagonal d(dX_j)/dX_j <= 0 (float), used if PRECOND = 1
//>   PRECOND = 1: diagonally preconditioned state update X -= r/(1 - dt Jd) (Jacobi-Newton; same fixed
//>                point as FPI, converges for stiff self-coupled state such as hardening of regions with
//>                small volume fraction)
//>   FI_STATE = 1: isotropic eigenstrain of the state, Fi = j(X) I (DAMASK's converged Li loop:
//>                invFi_new = I/j, independent of Fi0); fi_j2m1<H>(p,X) = j^2 - 1. The core uses the
//>                elastic strain Fi^T E Fi = ((AB)^T (AB) - j^2 I)/2 and outputs Fe = F invFp_new / j;
//>                the policy itself resolves Mp = Fi^T Fi S = j^2 S (and scales its dLp/dS)
//>   ACCEL = 1 (with PRECOND): Anderson(1) acceleration of the preconditioned state iteration instead of
//>                DAMASK's damper, for slowly (linearly) converging coupled state such as dislocation
//>                densities with multiplication through the mean free path; same fixed point and tolerances
//> FLOW policies receive dt as well: flow<H>(p,S,X,Y,D,T,dt, Lp, K, mode)

//> work space of one evaluation of Lp(S) (sizes depend on the trait)
template<typename H, class M, int FLOW = M::FLOW> struct cp_work;
template<typename H, class M> struct cp_work<H,M,0> {
  typename cp_ops<H>::J dgd[M::N];
  H    gd[M::N];
  bool pos[M::N];
};
template<typename H, class M> struct cp_work<H,M,1> {
  typename cp_ops<H>::J G[54];
};

//> Lp(S) and the columns of dB/dS (B = I - dt Lp), for both traits
template<typename H, class M, int FLOW = M::FLOW> struct cp_lp;
template<typename H, class M> struct cp_lp<H,M,0> {
  typedef typename cp_ops<H>::J J;
  static void eval(CP_CONSTANT typename M::param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y,
                   CP_THREAD const H* D, H T, H, CP_THREAD H* Lp, CP_THREAD cp_work<H,M>& w, bool) {
    typedef cp_ops<H> O;
    M::template kinetics<H>(p, S, X, Y, D, T, w.gd, w.dgd, w.pos);
    for (int k = 0; k < 9; ++k) {
      H s = O::c(0.0f);
      for (int i = 0; i < M::N; ++i) s = s + w.gd[i] * O::ld(p.P_sl[9*i+k]);
      Lp[k] = s;
    }
  }
  static void dB(CP_CONSTANT typename M::param& p, CP_THREAD const cp_work<H,M>& w, J dtj, int k, CP_THREAD J* dB) {
    int a = cp_voigt_i(k), b = cp_voigt_j(k);
    for (int m = 0; m < 9; ++m) dB[m] = J(0);
    for (int i = 0; i < M::N; ++i) {
      CP_CONSTANT cp_pair* Pns = w.pos[i] ? p.P_nS_pos : p.P_nS_neg;
      J c = k < 3 ? cp_ldJ<J>(Pns[9*i+CP_IDX(a,a)])
                  : cp_ldJ<J>(Pns[9*i+CP_IDX(a,b)]) + cp_ldJ<J>(Pns[9*i+CP_IDX(b,a)]);
      J v = -dtj * w.dgd[i] * c;
      for (int m = 0; m < 9; ++m) dB[m] += v * cp_ldJ<J>(p.P_sl[9*i+m]);
    }
  }
};
template<typename H, class M> struct cp_lp<H,M,1> {
  typedef typename cp_ops<H>::J J;
  static void eval(CP_CONSTANT typename M::param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y,
                   CP_THREAD const H* D, H T, H dt, CP_THREAD H* Lp, CP_THREAD cp_work<H,M>& w, bool deriv) {
    M::template flow<H>(p, S, X, Y, D, T, dt, Lp, w.G, deriv ? 1 : 0);
  }
  static void dB(CP_CONSTANT typename M::param&, CP_THREAD const cp_work<H,M>& w, J dtj, int k, CP_THREAD J* dB) {
    for (int m = 0; m < 9; ++m) dB[m] = -dtj * w.G[m+9*k];
  }
};

//> rate of the dot state, for both DT_STATE traits
template<typename H, class M, int DTS = M::DT_STATE> struct cp_dot;
template<typename H, class M> struct cp_dot<H,M,0> {
  static void eval(CP_CONSTANT typename M::param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y,
                   CP_THREAD const H* D, H T, H, CP_THREAD H* dX, CP_THREAD float*) {
    M::template dot_state<H>(p, S, X, Y, D, T, dX);
  }
};
template<typename H, class M> struct cp_dot<H,M,1> {
  static void eval(CP_CONSTANT typename M::param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y,
                   CP_THREAD const H* D, H T, H dt, CP_THREAD H* dX, CP_THREAD float* Jd) {
    M::template dot_state_dt<H>(p, S, X, Y, D, T, dt, dX, Jd);
  }
};

//> stiffness entry C66[idx] (column-major) in precision H / J, for both traits
template<typename H, class M, int SC = M::STATE_C> struct cp_stiff;
template<typename H, class M> struct cp_stiff<H,M,0> {
  static H h(CP_CONSTANT typename M::param& p, CP_THREAD const H*, int idx) { return cp_ops<H>::ld(p.C66[idx]); }
  template<typename J>
  static J j(CP_CONSTANT typename M::param& p, CP_THREAD const H*, int idx) { return cp_ldJ<J>(p.C66[idx]); }
};
template<typename H, class M> struct cp_stiff<H,M,1> {
  static H h(CP_CONSTANT typename M::param&, CP_THREAD const H* D, int idx) { return D[M::DC+idx]; }
  template<typename J>
  static J j(CP_CONSTANT typename M::param&, CP_THREAD const H* D, int idx) { return J(cp_ops<H>::f(D[M::DC+idx])); }
};


//> j^2 - 1 of the eigenstrain Fi = j I, for both FI_STATE traits
template<class M, int FI = M::FI_STATE> struct cp_fi {
  template<typename H>
  static H j2m1(CP_CONSTANT typename M::param&, CP_THREAD const H*) { return cp_ops<H>::c(0.0f); }
};
template<class M> struct cp_fi<M,1> {
  template<typename H>
  static H j2m1(CP_CONSTANT typename M::param& p, CP_THREAD const H* X) { return M::template fi_j2m1<H>(p, X); }
};

//> Ea = Ea0 - (j^2 - 1)/2 I: elastic strain (A^T A - j^2 I)/2 of the eigenstrain Fi = j I
template<typename H>
inline void cp_fi_strain(CP_THREAD const H* Ea0, H j2m1, CP_THREAD H* Ea) {
  for (int k = 0; k < 9; ++k) Ea[k] = k % 4 == 0 ? Ea0[k] - j2m1 * 0.5f : Ea0[k];
}


//--------------------------------------------------------------------------------------------------
//> elastic stress residual R(S) = S - C:E(A B), B = I - dt Lp(S)
template<typename H, class M>
inline void cp_core_residual(CP_CONSTANT typename M::param& p, CP_THREAD const H* Mt, CP_THREAD const H* Ea, H dt,
                             CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y, CP_THREAD const H* D, H T,
                             CP_THREAD H* R, CP_THREAD H* Lp, CP_THREAD H* B, CP_THREAD H* MB,
                             CP_THREAD cp_work<H,M>& w, bool deriv = true) {
  typedef cp_ops<H> O;
  cp_lp<H,M>::eval(p, S, X, Y, D, T, dt, Lp, w, deriv);
  H Dm[9];
  for (int k = 0; k < 9; ++k) {
    Dm[k] = -(dt * Lp[k]);
    B[k]  = (k % 4 == 0 ? O::c(1.0f) : O::c(0.0f)) + Dm[k];
  }
  H e[6];
  cp_strain(Mt, Ea, Dm, MB, e);
  for (int r = 0; r < 6; ++r) {
    H s = O::c(0.0f);
    for (int c = 0; c < 6; ++c) s = s + cp_stiff<H,M>::h(p, D, r+6*c) * e[c];
    R[r] = S[r] - s;
  }
}

//> dR/dS (6x6) in precision J
template<typename H, class M>
inline void cp_core_jacobian(CP_CONSTANT typename M::param& p, CP_THREAD const H* MB, H dt, CP_THREAD const H* D,
                             CP_THREAD const cp_work<H,M>& w, CP_THREAD typename cp_ops<H>::J* Jac) {
  typedef cp_ops<H> O;
  typedef typename O::J J;
  J mb[9];
  for (int k = 0; k < 9; ++k) mb[k] = J(O::f(MB[k]));
  J dtj = J(O::f(dt));
  for (int k = 0; k < 6; ++k) {
    J dB[9];
    cp_lp<H,M>::dB(p, w, dtj, k, dB);
    J dE[9];
    cp_mulT33(dB, mb, dE);
    J de[6];
    for (int q = 0; q < 6; ++q) {
      int u = cp_voigt_i(q), v = cp_voigt_j(q);
      de[q] = q < 3 ? dE[CP_IDX(u,u)] : dE[CP_IDX(u,v)] + dE[CP_IDX(v,u)];
    }
    for (int r = 0; r < 6; ++r) {
      J s = J(0);
      for (int c = 0; c < 6; ++c) s += cp_stiff<H,M>::template j<J>(p, D, r+6*c) * de[c];
      Jac[r+6*k] = (r == k ? J(1) : J(0)) - s;
    }
  }
}

//> mixed-precision iterative refinement, first stage: Newton on S entirely in FP32 (kinetics, residual,
//> Jacobian) until the FP32 residual reaches 1e-5*|S| or stagnates; the best iterate seeds the H iterations,
//> which then only polish (the converged result has the accuracy of H, at a fraction of the cost)
template<typename H, class M>
inline void cp_core_presolve_float(CP_CONSTANT typename M::param& p, CP_CONSTANT cp_numerics& num,
                                   CP_THREAD const H* Mt, CP_THREAD const H* Ea, H dt,
                                   CP_THREAD const H* X, CP_THREAD const H* Y, CP_THREAD const H* D, H T,
                                   CP_THREAD H* S, CP_THREAD int& iters) {
  typedef cp_ops<H> O;
  const int ND = M::ND, NY = M::NY > 0 ? M::NY : 1, NDEP = M::NDEP > 0 ? M::NDEP : 1;
  float Mf[9], Eaf[9], Xf[ND], Yf[NY], Df[NDEP], Sf[6], S_best[6];
  for (int k = 0; k < 9; ++k) { Mf[k] = O::f(Mt[k]); Eaf[k] = O::f(Ea[k]); }                       // Ea: accurate, no cancellation in FP32
  for (int j = 0; j < ND; ++j) Xf[j] = O::f(X[j]);
  for (int j = 0; j < NY; ++j) Yf[j] = M::NY > 0 ? O::f(Y[j]) : 0.0f;
  for (int j = 0; j < NDEP; ++j) Df[j] = M::NDEP > 0 ? O::f(D[j]) : 0.0f;
  for (int k = 0; k < 6; ++k) { Sf[k] = O::f(S[k]); S_best[k] = Sf[k]; }
  const float dtf = O::f(dt), Tf = O::f(T);
  float R[6], Lp[9], B[9], MB[9], Jac[36], dS[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  cp_work<float,M> w;
  float r_best = INFINITY, step = 1.0f;
  int stall = 0;
  for (int it = 0; it < num.n_stress; ++it) {
    ++iters;
#ifdef CP_ITER_SPLIT
    iters += 999;                                                                                   // bench diagnostics: presolve counts 1000
#endif
    cp_core_residual<float,M>(p, Mf, Eaf, dtf, Sf, Xf, Yf, Df, Tf, R, Lp, B, MB, w);
    float rn = cp_norm6(R);
    if (rn < r_best) {
      r_best = rn;
      step   = 1.0f;
      for (int k = 0; k < 6; ++k) S_best[k] = Sf[k];
      if (rn <= 1.0e-5f * cp_norm6(Sf)) break;                                                     // FP32 has done its part
      cp_core_jacobian<float,M>(p, MB, dtf, Df, w, Jac);
      float b[6];
      for (int k = 0; k < 6; ++k) b[k] = -R[k];
      if (!cp_solve66(Jac, b)) break;
      float nd = 0.0f;
      for (int k = 0; k < 6; ++k) nd += b[k]*b[k];
      nd = DF_SQRT(nd);
      float lim = num.trust_S * CP_FMAX(cp_norm6(Sf), M::template stress_scale<float>(p, Xf));
      float sc = nd > lim ? lim / nd : 1.0f;
      for (int k = 0; k < 6; ++k) dS[k] = b[k] * sc;
    } else {                                                                                        // not improved (incl. NaN): backtrack, give up on stagnation
      step *= num.stepsize_ls;
      if (++stall > 4) break;
    }
    for (int k = 0; k < 6; ++k) Sf[k] = S_best[k] + dS[k] * step;
  }
  for (int k = 0; k < 6; ++k) S[k] = O::c(S_best[k]);
}

//> Newton iteration on S with backtracking and trust region
template<typename H, class M>
inline int cp_core_solve_stress(CP_CONSTANT typename M::param& p, CP_CONSTANT cp_numerics& num,
                                CP_THREAD const H* Mt, CP_THREAD const H* Ea, H dt,
                                CP_THREAD const H* X, CP_THREAD const H* Y, CP_THREAD const H* D, H T,
                                CP_THREAD H* S, CP_THREAD H* Lp, CP_THREAD H* B, CP_THREAD H* MB,
                                CP_THREAD cp_work<H,M>& w, CP_THREAD int& iters) {
  typedef cp_ops<H> O;
  typedef typename O::J J;
  if (num.mixed && sizeof(H) != sizeof(float)) cp_core_presolve_float<H,M>(p, num, Mt, Ea, dt, X, Y, D, T, S, iters);
  H R[6], S_old[6];
  J dS[6], Jac[36];
  float r_old = INFINITY, step = 1.0f;
  for (int k = 0; k < 6; ++k) { S_old[k] = S[k]; dS[k] = J(0); }

  for (int it = 0; it < num.n_stress; ++it) {
    ++iters;
    cp_core_residual<H,M>(p, Mt, Ea, dt, S, X, Y, D, T, R, Lp, B, MB, w);
    float rn = cp_norm6(R);
    if (!DF_ISFINITE(rn) && it == 0) return CP_FAIL_NAN;
    if (rn <= num.rtol_S * cp_norm6(S) + num.atol_S) return CP_OK;
    if (rn < r_old) {
      r_old = rn;
      step  = 1.0f;
      for (int k = 0; k < 6; ++k) S_old[k] = S[k];
      cp_core_jacobian<H,M>(p, MB, dt, D, w, Jac);
      J b[6];
      for (int k = 0; k < 6; ++k) b[k] = -J(O::f(R[k]));
      if (!cp_solve66(Jac, b)) return CP_FAIL_SINGULAR;
      float nd = 0.0f;
      for (int k = 0; k < 6; ++k) { float x = float(b[k]); nd += x*x; }
      nd = DF_SQRT(nd);
      float lim = num.trust_S * CP_FMAX(cp_norm6(S), M::template stress_scale<H>(p, X));            // floor for S ~ 0
      J scale = J(nd > lim ? lim / nd : 1.0f);
      for (int k = 0; k < 6; ++k) dS[k] = b[k] * scale;
    } else {
      step *= num.stepsize_ls;
    }
    for (int k = 0; k < 6; ++k) S[k] = S_old[k] + O::fj(dS[k] * J(step));
  }
  return CP_FAIL_STRESS;
}


//--------------------------------------------------------------------------------------------------
//> integrate stress and state of one material point over dt (DAMASK's integrateStateFPI)
template<typename H, class M>
inline int cp_core_integrate(CP_CONSTANT typename M::param& p, CP_CONSTANT cp_numerics& num, H dt,
                             cp_soa_const<H> in, cp_soa<H> out, CP_THREAD int& iters) {
  typedef cp_ops<H> O;
  typedef cp_layout<M> L;
  const int ND = M::ND, NY = M::NY > 0 ? M::NY : 1, NDEP = M::NDEP > 0 ? M::NDEP : 1;
  iters = 0;
  H T = in.get(L::T);

  H Mt[9], Ea[9], Ea0[9];
  {
    H Fp0[9], invFp0[9], F[9], A[9];
    for (int k = 0; k < 9; ++k) { Fp0[k] = in.get(L::Fp0+k); F[k] = in.get(L::F+k); }
    H detFp0 = cp_inv33(Fp0, invFp0);
    if (!(O::f(O::ab(detFp0)) > 0.0f)) return CP_FAIL_SINGULAR;
    cp_mul33(F, invFp0, A);
    cp_mulT33(A, A, Mt);
    for (int k = 0; k < 9; ++k) Ea[k] = Ea0[k] = (Mt[k] - (k % 4 == 0 ? O::c(1.0f) : O::c(0.0f))) * 0.5f;
  }

  H S[6], X[ND], Y[NY], D[NDEP];
  for (int j = 0; j < M::NY; ++j) Y[j] = in.get(L::Y0+j);                                          // constant during the step
  for (int j = 0; j < M::NDEP; ++j) D[j] = in.get(L::D0+j);                                        // as stored by DAMASK (first predictor)
  if (M::STATE_C) {                                                                                 // stiffness of the initial state (not passed by the host)
    H X0_[ND];
    for (int j = 0; j < ND; ++j) X0_[j] = in.get(L::X0+j);
    M::template dependent<H>(p, X0_, Y, T, D);
  }
  if (M::FI_STATE) {                                                                                // eigenstrain of the initial state
    H X0_[ND];
    for (int j = 0; j < ND; ++j) X0_[j] = in.get(L::X0+j);
    cp_fi_strain<H>(Ea0, cp_fi<M>::template j2m1<H>(p, X0_), Ea);
  }
  {                                                                                                 // initial stress: elastic response to the Lp guess
    H D[9], MB[9], e[6];
    for (int k = 0; k < 9; ++k) D[k] = -(dt * in.get(L::Lp0+k));
    cp_strain(Mt, Ea, D, MB, e);
    for (int r = 0; r < 6; ++r) {
      H s = O::c(0.0f);
      for (int c = 0; c < 6; ++c) s = s + cp_stiff<H,M>::h(p, D, r+6*c) * e[c];
      S[r] = s;
    }
  }

  H dX[ND], dl1[ND];
  float Jd[M::PRECOND ? ND : 1];
  float dl2[ND];
  {
    H S0[6], X0[ND];
    for (int k = 0; k < 6; ++k) S0[k] = in.get(L::S0+k);
    for (int j = 0; j < ND; ++j) X0[j] = in.get(L::X0+j);
    cp_dot<H,M>::eval(p, S0, X0, Y, D, T, dt, dX, Jd);
    for (int j = 0; j < ND; ++j) {
      if (!O::finite(dX[j])) return CP_FAIL_NAN;
      X[j]   = X0[j] + dX[j] * dt;
      dl1[j] = O::c(0.0f);
    }

    // warm start: better of {elastic response to Lp guess, previous stress}, each capped
    float kappa = 0.0f;
    for (int k = 0; k < 9; ++k) { float x = O::f(Ea[k]); kappa += x*x; }
    kappa = DF_SQRT(kappa);
    M::template dependent<H>(p, X, Y, T, D);                                                         // as the first stress solve
    if (M::FI_STATE) cp_fi_strain<H>(Ea0, cp_fi<M>::template j2m1<H>(p, X), Ea);
    float q0 = M::template cap_ratio<H>(p, S,  X, Y, D, T, O::f(dt), kappa);
    float q1 = M::template cap_ratio<H>(p, S0, X, Y, D, T, O::f(dt), kappa);
    for (int k = 0; k < 6; ++k) { S[k] = S[k] * (q0 > 1.0f ? 1.0f/q0 : 1.0f); S0[k] = S0[k] * (q1 > 1.0f ? 1.0f/q1 : 1.0f); }
    H R0[6], R1[6], Lp_[9], B_[9], MB_[9];
    cp_work<H,M> w_;
    cp_core_residual<H,M>(p, Mt, Ea, dt, S,  X, Y, D, T, R0, Lp_, B_, MB_, w_, false);
    cp_core_residual<H,M>(p, Mt, Ea, dt, S0, X, Y, D, T, R1, Lp_, B_, MB_, w_, false);
    float r0 = cp_norm6(R0), r1 = cp_norm6(R1);
    if (r1 < r0 || !DF_ISFINITE(r0)) for (int k = 0; k < 6; ++k) S[k] = S0[k];
  }

  H Lp[9], B[9], MB[9];
  cp_work<H,M> w;
  float Up[M::ACCEL ? ND : 1], Fp[M::ACCEL ? ND : 1], Fk[M::ACCEL ? ND : 1];                       // ACCEL: last update and step
  int status = CP_FAIL_STATE;
  for (int it = 0; it < num.n_state; ++it) {
#ifdef CP_ITER_SPLIT
    iters += 1000000;                                                                               // bench diagnostics: state iterations
#endif
    for (int j = 0; j < ND; ++j) {
      dl2[j] = it > 0 ? O::f(dl1[j]) : 0.0f;
      dl1[j] = dX[j];
    }

    M::template dependent<H>(p, X, Y, T, D);                                                         // plastic_dependentState in integrateStress
    if (M::FI_STATE) cp_fi_strain<H>(Ea0, cp_fi<M>::template j2m1<H>(p, X), Ea);                    // Fi of the current state (Li loop)
    int st = cp_core_solve_stress<H,M>(p, num, Mt, Ea, dt, X, Y, D, T, S, Lp, B, MB, w, iters);
    if (st != CP_OK) return st;

    cp_dot<H,M>::eval(p, S, X, Y, D, T, dt, dX, Jd);
    if (M::ACCEL) {                                                                                 // Anderson(1) on the preconditioned step
      bool conv = true;
      float sn = 0.0f, sd = 0.0f;
      for (int j = 0; j < ND; ++j) {
        H r = X[j] - in.get(L::X0+j) - dX[j] * dt;
        if (!O::finite(r)) return CP_FAIL_NAN;
        float tol = CP_FMAX(M::atol(p, j), num.rtol_state * O::f(O::ab(X[j]))), rf = O::f(r);
        conv = conv && CP_FABS(rf) <= tol;
        Fk[j] = -rf / (1.0f - O::f(dt) * Jd[M::PRECOND ? j : 0]);
        if (it > 0) { float dF = (Fk[j] - Fp[j]) / tol; sn += dF * Fk[j] / tol; sd += dF * dF; }
      }
      float g = it > 0 && sd > 0.0f ? sn / sd : 0.0f;
      if (!(CP_FABS(g) < 4.0f)) g = 0.0f;
      for (int j = 0; j < ND; ++j) {
        float U = it > 0 ? Fk[j] - g * (Up[j] + Fk[j] - Fp[j]) : Fk[j];
        X[j] = X[j] + O::c(U);
        Up[j] = U; Fp[j] = Fk[j];
      }
      if (conv) { status = CP_OK; break; }
      continue;
    }

    float d01 = 0.0f, d12 = 0.0f, d22 = 0.0f;
    for (int j = 0; j < ND; ++j) {
      float o0 = O::f(dX[j]), o1 = O::f(dl1[j]), o2 = dl2[j];
      d01 += o0*o1;
      d12 += (o0-o1)*(o1-o2);
      d22 += (o1-o2)*(o1-o2);
    }
    float zeta = (CP_FMIN(d01, d12) < 0.0f && d22 > 0.0f) ? 0.75f + 0.25f * CP_TANH(2.0f + 4.0f*d12/d22) : 1.0f;

    bool conv = true;
    for (int j = 0; j < ND; ++j) {
      dX[j] = dX[j] * zeta + dl1[j] * (1.0f - zeta);
      H r = X[j] - in.get(L::X0+j) - dX[j] * dt;
      if (!O::finite(r)) return CP_FAIL_NAN;
      X[j] = M::PRECOND ? X[j] - r * (1.0f / (1.0f - O::f(dt) * Jd[M::PRECOND ? j : 0])) : X[j] - r;
      conv = conv && O::f(O::ab(r)) <= CP_FMAX(M::atol(p, j), num.rtol_state * O::f(O::ab(X[j])));
    }
    if (conv) { status = CP_OK; break; }
  }
  if (status != CP_OK) return status;

  M::template delta_state<H>(p, S, X, Y, D, T);                                                      // after convergence (plastic_deltaState)
  for (int j = 0; j < M::NY; ++j) if (!O::finite(Y[j])) return CP_FAIL_NAN;

  // outputs (last stress solve; as in DAMASK, P etc. belong to the state before the final update)
  H invFp_new[9];
  {
    H Fp0[9], invFp0[9];
    for (int k = 0; k < 9; ++k) Fp0[k] = in.get(L::Fp0+k);
    cp_inv33(Fp0, invFp0);
    cp_mul33(invFp0, B, invFp_new);
  }
  if (!(O::f(cp_det33(invFp_new)) > 0.0f)) return CP_FAIL_SINGULAR;
  {
    H F[9], Fe[9], Sm[9], T1[9], P[9];
    for (int k = 0; k < 9; ++k) F[k] = in.get(L::F+k);
    cp_mul33(F, invFp_new, Fe);
    for (int k = 0; k < 9; ++k) Sm[k] = O::c(0.0f);
    Sm[CP_IDX(0,0)] = S[0]; Sm[CP_IDX(1,1)] = S[1]; Sm[CP_IDX(2,2)] = S[2];
    Sm[CP_IDX(1,2)] = Sm[CP_IDX(2,1)] = S[3];
    Sm[CP_IDX(0,2)] = Sm[CP_IDX(2,0)] = S[4];
    Sm[CP_IDX(0,1)] = Sm[CP_IDX(1,0)] = S[5];
    cp_mul33(Fe, Sm, T1);
    cp_mul3T3(T1, invFp_new, P);
    if (M::FI_STATE) {                                                                              // Fe = F invFp_new invFi_new
      H ij = O::c(1.0f) / O::sq(O::c(1.0f) + cp_fi<M>::template j2m1<H>(p, X));
      for (int k = 0; k < 9; ++k) Fe[k] = Fe[k] * ij;
    }
    for (int k = 0; k < 9; ++k) {
      out.set(L::P+k,   P[k]);
      out.set(L::Fe+k,  Fe[k]);
      out.set(L::iFp+k, invFp_new[k]);
      out.set(L::Lp+k,  Lp[k]);
    }
  }
  for (int k = 0; k < 6; ++k)     out.set(L::S+k, S[k]);
  for (int j = 0; j < ND; ++j)    out.set(L::X+j, X[j]);
  for (int j = 0; j < M::NY; ++j) out.set(L::Y+j, Y[j]);
  for (int j = 0; j < M::NDEP; ++j) out.set(L::D+j, D[j]);
  return CP_OK;
}


//--------------------------------------------------------------------------------------------------
//> fast precision mode: stress solve, kinetics, dependent state and state rates in FP32; df64 only where
//> small increments meet large values: the elastic strain Ea = (A^T A - I)/2, the state update
//> X = X0 + dt dX, the delta state and the kinematics (invFp_new, Fe, P).
//> The FP32 stress residual has a noise floor of a few ulp of |S|; converged at CP_FAST_RTOL |S| or, if
//> stagnating there, at CP_FAST_RTOL_STALL |S|. In terms of Lp this is the accuracy of DAMASK's own
//> integrateStress (eps_rel_Lp = 1e-6), not the 1e-10 of the df64 polish.
#ifndef CP_FAST_RTOL
#define CP_FAST_RTOL       5.0e-7f
#endif
#ifndef CP_FAST_RTOL_STALL
#define CP_FAST_RTOL_STALL 5.0e-6f
#endif
#ifndef CP_FAST_RTOL_RATE
#define CP_FAST_RTOL_RATE  1.0e-5f                                                                  // state increments are converged to 1e-5 of themselves (FP32 rate noise ~1e-6)
#endif

template<class M>
inline int cp_core_solve_stress_fast(CP_CONSTANT typename M::param& p, CP_CONSTANT cp_numerics& num,
                                     CP_THREAD const float* Mt, CP_THREAD const float* Ea, float dt,
                                     CP_THREAD const float* X, CP_THREAD const float* Y, CP_THREAD const float* D, float T,
                                     CP_THREAD float* S, CP_THREAD float* Lp, CP_THREAD cp_work<float,M>& w, CP_THREAD int& iters) {
  float R[6], B[9], MB[9], Jac[36], S_best[6], Lp_best[9], dS[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float r_best = INFINITY, step = 1.0f;
  int stall = 0;
  for (int k = 0; k < 6; ++k) S_best[k] = S[k];
  float sE = 0.0f;                                                                                  // |C:E| of the total strain: the residual's terms (noise ~ ulp of max(|S|, |C:E|))
  for (int r = 0; r < 6; ++r) {
    float t = 0.0f;
    for (int c = 0; c < 6; ++c) {
      int a = cp_voigt_i(c), b = cp_voigt_j(c);
      t += cp_stiff<float,M>::h(p, D, r+6*c) * (c < 3 ? Ea[CP_IDX(a,a)] : Ea[CP_IDX(a,b)] + Ea[CP_IDX(b,a)]);
    }
    sE += t*t;
  }
  sE = DF_SQRT(sE);
  for (int it = 0; it < num.n_stress; ++it) {
    ++iters;
    cp_core_residual<float,M>(p, Mt, Ea, dt, S, X, Y, D, T, R, Lp, B, MB, w);
    float rn = cp_norm6(R), sn = cp_norm6(S), sc_n = CP_FMAX(sn, sE);
    if (!DF_ISFINITE(rn) && it == 0) return CP_FAIL_NAN;
    if (rn <= CP_FAST_RTOL * sc_n + num.atol_S) return CP_OK;
    if (rn < r_best) {
      r_best = rn;
      step   = 1.0f;
      stall  = 0;
      for (int k = 0; k < 6; ++k) S_best[k] = S[k];
      for (int k = 0; k < 9; ++k) Lp_best[k] = Lp[k];
      cp_core_jacobian<float,M>(p, MB, dt, D, w, Jac);
      for (int k = 0; k < 6; ++k) dS[k] = -R[k];
      if (!cp_solve66(Jac, dS)) return CP_FAIL_SINGULAR;
      float nd = cp_norm6(dS);
      float lim = num.trust_S * CP_FMAX(sn, M::template stress_scale<float>(p, X));
      float sc = nd > lim ? lim / nd : 1.0f;
      for (int k = 0; k < 6; ++k) dS[k] *= sc;
    } else {
      if (r_best <= CP_FAST_RTOL_STALL * CP_FMAX(cp_norm6(S_best), sE) + num.atol_S) {             // at the noise floor: best iterate
        for (int k = 0; k < 6; ++k) S[k] = S_best[k];
        for (int k = 0; k < 9; ++k) Lp[k] = Lp_best[k];
        return CP_OK;
      }
      step *= num.stepsize_ls;
      if (++stall > 8) return CP_FAIL_STRESS;
    }
    for (int k = 0; k < 6; ++k) S[k] = S_best[k] + dS[k] * step;
  }
  return CP_FAIL_STRESS;
}

//> integrate stress and state of one material point over dt in the fast precision mode (same algorithm
//> as cp_core_integrate)
template<typename H, class M>
inline int cp_core_integrate_fast(CP_CONSTANT typename M::param& p, CP_CONSTANT cp_numerics& num, H dt,
                                  cp_soa_const<H> in, cp_soa<H> out, CP_THREAD int& iters) {
  typedef cp_ops<H> O;
  typedef cp_layout<M> L;
  const int ND = M::ND, NY = M::NY > 0 ? M::NY : 1, NDEP = M::NDEP > 0 ? M::NDEP : 1;
  iters = 0;
  const float T = O::f(in.get(L::T)), dtf = O::f(dt);

  float Mt[9], Ea[9];
  H Ea0[9];                                                                                         // FI_STATE: (A^T A - I)/2, adjusted per state
  {
    H Fp0[9], invFp0[9], F[9], A[9], MtH[9];
    for (int k = 0; k < 9; ++k) { Fp0[k] = in.get(L::Fp0+k); F[k] = in.get(L::F+k); }
    H detFp0 = cp_inv33(Fp0, invFp0);
    if (!(O::f(O::ab(detFp0)) > 0.0f)) return CP_FAIL_SINGULAR;
    cp_mul33(F, invFp0, A);
    cp_mulT33(A, A, MtH);
    for (int k = 0; k < 9; ++k) {
      Mt[k]  = O::f(MtH[k]);
      Ea0[k] = (MtH[k] - (k % 4 == 0 ? O::c(1.0f) : O::c(0.0f))) * 0.5f;
      Ea[k]  = O::f(Ea0[k]);
    }
  }

  H X[ND];
  float Xf[ND], Yf[NY], Df[NDEP], S[6];
  for (int j = 0; j < M::NY; ++j) Yf[j] = O::f(in.get(L::Y0+j));
  for (int j = 0; j < M::NDEP; ++j) Df[j] = O::f(in.get(L::D0+j));
  for (int j = 0; j < ND; ++j) Xf[j] = O::f(in.get(L::X0+j));
  if (M::STATE_C) M::template dependent<float>(p, Xf, Yf, T, Df);                                  // stiffness of the initial state (not passed by the host)
  if (M::FI_STATE) {                                                                                // eigenstrain of the initial state
    H X0_[ND], Ej[9];
    for (int j = 0; j < ND; ++j) X0_[j] = in.get(L::X0+j);
    cp_fi_strain<H>(Ea0, cp_fi<M>::template j2m1<H>(p, X0_), Ej);
    for (int k = 0; k < 9; ++k) Ea[k] = O::f(Ej[k]);
  }
  {                                                                                                 // initial stress: elastic response to the Lp guess
    float Dm[9], MB[9], e[6];
    for (int k = 0; k < 9; ++k) Dm[k] = -(dtf * O::f(in.get(L::Lp0+k)));
    cp_strain(Mt, Ea, Dm, MB, e);
    for (int r = 0; r < 6; ++r) {
      float s = 0.0f;
      for (int c = 0; c < 6; ++c) s += cp_stiff<float,M>::h(p, Df, r+6*c) * e[c];
      S[r] = s;
    }
  }

  float dX[ND], dl1[ND], dl2[ND];
  float Jd[M::PRECOND ? ND : 1];
  {
    float S0[6];
    for (int k = 0; k < 6; ++k) S0[k] = O::f(in.get(L::S0+k));
    cp_dot<float,M>::eval(p, S0, Xf, Yf, Df, T, dtf, dX, Jd);
    for (int j = 0; j < ND; ++j) {
      if (!DF_ISFINITE(dX[j])) return CP_FAIL_NAN;
      X[j]   = in.get(L::X0+j) + O::c(dX[j]) * dt;
      Xf[j]  = O::f(X[j]);
      dl1[j] = 0.0f;
    }

    float kappa = 0.0f;
    for (int k = 0; k < 9; ++k) kappa += Ea[k]*Ea[k];
    kappa = DF_SQRT(kappa);
    M::template dependent<float>(p, Xf, Yf, T, Df);
    if (M::FI_STATE) { H Ej[9]; cp_fi_strain<H>(Ea0, cp_fi<M>::template j2m1<H>(p, X), Ej); for (int k = 0; k < 9; ++k) Ea[k] = O::f(Ej[k]); }
    float q0 = M::template cap_ratio<float>(p, S,  Xf, Yf, Df, T, dtf, kappa);
    float q1 = M::template cap_ratio<float>(p, S0, Xf, Yf, Df, T, dtf, kappa);
    for (int k = 0; k < 6; ++k) { S[k] *= q0 > 1.0f ? 1.0f/q0 : 1.0f; S0[k] *= q1 > 1.0f ? 1.0f/q1 : 1.0f; }
    float R0[6], R1[6], Lp_[9], B_[9], MB_[9];
    cp_work<float,M> w_;
    cp_core_residual<float,M>(p, Mt, Ea, dtf, S,  Xf, Yf, Df, T, R0, Lp_, B_, MB_, w_, false);
    cp_core_residual<float,M>(p, Mt, Ea, dtf, S0, Xf, Yf, Df, T, R1, Lp_, B_, MB_, w_, false);
    float r0 = cp_norm6(R0), r1 = cp_norm6(R1);
    if (r1 < r0 || !DF_ISFINITE(r0)) for (int k = 0; k < 6; ++k) S[k] = S0[k];
  }

  float Lp[9];
  cp_work<float,M> w;
  float Up[M::ACCEL ? ND : 1], Fp[M::ACCEL ? ND : 1], Fk[M::ACCEL ? ND : 1];                       // ACCEL: last update and step
  int status = CP_FAIL_STATE;
  for (int it = 0; it < num.n_state; ++it) {
#ifdef CP_ITER_SPLIT
    iters += 1000000;
#endif
    for (int j = 0; j < ND; ++j) {
      dl2[j] = it > 0 ? dl1[j] : 0.0f;
      dl1[j] = dX[j];
    }

    M::template dependent<float>(p, Xf, Yf, T, Df);
    if (M::FI_STATE) { H Ej[9]; cp_fi_strain<H>(Ea0, cp_fi<M>::template j2m1<H>(p, X), Ej); for (int k = 0; k < 9; ++k) Ea[k] = O::f(Ej[k]); }
    int st = cp_core_solve_stress_fast<M>(p, num, Mt, Ea, dtf, Xf, Yf, Df, T, S, Lp, w, iters);
    if (st != CP_OK) return st;

    cp_dot<float,M>::eval(p, S, Xf, Yf, Df, T, dtf, dX, Jd);
    if (M::ACCEL) {                                                                                 // Anderson(1) on the preconditioned step
      bool conv = true;
      float sn = 0.0f, sd = 0.0f;
      for (int j = 0; j < ND; ++j) {
        H r = X[j] - in.get(L::X0+j) - O::c(dX[j]) * dt;
        if (!O::finite(r)) return CP_FAIL_NAN;
        float tol = CP_FMAX(CP_FMAX(M::atol(p, j), num.rtol_state * O::f(O::ab(X[j]))), CP_FAST_RTOL_RATE * CP_FABS(dtf * dX[j]));
        float rf = O::f(r);
        conv = conv && CP_FABS(rf) <= tol;
        Fk[j] = -rf / (1.0f - dtf * Jd[M::PRECOND ? j : 0]);
        if (it > 0) { float dF = (Fk[j] - Fp[j]) / tol; sn += dF * Fk[j] / tol; sd += dF * dF; }
      }
      float g = it > 0 && sd > 0.0f ? sn / sd : 0.0f;
      if (!(CP_FABS(g) < 4.0f)) g = 0.0f;
      for (int j = 0; j < ND; ++j) {
        float U = it > 0 ? Fk[j] - g * (Up[j] + Fk[j] - Fp[j]) : Fk[j];
        X[j] = X[j] + O::c(U);
        Xf[j] = O::f(X[j]);
        Up[j] = U; Fp[j] = Fk[j];
      }
      if (conv) { status = CP_OK; break; }
      continue;
    }

    float d01 = 0.0f, d12 = 0.0f, d22 = 0.0f;
    for (int j = 0; j < ND; ++j) {
      float o0 = dX[j], o1 = dl1[j], o2 = dl2[j];
      d01 += o0*o1;
      d12 += (o0-o1)*(o1-o2);
      d22 += (o1-o2)*(o1-o2);
    }
    float zeta = (CP_FMIN(d01, d12) < 0.0f && d22 > 0.0f) ? 0.75f + 0.25f * CP_TANH(2.0f + 4.0f*d12/d22) : 1.0f;

    bool conv = true;
    for (int j = 0; j < ND; ++j) {
      dX[j] = dX[j] * zeta + dl1[j] * (1.0f - zeta);
      H r = X[j] - in.get(L::X0+j) - O::c(dX[j]) * dt;
      if (!O::finite(r)) return CP_FAIL_NAN;
      X[j] = M::PRECOND ? X[j] - r * (1.0f / (1.0f - dtf * Jd[M::PRECOND ? j : 0])) : X[j] - r;
      Xf[j] = O::f(X[j]);
      conv = conv && O::f(O::ab(r)) <= CP_FMAX(CP_FMAX(M::atol(p, j), num.rtol_state * O::f(O::ab(X[j]))),
                                               CP_FAST_RTOL_RATE * CP_FABS(dtf * dX[j]));
    }
    if (conv) { status = CP_OK; break; }
  }
  if (status != CP_OK) return status;

  H Sh[6], Y[NY], D[NDEP];
  for (int k = 0; k < 6; ++k) Sh[k] = O::c(S[k]);
  for (int j = 0; j < M::NY; ++j) Y[j] = in.get(L::Y0+j);
  for (int j = 0; j < M::NDEP; ++j) D[j] = O::c(Df[j]);
  M::template delta_state<H>(p, Sh, X, Y, D, O::c(T));
  for (int j = 0; j < M::NY; ++j) if (!O::finite(Y[j])) return CP_FAIL_NAN;

  H invFp_new[9], LpH[9];
  {
    H Fp0[9], invFp0[9], B[9];
    for (int k = 0; k < 9; ++k) {
      Fp0[k] = in.get(L::Fp0+k);
      LpH[k] = O::c(Lp[k]);
      B[k]   = (k % 4 == 0 ? O::c(1.0f) : O::c(0.0f)) - dt * LpH[k];
    }
    cp_inv33(Fp0, invFp0);
    cp_mul33(invFp0, B, invFp_new);
  }
  if (!(O::f(cp_det33(invFp_new)) > 0.0f)) return CP_FAIL_SINGULAR;
  {
    H F[9], Fe[9], Sm[9], T1[9], P[9];
    for (int k = 0; k < 9; ++k) F[k] = in.get(L::F+k);
    cp_mul33(F, invFp_new, Fe);
    for (int k = 0; k < 9; ++k) Sm[k] = O::c(0.0f);
    Sm[CP_IDX(0,0)] = Sh[0]; Sm[CP_IDX(1,1)] = Sh[1]; Sm[CP_IDX(2,2)] = Sh[2];
    Sm[CP_IDX(1,2)] = Sm[CP_IDX(2,1)] = Sh[3];
    Sm[CP_IDX(0,2)] = Sm[CP_IDX(2,0)] = Sh[4];
    Sm[CP_IDX(0,1)] = Sm[CP_IDX(1,0)] = Sh[5];
    cp_mul33(Fe, Sm, T1);
    cp_mul3T3(T1, invFp_new, P);
    if (M::FI_STATE) {                                                                              // Fe = F invFp_new invFi_new
      H ij = O::c(1.0f) / O::sq(O::c(1.0f) + cp_fi<M>::template j2m1<H>(p, X));
      for (int k = 0; k < 9; ++k) Fe[k] = Fe[k] * ij;
    }
    for (int k = 0; k < 9; ++k) {
      out.set(L::P+k,   P[k]);
      out.set(L::Fe+k,  Fe[k]);
      out.set(L::iFp+k, invFp_new[k]);
      out.set(L::Lp+k,  LpH[k]);
    }
  }
  for (int k = 0; k < 6; ++k)     out.set(L::S+k, Sh[k]);
  for (int j = 0; j < ND; ++j)    out.set(L::X+j, X[j]);
  for (int j = 0; j < M::NY; ++j) out.set(L::Y+j, Y[j]);
  for (int j = 0; j < M::NDEP; ++j) out.set(L::D+j, D[j]);
  return CP_OK;
}


//--------------------------------------------------------------------------------------------------
//> dLp/dMp contractions of the tangent, for both traits:
//>   add_A: A[9k] += dt * sum_mn dS_mn (T2 dLp/dMp_k)_mn,  dLp: tLS = dLp/dMp : dSdF
template<typename T, class M, int FLOW = M::FLOW> struct cp_tan;
template<typename T, class M> struct cp_tan<T,M,0> {
  T gd[M::N], dgd[M::N];
  bool pos[M::N];
  void init(CP_CONSTANT typename M::param& p, CP_THREAD const T* Sv, CP_THREAD const T* X, CP_THREAD const T* Y,
            CP_THREAD const T* D, T Temp, T) CP_THREAD {
    M::template kinetics<T>(p, Sv, X, Y, D, Temp, gd, dgd, pos);
  }
  void add_A(CP_CONSTANT typename M::param& p, CP_THREAD const T* dS, CP_THREAD const T* T2, T dt, CP_THREAD T* A) CP_THREAD {
    for (int s = 0; s < M::N; ++s) {
      if (dgd[s] == T(0)) continue;
      T c = T(0);
      for (int m = 0; m < 3; ++m) for (int n = 0; n < 3; ++n) {
        T tp = T(0);
        for (int o = 0; o < 3; ++o) tp += T2[CP_IDX(m,o)] * cp_ldJ<T>(p.P_sl[9*s+CP_IDX(o,n)]);
        c += dS[CP_IDX(m,n)] * tp;
      }
      CP_CONSTANT cp_pair* Pns = pos[s] ? p.P_nS_pos : p.P_nS_neg;
      T w = dt * dgd[s] * c;
      for (int k = 0; k < 9; ++k) A[9*k] += w * cp_ldJ<T>(Pns[9*s+k]);
    }
  }
  void dLp(CP_CONSTANT typename M::param& p, CP_THREAD const T* dSdF, CP_THREAD T* tLS) CP_THREAD {
    for (int q = 0; q < 9; ++q) tLS[q] = T(0);
    for (int s = 0; s < M::N; ++s) {
      if (dgd[s] == T(0)) continue;
      CP_CONSTANT cp_pair* Pns = pos[s] ? p.P_nS_pos : p.P_nS_neg;
      T c = T(0);
      for (int q = 0; q < 9; ++q) c += cp_ldJ<T>(Pns[9*s+q]) * dSdF[q];
      for (int q = 0; q < 9; ++q) tLS[q] += dgd[s] * c * cp_ldJ<T>(p.P_sl[9*s+q]);
    }
  }
};
template<typename T, class M> struct cp_tan<T,M,1> {
  T K[81];                                                                                          // dLp/dMp, K[q+9*r]
  void init(CP_CONSTANT typename M::param& p, CP_THREAD const T* Sv, CP_THREAD const T* X, CP_THREAD const T* Y,
            CP_THREAD const T* D, T Temp, T dt) CP_THREAD {
    T Lp[9];
    M::template flow<T>(p, Sv, X, Y, D, Temp, dt, Lp, K, 2);
  }
  void add_A(CP_CONSTANT typename M::param&, CP_THREAD const T* dS, CP_THREAD const T* T2, T dt, CP_THREAD T* A) CP_THREAD {
    T U[9];                                                                                         // U = T2^T dS: c_k = sum_on U_on K_(on),k
    for (int o = 0; o < 3; ++o) for (int n = 0; n < 3; ++n) {
      T s = T(0);
      for (int m = 0; m < 3; ++m) s += T2[CP_IDX(m,o)] * dS[CP_IDX(m,n)];
      U[CP_IDX(o,n)] = s;
    }
    for (int k = 0; k < 9; ++k) {
      T c = T(0);
      for (int q = 0; q < 9; ++q) c += U[q] * K[q+9*k];
      A[9*k] += dt * c;
    }
  }
  void dLp(CP_CONSTANT typename M::param&, CP_THREAD const T* dSdF, CP_THREAD T* tLS) CP_THREAD {
    for (int q = 0; q < 9; ++q) {
      T s = T(0);
      for (int k = 0; k < 9; ++k) s += K[q+9*k] * dSdF[k];
      tLS[q] = s;
    }
  }
};


//--------------------------------------------------------------------------------------------------
//> DAMASK's analytic tangent (phase_mechanical_dPdF, Fi = I) in precision T for model M; inputs as
//> stored after integrateStress (F, Fp0, invFp_new, Fe, S Voigt, final state X, Y)
template<typename T, class M>
inline bool cp_core_tangent(CP_CONSTANT typename M::param& p, T dt,
                            CP_THREAD const T* F, CP_THREAD const T* Fp0, CP_THREAD const T* invFp_new,
                            CP_THREAD const T* Fe, CP_THREAD const T* Sv, CP_THREAD const T* X,
                            CP_THREAD const T* Y, CP_THREAD const T* D, T Temp, CP_THREAD T* dPdF) {
  T invFp[9], invFp0[9], Sm[9];
  {
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
  T FiFp[9], T2[9], Fe_[9];                                                                        // FI_STATE: dS/dF via j Fe = F invFp (Fi = j I)
  cp_mul33(F, invFp, FiFp);
  cp_mul33(F, invFp0, T2);
  {
    T jf = M::FI_STATE ? CP_POW(T(1) + cp_fi<M>::template j2m1<T>(p, X), T(0.5)) : T(1);
    for (int k = 0; k < 9; ++k) Fe_[k] = Fe[k] * jf;
  }

  cp_tan<T,M> tw;
  tw.init(p, Sv, X, Y, D, Temp, dt);

  T A99[81], R99[81];
  for (int q = 0; q < 81; ++q) { A99[q] = T(0); R99[q] = T(0); }
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    int a = i + 3*j;
    T dS[9];
    for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
      T s = T(0);
      for (int m = 0; m < 3; ++m) s += Fe_[CP_IDX(k,m)] * cp_stiff<T,M>::template j<T>(p, D, cp_voigt(i,j) + 6*cp_voigt(m,l));
      dS[CP_IDX(k,l)] = s;
    }
    for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
      T s = T(0);
      for (int m = 0; m < 3; ++m) s += dS[CP_IDX(k,m)] * invFp[CP_IDX(l,m)];
      R99[a + 9*(k+3*l)] = s;
    }
    tw.add_A(p, dS, T2, dt, A99 + a);
  }
  for (int q = 0; q < 9; ++q) A99[q + 9*q] += T(1);
  if (!cp_solve99(A99, R99)) return false;

  T Xm[9], Ym[9], T1[9], T3[9], W[9];
  cp_mul3T3(Sm, invFp, T1);
  cp_mul33(FiFp, Sm, T3);
  cp_mul33(invFp, T1, W);
  for (int q = 0; q < 81; ++q) dPdF[q] = T(0);
  for (int pp = 0; pp < 3; ++pp)
    for (int j = 0; j < 3; ++j) for (int l = 0; l < 3; ++l)
      dPdF[pp + 3*j + 9*pp + 27*l] = W[CP_IDX(l,j)];
  for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) {
    int b = k + 3*l;
    T dSdF[9], tLS[9], dFpi[9];
    for (int q = 0; q < 9; ++q) dSdF[q] = R99[q + 9*b];
    tw.dLp(p, dSdF, tLS);
    cp_mul33(invFp0, tLS, dFpi);
    for (int q = 0; q < 9; ++q) dFpi[q] = -dt * dFpi[q];
    cp_mul33(F, dFpi, Xm);    cp_mul33(Xm, T1, Ym);
    for (int q = 0; q < 9; ++q) dPdF[q + 9*b] += Ym[q];
    cp_mul33(FiFp, dSdF, Xm); cp_mul3T3(Xm, invFp, Ym);
    for (int q = 0; q < 9; ++q) dPdF[q + 9*b] += Ym[q];
    cp_mul3T3(T3, dFpi, Ym);
    for (int q = 0; q < 9; ++q) dPdF[q + 9*b] += Ym[q];
  }
  return true;
}


//==================================================================================================
// model policies
//==================================================================================================

//> resolved stresses of both branches (non-Schmid) in precision H
template<typename H, typename P>
inline void cp_resolved(CP_CONSTANT P& p, int i, CP_THREAD const H* S, CP_THREAD H& tp, CP_THREAD H& tn) {
  typedef cp_ops<H> O;
  tp = O::c(0.0f); tn = O::c(0.0f);
  for (int k = 0; k < 6; ++k) {
    int a = cp_voigt_i(k), b = cp_voigt_j(k);
    H cp_ = k < 3 ? O::ld(p.P_nS_pos[9*i+CP_IDX(a,a)])
                  : O::ld(p.P_nS_pos[9*i+CP_IDX(a,b)]) + O::ld(p.P_nS_pos[9*i+CP_IDX(b,a)]);
    H cn_ = k < 3 ? O::ld(p.P_nS_neg[9*i+CP_IDX(a,a)])
                  : O::ld(p.P_nS_neg[9*i+CP_IDX(a,b)]) + O::ld(p.P_nS_neg[9*i+CP_IDX(b,a)]);
    tp = tp + S[k]*cp_;
    tn = tn + S[k]*cn_;
  }
}

//--------------------------------------------------------------------------------------------------
//> phenopowerlaw, slip only. X = [xi(N), gamma(N)], no delta state
template<int NSL>
struct cp_model_pheno {
  enum { N = NSL, ND = 2*NSL, NY = 0, NDEP = 0, NBX = 2, NBY = 0, NBD = 0, FLOW = 0, STATE_C = 0, FLAT = 0, DC = 0, DT_STATE = 0, PRECOND = 0, FI_STATE = 0, ACCEL = 0 };
  typedef cp_pheno_param param;
  static float pad_value(int group, int block) { return group == 0 && block == 0 ? 1.0f : 0.0f; }   // xi = 1, gamma = 0

  template<typename H>
  static void kinetics(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                       CP_THREAD const H*, H, CP_THREAD H* gd, CP_THREAD typename cp_ops<H>::J* dgd, CP_THREAD bool* pos) {
    cp_kinetics<H,N>(p, S, X, gd, dgd, pos);
  }
  template<typename H>
  static void dot_state(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                        CP_THREAD const H*, H, CP_THREAD H* dX) {
    cp_dotState<H,N>(p, S, X, dX, dX + N);
  }
  template<typename H>
  static void delta_state(CP_CONSTANT param&, CP_THREAD const H*, CP_THREAD const H*, CP_THREAD H*, CP_THREAD const H*, H) {}
  template<typename H>
  static void dependent(CP_CONSTANT param&, CP_THREAD const H*, CP_THREAD const H*, H, CP_THREAD H*) {}
  static float atol(CP_CONSTANT param& p, int j) { return j < N ? p.atol_xi : p.atol_gamma; }
  template<typename H>
  static float stress_scale(CP_CONSTANT param&, CP_THREAD const H* X) { return cp_ops<H>::f(X[0]); }
  template<typename H>
  static float cap_ratio(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                         CP_THREAD const H*, H, float dt, float kappa) {
    typedef cp_ops<H> O;
    float q = 0.0f;
    for (int i = 0; i < N; ++i) {
      float g0 = cp_ldJ<float>(p.dot_gamma_0[i]);
      if (!(g0 > 0.0f)) continue;                                                                   // padded system
      float t = 0.0f;
      for (int k = 0; k < 6; ++k) {
        int a = cp_voigt_i(k), b = cp_voigt_j(k);
        float c = k < 3 ? cp_ldJ<float>(p.P_sl[9*i+CP_IDX(a,a)])
                        : cp_ldJ<float>(p.P_sl[9*i+CP_IDX(a,b)]) + cp_ldJ<float>(p.P_sl[9*i+CP_IDX(b,a)]);
        t += O::f(S[k]) * c;
      }
      float r_cap = DF_EXP(DF_LOG(kappa / (dt * g0)) / cp_ldJ<float>(p.n[i]));
      q = CP_FMAX(q, (t < 0.0f ? -t : t) / O::f(X[i]) / r_cap);
    }
    return q;
  }
};

//--------------------------------------------------------------------------------------------------
//> kinehardening (phase_mechanical_plastic_kinehardening.f90), Schmid law.
//> X = [xi(N), chi(N), gamma(N)], Y = [sgn_gamma(N), chi_flip(N), gamma_flip(N)]
struct cp_kineh_param {
  cp_pair C66[36];
  cp_pair P_sl    [CP_MAX_SL*9];
  cp_pair P_nS_pos[CP_MAX_SL*9];
  cp_pair P_nS_neg[CP_MAX_SL*9];
  cp_pair h_sl_sl [CP_MAX_SL*CP_MAX_SL];                                                            // (i,j) at i+N*j (N: padded count)
  cp_pair dot_gamma_0[CP_MAX_SL];
  cp_pair n          [CP_MAX_SL];
  cp_pair h_0_xi     [CP_MAX_SL];
  cp_pair h_0_chi    [CP_MAX_SL];
  cp_pair h_inf_xi   [CP_MAX_SL];
  cp_pair h_inf_chi  [CP_MAX_SL];
  cp_pair xi_inf     [CP_MAX_SL];
  cp_pair chi_inf    [CP_MAX_SL];
  float   atol_xi;
  float   atol_gamma;
  int     N_sl;
  int     pad;
};

template<int NSL>
struct cp_model_kineh {
  enum { N = NSL, ND = 3*NSL, NY = 3*NSL, NDEP = 0, NBX = 3, NBY = 3, NBD = 0, FLOW = 0, STATE_C = 0, FLAT = 0, DC = 0, DT_STATE = 0, PRECOND = 0, FI_STATE = 0, ACCEL = 0 };
  typedef cp_kineh_param param;
  static float pad_value(int group, int block) { return group == 0 && block == 0 ? 1.0f : 0.0f; }   // xi = 1, others 0

  template<typename H>
  static void kinetics(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                       CP_THREAD const H*, H, CP_THREAD H* gd, CP_THREAD typename cp_ops<H>::J* dgd, CP_THREAD bool* pos) {
    typedef cp_ops<H> O;
    for (int i = 0; i < N; ++i) {
      H tp, tn;
      cp_resolved<H>(p, i, S, tp, tn);
      tp = tp - X[N+i];                                                                             // tau_pos = Mp:P_nS_pos - chi
      tn = tn + X[N+i];                                                                             // tau_neg = Mp:P_nS_neg + chi
      pos[i] = tp > tn;
      H tm = pos[i] ? tp : tn;
      H r  = tm / X[i];
      bool active = O::f(r) > 0.0f;
      H mag = active ? O::ld(p.dot_gamma_0[i]) * O::ex(O::ld(p.n[i]) * O::lg(r)) : O::c(0.0f);
      gd[i]  = pos[i] ? mag : -mag;
      dgd[i] = active ? O::f(gd[i]) * O::f(O::ld(p.n[i])) / O::f(tm) : 0.0f;
    }
  }
  template<typename H>
  static void dot_state(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y,
                        CP_THREAD const H* D, H T, CP_THREAD H* dX) {
    typedef cp_ops<H> O;
    H gd[N];
    typename O::J dgd[N];
    bool pos[N];
    kinetics<H>(p, S, X, Y, D, T, gd, dgd, pos);
    H sumGamma = O::c(0.0f);
    for (int i = 0; i < N; ++i) sumGamma = sumGamma + X[2*N+i];
    for (int i = 0; i < N; ++i) dX[2*N+i] = O::ab(gd[i]);                                           // dot_gamma = |dot_gamma|
    for (int i = 0; i < N; ++i) {
      H h = O::c(0.0f);
      for (int j = 0; j < N; ++j) h = h + O::ld(p.h_sl_sl[i+N*j]) * dX[2*N+j];
      H x = sumGamma * O::ld(p.h_0_xi[i]) / O::ld(p.xi_inf[i]);
      dX[i] = h * (O::ld(p.h_inf_xi[i]) + (O::ld(p.h_0_xi[i]) - O::ld(p.h_inf_xi[i]) * (O::c(1.0f) - x)) * O::ex(-x));
      H y = (X[2*N+i] - Y[2*N+i]) * O::ld(p.h_0_chi[i]) / (O::ld(p.chi_inf[i]) + Y[N+i]);
      dX[N+i] = Y[i] * dX[2*N+i]
              * (O::ld(p.h_inf_chi[i]) + (O::ld(p.h_0_chi[i]) - O::ld(p.h_inf_chi[i]) * (O::c(1.0f) - y)) * O::ex(-y));
    }
  }
  template<typename H>
  static void delta_state(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD H* Y,
                          CP_THREAD const H* D, H T) {
    typedef cp_ops<H> O;
    H gd[N];
    typename O::J dgd[N];
    bool pos[N];
    kinetics<H>(p, S, X, Y, D, T, gd, dgd, pos);
    for (int i = 0; i < N; ++i) {
      H sgn = O::f(O::ab(gd[i])) <= 1.0e-10f ? Y[i]                                                  // merge(sgn, sign(1,gd), dEq0(gd,1e-10))
                                             : (O::f(gd[i]) < 0.0f ? O::c(-1.0f) : O::c(1.0f));
      if (O::f(O::ab(sgn - Y[i])) > 0.1f) {                                                         // dNeq(sgn, sgn_old, 0.1): reversal
        Y[N+i]   = O::ab(X[N+i]);                                                                   // chi_flip   = |chi|
        Y[2*N+i] = X[2*N+i];                                                                        // gamma_flip = gamma
        Y[i]     = sgn;
      }
    }
  }
  static float atol(CP_CONSTANT param& p, int j) { return j < 2*N ? p.atol_xi : p.atol_gamma; }
  template<typename H>
  static void dependent(CP_CONSTANT param&, CP_THREAD const H*, CP_THREAD const H*, H, CP_THREAD H*) {}
  template<typename H>
  static float stress_scale(CP_CONSTANT param&, CP_THREAD const H* X) { return cp_ops<H>::f(X[0]); }
  template<typename H>
  static float cap_ratio(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                         CP_THREAD const H*, H, float dt, float kappa) {
    typedef cp_ops<H> O;
    float q = 0.0f;
    for (int i = 0; i < N; ++i) {
      float g0 = cp_ldJ<float>(p.dot_gamma_0[i]);
      if (!(g0 > 0.0f)) continue;
      float t = 0.0f;
      for (int k = 0; k < 6; ++k) {
        int a = cp_voigt_i(k), b = cp_voigt_j(k);
        float c = k < 3 ? cp_ldJ<float>(p.P_sl[9*i+CP_IDX(a,a)])
                        : cp_ldJ<float>(p.P_sl[9*i+CP_IDX(a,b)]) + cp_ldJ<float>(p.P_sl[9*i+CP_IDX(b,a)]);
        t += O::f(S[k]) * c;
      }
      t -= O::f(X[N+i]);                                                                            // effective stress w.r.t. back stress
      float r_cap = DF_EXP(DF_LOG(kappa / (dt * g0)) / cp_ldJ<float>(p.n[i]));
      q = CP_FMAX(q, (t < 0.0f ? -t : t) / O::f(X[i]) / r_cap);
    }
    return q;
  }
};

//--------------------------------------------------------------------------------------------------
//> dislotungsten (phase_mechanical_plastic_dislotungsten.f90): thermally activated kink-pair kinetics,
//> dislocation multiplication/annihilation, dipole formation/climb; non-Schmid allowed (tau_eff >= 0).
//> X = [rho_mob(N), rho_dip(N), gamma_sl(N)], no delta state, D = [tau_pass(N), Lambda_sl(N)]
struct cp_dtung_param {
  cp_pair C66[36];
  cp_pair P_sl    [CP_MAX_SL*9];
  cp_pair P_nS_pos[CP_MAX_SL*9];
  cp_pair P_nS_neg[CP_MAX_SL*9];
  cp_pair h_sl_sl [CP_MAX_SL*CP_MAX_SL];                                                            // (i,j) at i+N*j
  cp_pair forest  [CP_MAX_SL*CP_MAX_SL];                                                            // forestProjection (i,j) at i+N*j
  cp_pair b_sl[CP_MAX_SL], Q_s[CP_MAX_SL], i_sl[CP_MAX_SL], tau_P[CP_MAX_SL], p[CP_MAX_SL], q[CP_MAX_SL];
  cp_pair h[CP_MAX_SL], w[CP_MAX_SL], omega[CP_MAX_SL], B[CP_MAX_SL], d_caron[CP_MAX_SL], f_at[CP_MAX_SL];
  cp_pair D, D_0, Q_cl, mu, nu, k_B;
  float   atol_xi;                                                                                  // atol_rho
  float   atol_gamma;
  int     N_sl;
  int     dipoleformation;
};

template<int NSL>
struct cp_model_dtung {
  enum { N = NSL, ND = 3*NSL, NY = 0, NDEP = 2*NSL, NBX = 3, NBY = 0, NBD = 2, FLOW = 0, STATE_C = 0, FLAT = 0, DC = 0, DT_STATE = 0, PRECOND = 0, FI_STATE = 0, ACCEL = 0 };
  typedef cp_dtung_param param;
  static float pad_value(int group, int block) { return group == 2 && block == 1 ? 1.0f : 0.0f; }   // Lambda = 1

  //> shear rates, derivatives and effective stress (kinetics in the Fortran)
  template<typename H>
  static void rates(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* D, H T,
                    CP_THREAD H* gd, CP_THREAD typename cp_ops<H>::J* dgd, CP_THREAD bool* pos, CP_THREAD H* teff) {
    typedef cp_ops<H> O;
    typedef typename O::J J;
    H kT = O::ld(p.k_B) * T;
    for (int i = 0; i < N; ++i) {
      H tp, tn;
      cp_resolved<H>(p, i, S, tp, tn);
      pos[i] = tp > tn;
      H te = (pos[i] ? tp : tn) - D[i];                                                             // max(tau_pos,tau_neg) - tau_pass
      if (O::f(te) < 0.0f) te = O::c(0.0f);                                                         // math_clip(left=0)
      teff[i] = te;
      if (!(O::f(te) > 1.0e-8f)) { gd[i] = O::c(0.0f); dgd[i] = J(0); continue; }                   // tol_math_check
      H tauP = O::ld(p.tau_P[i]), pe = O::ld(p.p[i]), qe = O::ld(p.q[i]), b = O::ld(p.b_sl[i]);
      H lnSR  = O::lg(te / tauP);
      H SRp   = O::ex(pe * lnSR);
      H SRpm1 = O::ex((pe - O::c(1.0f)) * lnSR);
      H BR    = O::ld(p.Q_s[i]) / kT;
      H lnb   = O::lg(O::c(1.0f) - SRp);                                                            // NaN for SRp > 1, as (1-SRp)**q in Fortran
      H effL  = D[N+i] - O::ld(p.w[i]);
      H t_n   = b * O::ex(BR * O::ex(qe * lnb)) / (O::ld(p.omega[i]) * effL);
      H t_k   = effL * O::ld(p.B[i]) / (O::c(2.0f) * b * te);
      H g     = X[i] * b * O::ld(p.h[i]) / (t_n + t_k);
      gd[i]   = pos[i] ? g : -g;
      H dtn   = -(t_n * BR * pe * qe * O::ex((qe - O::c(1.0f)) * lnb) * SRpm1 / tauP);
      H dtk   = -(t_k / te);
      dgd[i]  = J(-O::f(gd[i]) * O::f(dtn + dtk) / O::f(t_n + t_k));
    }
  }
  template<typename H>
  static void kinetics(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                       CP_THREAD const H* D, H T, CP_THREAD H* gd, CP_THREAD typename cp_ops<H>::J* dgd, CP_THREAD bool* pos) {
    H teff[N];
    rates<H>(p, S, X, D, T, gd, dgd, pos, teff);
  }
  template<typename H>
  static void dot_state(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                        CP_THREAD const H* D, H T, CP_THREAD H* dX) {
    typedef cp_ops<H> O;
    H gd[N], teff[N];
    typename O::J dgd[N];
    bool pos[N];
    rates<H>(p, S, X, D, T, gd, dgd, pos, teff);
    H mu = O::ld(p.mu), nu = O::ld(p.nu), kT = O::ld(p.k_B) * T, pi = O::c(3.14159265358979f);
    H climb_c = O::c(3.0f) * mu * O::ld(p.D_0) * O::ex(-(O::ld(p.Q_cl) / kT)) / (O::c(2.0f) * pi * kT);
    for (int i = 0; i < N; ++i) {
      H ag = O::ab(gd[i]), b = O::ld(p.b_sl[i]), dc = O::ld(p.d_caron[i]), Lam = D[N+i];
      H d_hat, form;
      if (O::f(ag) == 0.0f) {                                                                       // dEq0(dot_gamma)
        d_hat = Lam;
        form  = O::c(0.0f);
      } else {
        d_hat = mu * b / (O::c(8.0f) * pi * (O::c(1.0f) - nu) * teff[i]);
        if (O::f(d_hat - Lam) > 0.0f) d_hat = Lam;                                                  // math_clip(right=Lambda)
        if (O::f(d_hat - dc) < 0.0f)  d_hat = dc;                                                   // math_clip(left=d_caron)
        form = p.dipoleformation ? ag * O::c(2.0f) * (d_hat - dc) / b * X[i] : O::c(0.0f);
      }
      H climb = O::c(0.0f);
      if (O::f(d_hat - dc) != 0.0f) {                                                               // dEq0(d_hat-d_caron)
        H v_cl = climb_c * O::ld(p.f_at[i]) / (d_hat + dc);
        climb = O::c(4.0f) * v_cl * X[N+i] / (d_hat - dc);
      }
      dX[i]     = ag / (b * Lam) - form - ag * O::c(2.0f) * dc / b * X[i];
      dX[N+i]   = form - climb - ag * O::c(2.0f) * dc / b * X[N+i];
      dX[2*N+i] = ag;
    }
  }
  template<typename H>
  static void delta_state(CP_CONSTANT param&, CP_THREAD const H*, CP_THREAD const H*, CP_THREAD H*, CP_THREAD const H*, H) {}
  //> dislotungsten_dependentState
  template<typename H>
  static void dependent(CP_CONSTANT param& p, CP_THREAD const H* X, CP_THREAD const H*, H, CP_THREAD H* D) {
    typedef cp_ops<H> O;
    for (int i = 0; i < N; ++i) {
      H s = O::c(0.0f), f = O::c(0.0f);
      for (int j = 0; j < N; ++j) {
        H r = X[j] + X[N+j];
        s = s + O::ld(p.h_sl_sl[i+N*j]) * r;
        f = f + O::ld(p.forest[i+N*j]) * r;
      }
      D[i]   = O::ld(p.mu) * O::ld(p.b_sl[i]) * O::sq(s);                                           // tau_pass
      D[N+i] = O::c(1.0f) / (O::c(1.0f) / O::ld(p.D) + O::sq(f) / O::ld(p.i_sl[i]));                // Lambda_sl
    }
  }
  static float atol(CP_CONSTANT param& p, int j) { return j < 2*N ? p.atol_xi : p.atol_gamma; }
  template<typename H>
  static float stress_scale(CP_CONSTANT param& p, CP_THREAD const H*) { return 0.01f * cp_ldJ<float>(p.tau_P[0]); }
  //> no power law: shrink S such that the fastest system does not exceed kappa (heuristic exponent)
  template<typename H>
  static float cap_ratio(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                         CP_THREAD const H* D, H T, float dt, float kappa) {
    typedef cp_ops<H> O;
    H gd[N], teff[N];
    typename O::J dgd[N];
    bool pos[N];
    rates<H>(p, S, X, D, T, gd, dgd, pos, teff);
    float m = 0.0f;
    for (int i = 0; i < N; ++i) m = CP_FMAX(m, O::f(O::ab(gd[i])));
    float r = dt * m / kappa;
    return r > 1.0f ? DF_EXP(DF_LOG(r) / 10.0f) : 0.0f;
  }
};

//--------------------------------------------------------------------------------------------------
//> stock phenopowerlaw WITH twinning (phase_mechanical_plastic_phenopowerlaw.f90): twin systems are
//> appended to the slip systems (rank-1, unidirectional: tau > 0 only, factor 1 - sum(gamma_tw/gamma_char)).
//> Systems 0..N_slip-1: slip, N_slip..N_slip+N_twin-1: twin. X = [xi_sl, xi_tw | gamma_sl, gamma_tw]
//> (DAMASK's state order). h_sl_sl holds the combined interaction [h_sl_sl h_sl_tw; h_tw_sl h_tw_tw].
struct cp_phenotw_param {
  cp_pair C66[36];
  cp_pair P_sl    [CP_MAX_SL*9];                                                                    // [P_sl, P_tw]
  cp_pair P_nS_pos[CP_MAX_SL*9];                                                                    // [P_nS_pos, P_tw]
  cp_pair P_nS_neg[CP_MAX_SL*9];                                                                    // [P_nS_neg, P_tw]
  cp_pair h_sl_sl [CP_MAX_SL*CP_MAX_SL];                                                            // combined, (i,j) at i+N*j
  cp_pair dot_gamma_0[CP_MAX_SL], n[CP_MAX_SL], a[CP_MAX_SL], h_0[CP_MAX_SL], xi_inf[CP_MAX_SL];
  cp_pair c_1[CP_MAX_SL], c_2[CP_MAX_SL], f_sat[CP_MAX_SL];                                         // slip entries
  cp_pair h_0_tw_sl[CP_MAX_SL], h_0_tw_tw[CP_MAX_SL], c_3[CP_MAX_SL], c_4[CP_MAX_SL], gamma_char[CP_MAX_SL]; // twin entries
  float   atol_xi;
  float   atol_gamma;
  int     N_sl;                                                                                     // total (slip + twin)
  int     N_slip;
  int     N_twin;
  int     pad;
};

//> x**c as Fortran real**real for x >= 0 (0**0 = 1, 0**c = 0 for c > 0)
template<typename H>
inline H cp_pow0(H x, H c) {
  typedef cp_ops<H> O;
  return O::f(x) > 0.0f ? O::ex(c * O::lg(x)) : (O::f(c) == 0.0f ? O::c(1.0f) : O::c(0.0f));
}

template<int NSL>
struct cp_model_phenotw {
  enum { N = NSL, ND = 2*NSL, NY = 0, NDEP = 0, NBX = 2, NBY = 0, NBD = 0, FLOW = 0, STATE_C = 0, FLAT = 0, DC = 0, DT_STATE = 0, PRECOND = 0, FI_STATE = 0, ACCEL = 0 };
  typedef cp_phenotw_param param;
  static float pad_value(int group, int block) { return group == 0 && block == 0 ? 1.0f : 0.0f; }

  template<typename H>
  static H sum_F(CP_CONSTANT param& p, CP_THREAD const H* X) {                                     // sum(gamma_tw/gamma_char)
    typedef cp_ops<H> O;
    H s = O::c(0.0f);
    for (int k = 0; k < p.N_twin; ++k) s = s + X[N + p.N_slip + k] / O::ld(p.gamma_char[p.N_slip + k]);
    return s;
  }
  template<typename H>
  static void kinetics(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                       CP_THREAD const H*, H, CP_THREAD H* gd, CP_THREAD typename cp_ops<H>::J* dgd, CP_THREAD bool* pos) {
    typedef cp_ops<H> O;
    H f_untw = O::c(1.0f) - sum_F<H>(p, X);
    for (int i = 0; i < N; ++i) {
      H tp, tn;
      cp_resolved<H>(p, i, S, tp, tn);
      if (i < p.N_slip) {                                                                           // kinetics_sl
        pos[i] = tp > tn;
        H tm = pos[i] ? tp : tn;
        H r  = tm / X[i];
        bool active = O::f(r) > 0.0f;
        H mag = active ? O::ld(p.dot_gamma_0[i]) * O::ex(O::ld(p.n[i]) * O::lg(r)) : O::c(0.0f);
        gd[i]  = pos[i] ? mag : -mag;
        dgd[i] = active ? O::f(gd[i]) * O::f(O::ld(p.n[i])) / O::f(tm) : 0.0f;
      } else if (i < p.N_slip + p.N_twin) {                                                         // kinetics_tw (P_nS_pos = P_tw)
        pos[i] = true;
        bool active = O::f(tp) > 0.0f;
        gd[i]  = active ? f_untw * O::ld(p.dot_gamma_0[i]) * O::ex(O::ld(p.n[i]) * O::lg(tp / X[i])) : O::c(0.0f);
        dgd[i] = active && O::f(gd[i]) != 0.0f ? O::f(gd[i]) * O::f(O::ld(p.n[i])) / O::f(tp) : 0.0f;
      } else {
        pos[i] = true; gd[i] = O::c(0.0f); dgd[i] = 0.0f;
      }
    }
  }
  //> phenopowerlaw_dotState with twins
  template<typename H>
  static void dot_state(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y,
                        CP_THREAD const H* D, H T, CP_THREAD H* dX) {
    typedef cp_ops<H> O;
    H gd[N];
    typename O::J dgd[N];
    bool pos[N];
    kinetics<H>(p, S, X, Y, D, T, gd, dgd, pos);
    const int ns = p.N_slip, nt = p.N_twin;
    H sumF = sum_F<H>(p, X), sumG = O::c(0.0f);
    for (int i = 0; i < ns; ++i) sumG = sumG + X[N+i];
    for (int i = 0; i < N; ++i) dX[N+i] = i < ns ? O::ab(gd[i]) : gd[i];                             // |dot_gamma_sl|, dot_gamma_tw (>= 0)
    for (int i = 0; i < ns; ++i) {
      H x = O::c(1.0f) - X[i] / (O::ld(p.xi_inf[i]) + O::ld(p.f_sat[i]) * O::sq(sumF));
      H ax = O::ab(x);
      H left = O::f(ax) > 0.0f ? O::ex(O::ld(p.a[i]) * O::lg(ax)) : O::c(0.0f);
      if (O::f(x) < 0.0f) left = -left;
      H hs = O::c(0.0f), ht = O::c(0.0f);
      for (int j = 0; j < ns; ++j)      hs = hs + O::ld(p.h_sl_sl[i+N*j]) * dX[N+j];
      for (int k = ns; k < ns+nt; ++k)  ht = ht + O::ld(p.h_sl_sl[i+N*k]) * dX[N+k];
      dX[i] = O::ld(p.h_0[i]) * (O::c(1.0f) + O::ld(p.c_1[i]) * cp_pow0<H>(sumF, O::ld(p.c_2[i]))) * left * hs + ht;
    }
    for (int k = ns; k < ns+nt; ++k) {
      H hs = O::c(0.0f), ht = O::c(0.0f);
      for (int j = 0; j < ns; ++j)      hs = hs + O::ld(p.h_sl_sl[k+N*j]) * dX[N+j];
      for (int l = ns; l < ns+nt; ++l)  ht = ht + O::ld(p.h_sl_sl[k+N*l]) * dX[N+l];
      dX[k] = O::ld(p.h_0_tw_sl[k]) * cp_pow0<H>(sumG, O::ld(p.c_3[k])) * hs
            + O::ld(p.h_0_tw_tw[k]) * cp_pow0<H>(sumF, O::ld(p.c_4[k])) * ht;
    }
    for (int i = ns+nt; i < N; ++i) dX[i] = O::c(0.0f);
  }
  template<typename H>
  static void delta_state(CP_CONSTANT param&, CP_THREAD const H*, CP_THREAD const H*, CP_THREAD H*, CP_THREAD const H*, H) {}
  template<typename H>
  static void dependent(CP_CONSTANT param&, CP_THREAD const H*, CP_THREAD const H*, H, CP_THREAD H*) {}
  static float atol(CP_CONSTANT param& p, int j) { return j < N ? p.atol_xi : p.atol_gamma; }
  template<typename H>
  static float stress_scale(CP_CONSTANT param&, CP_THREAD const H* X) { return cp_ops<H>::f(X[0]); }
  template<typename H>
  static float cap_ratio(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                         CP_THREAD const H*, H, float dt, float kappa) {
    typedef cp_ops<H> O;
    float q = 0.0f;
    for (int i = 0; i < p.N_slip + p.N_twin; ++i) {
      float g0 = cp_ldJ<float>(p.dot_gamma_0[i]);
      if (!(g0 > 0.0f)) continue;
      float t = 0.0f;
      for (int k = 0; k < 6; ++k) {
        int a = cp_voigt_i(k), b = cp_voigt_j(k);
        float c = k < 3 ? cp_ldJ<float>(p.P_sl[9*i+CP_IDX(a,a)])
                        : cp_ldJ<float>(p.P_sl[9*i+CP_IDX(a,b)]) + cp_ldJ<float>(p.P_sl[9*i+CP_IDX(b,a)]);
        t += O::f(S[k]) * c;
      }
      if (i >= p.N_slip && t < 0.0f) continue;                                                       // twins: unidirectional
      float r_cap = DF_EXP(DF_LOG(kappa / (dt * g0)) / cp_ldJ<float>(p.n[i]));
      q = CP_FMAX(q, (t < 0.0f ? -t : t) / O::f(X[i]) / r_cap);
    }
    return q;
  }
};

#endif
