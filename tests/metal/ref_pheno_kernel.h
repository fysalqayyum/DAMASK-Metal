// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Reference implementations of the phenopowerlaw constitutive update (host only, tests):
//>  - cp_integrate (per-point struct cp_point) with cp_residual, cp_jacobian, cp_cap_stress and
//>    cp_solve_stress: S-formulation of DAMASK's integrateStress/integrateStateFPI, incl. tangent
//>  - cp_integrate_slim (component-major arrays, cp_slim_layout) and DAMASK's analytic tangent
//>    cp_tangent (phase_mechanical_dPdF for Fi = I)
//> The production GPU code (cp_core.h, cp_model_pheno) implements the same equations; these
//> versions are kept as independent oracles for test_cp_kernel, test_cp_slim, test_cp_core and
//> test_cp_gpu. They are not part of the shaders.
//--------------------------------------------------------------------------------------------------
#ifndef DAMASK_REF_PHENO_KERNEL_H
#define DAMASK_REF_PHENO_KERNEL_H

#include "../../src/grid/metal/cp_slim.h"


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
}

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
