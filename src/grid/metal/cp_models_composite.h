// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Composite (parent + rotated product regions) model policies of the GPU constitutive update,
//> see cp_core.h for the policy interface (FLOW, STATE_C traits).
//> tdt (phase_mechanical_plastic_tdt.f90): twinning/detwinning/retwinning with slip inside twins.
//> Template parameters are capacities (padded); loops run over the actual counts in the parameters.
//--------------------------------------------------------------------------------------------------
#ifndef DAMASK_CP_MODELS_COMPOSITE_H
#define DAMASK_CP_MODELS_COMPOSITE_H

#include "cp_core.h"

#define CP_TDT_MAX_SL  36
#define CP_TDT_MAX_TW  12
#define CP_TDT_MAX_FAM 6

//> 6x6 inverse (Gauss-Jordan, partial pivoting; same algorithm as inv66 in the Fortran), A column-major
template<typename H>
inline bool cp_inv66(CP_THREAD const H* A, CP_THREAD H* B) {
  typedef cp_ops<H> O;
  H W[6][12];
  for (int r = 0; r < 6; ++r) for (int c = 0; c < 12; ++c)
    W[r][c] = c < 6 ? A[r+6*c] : (c-6 == r ? O::c(1.0f) : O::c(0.0f));
  for (int c = 0; c < 6; ++c) {
    int p = c;
    float best = O::f(O::ab(W[c][c]));
    for (int r = c+1; r < 6; ++r) { float v = O::f(O::ab(W[r][c])); if (v > best) { best = v; p = r; } }
    if (!(best > 0.0f)) return false;
    if (p != c) for (int k = 0; k < 12; ++k) { H t = W[c][k]; W[c][k] = W[p][k]; W[p][k] = t; }
    H q = O::c(1.0f) / W[c][c];
    for (int k = 0; k < 12; ++k) W[c][k] = W[c][k] * q;
    for (int r = 0; r < 6; ++r) {
      if (r == c) continue;
      H f = W[r][c];
      for (int k = 0; k < 12; ++k) W[r][k] = W[r][k] - f * W[c][k];
    }
  }
  for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) B[r+6*c] = W[r][6+c];
  return true;
}

//> x**c for x >= 0 via exp/log (0 for x = 0)
template<typename H>
inline H cp_powp(H x, H c) {
  typedef cp_ops<H> O;
  return O::f(x) > 0.0f ? O::ex(c * O::lg(x)) : O::c(0.0f);
}

//> sign(|x|**c, x)
template<typename H>
inline H cp_spow(H x, H c) {
  typedef cp_ops<H> O;
  H m = cp_powp<H>(O::ab(x), c);
  return O::f(x) < 0.0f ? -m : m;
}

//--------------------------------------------------------------------------------------------------
//> tdt parameters. Tables use the capacities of the kernel (NSL, NTW, NFAM) as strides.
struct cp_tdt_param {
  cp_pair C66[36];                                                                                  // parent stiffness (reference)
  cp_pair Mel[36*(1+CP_TDT_MAX_TW)];                                                                // compliances (Reuss) or stiffnesses (Voigt): parent, twins
  cp_pair P9[(1+CP_TDT_MAX_TW)*CP_TDT_MAX_SL*9];                                                    // region r (0: parent), system i at 9*(NSL*r+i)
  cp_pair Pv[(1+CP_TDT_MAX_TW)*CP_TDT_MAX_SL*6];                                                    // Voigt contraction: tau = sum_k S_k Pv_k
  cp_pair P9_tw[CP_TDT_MAX_TW*9];
  cp_pair Pv_tw[CP_TDT_MAX_TW*6];
  cp_pair h_ss[CP_TDT_MAX_SL*CP_TDT_MAX_SL];                                                        // (i,j) at i+NSL*j
  cp_pair h_st[CP_TDT_MAX_SL*CP_TDT_MAX_TW];                                                        // h_0_sl-tw h_sl-tw gamma_char, (i,k) at i+NSL*k
  cp_pair h_ts[CP_TDT_MAX_TW*CP_TDT_MAX_SL];                                                        // h_0_tw-sl h_tw-sl, (k,j) at k+NTW*j
  cp_pair h_tt[CP_TDT_MAX_TW*CP_TDT_MAX_TW];                                                        // h_0_tw-tw h_tw-tw gamma_char, (k,l) at k+NTW*l
  cp_pair g0[CP_TDT_MAX_SL], n[CP_TDT_MAX_SL], a[CP_TDT_MAX_SL], h0[CP_TDT_MAX_SL], xi_inf[CP_TDT_MAX_SL];
  cp_pair xi_inf_tw[CP_TDT_MAX_FAM], h0_tw[CP_TDT_MAX_FAM], a_tw[CP_TDT_MAX_FAM];
  cp_pair g0_tw[CP_TDT_MAX_TW], n_tw[CP_TDT_MAX_TW], gch[CP_TDT_MAX_TW], dxi_nuc[CP_TDT_MAX_TW];
  cp_pair dxi_retw[CP_TDT_MAX_TW], H_chi[CP_TDT_MAX_TW], h0_dtw[CP_TDT_MAX_TW];
  cp_pair f_n, f_sat, f_min;
  int     fam[CP_TDT_MAX_SL];                                                                       // slip family (0-based)
  float   atol_xi, atol_gamma, atol_f;
  int     N_sl, N_tw, N_fam, reuss;
};

//> X = [xi_sl(NSL), gamma_sl(NSL), f(NTW), xi_tw(NTW), xi_dtw(NTW), xi_sl_tw(NFAM,NTW), gamma_sl_tw(NTW)],
//> Y = f at the start of the step (delta state), D = homogenized C66.
//> Twin volume fractions: closed-form backward Euler over dt (no stiffness limit for saturation and
//> complete detwinning), with f0 = Y:
//>   shrinkage (tau_k < 0): fd_k = -lambda_k f0_k / (1 + lambda_k dt)
//>   growth    (tau_k > 0): fd_k = a_k R / (f_sat + A dt), A = sum a, R = f_sat - F_end(others) - F0(growing)
//> nucleation threshold and back stress use f0 (lagged), so tau_k depends on the stress only.
template<int NSL, int NTW, int NFAM>
struct cp_model_tdt {
  enum { N = NSL, ND = 2*NSL + (4+NFAM)*NTW, NY = NTW, NDEP = 36, NBX = 0, NBY = 0, NBD = 0,
         FLOW = 1, STATE_C = 1, FLAT = 0, DC = 0, DT_STATE = 1, PRECOND = 1, FI_STATE = 0, ACCEL = 0,
         iXI = 0, iG = NSL, iF = 2*NSL, iXT = 2*NSL+NTW, iXD = 2*NSL+2*NTW, iXR = 2*NSL+3*NTW,
         iGR = 2*NSL+3*NTW+NFAM*NTW };
  typedef cp_tdt_param param;
  static float pad_value(int, int) { return 0.0f; }                                                 // state is mapped by the host (see cp_tdt_map)

  template<typename H>
  static H resolved(CP_CONSTANT cp_pair* Pv, CP_THREAD const H* S) {
    typedef cp_ops<H> O;
    H t = O::c(0.0f);
    for (int k = 0; k < 6; ++k) t = t + S[k] * O::ld(Pv[k]);
    return t;
  }

  //> K += c P9 (x) (Pv: mode 1, P9: mode 2)
  template<typename J>
  static void rank1(CP_THREAD J* K, J c, CP_CONSTANT cp_pair* P9, CP_CONSTANT cp_pair* Pv, int mode) {
    if (mode == 1) for (int k = 0; k < 6; ++k) { J ck = c * cp_ldJ<J>(Pv[k]); for (int m = 0; m < 9; ++m) K[m+9*k] += ck * cp_ldJ<J>(P9[m]); }
    else           for (int k = 0; k < 9; ++k) { J ck = c * cp_ldJ<J>(P9[k]); for (int m = 0; m < 9; ++m) K[m+9*k] += ck * cp_ldJ<J>(P9[m]); }
  }

  //> rates of all regions: gd parent slip (signed), gr sum |slip rate| in twin k, fd twin fraction rate;
  //> mode >= 0: also Lp and K (mode 1: dLp/dS_Voigt 9x6, mode 2: dLp/dMp 9x9)
  template<typename H>
  static void rates(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y, H dt,
                    CP_THREAD H* gd, CP_THREAD H* gr, CP_THREAD H* fd,
                    CP_THREAD H* Lp, CP_THREAD typename cp_ops<H>::J* K, int mode, CP_THREAD float* grn = nullptr) {
    typedef cp_ops<H> O;
    typedef typename O::J J;
    const bool L = mode >= 0;
    if (L) for (int k = 0; k < 9; ++k) Lp[k] = O::c(0.0f);
    if (mode > 0) for (int q = 0; q < 9*(mode == 1 ? 6 : 9); ++q) K[q] = J(0);
    if (grn) for (int q = 0; q < NTW*NFAM; ++q) grn[q] = 0.0f;                                     // sum n |gd| per twin and slip family
    H sumf = O::c(0.0f);
    for (int k = 0; k < p.N_tw; ++k) sumf = sumf + X[iF+k];
    const H w0 = O::c(1.0f) - sumf;

    for (int r = 0; r <= p.N_tw; ++r) {                                                             // r = 0: parent, r = k+1: twin k
      H w = r == 0 ? w0 : X[iF+r-1];
      if (r > 0) gr[r-1] = O::c(0.0f);
      if (r > 0 && !(O::f(w) > cp_ldJ<float>(p.f_min))) continue;                                   // twin absent: no slip
      for (int i = 0; i < p.N_sl; ++i) {
        CP_CONSTANT cp_pair* P9 = p.P9 + 9*(NSL*r+i);
        H tau = resolved<H>(p.Pv + 6*(NSL*r+i), S);
        H xi  = r == 0 ? X[iXI+i] : X[iXR + NFAM*(r-1) + p.fam[i]];
        H g   = O::ld(p.g0[i]) * cp_powp<H>(O::ab(tau) / xi, O::ld(p.n[i]));
        if (O::f(tau) < 0.0f) g = -g;
        if (r == 0) gd[i] = g;
        else {
          gr[r-1] = gr[r-1] + O::ab(g);
          if (grn) grn[NFAM*(r-1)+p.fam[i]] += cp_ldJ<float>(p.n[i]) * O::f(O::ab(g));
        }
        if (!L || O::f(g) == 0.0f) continue;
        H wg = w * g;
        for (int k = 0; k < 9; ++k) Lp[k] = Lp[k] + wg * O::ld(P9[k]);
        if (mode > 0) rank1<J>(K, J(O::f(wg)) * cp_ldJ<J>(p.n[i]) / J(O::f(tau)), P9, p.Pv + 6*(NSL*r+i), mode);
      }
    }

    // twin volume fractions (closed-form backward Euler, see above)
    H f0s = O::c(0.0f);
    for (int k = 0; k < p.N_tw; ++k) f0s = f0s + Y[k];
    const H w00 = O::c(1.0f) - f0s;
    H tau[NTW], lr[NTW];                                                                            // resolved stress, ln(a_k) (growth) or ln(lambda_k) (shrinkage)
    J al[NTW], ga[NTW];                                                                             // d fd_k/d tau_k (diagonal), coupling via R
    H Fe = O::c(0.0f), lmax = O::c(-1.0e30f);                                                       // F_end(not growing) + F0(growing); max ln(a)
    const H ldt = O::f(dt) > 0.0f ? O::lg(dt) : O::c(-1.0e30f);                                     // dt = 0: initial evaluation
    bool any_grow = false;
    for (int k = 0; k < p.N_tw; ++k) {
      H f0 = Y[k];
      tau[k] = resolved<H>(p.Pv_tw + 6*k, S) - O::ld(p.H_chi[k]) * f0 * w00;
      H lc = O::lg(O::ld(p.g0_tw[k]) / O::ld(p.gch[k]));
      al[k] = J(0); ga[k] = J(0); fd[k] = O::c(0.0f); lr[k] = O::c(0.0f);
      if (O::f(tau[k]) > 0.0f) {
        H xi = X[iXT+k] + O::ld(p.dxi_nuc[k]) * O::ex(-(f0 / O::ld(p.f_n)));
        lr[k] = lc + O::ld(p.n_tw[k]) * O::lg(tau[k] / xi);
        if (!any_grow || O::f(lr[k]) > O::f(lmax)) lmax = lr[k];
        any_grow = true;
        Fe = Fe + f0;
      } else if (O::f(tau[k]) < 0.0f) {
        H xi = X[iXD+k] + O::ld(p.dxi_retw[k]) * O::ex(-(w00 / O::ld(p.f_n)));
        lr[k] = lc + O::ld(p.n_tw[k]) * O::lg(-tau[k] / xi);
        H fp = O::f(f0) > 0.0f ? f0 : O::c(0.0f);
        H x = lr[k] + ldt, sh;                                                                      // x = ln(lambda dt); sh = 1/(1 + lambda dt)
        if (O::f(x) < 0.0f) { H ld_ = O::ex(x); sh = O::c(1.0f) / (O::c(1.0f) + ld_); fd[k] = -(fp * O::ex(lr[k]) * sh); }
        else                { H u = O::ex(-x);  sh = u / (O::c(1.0f) + u);           fd[k] = -(fp / dt / (O::c(1.0f) + u)); }
        al[k] = cp_ldJ<J>(p.n_tw[k]) * J(O::f(fd[k])) * J(O::f(sh)) / J(O::f(tau[k]));
        ga[k] = -(J(O::f(dt)) * al[k]);                                                             // dR/dtau_k
        Fe = Fe + f0 + fd[k] * dt;
      } else {
        Fe = Fe + f0;
      }
    }
    const H R = O::ld(p.f_sat) - Fe;
    const bool grow = any_grow && O::f(R) > 0.0f;
    J u[9];                                                                                         // sum_growing gamma_k beta_k P_k (coupling term)
    for (int q = 0; q < 9; ++q) u[q] = J(0);
    if (grow) {                                                                                     // sum fd = R A/(f_sat + A dt), shares a_k/A (overflow free)
      H As = O::c(0.0f);
      for (int k = 0; k < p.N_tw; ++k) if (O::f(tau[k]) > 0.0f) As = As + O::ex(lr[k] - lmax);
      H x = lmax + ldt + O::lg(As), tot;                                                            // x = ln(A dt)
      if (!(O::f(dt) > 0.0f)) tot = R * O::ex(lmax) * As / O::ld(p.f_sat);                          // dt = 0 (initial evaluation)
      else if (O::f(x) < 0.0f) { H Ad = O::ex(x);  tot = R * Ad / dt / (O::ld(p.f_sat) + Ad); }
      else                     { H v = O::ex(-x);  tot = R / dt / (O::ld(p.f_sat) * v + O::c(1.0f)); }
      for (int k = 0; k < p.N_tw; ++k) {
        if (!(O::f(tau[k]) > 0.0f)) continue;
        fd[k] = tot * O::ex(lr[k] - lmax) / As;
        J nf = cp_ldJ<J>(p.n_tw[k]) * J(O::f(fd[k])) / J(O::f(tau[k]));
        al[k] = nf;
        ga[k] = -(J(O::f(dt)) * nf);                                                                // via A
        if (mode > 0) { J b = cp_ldJ<J>(p.gch[k]) * J(O::f(fd[k])) / J(O::f(R)); for (int q = 0; q < 9; ++q) u[q] += b * cp_ldJ<J>(p.P9_tw[9*k+q]); }
      }
    } else {
      for (int k = 0; k < p.N_tw; ++k) if (O::f(tau[k]) > 0.0f) ga[k] = J(0);
    }
    if (!L) return;
    for (int k = 0; k < p.N_tw; ++k) {
      if (O::f(fd[k]) == 0.0f) continue;
      H gf = O::ld(p.gch[k]) * fd[k];
      for (int q = 0; q < 9; ++q) Lp[q] = Lp[q] + gf * O::ld(p.P9_tw[9*k+q]);
    }
    if (mode == 0) return;
    for (int k = 0; k < p.N_tw; ++k) {
      if (al[k] != J(0)) rank1<J>(K, cp_ldJ<J>(p.gch[k]) * al[k], p.P9_tw + 9*k, p.Pv_tw + 6*k, mode);
      if (!grow || ga[k] == J(0)) continue;
      const int nc = mode == 1 ? 6 : 9;
      for (int c = 0; c < nc; ++c) {
        J v = ga[k] * (mode == 1 ? cp_ldJ<J>(p.Pv_tw[6*k+c]) : cp_ldJ<J>(p.P9_tw[9*k+c]));
        for (int q = 0; q < 9; ++q) K[q+9*c] += u[q] * v;
      }
    }
  }

  template<typename H>
  static void flow(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y,
                   CP_THREAD const H*, H, H dt, CP_THREAD H* Lp, CP_THREAD typename cp_ops<H>::J* K, int mode) {
    H gd[NSL], gr[NTW], fd[NTW];
    rates<H>(p, S, X, Y, dt, gd, gr, fd, Lp, K, mode);
  }

  template<typename H>
  static void dot_state_dt(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* Y,
                           CP_THREAD const H*, H, H dt, CP_THREAD H* dX, CP_THREAD float* Jd) {
    typedef cp_ops<H> O;
    H gd[NSL], gr[NTW], fd[NTW];
    float grn[NTW*NFAM];
    rates<H>(p, S, X, Y, dt, gd, gr, fd, nullptr, nullptr, -1, grn);
    if (Jd) for (int j = 0; j < ND; ++j) Jd[j] = 0.0f;
    for (int j = 0; j < ND; ++j) dX[j] = O::c(0.0f);
    const int ns = p.N_sl, nt = p.N_tw;
    for (int i = 0; i < ns; ++i) {
      H hs = O::c(0.0f), ht = O::c(0.0f);
      for (int j = 0; j < ns; ++j) hs = hs + O::ld(p.h_ss[i+NSL*j]) * O::ab(gd[j]);
      for (int k = 0; k < nt; ++k) ht = ht + O::ld(p.h_st[i+NSL*k]) * O::ab(fd[k]);
      H x = O::c(1.0f) - X[iXI+i] / O::ld(p.xi_inf[i]);
      dX[iXI+i] = O::ld(p.h0[i]) * cp_spow<H>(x, O::ld(p.a[i])) * hs + ht;
      if (Jd) Jd[iXI+i] = slope(cp_ldJ<float>(p.h0[i]), O::f(x), cp_ldJ<float>(p.a[i]), cp_ldJ<float>(p.xi_inf[i]), O::f(hs),
                                O::f(X[iXI+i]), cp_ldJ<float>(p.h_ss[i+NSL*i]) * cp_ldJ<float>(p.n[i]) * O::f(O::ab(gd[i])));
      dX[iG+i]  = O::ab(gd[i]);
    }
    for (int k = 0; k < nt; ++k) {
      H htt = O::c(0.0f), hts = O::c(0.0f);
      for (int l = 0; l < nt; ++l) htt = htt + O::ld(p.h_tt[k+NTW*l]) * (O::f(fd[l]) > 0.0f ? fd[l] : O::c(0.0f));
      for (int j = 0; j < ns; ++j) hts = hts + O::ld(p.h_ts[k+NTW*j]) * O::ab(gd[j]);
      dX[iF+k]  = fd[k];
      dX[iXT+k] = htt + hts;
      dX[iXD+k] = O::ld(p.h0_dtw[k]) * gr[k];
      for (int f = 0; f < p.N_fam; ++f) {
        H y = O::c(1.0f) - X[iXR+NFAM*k+f] / O::ld(p.xi_inf_tw[f]);
        dX[iXR+NFAM*k+f] = O::ld(p.h0_tw[f]) * cp_spow<H>(y, O::ld(p.a_tw[f])) * gr[k];
        if (Jd) Jd[iXR+NFAM*k+f] = slope(cp_ldJ<float>(p.h0_tw[f]), O::f(y), cp_ldJ<float>(p.a_tw[f]), cp_ldJ<float>(p.xi_inf_tw[f]),
                                         O::f(gr[k]), O::f(X[iXR+NFAM*k+f]), grn[NFAM*k+f]);
      }
      dX[iGR+k] = gr[k];
    }
  }

  //> d/dxi of h0 sign(|x|^a,x) G(xi), x = 1 - xi/xi_inf, dG/dxi = -nG/xi (power law): diagonal of the
  //> state Jacobian (clamped <= 0) for the preconditioned state update
  static float slope(float h0, float x, float a, float xi_inf, float G, float xi, float nG) {
    float ax = x < 0.0f ? -x : x;
    float l  = ax > 0.0f ? DF_EXP(a * DF_LOG(ax)) : 0.0f;
    float dl = ax > 0.0f ? a * l / ax : 0.0f;                                                       // d|x|^a/d|x|
    float d  = h0 * (-(dl / xi_inf) * G - (x < 0.0f ? -l : l) * nG / xi);
    return d < 0.0f ? d : 0.0f;
  }

  //> f at the start of the next step
  template<typename H>
  static void delta_state(CP_CONSTANT param& p, CP_THREAD const H*, CP_THREAD const H* X, CP_THREAD H* Y, CP_THREAD const H*, H) {
    for (int k = 0; k < p.N_tw; ++k) Y[k] = X[iF+k];
  }

  //> homogenized stiffness (Reuss: inverse of the mixed compliances, Voigt: mixed stiffnesses)
  template<typename H>
  static void dependent(CP_CONSTANT param& p, CP_THREAD const H* X, CP_THREAD const H*, H, CP_THREAD H* D) {
    typedef cp_ops<H> O;
    H sumf = O::c(0.0f);
    for (int k = 0; k < p.N_tw; ++k) sumf = sumf + X[iF+k];
    H Mx[36];
    for (int q = 0; q < 36; ++q) Mx[q] = (O::c(1.0f) - sumf) * O::ld(p.Mel[q]);
    for (int k = 0; k < p.N_tw; ++k)
      for (int q = 0; q < 36; ++q) Mx[q] = Mx[q] + X[iF+k] * O::ld(p.Mel[36*(k+1)+q]);
    if (p.reuss) { if (!cp_inv66<H>(Mx, D + DC)) for (int q = 0; q < 36; ++q) D[DC+q] = O::c(NAN); }
    else for (int q = 0; q < 36; ++q) D[DC+q] = Mx[q];
  }

  static float atol(CP_CONSTANT param& p, int j) {
    return j < iG ? p.atol_xi : j < iF ? p.atol_gamma : j < iXT ? p.atol_f : j < iGR ? p.atol_xi : p.atol_gamma;
  }

  //> certificate: rate of entry j switches on when the twin fraction X[result] exceeds thr (twin-region
  //> slip resistances and slip); -1: continuous
  static int switch_entry(CP_CONSTANT param& p, int j, CP_THREAD float& thr) {
    thr = cp_ldJ<float>(p.f_min);
    if (j >= iXD && j < iXR) return iF + (j - iXD);
    if (j >= iXR && j < iGR) return iF + (j - iXR) / NFAM;
    if (j >= iGR)            return iF + (j - iGR);
    return -1;
  }
  template<typename H>
  static float stress_scale(CP_CONSTANT param&, CP_THREAD const H* X) { return cp_ops<H>::f(X[iXI]); }
  template<typename H>
  static float cap_ratio(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                         CP_THREAD const H*, H, float dt, float kappa) {
    typedef cp_ops<H> O;
    float q = 0.0f, Sf[6];
    for (int k = 0; k < 6; ++k) Sf[k] = O::f(S[k]);
    for (int i = 0; i < p.N_sl; ++i) {
      float t = resolved<float>(p.Pv + 6*i, Sf);
      float r_cap = DF_EXP(DF_LOG(kappa / (dt * cp_ldJ<float>(p.g0[i]))) / cp_ldJ<float>(p.n[i]));
      q = CP_FMAX(q, (t < 0.0f ? -t : t) / O::f(X[iXI+i]) / r_cap);
    }
    for (int k = 0; k < p.N_tw; ++k) {
      float t = resolved<float>(p.Pv_tw + 6*k, Sf);
      float xi = O::f(t > 0.0f ? X[iXT+k] : X[iXD+k]);
      float r_cap = DF_EXP(DF_LOG(kappa * cp_ldJ<float>(p.gch[k]) / (dt * cp_ldJ<float>(p.g0_tw[k]))) / cp_ldJ<float>(p.n_tw[k]));
      q = CP_FMAX(q, (t < 0.0f ? -t : t) / xi / r_cap);
    }
    return q;
  }
};


//--------------------------------------------------------------------------------------------------
//> twintrip parameters (phase_mechanical_plastic_twintrip.f90, blob of plastic_twintrip_GPUadd).
//> Product region kinds k: 0 twin, 1 epsilon, 2 alpha'; plane p = 0..3; region slip system j of kind k
//> on plane p at (4k+p)*CP_TT_RMAX + j; family parameters of kind k at k*CP_TT_FMAX + f.
#define CP_BLOB_MAX 65536                                                                          // host: padded blob length; all supported
                                                                                                    // tdt/twintrip configurations read < 10^4
#define CP_TT_RMAX  24
#define CP_TT_FMAX  4
#define CP_TT_NDMAX 128
struct cp_twintrip_param {
  cp_pair C66[36];                                                                                  // austenite stiffness (reference)
  cp_pair Mel[13*36];                                                                               // compliances (Reuss) / stiffnesses (Voigt): austenite, twin p, epsilon p, alpha' p
  cp_pair P9_sl[12*9], Pv_sl[12*6], Nv_sl[12*6];                                                    // Schmid tensors, Voigt contractions of P and n (x) n
  cp_pair P9_tw[12*9], Pv_tw[12*6];                                                                 // {111}<112> (twinning and transformation)
  cp_pair forest[144], h_ss[144], h_st[144], h_sr[144], h_tt[144], h_rr[144];                       // (i,j) at i+12*j
  cp_pair b_sl[12], Q_sl[12], v_0[12], i_sl[12], p_sl[12], q_sl[12], tau_0[12], B[12], d_caron[12];
  cp_pair t_tw[12], r_tw[12], gch_tw[12], t_tr[12], r_tr[12];
  cp_pair rP9[3*4*CP_TT_RMAX*9], rPv[3*4*CP_TT_RMAX*6];
  cp_pair rg0[3*CP_TT_FMAX], rn[3*CP_TT_FMAX], rxi_inf[3*CP_TT_FMAX], rh0[3*CP_TT_FMAX], ra[3*CP_TT_FMAX];
  cp_pair poly[3*5], poly_x0[3];                                                                    // Gamma_sf(T), Delta_G(T), Delta_G_alpha(T)
  cp_pair b_tw, b_tr, Q_cl, omega, Dg, i_tw, i_tr, L_tw, L_tr, x_c, V_cs, h, V_mol, rho, delta_sl,
          k_alpha, G_s, Delta_V, mu, nu;
  cp_pair k_B, pi, pi_4, gamma_eps, two_thirds, w_floor;                                            // constants (exact in df64)
  float   atol[CP_TT_NDMAX];
  int     rfam[3*CP_TT_RMAX];
  int     pair[24];                                                                                 // nucleation slip pair of each twin system (0-based)
  int     poly_n[3];
  int     extended, omit_dipoles, reuss, pad;
};

//> X = DAMASK's state [rho_mob, rho_dip, gamma_sl, f_tw, f_tr, f_a | xi (4 N_fam), gamma (4) per plastic region
//> kind], D = [C66, Lambda_sl, Lambda_tw, Lambda_tr, tau_pass, (j^2 of Fi = j I if alpha')]. Mp = j^2 S.
//> NTW, NTR: 0 or 12; AL: alpha'; RT/FT, RE/FE, RA/FA: slip systems/families per plane in twin, epsilon,
//> alpha' regions (0: elastic).
template<int NTW, int NTR, int AL, int RT, int FT, int RE, int FE, int RA, int FA>
struct cp_model_twintrip {
  enum { NS = 12,
         iRM = 0, iRD = NS, iGS = 2*NS, iFT = 3*NS, iFR = 3*NS+NTW, iFA = 3*NS+NTW+NTR,
         oT = iFA + 4*AL, oE = oT + (RT > 0 ? 4*FT+4 : 0), oA = oE + (RE > 0 ? 4*FE+4 : 0),
         N = NS, ND = oA + (RA > 0 ? 4*FA+4 : 0), NY = 0, NDEP = 60+NTW+NTR+AL, NBX = 0, NBY = 0, NBD = 0,
         dLS = 36, dLT = 48, dLR = 48+NTW, dTP = 48+NTW+NTR, dJ2 = 60+NTW+NTR,
         FLOW = 1, STATE_C = 1, FLAT = 1, DC = 0, DT_STATE = 1, PRECOND = 1, FI_STATE = AL, ACCEL = 1,
         NG = 4*(FT > FE ? (FT > FA ? FT : FA) : (FE > FA ? FE : FA)) };
  typedef cp_twintrip_param param;
  static float pad_value(int, int) { return 0.0f; }

  template<typename H>
  static H resolved(CP_CONSTANT cp_pair* Pv, CP_THREAD const H* S) {
    typedef cp_ops<H> O;
    H t = O::c(0.0f);
    for (int k = 0; k < 6; ++k) t = t + S[k] * O::ld(Pv[k]);
    return t;
  }
  //> K += c L (x) R (R: Pv mode 1, P9 mode 2)
  template<typename J>
  static void rank1(CP_THREAD J* K, J c, CP_CONSTANT cp_pair* L9, CP_CONSTANT cp_pair* Rv, CP_CONSTANT cp_pair* R9, int mode) {
    if (mode == 1) for (int k = 0; k < 6; ++k) { J ck = c * cp_ldJ<J>(Rv[k]); for (int m = 0; m < 9; ++m) K[m+9*k] += ck * cp_ldJ<J>(L9[m]); }
    else           for (int k = 0; k < 9; ++k) { J ck = c * cp_ldJ<J>(R9[k]); for (int m = 0; m < 9; ++m) K[m+9*k] += ck * cp_ldJ<J>(L9[m]); }
  }
  template<typename H>
  static H poly(CP_CONSTANT param& p, int i, H T) {
    typedef cp_ops<H> O;
    const int n = p.poly_n[i];
    H y = O::ld(p.poly[5*i+n]), x = T - O::ld(p.poly_x0[i]);
    for (int o = n-1; o >= 0; --o) y = y * x + O::ld(p.poly[5*i+o]);
    return y;
  }
  //> volume fractions of the product regions per plane (fractions in the Fortran)
  template<typename H>
  static void fractions(CP_THREAD const H* X, CP_THREAD H* wT, CP_THREAD H* wE, CP_THREAD H* wA) {
    typedef cp_ops<H> O;
    for (int q = 0; q < 4; ++q) {
      H t = O::c(0.0f), e = O::c(0.0f), a = O::c(0.0f);
      if (NTW > 0) { t = X[iFT+3*q] + X[iFT+3*q+1] + X[iFT+3*q+2]; if (O::f(t) < 0.0f) t = O::c(0.0f); }
      if (NTR > 0) e = X[iFR+3*q] + X[iFR+3*q+1] + X[iFR+3*q+2];
      if (AL)      { a = X[iFA+q]; if (O::f(a) < 0.0f) a = O::c(0.0f); }
      e = e - a;
      if (O::f(e) < 0.0f) e = O::c(0.0f);
      wT[q] = t; wE[q] = e; wA[q] = a;
    }
  }

  //> power-law slip in the product regions of kind K (regions_kinetics)
  template<typename H, int K, int NR, int NF, int OX>
  static void region_rates(CP_CONSTANT param& p, CP_THREAD const H* Sm, CP_THREAD const H* X, CP_THREAD const H* w,
                           CP_THREAD H* g, CP_THREAD float* nG, CP_THREAD H* Lp, CP_THREAD typename cp_ops<H>::J* Km, int mode) {
    typedef cp_ops<H> O;
    typedef typename O::J J;
    for (int q = 0; q < 4; ++q) g[q] = O::c(0.0f);
    if (NR == 0) return;
    if (nG) for (int q = 0; q < 4*NF; ++q) nG[q] = 0.0f;
    for (int q = 0; q < 4; ++q) {
      if (!(O::f(w[q]) > 1.0e-12f)) continue;                                                       // F_MIN
      for (int j = 0; j < NR; ++j) {
        const int s = (4*K+q)*CP_TT_RMAX + j, f = p.rfam[K*CP_TT_RMAX+j], fi = K*CP_TT_FMAX + f;
        H tau = resolved<H>(p.rPv + 6*s, Sm);
        H gg  = O::ld(p.rg0[fi]) * cp_powp<H>(O::ab(tau) / X[OX + NF*q + f], O::ld(p.rn[fi]));
        if (O::f(tau) < 0.0f) gg = -gg;
        g[q] = g[q] + O::ab(gg);
        if (nG) nG[NF*q+f] += cp_ldJ<float>(p.rn[fi]) * O::f(O::ab(gg));
        if (mode < 0 || O::f(gg) == 0.0f) continue;
        H wg = w[q] * gg;
        for (int m = 0; m < 9; ++m) Lp[m] = Lp[m] + wg * O::ld(p.rP9[9*s+m]);
        if (mode > 0) rank1<J>(Km, J(O::f(wg)) * cp_ldJ<J>(p.rn[fi]) / J(O::f(tau)), p.rP9 + 9*s, p.rPv + 6*s, p.rP9 + 9*s, mode);
      }
    }
  }

  //> all rates (kinetics, kinetics_sl, nucleation): gd austenite slip (signed), fd_tw, fd_tr, gR region slip
  //> (sum |rate| per plane, kinds 0..2 at 4k+q), fd_a; mode >= 0: also Lp and K (dLp/dS: mode 1 Voigt
  //> columns 9x6, mode 2 9x9); nG: sum n |rate| per family and plane of each kind (at k*NG + NF*q + f)
  template<typename H>
  static void rates(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H* D, H T,
                    CP_THREAD H* gd, CP_THREAD H* fd_tw, CP_THREAD H* fd_tr, CP_THREAD H* gR, CP_THREAD H* fd_a,
                    CP_THREAD H* Lp, CP_THREAD typename cp_ops<H>::J* Km, int mode, CP_THREAD float* nG = nullptr) {
    typedef cp_ops<H> O;
    typedef typename O::J J;
    if (mode >= 0) for (int k = 0; k < 9; ++k) Lp[k] = O::c(0.0f);
    if (mode > 0) for (int q = 0; q < 9*(mode == 1 ? 6 : 9); ++q) Km[q] = J(0);
    H Sm[6];                                                                                        // Mp = Fi^T Fi S
    const H j2 = AL ? D[dJ2] : O::c(1.0f);
    for (int k = 0; k < 6; ++k) Sm[k] = AL ? S[k] * j2 : S[k];
    H wT[4], wE[4], wA[4];
    fractions<H>(X, wT, wE, wA);
    H w_g = O::c(1.0f);
    for (int k = 0; k < NTW; ++k) w_g = w_g - X[iFT+k];
    for (int k = 0; k < NTR; ++k) w_g = w_g - X[iFR+k];
    if (O::f(w_g) < 0.0f) w_g = O::c(0.0f);
    const H kT = O::ld(p.k_B) * T;

    // austenite slip (kinetics_sl): rate = rho b^2 u tau_eff/(tau_eff b + u B), u = v_0 exp(-Q/kT x^q)
    J dgd[NS];
    for (int i = 0; i < NS; ++i) {
      H tau = resolved<H>(p.Pv_sl + 6*i, Sm);
      H te  = O::ab(tau) - D[dTP+i];
      gd[i] = O::c(0.0f); dgd[i] = J(0);
      if (!(O::f(te) > 1.0e-8f)) continue;                                                          // tol_math_check
      H b = O::ld(p.b_sl[i]), t0 = O::ld(p.tau_0[i]), pe = O::ld(p.p_sl[i]), qe = O::ld(p.q_sl[i]), dl = O::ld(p.delta_sl);
      H QkT = O::ld(p.Q_sl[i]) / kT;
      H lSR = O::lg(te / t0);
      H SRp = O::ex(pe * lSR);
      H y   = (O::c(1.0f) - SRp) / dl, x, sg;
      if (O::f(y) > 30.0f) { x = y * dl; sg = O::c(1.0f); }
      else { H ey = O::ex(y); x = dl * O::lg(O::c(1.0f) + ey); sg = ey / (O::c(1.0f) + ey); }
      H xq1 = cp_powp<H>(x, qe - O::c(1.0f));
      H A   = QkT * x * xq1;
      H u   = O::ld(p.v_0[i]) * O::ex(-A);
      H dA  = -(QkT * qe * xq1 * sg * pe * O::ex((pe - O::c(1.0f)) * lSR) / t0);
      H den = te * b + u * O::ld(p.B[i]);
      H rho = X[iRM+i];
      H rate = rho * b * u * te * b / den;
      gd[i]  = O::f(tau) < 0.0f ? -rate : rate;
      dgd[i] = J(O::f(rho * b * ((-(u * dA)) * (te * b) * (te * b) + u * u * b * O::ld(p.B[i])) / (den * den)));
      if (mode < 0) continue;
      for (int m = 0; m < 9; ++m) Lp[m] = Lp[m] + w_g * gd[i] * O::ld(p.P9_sl[9*i+m]);
      if (mode > 0) rank1<J>(Km, J(O::f(w_g)) * dgd[i], p.P9_sl + 9*i, p.Pv_sl + 6*i, p.P9_sl + 9*i, mode);
    }

    // twinning and gamma->epsilon: nucleation on the twin systems (Wong et al. 2016, |gd| in N0)
    if (NTW + NTR > 0) {
      const H mu = O::ld(p.mu), nu = O::ld(p.nu), pi = O::ld(p.pi), b1 = O::ld(p.b_sl[0]);
      const H Gsf = poly<H>(p, 0, T);
      const H x0  = mu * b1 * b1 * (O::c(2.0f) + nu) / (Gsf * O::c(8.0f) * pi * (O::c(1.0f) - nu));
      const H tau_r = mu * b1 / (O::c(2.0f) * pi) * (O::c(1.0f) / (x0 + O::ld(p.x_c)) + O::c(0.5f) / x0);
      const H aV = O::ld(p.V_cs) / kT;
      for (int kind = 0; kind < 2; ++kind) {
        if ((kind == 0 && NTW == 0) || (kind == 1 && NTR == 0)) continue;
        CP_THREAD H* fd = kind == 0 ? fd_tw : fd_tr;
        H tau_hat, Ln;
        if (kind == 0) {
          Ln = O::ld(p.L_tw);
          tau_hat = O::c(3.0f) * O::ld(p.b_tw) * mu / Ln + Gsf / (O::c(3.0f) * O::ld(p.b_tw));
        } else {
          Ln = O::ld(p.L_tr);
          tau_hat = O::c(3.0f) * O::ld(p.b_tr) * mu / Ln
                  + (Gsf + (O::ld(p.h) / O::ld(p.V_mol) - O::c(2.0f) * O::ld(p.rho)) * poly<H>(p, 1, T)) / (O::c(3.0f) * O::ld(p.b_tr));
        }
        for (int k = 0; k < 12; ++k) {
          fd[k] = O::c(0.0f);
          H tau = resolved<H>(p.Pv_tw + 6*k, Sm);
          if (!(O::f(tau) > 1.0e-8f && O::f(tau - tau_r) < 0.0f)) continue;
          H r  = kind == 0 ? O::ld(p.r_tw[k]) : O::ld(p.r_tr[k]);
          H z  = r * O::lg(tau_hat / tau);                                                          // (tau_hat/tau)^r = exp(z)
          H P = O::c(0.0f), dP = O::c(0.0f);
          if (O::f(z) < 80.0f) { H zr = O::ex(z); P = O::ex(-zr); dP = r * zr / tau * P; }
          const int s1 = p.pair[2*k], s2 = p.pair[2*k+1];
          H r1 = X[iRM+s1] + X[iRD+s1], r2 = X[iRM+s2] + X[iRD+s2];
          H N0 = (O::ab(gd[s2]) * r1 + O::ab(gd[s1]) * r2) / (O::c(3.0f) * Ln);
          H xn = aV * (tau_r - tau), Pn, dPn;                                                       // P_ncs = 1 - exp(-xn)
          if (O::f(xn) < 1.0e-3f) Pn = xn * (O::c(1.0f) - xn * (O::c(0.5f) - xn / O::c(6.0f)));
          else                    Pn = O::c(1.0f) - O::ex(-xn);
          dPn = aV * (Pn - O::c(1.0f));
          H Lam = kind == 0 ? D[dLT+k] : D[dLR+k];
          H V = O::ld(p.pi_4) * Lam * Lam * (kind == 0 ? O::ld(p.t_tw[k]) : O::ld(p.t_tr[k]));
          H c = w_g * V * Pn * P;
          fd[k] = c * N0;
          if (mode < 0 || O::f(fd[k]) == 0.0f && O::f(c) == 0.0f) continue;
          H gch = kind == 0 ? O::ld(p.gch_tw[k]) : O::ld(p.gamma_eps);
          for (int m = 0; m < 9; ++m) Lp[m] = Lp[m] + gch * fd[k] * O::ld(p.P9_tw[9*k+m]);
          if (mode == 0) continue;
          J dfd = J(O::f(w_g * V * N0 * (P * dPn + Pn * dP)));
          rank1<J>(Km, J(O::f(gch)) * dfd, p.P9_tw + 9*k, p.Pv_tw + 6*k, p.P9_tw + 9*k, mode);
          J cn = J(O::f(gch * c)) / J(3.0f * O::f(Ln));
          J a2 = cn * (O::f(gd[s2]) < 0.0f ? -dgd[s2] : dgd[s2]) * J(O::f(r1));
          J a1 = cn * (O::f(gd[s1]) < 0.0f ? -dgd[s1] : dgd[s1]) * J(O::f(r2));
          if (a2 != J(0)) rank1<J>(Km, a2, p.P9_tw + 9*k, p.Pv_sl + 6*s2, p.P9_sl + 9*s2, mode);
          if (a1 != J(0)) rank1<J>(Km, a1, p.P9_tw + 9*k, p.Pv_sl + 6*s1, p.P9_sl + 9*s1, mode);
        }
      }
    }

    // slip in the product regions
    region_rates<H,0,RT,FT,oT>(p, Sm, X, wT, gR,   nG ? nG        : nullptr, Lp, Km, mode);
    region_rates<H,1,RE,FE,oE>(p, Sm, X, wE, gR+4, nG ? nG + NG   : nullptr, Lp, Km, mode);
    region_rates<H,2,RA,FA,oA>(p, Sm, X, wA, gR+8, nG ? nG + 2*NG : nullptr, Lp, Km, mode);

    // alpha' at epsilon lamellae sheared on the other planes
    if (AL && fd_a) {
      H Sp[4], St = O::c(0.0f);
      for (int q = 0; q < 4; ++q) {
        Sp[q] = wE[q] * gR[4+q];
        if (NTR > 0) Sp[q] = Sp[q] + O::ld(p.gamma_eps) * (fd_tr[3*q] + fd_tr[3*q+1] + fd_tr[3*q+2]);
        St = St + Sp[q];
      }
      H ka = O::ld(p.k_alpha) / (O::c(1.0f) + O::ex(poly<H>(p, 2, T) / O::ld(p.G_s)));
      for (int q = 0; q < 4; ++q) fd_a[q] = ka * wE[q] * (St - Sp[q]);
    }
    if (AL && mode > 0) for (int q = 0; q < 9*(mode == 1 ? 6 : 9); ++q) Km[q] *= J(O::f(j2));       // dLp/dS = j^2 dLp/dMp
  }

  template<typename H>
  static void flow(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                   CP_THREAD const H* D, H T, H, CP_THREAD H* Lp, CP_THREAD typename cp_ops<H>::J* K, int mode) {
    H gd[NS], ft[NTW > 0 ? NTW : 1], fr[NTR > 0 ? NTR : 1], gR[12];
    rates<H>(p, S, X, D, T, gd, ft, fr, gR, nullptr, Lp, K, mode);
  }

  //> twintrip_dotState and twintrip_dotStateDiag (Jd: slip resistances of the regions)
  template<typename H>
  static void dot_state_dt(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                           CP_THREAD const H* D, H T, H, CP_THREAD H* dX, CP_THREAD float* Jd) {
    typedef cp_ops<H> O;
    H gd[NS], ft[NTW > 0 ? NTW : 1], fr[NTR > 0 ? NTR : 1], gR[12], fa[4];
    float nG[3*NG > 0 ? 3*NG : 1];
    rates<H>(p, S, X, D, T, gd, ft, fr, gR, fa, nullptr, nullptr, -1, Jd ? nG : nullptr);
    const H j2 = AL ? D[dJ2] : O::c(1.0f);
    H Sm[6];
    for (int k = 0; k < 6; ++k) Sm[k] = AL ? S[k] * j2 : S[k];
    const H mu = O::ld(p.mu), nu = O::ld(p.nu), pi = O::ld(p.pi), kT = O::ld(p.k_B) * T;
    const H lQ = -(O::ld(p.Q_cl) / kT);
    H bd = O::c(1.0f);
    for (int i = 0; i < NS; ++i) {
      H b = O::ld(p.b_sl[i]), dc = O::ld(p.d_caron[i]), Lam = D[dLS+i], ag = O::ab(gd[i]);
      H tau = resolved<H>(p.Pv_sl + 6*i, Sm), d_hat, form = O::c(0.0f), climb = O::c(0.0f);
      if (O::f(tau) == 0.0f || p.omit_dipoles) d_hat = Lam;
      else {
        d_hat = O::c(3.0f) * mu * b / (O::c(16.0f) * pi * O::ab(tau));
        if (O::f(d_hat - Lam) > 0.0f) d_hat = Lam;
        if (O::f(d_hat - dc) < 0.0f)  d_hat = dc;
        form = O::c(2.0f) * (d_hat - dc) / b * X[iRM+i] * ag;
      }
      if (O::f(d_hat - dc) != 0.0f) {
        H sc = O::ab(resolved<H>(p.Nv_sl + 6*i, Sm)) * b * b * b / kT, em1;                         // exp(|sigma_cl| b^3/kT) - 1
        em1 = O::f(sc) < 1.0e-3f ? sc * (O::c(1.0f) + sc * (O::c(0.5f) + sc / O::c(6.0f))) : O::ex(sc) - O::c(1.0f);
        if (p.extended) bd = O::c(24.0f) * pi * (O::c(1.0f) - nu) / (O::c(2.0f) + nu) * poly<H>(p, 0, T) / (mu * b);
        if (O::f(em1) > 0.0f) {
          H a = lQ + O::lg(em1);
          H v_cl = O::f(a) > -100.0f ? O::c(2.0f) * O::ld(p.omega) * bd * bd * O::ex(a) : O::c(0.0f);
          climb = O::c(4.0f) * v_cl * X[iRD+i] / (d_hat - dc);
        }
      }
      dX[iRM+i] = ag / (b * Lam) - form - O::c(2.0f) * dc / b * X[iRM+i] * ag;
      dX[iRD+i] = form - O::c(2.0f) * dc / b * X[iRD+i] * ag - climb;
      dX[iGS+i] = ag;
    }
    for (int k = 0; k < NTW; ++k) dX[iFT+k] = ft[k];
    for (int k = 0; k < NTR; ++k) dX[iFR+k] = fr[k];
    if (AL) for (int q = 0; q < 4; ++q) dX[iFA+q] = fa[q];
    if (Jd) for (int j = 0; j < ND; ++j) Jd[j] = 0.0f;
    region_dot<H,0,RT,FT,oT>(p, X, gR,   Jd ? nG : nullptr,        dX, Jd);
    region_dot<H,1,RE,FE,oE>(p, X, gR+4, Jd ? nG + NG : nullptr,   dX, Jd);
    region_dot<H,2,RA,FA,oA>(p, X, gR+8, Jd ? nG + 2*NG : nullptr, dX, Jd);
  }
  //> Voce hardening per family driven by the total slip rate of the region (regions_dot, regions_diag)
  template<typename H, int K, int NR, int NF, int OX>
  static void region_dot(CP_CONSTANT param& p, CP_THREAD const H* X, CP_THREAD const H* g, CP_THREAD const float* nG,
                         CP_THREAD H* dX, CP_THREAD float* Jd) {
    typedef cp_ops<H> O;
    if (NR == 0) return;
    for (int q = 0; q < 4; ++q) {
      for (int f = 0; f < NF; ++f) {
        const int o = OX + NF*q + f, fi = K*CP_TT_FMAX + f;
        H x = O::c(1.0f) - X[o] / O::ld(p.rxi_inf[fi]);
        dX[o] = O::ld(p.rh0[fi]) * cp_spow<H>(x, O::ld(p.ra[fi])) * g[q];
        if (Jd) Jd[o] = cp_model_tdt<1,1,1>::slope(cp_ldJ<float>(p.rh0[fi]), O::f(x), cp_ldJ<float>(p.ra[fi]),
                                                   cp_ldJ<float>(p.rxi_inf[fi]), O::f(g[q]), O::f(X[o]), nG[NF*q+f]);
      }
      dX[OX + 4*NF + q] = g[q];
    }
  }

  template<typename H>
  static void delta_state(CP_CONSTANT param&, CP_THREAD const H*, CP_THREAD const H*, CP_THREAD H*, CP_THREAD const H*, H) {}

  //> twintrip_dependentState and plastic_twintrip_homogenizedC (and j^2 of the alpha' eigenstrain)
  template<typename H>
  static void dependent(CP_CONSTANT param& p, CP_THREAD const H* X, CP_THREAD const H*, H, CP_THREAD H* D) {
    typedef cp_ops<H> O;
    H sft = O::c(0.0f), sfr = O::c(0.0f);
    for (int k = 0; k < NTW; ++k) sft = sft + X[iFT+k];
    for (int k = 0; k < NTR; ++k) sfr = sfr + X[iFR+k];
    H at = O::c(1.0f) - sft, ar = O::c(1.0f) - sfr;
    if (O::f(at - O::ld(p.w_floor)) < 0.0f) at = O::ld(p.w_floor);                                  // W_FLOOR
    if (O::f(ar - O::ld(p.w_floor)) < 0.0f) ar = O::ld(p.w_floor);
    const H Dg = O::ld(p.Dg);
    for (int i = 0; i < NS; ++i) {
      H fo = O::c(0.0f), hs = O::c(0.0f), il;
      for (int j = 0; j < NS; ++j) {
        H r = X[iRM+j] + X[iRD+j];
        fo = fo + O::ld(p.forest[i+12*j]) * r;
        hs = hs + O::ld(p.h_ss[i+12*j]) * r;
      }
      il = O::sq(fo) / O::ld(p.i_sl[i]);
      if (NTW > 0) { H s = O::c(0.0f); for (int k = 0; k < NTW; ++k) s = s + O::ld(p.h_st[i+12*k]) * X[iFT+k] / O::ld(p.t_tw[k]); il = il + s / at; }
      if (NTR > 0) { H s = O::c(0.0f); for (int k = 0; k < NTR; ++k) s = s + O::ld(p.h_sr[i+12*k]) * sfr / O::ld(p.t_tr[k]); il = il + s / ar; }
      D[dLS+i] = Dg / (O::c(1.0f) + Dg * il);
      D[dTP+i] = O::ld(p.mu) * O::ld(p.b_sl[i]) * O::sq(hs);
    }
    for (int k = 0; k < NTW; ++k) {
      H s = O::c(0.0f);
      for (int l = 0; l < NTW; ++l) s = s + O::ld(p.h_tt[k+12*l]) * X[iFT+l] / O::ld(p.t_tw[l]);
      D[dLT+k] = O::ld(p.i_tw) * Dg / (O::c(1.0f) + Dg * s / at);
    }
    for (int k = 0; k < NTR; ++k) {
      H s = O::c(0.0f);
      for (int l = 0; l < NTR; ++l) s = s + O::ld(p.h_rr[k+12*l]) * sfr / O::ld(p.t_tr[l]);
      D[dLR+k] = O::ld(p.i_tr) * Dg / (O::c(1.0f) + Dg * s / ar);
    }
    // homogenized stiffness
    H wT[4], wE[4], wA[4], ws = O::c(0.0f);
    fractions<H>(X, wT, wE, wA);
    for (int q = 0; q < 4; ++q) ws = ws + wT[q] + wE[q] + wA[q];
    if (!(O::f(ws) > 0.0f)) for (int q = 0; q < 36; ++q) D[q] = O::ld(p.C66[q]);                    // austenite only: exactly C
    else {
      H wg = O::c(1.0f) - ws, Mx[36];
      if (O::f(wg) < 0.0f) wg = O::c(0.0f);
      for (int q = 0; q < 36; ++q) Mx[q] = wg * O::ld(p.Mel[q]);
      for (int r = 0; r < 4; ++r) {
        if (O::f(wT[r]) > 0.0f) for (int q = 0; q < 36; ++q) Mx[q] = Mx[q] + wT[r] * O::ld(p.Mel[36*(1+r)+q]);
        if (O::f(wE[r]) > 0.0f) for (int q = 0; q < 36; ++q) Mx[q] = Mx[q] + wE[r] * O::ld(p.Mel[36*(5+r)+q]);
        if (O::f(wA[r]) > 0.0f) for (int q = 0; q < 36; ++q) Mx[q] = Mx[q] + wA[r] * O::ld(p.Mel[36*(9+r)+q]);
      }
      if (p.reuss) { if (!cp_inv66<H>(Mx, D)) for (int q = 0; q < 36; ++q) D[q] = O::c(NAN); }
      else for (int q = 0; q < 36; ++q) D[q] = Mx[q];
    }
    if (AL) D[dJ2] = O::c(1.0f) + fi_j2m1<H>(p, X);
  }
  //> j^2 - 1 of Fi = (1 + Delta_V sum(f_a))^(1/3) I
  template<typename H>
  static H fi_j2m1(CP_CONSTANT param& p, CP_THREAD const H* X) {
    typedef cp_ops<H> O;
    if (!AL) return O::c(0.0f);
    H s = O::c(0.0f);
    for (int q = 0; q < 4; ++q) s = s + X[iFA+q];
    H x = O::ld(p.Delta_V) * s;
    return O::ex(O::lg(O::c(1.0f) + x) * O::ld(p.two_thirds)) - O::c(1.0f);
  }

  static float atol(CP_CONSTANT param& p, int j) { return p.atol[j]; }
  template<typename H>
  static float stress_scale(CP_CONSTANT param& p, CP_THREAD const H*) { return 0.01f * cp_ldJ<float>(p.tau_0[0]); }
  //> no power law in the austenite: shrink S such that the fastest system does not exceed kappa (as dislotungsten)
  template<typename H>
  static float cap_ratio(CP_CONSTANT param& p, CP_THREAD const H* S, CP_THREAD const H* X, CP_THREAD const H*,
                         CP_THREAD const H* D, H T, float dt, float kappa) {
    typedef cp_ops<H> O;
    H gd[NS], ft[NTW > 0 ? NTW : 1], fr[NTR > 0 ? NTR : 1], gR[12], Lp[9];
    rates<H>(p, S, X, D, T, gd, ft, fr, gR, nullptr, Lp, nullptr, -1);
    float m = 0.0f;
    for (int i = 0; i < NS; ++i) m = CP_FMAX(m, O::f(O::ab(gd[i])));
    for (int q = 0; q < 12; ++q) m = CP_FMAX(m, O::f(gR[q]));
    float r = dt * m / kappa;
    return r > 1.0f ? DF_EXP(DF_LOG(r) / 10.0f) : 0.0f;
  }
};

#if !defined(__METAL_VERSION__)
#include <cstring>
#include <vector>
//> host: unpack the blob of plastic_tdt_GPUadd into the parameters of kernel capacities NSL, NTW, NFAM;
//> smap: GPU state component (relative to X) of each DAMASK state entry (incl. the 36 of D), sinit:
//> values of the padded GPU state components. Returns the number of blob values read (the caller
//> passes a blob padded to CP_BLOB_MAX and checks this against the actual length), -1 for an index out
//> of range.
inline int cp_tdt_pack(const int* dims, const double* C66, const double* blob, const double* atols,
                      int NSL, int NTW, int NFAM, cp_tdt_param& prm, std::vector<int>& smap, std::vector<double>& sinit) {
  const int ns = dims[0], nt = dims[1], nf = dims[2];
  cp_tdt_param* q = &prm;
  std::memset(q, 0, sizeof(*q));
  const double* b = blob;
  auto voigt = [&](const double* P9, cp_pair* Pv) {                                               // tau = sum_k S_k Pv_k
    for (int k = 0; k < 6; ++k) {
      int a = cp_voigt_i(k), c = cp_voigt_j(k);
      Pv[k] = cp_to_pair(k < 3 ? P9[CP_IDX(a,a)] : P9[CP_IDX(a,c)] + P9[CP_IDX(c,a)]);
    }
  };
  for (int i = 0; i < ns; ++i) { for (int k = 0; k < 9; ++k) q->P9[9*i+k] = cp_to_pair(b[9*i+k]); voigt(b+9*i, q->Pv+6*i); }
  b += 9*ns;
  for (int t = 0; t < nt; ++t) for (int i = 0; i < ns; ++i) {
    const double* P = b + 9*(i + ns*t);
    for (int k = 0; k < 9; ++k) q->P9[9*(NSL*(t+1)+i)+k] = cp_to_pair(P[k]);
    voigt(P, q->Pv + 6*(NSL*(t+1)+i));
  }
  b += 9*ns*nt;
  for (int t = 0; t < nt; ++t) { for (int k = 0; k < 9; ++k) q->P9_tw[9*t+k] = cp_to_pair(b[9*t+k]); voigt(b+9*t, q->Pv_tw+6*t); }
  b += 9*nt;
  for (int k = 0; k < 36*(1+nt); ++k) q->Mel[k] = cp_to_pair(b[k]);
  b += 36*(1+nt);
  auto vec = [&](cp_pair* dst, int n, int cap, double pad) { for (int i = 0; i < cap; ++i) dst[i] = cp_to_pair(i < n ? b[i] : pad); b += n; };
  vec(q->g0, ns, NSL, 0.0); vec(q->n, ns, NSL, 1.0); vec(q->a, ns, NSL, 1.0); vec(q->h0, ns, NSL, 0.0); vec(q->xi_inf, ns, NSL, 1.0);
  for (int i = 0; i < ns; ++i) {
    q->fam[i] = int(b[i]) - 1;
    if (q->fam[i] < 0 || q->fam[i] >= nf) return -1;                                                // index into the state
  }
  b += ns;
  vec(q->xi_inf_tw, nf, NFAM, 1.0); vec(q->h0_tw, nf, NFAM, 0.0); vec(q->a_tw, nf, NFAM, 1.0);
  vec(q->g0_tw, nt, NTW, 0.0); vec(q->n_tw, nt, NTW, 1.0); vec(q->gch, nt, NTW, 1.0); vec(q->dxi_nuc, nt, NTW, 0.0);
  vec(q->dxi_retw, nt, NTW, 0.0); vec(q->H_chi, nt, NTW, 0.0); vec(q->h0_dtw, nt, NTW, 0.0);
  for (int i = 0; i < ns; ++i) for (int j = 0; j < ns; ++j) q->h_ss[i+NSL*j] = cp_to_pair(b[i+ns*j]);
  b += ns*ns;
  for (int i = 0; i < ns; ++i) for (int k = 0; k < nt; ++k) q->h_st[i+NSL*k] = cp_to_pair(b[i+ns*k]);
  b += ns*nt;
  for (int k = 0; k < nt; ++k) for (int j = 0; j < ns; ++j) q->h_ts[k+NTW*j] = cp_to_pair(b[k+nt*j]);
  b += nt*ns;
  for (int k = 0; k < nt; ++k) for (int l = 0; l < nt; ++l) q->h_tt[k+NTW*l] = cp_to_pair(b[k+nt*l]);
  b += nt*nt;
  q->f_n = cp_to_pair(b[0]); q->f_sat = cp_to_pair(b[1]); q->f_min = cp_to_pair(b[2]);
  b += 3;
  for (int k = 0; k < 36; ++k) q->C66[k] = cp_to_pair(C66[k]);
  q->atol_xi = float(atols[0]); q->atol_gamma = float(atols[1]); q->atol_f = float(atols[2]);
  q->N_sl = ns; q->N_tw = nt; q->N_fam = nf; q->reuss = dims[3];

  // DAMASK state [xi_sl, gamma_sl, f, xi_tw, xi_dtw, xi_sl_tw(nf,nt), gamma_sl_tw | f0] + D(36) -> padded GPU state
  const int ND = 2*NSL + (4+NFAM)*NTW, NY = NTW;
  const int iG = NSL, iF = 2*NSL, iXT = iF+NTW, iXD = iXT+NTW, iXR = iXD+NTW, iGR = iXR+NFAM*NTW;
  smap.clear();
  sinit.assign(ND + NY + 36, 0.0);
  for (int i = 0; i < NSL; ++i) sinit[i] = 1.0;                                                // padded resistances
  for (int k = 0; k < NFAM*NTW; ++k) sinit[iXR+k] = 1.0;
  for (int k = 0; k < NTW; ++k) { sinit[iXT+k] = 1.0; sinit[iXD+k] = 1.0; }
  for (int i = 0; i < ns; ++i) smap.push_back(i);
  for (int i = 0; i < ns; ++i) smap.push_back(iG+i);
  for (int base : {iF, iXT, iXD}) for (int k = 0; k < nt; ++k) smap.push_back(base+k);
  for (int k = 0; k < nt; ++k) for (int f = 0; f < nf; ++f) smap.push_back(iXR + NFAM*k + f);
  for (int k = 0; k < nt; ++k) smap.push_back(iGR+k);
  for (int k = 0; k < nt; ++k) smap.push_back(ND+k);                                              // f0 (delta state)
  for (int k = 0; k < 36; ++k) smap.push_back(ND+NY+k);
  return int(b - blob);
}

//> supported twintrip configurations: N_tw, N_tr, alpha', twin/epsilon/alpha' region slip systems and families
#define CP_TT_CASES(X) X(0,0,0,0,0,0,0,0,0) \
                       X(0,12,0,0,0,0,0,0,0) \
                       X(0,12,1,0,0,0,0,0,0) \
                       X(0,12,1,0,0,0,0,12,1) \
                       X(0,12,1,0,0,0,0,24,2) \
                       X(0,12,0,0,0,3,1,0,0) \
                       X(0,12,1,0,0,3,1,0,0) \
                       X(0,12,1,0,0,3,1,12,1) \
                       X(0,12,1,0,0,3,1,24,2) \
                       X(0,12,0,0,0,6,2,0,0) \
                       X(0,12,1,0,0,6,2,0,0) \
                       X(0,12,1,0,0,6,2,12,1) \
                       X(0,12,1,0,0,6,2,24,2) \
                       X(0,12,0,0,0,12,3,0,0) \
                       X(0,12,1,0,0,12,3,0,0) \
                       X(0,12,1,0,0,12,3,12,1) \
                       X(0,12,1,0,0,12,3,24,2) \
                       X(12,0,0,0,0,0,0,0,0) \
                       X(12,12,0,0,0,0,0,0,0) \
                       X(12,12,1,0,0,0,0,0,0) \
                       X(12,12,1,0,0,0,0,12,1) \
                       X(12,12,1,0,0,0,0,24,2) \
                       X(12,12,0,0,0,3,1,0,0) \
                       X(12,12,1,0,0,3,1,0,0) \
                       X(12,12,1,0,0,3,1,12,1) \
                       X(12,12,1,0,0,3,1,24,2) \
                       X(12,12,0,0,0,6,2,0,0) \
                       X(12,12,1,0,0,6,2,0,0) \
                       X(12,12,1,0,0,6,2,12,1) \
                       X(12,12,1,0,0,6,2,24,2) \
                       X(12,12,0,0,0,12,3,0,0) \
                       X(12,12,1,0,0,12,3,0,0) \
                       X(12,12,1,0,0,12,3,12,1) \
                       X(12,12,1,0,0,12,3,24,2) \
                       X(12,0,0,12,1,0,0,0,0) \
                       X(12,12,0,12,1,0,0,0,0) \
                       X(12,12,1,12,1,0,0,0,0) \
                       X(12,12,1,12,1,0,0,12,1) \
                       X(12,12,1,12,1,0,0,24,2) \
                       X(12,12,0,12,1,3,1,0,0) \
                       X(12,12,1,12,1,3,1,0,0) \
                       X(12,12,1,12,1,3,1,12,1) \
                       X(12,12,1,12,1,3,1,24,2) \
                       X(12,12,0,12,1,6,2,0,0) \
                       X(12,12,1,12,1,6,2,0,0) \
                       X(12,12,1,12,1,6,2,12,1) \
                       X(12,12,1,12,1,6,2,24,2) \
                       X(12,12,0,12,1,12,3,0,0) \
                       X(12,12,1,12,1,12,3,0,0) \
                       X(12,12,1,12,1,12,3,12,1) \
                       X(12,12,1,12,1,12,3,24,2)

//> host: unpack the blob of plastic_twintrip_GPUadd (layout documented there);
//> dims = [N_tw, N_tr, alpha, N_sys/N_fam of twin, epsilon, alpha' regions (0: elastic), Reuss, sizeState].
//> Returns the number of blob values read (see cp_tdt_pack), -1 for an index out of range.
inline int cp_twintrip_pack(const int* dims, const double* C66, const double* blob, const double* atol,
                             cp_twintrip_param& prm) {
  cp_twintrip_param* q = &prm;
  std::memset(q, 0, sizeof(*q));
  const int nt = dims[0], nr = dims[1], NR[3] = {dims[3], dims[5], dims[7]}, NF[3] = {dims[4], dims[6], dims[8]};
  const double* b = blob;
  auto voigt = [&](const double* P9, cp_pair* Pv) {                                               // tau = sum_k S_k Pv_k
    for (int k = 0; k < 6; ++k) {
      int a = cp_voigt_i(k), c = cp_voigt_j(k);
      Pv[k] = cp_to_pair(k < 3 ? P9[CP_IDX(a,a)] : P9[CP_IDX(a,c)] + P9[CP_IDX(c,a)]);
    }
  };
  auto vec = [&](cp_pair* dst, int n) { for (int i = 0; i < n; ++i) dst[i] = cp_to_pair(b[i]); b += n; };
  vec(q->Mel, 13*36);
  for (int i = 0; i < 12; ++i) { for (int k = 0; k < 9; ++k) q->P9_sl[9*i+k] = cp_to_pair(b[9*i+k]); voigt(b+9*i, q->Pv_sl+6*i); }
  b += 108;
  for (int i = 0; i < 12; ++i) {
    double nn[9];
    for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) nn[CP_IDX(a,c)] = b[3*i+a]*b[3*i+c];
    voigt(nn, q->Nv_sl+6*i);
  }
  b += 36;
  for (int i = 0; i < 12; ++i) { for (int k = 0; k < 9; ++k) q->P9_tw[9*i+k] = cp_to_pair(b[9*i+k]); voigt(b+9*i, q->Pv_tw+6*i); }
  b += 108;
  vec(q->forest, 144); vec(q->h_ss, 144);
  if (nt) vec(q->h_st, 144);
  if (nr) vec(q->h_sr, 144);
  if (nt) vec(q->h_tt, 144);
  if (nr) vec(q->h_rr, 144);
  vec(q->b_sl, 12); vec(q->Q_sl, 12); vec(q->v_0, 12); vec(q->i_sl, 12); vec(q->p_sl, 12); vec(q->q_sl, 12);
  vec(q->tau_0, 12); vec(q->B, 12); vec(q->d_caron, 12);
  if (nt) { vec(q->t_tw, 12); vec(q->r_tw, 12); vec(q->gch_tw, 12); }
  if (nr) { vec(q->t_tr, 12); vec(q->r_tr, 12); }
  for (int k = 0; k < 24; ++k) {
    q->pair[k] = int(b[k]) - 1;
    if (q->pair[k] < 0 || q->pair[k] >= 12) return -1;                                              // slip system of the nucleation pair
  }
  b += 24;
  for (int k = 0; k < 3; ++k) {
    if (NR[k] == 0) continue;
    for (int pl = 0; pl < 4; ++pl) for (int j = 0; j < NR[k]; ++j) {
      const double* P = b + 9*(j + NR[k]*pl);
      const int sidx = (4*k+pl)*CP_TT_RMAX + j;
      for (int m = 0; m < 9; ++m) q->rP9[9*sidx+m] = cp_to_pair(P[m]);
      voigt(P, q->rPv + 6*sidx);
    }
    b += 36*NR[k];
    for (int j = 0; j < NR[k]; ++j) {
      q->rfam[k*CP_TT_RMAX+j] = int(b[j]) - 1;
      if (q->rfam[k*CP_TT_RMAX+j] < 0 || q->rfam[k*CP_TT_RMAX+j] >= NF[k]) return -1;            // family of a region slip system
    }
    b += NR[k];
    vec(q->rg0 + k*CP_TT_FMAX, NF[k]); vec(q->rn + k*CP_TT_FMAX, NF[k]); vec(q->rxi_inf + k*CP_TT_FMAX, NF[k]);
    vec(q->rh0 + k*CP_TT_FMAX, NF[k]); vec(q->ra + k*CP_TT_FMAX, NF[k]);
  }
  for (int k = 0; k < 3; ++k) {
    q->poly_n[k] = int(b[0]); q->poly_x0[k] = cp_to_pair(b[1]);
    for (int o = 0; o < 5; ++o) q->poly[5*k+o] = cp_to_pair(b[2+o]);
    b += 7;
  }
  cp_pair* sc[] = {&q->b_tw, &q->b_tr, &q->Q_cl, &q->omega, &q->Dg, &q->i_tw, &q->i_tr, &q->L_tw, &q->L_tr, &q->x_c,
                   &q->V_cs, &q->h, &q->V_mol, &q->rho, &q->delta_sl, &q->k_alpha, &q->G_s, &q->Delta_V, &q->mu, &q->nu};
  for (cp_pair* d : sc) *d = cp_to_pair(*b++);
  q->extended = int(b[0]); q->omit_dipoles = int(b[1]);
  b += 2;
  q->k_B = cp_to_pair(1.380649e-23); q->pi = cp_to_pair(3.14159265358979323846); q->pi_4 = cp_to_pair(0.25*3.14159265358979323846);
  q->gamma_eps = cp_to_pair(std::sqrt(0.125)); q->two_thirds = cp_to_pair(2.0/3.0); q->w_floor = cp_to_pair(1.0e-3);
  for (int k = 0; k < 36; ++k) q->C66[k] = cp_to_pair(C66[k]);
  for (int j = 0; j < dims[10] && j < CP_TT_NDMAX; ++j) q->atol[j] = float(atol[j]);
  q->reuss = dims[9];
  return int(b - blob);                                                                             // blob values read
}
#endif

#endif
