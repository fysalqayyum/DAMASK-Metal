// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief GPU entry points of the constitutive update (cp_core.h), one thread per material point.
//> @details Compiled once per policy type on first use: the host prepends
//>   #define CP_MODEL_TYPE <policy>   (cp_model_pheno<12>, cp_model_tdt<18,12,6>, ...)
//> Residuals/state in df64, Jacobian and LU in float, tangent in float. Component-major (SoA) device
//> arrays, see cp_layout. Concatenated after df64.h, cp_kernel.h, cp_slim.h, cp_core.h,
//> cp_models_composite.h (cmake/EmbedMetal.cmake).
//--------------------------------------------------------------------------------------------------
#include "df64.h"
#include "cp_kernel.h"
#include "cp_slim.h"
#include "cp_core.h"
#include "cp_models_composite.h"

struct cp_launch {
  uint  n;                                                                                          // number of points in this chunk (= array stride)
  uint  pad;
  df64  dt;
};

//> struct sizes as seen by the GPU compiler (checked against the host at initialisation)
kernel void cp_sizes(device uint* s [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i > 0) return;
  s[0] = sizeof(cp_pheno_param);
  s[1] = sizeof(cp_numerics);
  s[2] = sizeof(cp_launch);
  s[3] = sizeof(df64);
  s[4] = sizeof(cp_kineh_param);
  s[5] = sizeof(cp_dtung_param);
  s[6] = sizeof(cp_phenotw_param);
  s[7] = sizeof(cp_tdt_param);
  s[8] = sizeof(cp_twintrip_param);
}

#if defined(CP_MODEL_TYPE)
typedef CP_MODEL_TYPE cp_M;

kernel void cp_integrate_kernel(device const df64*            in   [[buffer(0)]],
                                device       df64*            out  [[buffer(1)]],
                                device       int*             stat [[buffer(2)]],
                                constant     cp_M::param&     p    [[buffer(3)]],
                                constant     cp_numerics&     num  [[buffer(4)]],
                                constant     cp_launch&       l    [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
  if (i >= l.n) return;
  cp_soa_const<df64> vi = {in, l.n, i};
  cp_soa<df64>       vo = {out, l.n, i};
  int its = 0;
#if defined(CP_FAST)
  stat[i] = cp_core_integrate_fast<df64,cp_M>(p, num, l.dt, vi, vo, its);                          // FP32 kinetics, df64 accumulation
#else
  stat[i] = cp_core_integrate<df64,cp_M>(p, num, l.dt, vi, vo, its);
#endif
  stat[l.n + i] = its;
}

//> tangent at the stored (converged, delta-updated) state, as phase_mechanical_dPdF
kernel void cp_tangent_kernel(device const df64*            in   [[buffer(0)]],
                              device const df64*            out  [[buffer(1)]],
                              device const int*             stat [[buffer(2)]],
                              constant     cp_M::param&     p    [[buffer(3)]],
                              constant     cp_launch&       l    [[buffer(5)]],
                              device       float*           dPdF [[buffer(6)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= l.n || stat[i] != CP_OK) return;
  typedef cp_layout<cp_M> L;
  const int NY = cp_M::NY > 0 ? cp_M::NY : 1;
  const int NDEP = cp_M::NDEP > 0 ? cp_M::NDEP : 1;
  float F[9], Fp0[9], iFp[9], Fe[9], S[6], X[cp_M::ND], Y[NY], D[NDEP], t[81];
  for (int k = 0; k < 9; ++k) {
    F[k]   = df_to_float(in[(L::F+k)*l.n + i]);    Fp0[k] = df_to_float(in[(L::Fp0+k)*l.n + i]);
    iFp[k] = df_to_float(out[(L::iFp+k)*l.n + i]); Fe[k]  = df_to_float(out[(L::Fe+k)*l.n + i]);
  }
  for (int k = 0; k < 6; ++k)         S[k] = df_to_float(out[(L::S+k)*l.n + i]);
  for (int j = 0; j < cp_M::ND; ++j)  X[j] = df_to_float(out[(L::X+j)*l.n + i]);
  for (int j = 0; j < cp_M::NY; ++j)  Y[j] = df_to_float(out[(L::Y+j)*l.n + i]);
  for (int j = 0; j < cp_M::NDEP; ++j) D[j] = df_to_float(out[(L::D+j)*l.n + i]);                  // of the last stress solve, as phase_mechanical_dPdF
  float T = df_to_float(in[L::T*l.n + i]);
  bool ok = cp_core_tangent<float,cp_M>(p, df_to_float(l.dt), F, Fp0, iFp, Fe, S, X, Y, D, T, t);
  for (int k = 0; k < 81; ++k) dPdF[k*l.n + i] = ok ? t[k] : NAN;
}
#endif
