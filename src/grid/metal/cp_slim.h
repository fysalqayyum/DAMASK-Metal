// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Component-major ("SoA") data access and helpers of the GPU constitutive update (cp_soa,
//> cp_solve99, cp_voigt). The slim reference pipeline (cp_integrate_slim, cp_tangent) built on them
//> is in tests/metal/ref_pheno_kernel.h.
//> @details Design for a small per-thread footprint (registers instead of spilled stack):
//>  - no per-point struct in private memory: inputs/outputs are component-major ("SoA") device
//>    arrays, element (c,i) at c*n + i (coalesced across a SIMD group)
//>  - inputs are re-read from device memory when needed instead of being kept alive
//>  - outputs are written directly when they become available
//>  - the tangent is a separate kernel (float), evaluated at the converged state
//> Component layout of the input/output arrays: cp_layout in cp_core.h.
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


#endif
