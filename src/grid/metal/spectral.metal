// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Fourier-space operators of the spectral solver (FP32), see spectral_utilities.f90
//> @details Fourier fields are stored component-major: f[c*M + w], c = l+3m (Fortran order of the
//> 3x3 tensor), w = x + h*(y + n2*z), h = n1/2+1, M = h*n2*n3 (non-transposed, single rank).
//> Wave vectors xi[3*w + l] are DAMASK's xi1st.
//--------------------------------------------------------------------------------------------------
#include <metal_stdlib>
using namespace metal;

struct gamma_params {
  uint  M;
  uint  pad[3];
  float C_hat[81];                                                                                  // C_ref/c0, (l,a,m,b) at l+3a+9m+27b
  float inv_c0;
  float aim[9];                                                                                     // fieldAim (3,3), Fortran order
};

inline float2 cmul (float2 a, float2 b) { return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
inline float2 cconj(float2 a)           { return float2(a.x, -a.y); }
inline float2 cdiv (float2 a, float2 b) { return cmul(a, cconj(b)) / dot(b, b); }

//--------------------------------------------------------------------------------------------------
//> f <- Gamma_hat : f with Gamma_hat(l,m,n,o) = inv(C_ref:xiDyad)(l,n) xiDyad(o,m),
//> xiDyad(a,b) = conj(-xi_a) xi_b, i.e. f <- inv(C_ref:xiDyad) . f . xiDyad.
//> Gamma_hat is homogeneous of degree 0 in xi and -1 in C_ref, so it is evaluated with unit xi and
//> C_ref/c0 (FP32-safe: the unscaled determinant reaches ~1e63) and scaled by 1/c0.
//--------------------------------------------------------------------------------------------------
kernel void gamma_convolution(device       float2*       f  [[buffer(0)]],
                              device const float2*       xi [[buffer(1)]],
                              constant     gamma_params& p  [[buffer(2)]],
                              uint w [[thread_position_in_grid]]) {
  if (w >= p.M) return;

  if (w == 0) {                                                                                     // DC: impose the average (cells3Offset == 0)
    for (uint c = 0; c < 9; ++c) f[c*p.M] = float2(p.aim[c], 0.0f);
    return;
  }

  float2 x[3] = {xi[3*w], xi[3*w+1], xi[3*w+2]};
  float n2 = dot(x[0], x[0]) + dot(x[1], x[1]) + dot(x[2], x[2]);
  if (!(n2 > 0.0f)) {
    for (uint c = 0; c < 9; ++c) f[c*p.M + w] = float2(0.0f);
    return;
  }
  float s = rsqrt(n2);
  for (uint l = 0; l < 3; ++l) x[l] *= s;

  float2 xd[9];                                                                                     // (a,b) at a+3b
  for (uint b = 0; b < 3; ++b)
    for (uint a = 0; a < 3; ++a)
      xd[a+3*b] = cmul(-cconj(x[a]), x[b]);

  float2 t[9];                                                                                      // C_hat : xiDyad, (l,m) at l+3m
  for (uint m = 0; m < 3; ++m)
    for (uint l = 0; l < 3; ++l) {
      float2 acc = float2(0.0f);
      for (uint b = 0; b < 3; ++b)
        for (uint a = 0; a < 3; ++a)
          acc += p.C_hat[l+3*a+9*m+27*b] * xd[a+3*b];
      t[l+3*m] = acc;
    }

  float detRe = t[0].x*(t[4].x*t[8].x - t[7].x*t[5].x)                                              // as DAMASK: invertibility of Re(C:xiDyad)
              - t[3].x*(t[1].x*t[8].x - t[7].x*t[2].x)
              + t[6].x*(t[1].x*t[5].x - t[4].x*t[2].x);
  if (!(abs(detRe) > 1.0e-30f)) {
    for (uint c = 0; c < 9; ++c) f[c*p.M + w] = float2(0.0f);
    return;
  }

  float2 adj[9];                                                                                    // complex inverse via adjugate
  adj[0] =  cmul(t[4],t[8]) - cmul(t[7],t[5]);
  adj[1] = -(cmul(t[1],t[8]) - cmul(t[7],t[2]));
  adj[2] =  cmul(t[1],t[5]) - cmul(t[4],t[2]);
  adj[3] = -(cmul(t[3],t[8]) - cmul(t[6],t[5]));
  adj[4] =  cmul(t[0],t[8]) - cmul(t[6],t[2]);
  adj[5] = -(cmul(t[0],t[5]) - cmul(t[3],t[2]));
  adj[6] =  cmul(t[3],t[7]) - cmul(t[6],t[4]);
  adj[7] = -(cmul(t[0],t[7]) - cmul(t[6],t[1]));
  adj[8] =  cmul(t[0],t[4]) - cmul(t[3],t[1]);
  float2 det = cmul(t[0],adj[0]) + cmul(t[3],adj[1]) + cmul(t[6],adj[2]);
  float2 sdet = cdiv(float2(p.inv_c0, 0.0f), det);

  float2 v[9], g[9];
  for (uint c = 0; c < 9; ++c) v[c] = f[c*p.M + w];
  for (uint m = 0; m < 3; ++m)                                                                      // g = f . xiDyad
    for (uint n = 0; n < 3; ++n)
      g[n+3*m] = cmul(v[n], xd[3*m]) + cmul(v[n+3], xd[1+3*m]) + cmul(v[n+6], xd[2+3*m]);
  for (uint m = 0; m < 3; ++m)                                                                      // f = inv(t)/c0 . g
    for (uint l = 0; l < 3; ++l)
      v[l+3*m] = cmul(sdet, cmul(adj[l], g[3*m]) + cmul(adj[l+3], g[1+3*m]) + cmul(adj[l+6], g[2+3*m]));
  for (uint c = 0; c < 9; ++c) f[c*p.M + w] = v[c];
}
