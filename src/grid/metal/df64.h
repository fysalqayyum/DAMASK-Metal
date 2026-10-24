// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief double-float ("df64") arithmetic: an unevaluated sum hi+lo of two IEEE floats, giving
//> ~48 bit mantissa on hardware without FP64 (Apple GPUs).
//> @details Single source for Metal Shading Language and C++ (CPU validation).
//> Requires IEEE-exact float arithmetic: compile MSL with MTLMathModeSafe, C++ with -ffp-contract=off
//> and without -ffast-math. Exponent range is that of float (~1e+-38); full ~48 bit precision
//> only for magnitudes in ~[1e-30,1e38] (the lo word becomes subnormal below).
//> Algorithms: Dekker/Knuth error-free transformations, cf. Hida, Li & Bailey (QD library).
//--------------------------------------------------------------------------------------------------
#ifndef DAMASK_DF64_H
#define DAMASK_DF64_H

#ifdef __METAL_VERSION__
  #include <metal_stdlib>
  #define DF_THREAD thread
  #define DF_FMA(a,b,c)  metal::precise::fma(a,b,c)
  #define DF_SQRT(a)     metal::precise::sqrt(a)
  #define DF_LOG(a)      metal::precise::log(a)
  #define DF_EXP(a)      metal::precise::exp(a)
  #define DF_RINT(a)     metal::rint(a)
  #define DF_LDEXP(a,e)  metal::ldexp(a,e)
  #define DF_ISFINITE(a) metal::isfinite(a)
#else
  #include <cmath>
  #define DF_THREAD
  #define DF_FMA(a,b,c)  std::fma(a,b,c)
  #define DF_SQRT(a)     std::sqrt(a)
  #define DF_LOG(a)      std::log(a)
  #define DF_EXP(a)      std::exp(a)
  #define DF_RINT(a)     std::rint(a)
  #define DF_LDEXP(a,e)  std::ldexp(a,e)
  #define DF_ISFINITE(a) std::isfinite(a)
#endif


struct df64 {
  float hi;
  float lo;
};

inline df64 df_make(float hi, float lo) { df64 r; r.hi = hi; r.lo = lo; return r; }
inline df64 df_from(float a) { return df_make(a, 0.0f); }

//--------------------------------------------------------------------------------------------------
// error-free transformations
inline df64 quick_two_sum(float a, float b) {                                                      // requires |a| >= |b|
  float s = a + b;
  float e = b - (s - a);
  return df_make(s, e);
}

inline df64 two_sum(float a, float b) {
  float s  = a + b;
  float bb = s - a;
  float e  = (a - (s - bb)) + (b - bb);
  return df_make(s, e);
}

inline df64 two_prod(float a, float b) {
  float p = a * b;
  float e = DF_FMA(a, b, -p);
  return df_make(p, e);
}

//--------------------------------------------------------------------------------------------------
// arithmetic
inline df64 operator+(df64 a, df64 b) {
  df64 s = two_sum(a.hi, b.hi);
  df64 t = two_sum(a.lo, b.lo);
  s.lo += t.hi;
  s = quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return quick_two_sum(s.hi, s.lo);
}

inline df64 operator-(df64 a) { return df_make(-a.hi, -a.lo); }
inline df64 operator-(df64 a, df64 b) { return a + (-b); }

inline df64 operator*(df64 a, df64 b) {
  df64 p = two_prod(a.hi, b.hi);
  p.lo += a.hi * b.lo + a.lo * b.hi;
  return quick_two_sum(p.hi, p.lo);
}

inline df64 operator*(df64 a, float b) {
  df64 p = two_prod(a.hi, b);
  p.lo += a.lo * b;
  return quick_two_sum(p.hi, p.lo);
}

inline df64 operator/(df64 a, df64 b) {
  float q1 = a.hi / b.hi;
  df64  r  = a - b * q1;
  float q2 = r.hi / b.hi;
  r = r - b * q2;
  float q3 = r.hi / b.hi;
  return quick_two_sum(q1, q2) + df_from(q3);
}

inline df64 operator+(df64 a, float b) { return a + df_from(b); }
inline df64 operator-(df64 a, float b) { return a - df_from(b); }
inline df64 operator/(df64 a, float b) { return a / df_from(b); }

inline DF_THREAD df64& operator+=(DF_THREAD df64& a, df64 b) { a = a + b; return a; }
inline DF_THREAD df64& operator-=(DF_THREAD df64& a, df64 b) { a = a - b; return a; }
inline DF_THREAD df64& operator*=(DF_THREAD df64& a, df64 b) { a = a * b; return a; }

inline bool operator<(df64 a, df64 b)  { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
inline bool operator>(df64 a, df64 b)  { return b < a; }
inline bool operator<=(df64 a, df64 b) { return !(b < a); }
inline bool operator>=(df64 a, df64 b) { return !(a < b); }

//--------------------------------------------------------------------------------------------------
// functions
inline float df_to_float(df64 a) { return a.hi + a.lo; }
inline df64  df_abs(df64 a) { return a.hi < 0.0f ? -a : a; }
inline bool  df_isfinite(df64 a) { return DF_ISFINITE(a.hi) && DF_ISFINITE(a.lo); }
inline df64  df_ldexp(df64 a, int e) { return df_make(DF_LDEXP(a.hi, e), DF_LDEXP(a.lo, e)); }

inline df64 df_sqrt(df64 a) {
  if (a.hi <= 0.0f) return df_from(0.0f);
  float s = DF_SQRT(a.hi);
  df64 r = a - two_prod(s, s);                                                                      // one Newton step
  return quick_two_sum(s, r.hi / (2.0f*s));
}

// ln(2) split into hi+lo floats
#define DF_LN2_HI   0.693147182464599609375f
#define DF_LN2_LO  -1.904654323148236e-09f

//> exp(a): a = k*ln2 + r, exp(r) via expm1 polynomial on r/2^3 followed by 3 squarings
//> (few squarings: each one doubles the relative error). expm1(r) = r(1 + r/2 + r^2/3! + ... + r^8/9!):
//> Horner in df64 for the leading terms, the tail (|.| <= 2e-9 relative) in float; no divisions.
inline df64 df_exp(df64 a) {
  if (a.hi >  88.7f) return df_from(INFINITY);
  if (a.hi < -87.3f) return df_from(0.0f);
  float k = DF_RINT(a.hi / DF_LN2_HI);
  df64 r = a - df_make(DF_LN2_HI, DF_LN2_LO) * k;                                                   // |r| <= ln2/2
  r = df_ldexp(r, -3);                                                                              // |r| <= 0.044
  float x = r.hi;
  float q = DF_FMA(x, DF_FMA(x, DF_FMA(x, DF_FMA(x, 2.755731884e-06f, 2.480158764e-05f),
                                       1.984127011e-04f), 1.388888923e-03f), 8.333333768e-03f);     // 1/5! + ... + r^4/9!
  df64 p = r * q + df_make(4.166666791e-02f, -1.241763470e-09f);                                    // 1/4!
  p = p * r + df_make(1.666666716e-01f, -4.967053879e-09f);                                         // 1/3!
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  df64 s = p * r;
  for (int i = 0; i < 3; ++i) s = s * (s + 2.0f);                                                   // expm1(2x) = expm1(x)*(expm1(x)+2)
  return df_ldexp(s + 1.0f, int(k));
}

//> log(a) for a > 0: float estimate y, refined by one Newton step y + a*exp(-y) - 1
inline df64 df_log(df64 a) {
  if (!(a.hi > 0.0f)) return df_from(a.hi == 0.0f ? -INFINITY : NAN);
  df64 y = df_from(DF_LOG(a.hi));
  return y + a * df_exp(-y) - 1.0f;
}

//> a^b for a > 0
inline df64 df_pow(df64 a, df64 b) { return df_exp(b * df_log(a)); }

#endif
