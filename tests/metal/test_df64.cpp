// CPU validation of df64.h against long double reference
#include <cstdio>
#include <cmath>
#include <random>
#include "../../src/grid/metal/df64.h"

static df64 to_df(long double x) { float hi = float(x); return df_make(hi, float(x - (long double)hi)); }
static long double val(df64 a) { return (long double)a.hi + (long double)a.lo; }
static long double relerr(df64 a, long double ref) { return fabsl(val(a) - ref) / fmaxl(fabsl(ref), 1e-300L); }

int main() {
  std::mt19937_64 g(42);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  long double e_add=0,e_mul=0,e_div=0,e_sqrt=0,e_exp=0,e_log=0,e_pow=0;
  for (int i = 0; i < 200000; ++i) {
    long double a = (long double)u(g)*1e3L, b = (long double)u(g)*1e2L+1e-3L;
    df64 A = to_df(a), B = to_df(b);
    long double ra = val(A), rb = val(B);
    e_add  = fmaxl(e_add,  relerr(A+B, ra+rb) * fabsl(ra+rb)/fmaxl(fabsl(ra),fabsl(rb)));
    e_mul  = fmaxl(e_mul,  relerr(A*B, ra*rb));
    e_div  = fmaxl(e_div,  relerr(A/B, ra/rb));
    e_sqrt = fmaxl(e_sqrt, relerr(df_sqrt(df_abs(A)), sqrtl(fabsl(ra))));
    long double x = (long double)u(g)*70.0L; df64 X = to_df(x); long double rx = val(X);
    e_exp  = fmaxl(e_exp,  relerr(df_exp(X), expl(rx)));
    long double y = fabsl(a)+1e-6L; df64 Y = to_df(y); long double ry = val(Y);
    e_log  = fmaxl(e_log,  fabsl(val(df_log(Y)) - logl(ry)) / fmaxl(1.0L, fabsl(logl(ry))));
    long double r = 0.05L + 2.0L*fabsl((long double)u(g)); df64 R = to_df(r); long double rr = val(R);
    e_pow  = fmaxl(e_pow,  relerr(df_pow(R, df_from(20.0f)), powl(rr, 20.0L)));
  }
  printf("max rel err  add %.2Le  mul %.2Le  div %.2Le  sqrt %.2Le\n", e_add, e_mul, e_div, e_sqrt);
  printf("             exp %.2Le  log %.2Le  pow(x,20) %.2Le\n", e_exp, e_log, e_pow);
  printf("(float eps = %.2e, double eps = %.2e)\n", 1.19e-7, 2.22e-16);
  return (e_mul < 1e-13 && e_exp < 1e-12 && e_pow < 1e-11) ? 0 : 1;
}
