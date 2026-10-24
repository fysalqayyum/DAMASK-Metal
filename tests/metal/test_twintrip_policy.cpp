// GPU policy cp_model_twintrip (cp_models_composite.h) in double on the CPU vs. the Fortran model
// (phase_mechanical_plastic_twintrip.f90) for synthetic states and stresses written by DAMASK with
// DAMASK_TWINTRIP_GOLDEN=<file> (plastic_twintrip_GPUadd): dependent state, homogenized stiffness, Fi,
// Lp, dLp/dMp, dotState, its diagonal; plus finite-difference check of the policy's dLp/dS (Voigt).
// build: clang++ -std=c++17 -O2 -ffp-contract=off test_twintrip_policy.cpp
// usage: test_twintrip_policy <golden file>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "../../src/grid/metal/cp_models_composite.h"

struct Reader {
  FILE* f;
  template<typename T> void get(T* v, size_t n) { if (std::fread(v, sizeof(T), n, f) != n) { std::printf("short file\n"); std::exit(2); } }
};

static double rel(const double* a, const double* b, int n, double floor = 0.0) {                    // |a-b|/max(|b|,floor)
  double d = 0.0, s = 0.0;
  for (int k = 0; k < n; ++k) { d += (a[k]-b[k])*(a[k]-b[k]); s += b[k]*b[k]; }
  return std::sqrt(d) / std::max(std::sqrt(s), floor);
}

template<class M>
static int run(Reader& r, const int* dims, const std::vector<double>& C66, const std::vector<double>& blob,
               const std::vector<double>& atol) {
  static cp_twintrip_param p;
  const int ND = M::ND, nt = dims[0], nr = dims[1];
  if (ND != dims[10]) { std::printf("state size: policy %d, DAMASK %d\n", ND, dims[10]); return 1; }
  std::vector<double> pb(CP_BLOB_MAX, 0.0);                                                         // padded as in the bridge
  std::copy(blob.begin(), blob.end(), pb.begin());
  const int used = cp_twintrip_pack(dims, C66.data(), pb.data(), atol.data(), p);
  if (used < 0) { std::printf("blob: index out of range\n"); return 1; }
  if (used != int(blob.size())) { std::printf("blob: %zu values, dims need %d\n", blob.size(), used); return 1; }
  int ns_ = 0; r.get(&ns_, 1);
  if (ns_ < 0 || ns_ > 100000) { std::printf("invalid sample count %d\n", ns_); return 1; }
  double eD = 0, eC = 0, eJ = 0, eL = 0, eK = 0, eX = 0, eJd = 0, eFD = 0;
  int worst_X = -1, worst_s = -1;
  for (int s = 0; s < ns_; ++s) {
    std::vector<double> X(ND), dotF(ND), JdF(ND), Dref(60 + nt + nr), Cref(36);
    double Mp[9], T, j, Lp_ref[9], K_ref[81];
    r.get(X.data(), ND); r.get(Mp, 9); r.get(&T, 1);
    r.get(Dref.data() + 36, 12); r.get(Dref.data() + 48, nt); r.get(Dref.data() + 48 + nt, nr); r.get(Dref.data() + 48 + nt + nr, 12);
    r.get(Cref.data(), 36); r.get(&j, 1); r.get(Lp_ref, 9); r.get(K_ref, 81); r.get(dotF.data(), ND); r.get(JdF.data(), ND);

    double D[M::NDEP];
    M::template dependent<double>(p, X.data(), nullptr, T, D);
    eD = std::max(eD, rel(D + 36, Dref.data() + 36, 24 + nt + nr));
    eC = std::max(eC, rel(D, Cref.data(), 36));
    const double j2m1 = cp_fi<M>::template j2m1<double>(p, X.data());
    eJ = std::max(eJ, std::fabs(std::sqrt(1.0 + j2m1) - j));
    double S[6], j2 = 1.0 + j2m1;                                                                   // S = Mp/j^2 (Voigt)
    for (int k = 0; k < 6; ++k) S[k] = 0.5*(Mp[CP_IDX(cp_voigt_i(k),cp_voigt_j(k))] + Mp[CP_IDX(cp_voigt_j(k),cp_voigt_i(k))]) / j2;
    double Lp[9], K[81], G[54];
    M::template flow<double>(p, S, X.data(), nullptr, D, T, 0.0, Lp, K, 2);
    for (int q = 0; q < 81; ++q) K[q] /= j2;                                                        // policy: dLp/dS = j^2 dLp/dMp
    double nL = 0; for (int k = 0; k < 9; ++k) nL += Lp_ref[k]*Lp_ref[k];
    eL = std::max(eL, rel(Lp, Lp_ref, 9, 1e-12));
    eK = std::max(eK, rel(K, K_ref, 81, 1e-30));
    std::vector<double> dX(ND);
    std::vector<float> Jd(ND);
    M::template dot_state_dt<double>(p, S, X.data(), nullptr, D, T, 0.0, dX.data(), Jd.data());
    for (int q = 0; q < ND; ++q) {
      double e = std::fabs(dX[q] - dotF[q]) / std::max(std::fabs(dotF[q]), 1e-12*std::max(1.0, std::fabs(X[q])));
      if (e > eX) { eX = e; worst_X = q; worst_s = s; }
      eJd = std::max(eJd, std::fabs(double(Jd[q]) - JdF[q]) / std::max(std::fabs(JdF[q]), 1e-6));
    }
    // dLp/dS (Voigt columns) vs central differences of the policy's Lp
    M::template flow<double>(p, S, X.data(), nullptr, D, T, 0.0, Lp, G, 1);
    double nS = 0; for (int k = 0; k < 6; ++k) nS += S[k]*S[k];
    const double h = 1e-6*std::sqrt(nS);
    double dmax = 0, gmax = 0;
    for (int c = 0; c < 6; ++c) {
      double Sp[6], Sm[6], Lpp[9], Lpm[9];
      for (int k = 0; k < 6; ++k) { Sp[k] = S[k]; Sm[k] = S[k]; }
      Sp[c] += h; Sm[c] -= h;
      M::template flow<double>(p, Sp, X.data(), nullptr, D, T, 0.0, Lpp, nullptr, 0);
      M::template flow<double>(p, Sm, X.data(), nullptr, D, T, 0.0, Lpm, nullptr, 0);
      for (int m = 0; m < 9; ++m) {
        double fd = (Lpp[m] - Lpm[m]) / (2*h);
        dmax = std::max(dmax, std::fabs(fd - G[m+9*c]));
        gmax = std::max(gmax, std::fabs(G[m+9*c]));
      }
    }
    if (gmax > 0) eFD = std::max(eFD, dmax/gmax);
  }
  std::printf("%d samples: dependent %.2e  C66 %.2e  j %.2e  Lp %.2e  dLp/dMp %.2e  dotState %.2e (entry %d, sample %d)  Jd %.2e  FD dLp/dS %.2e\n",
              ns_, eD, eC, eJ, eL, eK, eX, worst_X, worst_s, eJd, eFD);
  bool ok = eD < 1e-12 && eC < 1e-12 && eJ < 1e-14 && eL < 1e-12 && eK < 1e-11 && eX < 1e-10 && eJd < 1e-6 && eFD < 1e-4;
  std::printf("%s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <golden file>\n", argv[0]); return 2; }
  Reader r{std::fopen(argv[1], "rb")};
  if (!r.f) { std::printf("cannot open %s\n", argv[1]); return 2; }
  int dims[11], nb;
  r.get(dims, 11); r.get(&nb, 1);
  if (nb < 0 || nb > CP_BLOB_MAX || dims[10] < 0 || dims[10] > CP_TT_NDMAX) { std::printf("invalid golden file header\n"); return 2; }
  std::vector<double> C66(36), blob(nb), atol(dims[10]);
  r.get(C66.data(), 36); r.get(blob.data(), nb); r.get(atol.data(), dims[10]);
  std::printf("twintrip <%d,%d,%d,%d,%d,%d,%d,%d,%d> Reuss %d, state %d, blob %d\n",
              dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], dims[8], dims[9], dims[10], nb);
#define CP_TT_RUN(q1,q2,q3,q4,q5,q6,q7,q8,q9)                                                       \
  if (dims[0]==q1 && dims[1]==q2 && dims[2]==q3 && dims[3]==q4 && dims[4]==q5 && dims[5]==q6 && dims[6]==q7 \
      && dims[7]==q8 && dims[8]==q9) return run<cp_model_twintrip<q1,q2,q3,q4,q5,q6,q7,q8,q9>>(r, dims, C66, blob, atol);
  CP_TT_CASES(CP_TT_RUN)
#undef CP_TT_RUN
  std::printf("configuration not supported\n");
  return 1;
}
