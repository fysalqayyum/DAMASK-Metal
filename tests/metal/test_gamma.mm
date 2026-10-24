// GPU Gamma convolution (src/grid/metal, FP32) vs. a double-precision FFTW port of
// utilities_GammaConvolution (spectral_utilities.f90, memory_efficient branch, single rank).
// usage: test_gamma <path to spectral.metal> [n1 n2 n3]
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>
#include <chrono>
#include <fftw3.h>
#include "../../src/grid/metal/DAMASK_Metal.h"

typedef std::complex<double> cd;

static bool invert6(const double* A, double* Ai) {                                                 // Gauss-Jordan, (r,c) at r+6c
  double M[6][12];
  for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) { M[r][c] = A[r+6*c]; M[r][c+6] = r == c; }
  for (int c = 0; c < 6; ++c) {
    int p = c; for (int r = c+1; r < 6; ++r) if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r;
    if (M[p][c] == 0) return false;
    for (int k = 0; k < 12; ++k) std::swap(M[c][k], M[p][k]);
    double d = M[c][c]; for (int k = 0; k < 12; ++k) M[c][k] /= d;
    for (int r = 0; r < 6; ++r) if (r != c) { double f = M[r][c]; for (int k = 0; k < 12; ++k) M[r][k] -= f*M[c][k]; }
  }
  for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) Ai[r+6*c] = M[r][c+6];
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s spectral.metal [n1 n2 n3]\n", argv[0]); return 2; }
  std::ifstream fs(argv[1]); std::stringstream ss; ss << fs.rdbuf(); std::string src = ss.str();
  const int n1 = argc > 4 ? std::atoi(argv[2]) : 32, n2 = argc > 4 ? std::atoi(argv[3]) : 32, n3 = argc > 4 ? std::atoi(argv[4]) : 32;
  const int h = n1/2 + 1;
  const size_t N = size_t(n1)*n2*n3, M = size_t(h)*n2*n3;
  const int cells[3] = {n1, n2, n3};
  const double geom[3] = {1.0e-5, 1.2e-5, 0.9e-5};
  const double wgt = 1.0/double(N);

  // xi1st(3, h, n3, n2) exactly as spectral_utilities_init (continuous derivative)
  std::vector<cd> xi1st(3*M);
  for (int j = 1; j <= n2; ++j) for (int k = 1; k <= n3; ++k) for (int i = 1; i <= h; ++i) {
    int ks[3] = {i-1, j-1, k-1};
    if (j > n2/2+1) ks[1] -= n2;
    if (k > n3/2+1) ks[2] -= n3;
    int ijk[3] = {i, j, k};
    for (int l = 0; l < 3; ++l) {
      cd x(0.0, 2.0*M_PI*ks[l]/geom[l]);
      if (cells[l] % 2 == 0 && ijk[l] == cells[l]/2+1) x = 0.0;
      xi1st[l + 3*((i-1) + size_t(h)*((k-1) + size_t(n3)*(j-1)))] = x;
    }
  }

  // C_ref = C/wgt, cubic Al rotated (anisotropic, all components populated)
  double C66[3] = {106.75e9, 60.41e9, 28.34e9}, Cc[81] = {0}, C[81];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k) for (int l=0;l<3;++l)
    Cc[i+3*j+9*k+27*l] = C66[1]*(i==j)*(k==l) + C66[2]*((i==k)*(j==l)+(i==l)*(j==k))
                       + (C66[0]-C66[1]-2*C66[2])*(i==j)*(j==k)*(k==l);
  double R[9]; { double a=0.3,b=0.5,c=-0.2,d=0.6, s=std::sqrt(a*a+b*b+c*c+d*d); a/=s;b/=s;c/=s;d/=s;
    double Rm[9] = {a*a+b*b-c*c-d*d, 2*(b*c+a*d), 2*(b*d-a*c), 2*(b*c-a*d), a*a-b*b+c*c-d*d, 2*(c*d+a*b), 2*(b*d+a*c), 2*(c*d-a*b), a*a-b*b-c*c+d*d};
    std::memcpy(R, Rm, sizeof(R)); }
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k) for (int l=0;l<3;++l) {
    double s = 0;
    for (int p=0;p<3;++p) for (int q=0;q<3;++q) for (int r=0;r<3;++r) for (int t=0;t<3;++t)
      s += R[i+3*p]*R[j+3*q]*R[k+3*r]*R[l+3*t]*Cc[p+3*q+9*r+27*t];
    C[i+3*j+9*k+27*l] = s/wgt;
  }

  // fields: (a) random stress-like field, (b) near-equilibrium field (constant + 1e-3 smooth perturbation)
  std::mt19937_64 g(1); std::normal_distribution<double> nd;
  std::vector<double> fa(9*N), fb(9*N);
  for (size_t p = 0; p < N; ++p) {
    size_t x = p % n1, y = (p/n1) % n2, z = p/(size_t(n1)*n2);
    for (int c = 0; c < 9; ++c) {
      fa[9*p+c] = 1.0e8*nd(g);
      fb[9*p+c] = (c == 0 ? 1.0e8 : 1.0e6*c) * (1.0 + 1.0e-3*std::sin(2*M_PI*(x/double(n1) + 2*y/double(n2)) + c)*std::cos(2*M_PI*z/double(n3)));
    }
  }
  const double aim[9] = {1.0e-3, 2.0e-5, 0, 2.0e-5, -5.0e-4, 0, 0, 0, -5.0e-4};

  // reference (double, FFTW, DAMASK algorithm)
  auto reference = [&](const std::vector<double>& f) {
    std::vector<double> in(f), out(9*N);
    fftw_complex* F = fftw_alloc_complex(9*M);
    int n[3] = {n3, n2, n1};
    fftw_plan pf = fftw_plan_many_dft_r2c(3, n, 9, in.data(), nullptr, 9, 1, F, nullptr, 9, 1, FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_many_dft_c2r(3, n, 9, F, nullptr, 9, 1, out.data(), nullptr, 9, 1, FFTW_ESTIMATE);
    fftw_execute(pf);
    for (int z = 0; z < n3; ++z) for (int y = 0; y < n2; ++y) for (int x = 0; x < h; ++x) {
      size_t w = x + size_t(h)*(y + size_t(n2)*z);
      cd* v = reinterpret_cast<cd*>(F) + 9*w;
      if (w == 0) continue;
      const cd* xi = &xi1st[3*(x + size_t(h)*(z + size_t(n3)*y))];
      cd xd[9], t[9];
      for (int a=0;a<3;++a) for (int b=0;b<3;++b) xd[a+3*b] = std::conj(-xi[a])*xi[b];
      for (int l=0;l<3;++l) for (int m=0;m<3;++m) { cd s = 0; for (int a=0;a<3;++a) for (int b=0;b<3;++b) s += C[l+3*a+9*m+27*b]*xd[a+3*b]; t[l+3*m] = s; }
      double A[36], Ai[36];
      for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
        A[r+6*c] = t[r+3*c].real(); A[r+3+6*(c+3)] = t[r+3*c].real();
        A[r+6*(c+3)] = t[r+3*c].imag(); A[r+3+6*c] = -t[r+3*c].imag();
      }
      double det = t[0].real()*(t[4].real()*t[8].real()-t[7].real()*t[5].real()) - t[3].real()*(t[1].real()*t[8].real()-t[7].real()*t[2].real()) + t[6].real()*(t[1].real()*t[5].real()-t[4].real()*t[2].real());
      cd res[9];
      if (std::fabs(det) > 1e-16 && invert6(A, Ai)) {
        cd ti[9]; for (int r=0;r<3;++r) for (int c=0;c<3;++c) ti[r+3*c] = cd(Ai[r+6*c], Ai[r+6*(c+3)]);
        for (int l=0;l<3;++l) for (int m=0;m<3;++m) { cd s = 0; for (int nn=0;nn<3;++nn) for (int o=0;o<3;++o) s += ti[l+3*nn]*xd[o+3*m]*v[nn+3*o]; res[l+3*m] = s; }
      }
      for (int c = 0; c < 9; ++c) v[c] = res[c];
    }
    for (int c = 0; c < 9; ++c) reinterpret_cast<cd*>(F)[c] = aim[c];
    fftw_execute(pb);
    fftw_destroy_plan(pf); fftw_destroy_plan(pb); fftw_free(F);
    return out;
  };

  void* ctx = C_Metal_new(n1, n2, n3, src.c_str());
  if (!ctx) { std::fprintf(stderr, "no Metal device / init failed\n"); return 1; }
  std::printf("device: %s, grid %dx%dx%d\n", C_Metal_device_name(ctx), n1, n2, n3);
  C_Metal_set_xi(ctx, reinterpret_cast<const double*>(xi1st.data()));
  C_Metal_set_C_ref(ctx, C);

  int fails = 0;
  for (int which = 0; which < 2; ++which) {
    const std::vector<double>& f = which ? fb : fa;
    std::vector<double> ref = reference(f), gpu(9*N);
    if (C_Metal_gamma_convolution(ctx, f.data(), aim, gpu.data())) { std::printf("GPU call failed\n"); return 1; }
    double d2 = 0, r2 = 0, dmax = 0, fmax = 0;
    for (size_t k = 0; k < 9*N; ++k) {
      d2 += (gpu[k]-ref[k])*(gpu[k]-ref[k]); r2 += ref[k]*ref[k];
      dmax = std::max(dmax, std::fabs(gpu[k]-ref[k])); fmax = std::max(fmax, std::fabs(f[k]));
    }
    double strain_scale = fmax / 28.34e9;                                                           // |field|/C: natural scale of Gamma*field
    double rel = std::sqrt(d2/r2), abs_n = dmax/strain_scale;
    bool ok = rel < 1e-5 && abs_n < 1e-5;
    fails += !ok;
    std::printf("%s field: rel L2 err %.2e, max abs err %.2e (= %.2e of |field|/C)  %s\n",
                which ? "near-equilibrium" : "random          ", rel, dmax, abs_n, ok ? "OK" : "FAIL");
  }

  // timing
  std::vector<double> out(9*N);
  double tg = 0, th = 0, tg_min = 1e9;
  const int reps = 20;
  for (int r = 0; r < reps; ++r) {
    C_Metal_gamma_convolution(ctx, fa.data(), aim, out.data());
    double a, b; C_Metal_last_timing(ctx, &a, &b); tg += a; th += b; tg_min = std::min(tg_min, a);
  }
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < 3; ++r) reference(fa);
  double t_ref = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()/3;
  std::printf("time per call: GPU %.2f ms (min %.2f), host conversion %.2f ms; CPU FP64 reference (1 thread, FFTW_ESTIMATE) %.2f ms\n",
              1e3*tg/reps, 1e3*tg_min, 1e3*th/reps, 1e3*t_ref);
  C_Metal_delete(ctx);
  return fails;
}
