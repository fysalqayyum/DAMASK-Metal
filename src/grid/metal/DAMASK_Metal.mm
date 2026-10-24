// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Metal (Apple GPU) backend of the grid solver, see DAMASK_Metal.h
//> @details FFTs: MPSGraph, FP32, 9 tensor components batched, unnormalized (as FFTW).
//> Real fields on the GPU are component-major [9][n3][n2][n1], Fourier fields [9][n3][n2][h].
//> One command buffer per call: forward FFT -> Gamma kernel -> inverse FFT.
//--------------------------------------------------------------------------------------------------
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <dispatch/dispatch.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <mutex>

#include "DAMASK_Metal.h"
#include "cp_models_composite.h"                                                                   // host view of the GPU data structures and model policies
#include <map>
#include <string>
#include <vector>
#if __has_include("metal_sources.h")
#include "metal_sources.h"                                                                         // kMetalSource, kMetalSourceCP (cmake/EmbedMetal.cmake)
#define DAMASK_METAL_EMBEDDED
#endif

namespace {

struct GammaParams {                                                                                // must match gamma_params in spectral.metal
  uint32_t M;
  uint32_t pad[3];
  float    C_hat[81];
  float    inv_c0;
  float    aim[9];
};

struct CPLaunch {                                                                                   // must match cp_launch in cp.metal
  uint32_t n;
  uint32_t pad;
  df64     dt;
};

struct CPPhase {
  int                         model;                                                                // 1 phenopowerlaw, 2 kinehardening, 3 dislotungsten,
                                                                                                    // 4 phenopowerlaw + twins, 5 tdt, 6 twintrip
  int                         N;                                                                    // actual slip systems
  int                         Np;                                                                   // padded (kernel) count
  int                         Np_tw = 0, Np_fam = 0;                                                // tdt: twin system and slip family capacities
  int                         tt[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};                                 // twintrip: template arguments
  int                         n_state = 0;                                                          // FLAT models: DAMASK state + dependent state (capture)
  id<MTLComputePipelineState> integrate;
  id<MTLComputePipelineState> tangent;
  id<MTLBuffer>               param;                                                                // model parameter struct
  cp_numerics                 num;
  double                      C_norm;                                                               // Frobenius norm of C66 (tangent sanity check)
  std::vector<int>            smap;                                                                 // mapped models: GPU component (rel. to X) of DAMASK state entry
  std::vector<double>         sinit;                                                                // mapped models: GPU state values of unmapped (padded) components
};

struct CPBuffers {                                                                                  // one chunk, component-major
  id<MTLBuffer> in, out, stat, dPdF;
};

struct Context {
  id<NSObject>                activity;                                                             // keeps macOS from throttling the solver (App Nap, idle)
  id<MTLDevice>               device;
  id<MTLCommandQueue>         queue;
  id<MTLComputePipelineState> gamma;
  MPSGraphExecutable*         fft_forth;
  MPSGraphExecutable*         fft_back;
  id<MTLBuffer>               real;                                                                 // float [9][N]
  id<MTLBuffer>               fourier;                                                              // float2 [9][M]
  id<MTLBuffer>               xi;                                                                   // float2 [M][3]
  MPSGraphTensorData*         td_real;
  MPSGraphTensorData*         td_fourier;
  GammaParams                 params;
  int n1, n2, n3, h;
  size_t N, M;
  double t_gpu = 0.0, t_host = 0.0;
  char name[256];
  // constitutive update
  bool                        cp_failed = false;
  std::string                 cp_src;                                                               // constitutive source (compiled per model and slip system count)
  std::map<std::string, id<MTLLibrary>> cp_libs;
  std::vector<CPPhase>        cp_phases;
  CPBuffers                   cp_buf[2];
  size_t                      cp_buf_in = 0, cp_buf_out = 0;                                       // capacity of cp_buf: df64 components per point
  dispatch_queue_t            cp_queue = nullptr;                                                   // asynchronous constitutive jobs (serial: one at a time)
  dispatch_group_t            cp_group = nullptr;
  std::vector<int>            cp_job_rc;                                                            // return code of the last asynchronous job per phase
  std::vector<double>         cp_job_time;                                                          // its wall time [s] (submission to completion)
                                                                                                    // (sized when phases are added: no reallocation while jobs run)
  size_t                      cp_chunk = 16384;                                                     // points per command buffer (bounds GPU submission time)
  NSUInteger                  cp_tg = 32;                                                           // threads per threadgroup (pipeline hint)
};

typedef std::chrono::steady_clock Clock;
inline double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b-a).count(); }

inline df64    to_df(double x) { float h = float(x); return df_make(h, float(x - double(h))); }
inline double  from_df(df64 x) { return double(x.hi) + double(x.lo); }

//> run body(begin, end) over [0,n) in parallel chunks on the CPU
template<typename F>
void parallel_for(size_t n, F body) {
  static const size_t n_cpu = std::max<size_t>(1, [[NSProcessInfo processInfo] activeProcessorCount]);
  const size_t chunk = std::max<size_t>(64, (n + 8*n_cpu - 1) / (8*n_cpu));                         // ~8 pieces per core
  size_t n_chunks = (n + chunk - 1) / chunk;
  dispatch_apply(n_chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
    body(c*chunk, std::min(n, (c+1)*chunk));
  });
}

inline double norm9(const double* a) { double s = 0.0; for (int k = 0; k < 9; ++k) s += a[k]*a[k]; return std::sqrt(s); }
inline double dist9(const double* a, const double* b) {
  double s = 0.0; for (int k = 0; k < 9; ++k) s += (a[k]-b[k])*(a[k]-b[k]); return std::sqrt(s);
}

//> model has rates that switch on at a state threshold (M::switch_entry, used by the certificate)
template<class M, class = void> struct cp_has_switch : std::false_type {};
template<class M> struct cp_has_switch<M, std::void_t<decltype(M::switch_entry)>> : std::true_type {};

//> FP64/FP32 certificate of one GPU result (padded model space, read from the chunk buffers):
//> kinematics (FP64, exact up to rounding): invFp_new = invFp0 (I - dt Lp), Fe = F invFp_new,
//> P = Fe S invFp_new^T; physics (FP32, loose tolerances): stress residual, flow rule, dot-state update
//> (100x DAMASK's state tolerance); delta state recomputed in FP64 from the returned S and X.
//> Guards against silent corruption of GPU results (observed for whole SIMD groups; docs/design.md).
template<class M>
bool cp_verify(const typename M::param& prm, const cp_numerics& num, double dt,
               const df64* in, const df64* out, size_t m, size_t j) {
  typedef cp_layout<M> L;
  static const bool dbg = getenv("DAMASK_METAL_VERIFY_DEBUG") != nullptr;
  auto cp_reject = [&](int check) {
    if (dbg) std::fprintf(stderr, "DAMASK_Metal: certificate check %d failed (point %zu)\n", check, j);
    return false;
  };
  const int ND = M::ND, NY = M::NY > 0 ? M::NY : 1, NDEP = M::NDEP > 0 ? M::NDEP : 1;
  auto gi = [&](int c) { return from_df(in[c*m + j]); };
  auto go = [&](int c) { return from_df(out[c*m + j]); };
  double F[9], Fp0[9], iFp[9], Fe[9], P[9], Lp[9], Sv[6], X[ND], Y[NY], X0[ND], Y0[NY], T = gi(L::T);
  for (int k = 0; k < 9; ++k) {
    F[k] = gi(L::F+k); Fp0[k] = gi(L::Fp0+k);
    iFp[k] = go(L::iFp+k); Fe[k] = go(L::Fe+k); P[k] = go(L::P+k); Lp[k] = go(L::Lp+k);
  }
  for (int k = 0; k < 6; ++k)  Sv[k] = go(L::S+k);
  for (int k = 0; k < ND; ++k) { X[k] = go(L::X+k); X0[k] = gi(L::X0+k); }
  for (int k = 0; k < M::NY; ++k) { Y[k] = go(L::Y+k); Y0[k] = gi(L::Y0+k); }
  double Dd[NDEP];
  for (int k = 0; k < M::NDEP; ++k) { Dd[k] = go(L::D+k); if (!std::isfinite(Dd[k])) return cp_reject(1); }
  for (const double* v : {P, iFp, Fe, Lp}) for (int k = 0; k < 9; ++k) if (!std::isfinite(v[k])) return cp_reject(2);
  for (int k = 0; k < 6; ++k)  if (!std::isfinite(Sv[k])) return cp_reject(3);
  for (int k = 0; k < ND; ++k) if (!std::isfinite(X[k])) return cp_reject(4);
  for (int k = 0; k < M::NY; ++k) if (!std::isfinite(Y[k])) return cp_reject(5);

  double invFp0[9], B[9], iFpc[9], FiFp[9], Fec[9], Sm[9], T1[9], Pc[9];
  cp_inv33(Fp0, invFp0);
  for (int k = 0; k < 9; ++k) B[k] = (k % 4 == 0 ? 1.0 : 0.0) - dt*Lp[k];
  cp_mul33(invFp0, B, iFpc);
  if (dist9(iFp, iFpc) > 1e-9*norm9(iFpc)) return cp_reject(6);
  cp_mul33(F, iFp, FiFp);
  for (int k = 0; k < 9; ++k) Fec[k] = FiFp[k];
  const double j2m1 = cp_fi<M>::template j2m1<double>(prm, X);                                      // FI_STATE: Fi = j I of the returned state
  if (M::FI_STATE) { double ij = 1.0/std::sqrt(1.0 + j2m1); for (int k = 0; k < 9; ++k) Fec[k] *= ij; }
  if (dist9(Fe, Fec) > 1e-9*norm9(Fec)) return cp_reject(7);
  for (int k = 0; k < 9; ++k) Sm[k] = 0.0;
  Sm[0] = Sv[0]; Sm[4] = Sv[1]; Sm[8] = Sv[2];
  Sm[CP_IDX(1,2)] = Sm[CP_IDX(2,1)] = Sv[3];
  Sm[CP_IDX(0,2)] = Sm[CP_IDX(2,0)] = Sv[4];
  Sm[CP_IDX(0,1)] = Sm[CP_IDX(1,0)] = Sv[5];
  cp_mul33(FiFp, Sm, Pc);                                                                           // P = F invFp S invFp^T
  cp_mul3T3(Pc, iFp, T1);
  for (int k = 0; k < 9; ++k) Pc[k] = T1[k];
  if (dist9(P, Pc) > 1e-8*norm9(Pc) + 1e-6) return cp_reject(8);

  // physics at the returned state; delta-updated Y is not used by the dot state/kinetics within a step,
  // so the dot-state check uses Y0
  float Mt[9], Ea[9], Sf[6], Xf[ND], Y0f[NY], Df[NDEP], R[6], Lpc[9], Bc[9], MB[9], dX[ND];
  cp_work<float,M> w;
  for (int k = 0; k < 6; ++k) Sf[k] = float(Sv[k]);
  for (int k = 0; k < ND; ++k) Xf[k] = float(X[k]);
  for (int k = 0; k < NY; ++k) Y0f[k] = M::NY > 0 ? float(Y0[k]) : 0.0f;
  for (int k = 0; k < NDEP; ++k) Df[k] = M::NDEP > 0 ? float(Dd[k]) : 0.0f;                         // of the last stress solve
  {                                                                                                 // Ea = (A^T A - I)/2 in FP64: the only cancellation;
    double Ad[9], Md[9];                                                                            // in FP32 it leaves ~1e-7*|C| of stress noise
    cp_mul33(F, invFp0, Ad);
    cp_mulT33(Ad, Ad, Md);
    for (int k = 0; k < 9; ++k) { Mt[k] = float(Md[k]); Ea[k] = float(0.5*(Md[k] - (k % 4 == 0 ? 1.0 + j2m1 : 0.0))); }
  }
  cp_core_residual<float,M>(prm, Mt, Ea, float(dt), Sf, Xf, Y0f, Df, float(T), R, Lpc, Bc, MB, w, false);
  float nS = 0.0f, nR = 0.0f, nL = 0.0f, nD = 0.0f;
  for (int k = 0; k < 6; ++k) { nS += Sf[k]*Sf[k]; nR += R[k]*R[k]; }
  if (!(std::sqrt(nR) <= 1e-3f*std::sqrt(nS) + 1e-3f*M::template stress_scale<float>(prm, Xf))) {
    if (dbg) {                                                                                      // FP64 residual for comparison
      double Ead[9], Md[9], Ad[9], Rd[6], Lpd[9], Bd[9], MBd[9]; cp_work<double,M> wd;
      double Xd[ND], Y0d[NY];
      for (int k = 0; k < ND; ++k) Xd[k] = X[k];
      for (int k = 0; k < NY; ++k) Y0d[k] = M::NY > 0 ? Y0[k] : 0.0;
      cp_mul33(F, invFp0, Ad); cp_mulT33(Ad, Ad, Md);
      for (int k = 0; k < 9; ++k) Ead[k] = 0.5*(Md[k] - (k % 4 == 0 ? 1.0 + j2m1 : 0.0));
      cp_core_residual<double,M>(prm, Md, Ead, dt, Sv, Xd, Y0d, Dd, T, Rd, Lpd, Bd, MBd, wd, false);
      double nRd = 0; for (int k = 0; k < 6; ++k) nRd += Rd[k]*Rd[k];
      std::fprintf(stderr, "   |R|/|S|: FP32 %.3e  FP64 %.3e  |S| %.3e  |Lp| %.3e\n", std::sqrt(nR/nS), std::sqrt(nRd)/std::sqrt(nS),
                   std::sqrt(double(nS)), norm9(Lpd));
    }
    return cp_reject(9);
  }
  for (int k = 0; k < 9; ++k) { float d = float(Lp[k]) - Lpc[k]; nD += d*d; nL += Lpc[k]*Lpc[k]; }
  if (!(std::sqrt(nD) <= 1e-2f*std::sqrt(nL) + 1e-12f)) return cp_reject(10);
  cp_dot<float,M>::eval(prm, Sf, Xf, Y0f, Df, float(T), float(dt), dX, nullptr);
  auto tol_state = [&](int k) { return 100.0*std::max(double(M::atol(prm, k)), double(num.rtol_state)*std::fabs(X[k])); };
  for (int k = 0; k < ND; ++k) {
    double r = X[k] - X0[k] - dt*dX[k];
    double slack = 1e-2*std::fabs(dt*dX[k]);
    if constexpr (cp_has_switch<M>::value) {                                                        // rate switched on in the last state update
      float thr;                                                                                    // (converged at the previous iterate, as on
      const int s = M::switch_entry(prm, k, thr);                                                   // the CPU): rate of either side is valid
      if (s >= 0 && X[s] > thr && X[s] - thr <= tol_state(s)) slack = 1.01*std::fabs(dt*dX[k]);
    }
    if (!(std::fabs(r) <= tol_state(k) + slack)) {
      if (dbg) std::fprintf(stderr, "   entry %d: X %.6e X0 %.6e dt*dX %.6e r %.3e atol %.3e\n", k, X[k], X0[k], dt*dX[k], r,
                            double(M::atol(prm, k)));
      return cp_reject(11);
    }
  }
  if (M::NY > 0) {                                                                                  // delta state, FP64
    double Yc[NY];
    for (int k = 0; k < M::NY; ++k) Yc[k] = Y0[k];
    M::template delta_state<double>(prm, Sv, X, Yc, Dd, T);
    for (int k = 0; k < M::NY; ++k) if (!(std::fabs(Y[k] - Yc[k]) <= 1e-6*std::fabs(Yc[k]) + 1e-8)) return cp_reject(12);
  }
  if (M::STATE_C) {                                                                                 // stiffness used vs stiffness of the returned state
    double Dc[NDEP], d = 0.0, c = 0.0;
    for (int k = 0; k < M::NDEP; ++k) Dc[k] = Dd[k];
    M::template dependent<double>(prm, X, Y0, T, Dc);
    for (int k = 0; k < 36; ++k) { d += (Dc[M::DC+k] - Dd[M::DC+k])*(Dc[M::DC+k] - Dd[M::DC+k]); c += Dc[M::DC+k]*Dc[M::DC+k]; }
    if (!(std::sqrt(d) <= 1e-3*std::sqrt(c))) return cp_reject(13);
  }
  return true;
}

//> GPU input/output component of DAMASK state entry q (state followed by dependent state): FLAT models
//> are contiguous, others consist of blocks of N values padded to M::N
template<class M>
inline int cp_state_comp(int q, int N, int base_X) {
  if (M::FLAT) return base_X + q;
  const int blk = q / N, i = q % N;
  const int grp = blk < M::NBX ? 0 : blk < M::NBX + M::NBY ? 1 : 2;
  const int b0  = grp == 0 ? blk : grp == 1 ? blk - M::NBX : blk - M::NBX - M::NBY;
  return base_X + (grp == 0 ? 0 : grp == 1 ? M::ND : M::ND + M::NY) + b0*M::N + i;
}

//> compile Metal Shading Language source with IEEE semantics (no fast-math reassociation)
id<MTLLibrary> compile_msl(id<MTLDevice> dev, const char* src, NSError** err) {
  MTLCompileOptions* opt = [MTLCompileOptions new];
  opt.mathMode = MTLMathModeSafe;
  return [dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:opt error:err];
}

//> compile the constitutive base library and check that host and GPU agree on the data layout
bool cp_init(Context* ctx, const char* src) {
  if (!ctx->cp_src.empty()) return true;
  if (ctx->cp_failed || !src) return false;
  ctx->cp_failed = true;
  if (getenv("DAMASK_METAL_CP_TG")) ctx->cp_tg = NSUInteger(std::max(1, atoi(getenv("DAMASK_METAL_CP_TG"))));
  NSError* err = nil;
  id<MTLLibrary> lib = compile_msl(ctx->device, src, &err);
  if (!lib) {
    std::fprintf(stderr, "DAMASK_Metal: constitutive shader compilation failed:\n%s\n", err.localizedDescription.UTF8String);
    return false;
  }
  id<MTLComputePipelineState> ps = [ctx->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cp_sizes"] error:&err];
  if (!ps) return false;
  id<MTLBuffer> s = [ctx->device newBufferWithLength:9*sizeof(uint32_t) options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ps];
  [enc setBuffer:s offset:0 atIndex:0];
  [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  const uint32_t* g = static_cast<const uint32_t*>(s.contents);
  const uint32_t h[9] = {uint32_t(sizeof(cp_pheno_param)), uint32_t(sizeof(cp_numerics)),
                         uint32_t(sizeof(CPLaunch)), uint32_t(sizeof(df64)), uint32_t(sizeof(cp_kineh_param)),
                         uint32_t(sizeof(cp_dtung_param)), uint32_t(sizeof(cp_phenotw_param)), uint32_t(sizeof(cp_tdt_param)),
                         uint32_t(sizeof(cp_twintrip_param))};
  for (int k = 0; k < 9; ++k)
    if (g[k] != h[k]) {
      std::fprintf(stderr, "DAMASK_Metal: data layout mismatch host/GPU (item %d: %u vs %u)\n", k, h[k], g[k]);
      return false;
    }
  ctx->cp_src = src;
  ctx->cp_failed = false;
  return true;
}

//> precision mode of the constitutive update (DAMASK_METAL_PRECISION): 0 df64, 1 mixed (FP32 pre-solve,
//> df64 polish to rtol_S), 2 fast (FP32 kinetics and stress solve, df64 accumulation, see cp_core_integrate_fast)
static int g_cp_prec = -1;                                                                          // from C_Metal_cp_set_precision
int cp_precision() {
  if (const char* e = getenv("DAMASK_METAL_PRECISION")) {                                           // overrides the numerics
    if (!std::strcmp(e, "df64"))  return 0;
    if (!std::strcmp(e, "mixed")) return 1;
    if (!std::strcmp(e, "fast"))  return 2;
  }
  if (getenv("DAMASK_METAL_NO_MIXED")) return 0;
  return g_cp_prec >= 0 ? g_cp_prec : 1;
}

//> constitutive base library from the embedded shader sources (if built with them)
bool cp_init_embedded(Context* ctx) {
#ifdef DAMASK_METAL_EMBEDDED
  return cp_init(ctx, kMetalSourceCP);
#else
  return cp_init(ctx, nullptr);
#endif
}

//> numerics of a phase: [rtol_S, atol_S, rtol_state, stepsize_ls, trust_S, n_stress, n_state] and |C66|
void cp_set_numerics(CPPhase& ph, const double* C66, const double* numerics) {
  ph.C_norm = 0.0;
  for (int k = 0; k < 36; ++k) ph.C_norm += C66[k]*C66[k];
  ph.C_norm = std::sqrt(ph.C_norm);
  ph.num.rtol_S      = float(numerics[0]);
  ph.num.atol_S      = float(numerics[1]);
  ph.num.rtol_state  = float(numerics[2]);
  ph.num.stepsize_ls = float(numerics[3]);
  ph.num.trust_S     = float(numerics[4]);
  ph.num.n_stress    = int(numerics[5]);
  ph.num.n_state     = int(numerics[6]);
  ph.num.mixed       = cp_precision();                                                             // 0 df64, 1 mixed, 2 fast
}

//> add a phase; returns its handle
int cp_register_phase(Context* ctx, const CPPhase& ph) {
  ctx->cp_phases.push_back(ph);
  ctx->cp_job_rc.resize(ctx->cp_phases.size(), 1);
  ctx->cp_job_time.resize(ctx->cp_phases.size(), 0.0);
  return int(ctx->cp_phases.size()) - 1;
}

//> pipelines of one policy type (e.g. cp_model_pheno<12>): own library compiled with CP_MODEL_TYPE, cached;
//> the threadgroup size is bounded at pipeline creation (more registers per thread, fewer spills)
bool cp_pipelines(Context* ctx, const std::string& key, CPPhase& ph) {
  const bool fast = cp_precision() == 2;
  id<MTLLibrary> lib = ctx->cp_libs[key + (fast ? "/fast" : "")];
  if (!lib) {
    std::string src = std::string(fast ? "#define CP_FAST 1\n" : "") + "#define CP_MODEL_TYPE " + key + "\n" + ctx->cp_src;
    NSError* err = nil;
    auto t0 = Clock::now();
    lib = compile_msl(ctx->device, src.c_str(), &err);
    if (!lib) {
      std::fprintf(stderr, "DAMASK_Metal: %s shader compilation failed:\n%s\n", key.c_str(), err.localizedDescription.UTF8String);
      return false;
    }
    if (getenv("DAMASK_METAL_VERBOSE"))
      std::fprintf(stderr, "DAMASK_Metal: compiled %s in %.1f s\n", key.c_str(), seconds(t0, Clock::now()));
    ctx->cp_libs[key + (fast ? "/fast" : "")] = lib;
  }
  for (int k = 0; k < 2; ++k) {
    MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
    d.computeFunction = [lib newFunctionWithName:(k == 0 ? @"cp_integrate_kernel" : @"cp_tangent_kernel")];
    d.maxTotalThreadsPerThreadgroup = ctx->cp_tg;
    NSError* err = nil;
    id<MTLComputePipelineState> ps = [ctx->device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone
                                                                             reflection:nil error:&err];
    if (!ps) return false;
    (k == 0 ? ph.integrate : ph.tangent) = ps;
  }
  return true;
}

//> DAMASK state (2D: [block b of N values]...) <-> padded GPU state (blocks of Np values)
template<class M>
void cp_pack(const CPBuffers& buf, size_t m, size_t off, const CPPhase& ph, const double* F, const double* Fp0, const double* S0,
             const double* Lp_guess, const double* state0, const double* T) {
  typedef cp_layout<M> L;
  const int N = ph.N;
  const bool mapped = !ph.smap.empty();
  const int NS = mapped ? int(ph.smap.size()) : M::FLAT ? M::ND + M::NY + M::NDEP : (M::NBX + M::NBY + M::NBD) * N; // DAMASK state (+ dependent state) per point
  df64* in = static_cast<df64*>(buf.in.contents);
  parallel_for(m, [&](size_t b, size_t e) {
    for (size_t j = b; j < e; ++j) {
      size_t i = off + j;
      for (int k = 0; k < 9; ++k) {
        in[(L::F+k)*m + j]   = to_df(F[9*i+k]);
        in[(L::Fp0+k)*m + j] = to_df(Fp0[9*i+k]);
        in[(L::Lp0+k)*m + j] = to_df(Lp_guess[9*i+k]);
      }
      const double* s = S0 + 9*i;                                                                   // (3,3) -> Voigt 11,22,33,23,13,12
      const double sv[6] = {s[0], s[4], s[8], s[CP_IDX(1,2)], s[CP_IDX(0,2)], s[CP_IDX(0,1)]};
      for (int k = 0; k < 6; ++k) in[(L::S0+k)*m + j] = to_df(sv[k]);
      if (mapped) {
        for (int c = 0; c < M::ND + M::NY + M::NDEP; ++c) in[(L::X0+c)*m + j] = to_df(ph.sinit[c]);
        for (int q = 0; q < NS; ++q) in[(L::X0+ph.smap[q])*m + j] = to_df(state0[NS*i + q]);
      } else if (M::FLAT)
        for (int q = 0; q < NS; ++q) in[(L::X0+q)*m + j] = to_df(state0[NS*i + q]);
      else
        for (int blk = 0; blk < M::NBX + M::NBY + M::NBD; ++blk) {
          int grp = blk < M::NBX ? 0 : blk < M::NBX + M::NBY ? 1 : 2;
          int b0  = grp == 0 ? blk : grp == 1 ? blk - M::NBX : blk - M::NBX - M::NBY;
          int base = (grp == 0 ? L::X0 : grp == 1 ? L::Y0 : L::D0) + b0*M::N;
          double pad = M::pad_value(grp, b0);
          for (int q = 0; q < M::N; ++q)
            in[(base+q)*m + j] = to_df(q < N ? state0[NS*i + blk*N + q] : pad);
        }
      in[L::T*m + j] = to_df(T ? T[i] : 0.0);
    }
  });
}

//> sanity of a GPU tangent: finite and of the order of the elastic stiffness
inline bool cp_tangent_ok(const float* t, size_t stride, double C_norm) {
  double s = 0.0;
  for (int k = 0; k < 81; ++k) { double x = t[k*stride]; if (!std::isfinite(x)) return false; s += x*x; }
  s = std::sqrt(s);
  return s > 1e-4*C_norm && s < 1e2*C_norm;
}

template<class M>
void cp_unpack(const CPBuffers& buf, size_t m, size_t off, const CPPhase& ph, double dt,
               double* P, double* S, double* Fp, double* Fe, double* Lp, double* state, float* dPdF,
               int* status, int* iterations) {
  typedef cp_layout<M> L;
  const bool mapped = !ph.smap.empty();
  const int N = ph.N, NS = mapped ? int(ph.smap.size()) : M::FLAT ? M::ND + M::NY + M::NDEP : (M::NBX + M::NBY + M::NBD) * N;
  const df64*  in  = static_cast<const df64*>(buf.in.contents);
  const df64*  out = static_cast<const df64*>(buf.out.contents);
  const int*   st  = static_cast<const int*>(buf.stat.contents);
  const float* tg  = dPdF ? static_cast<const float*>(buf.dPdF.contents) : nullptr;
  const typename M::param& prm = *static_cast<const typename M::param*>(ph.param.contents);
  static const bool no_verify = getenv("DAMASK_METAL_NO_VERIFY") != nullptr;                        // diagnostics only
  parallel_for(m, [&](size_t b, size_t e) {
    for (size_t j = b; j < e; ++j) {
      size_t i = off + j;
      status[i] = st[j];
      if (iterations) iterations[i] = st[m + j];
      if (st[j] != CP_OK) {
        static const char* dump = getenv("DAMASK_METAL_DUMP");                                       // diagnostics: inputs of failed points
        if (dump) {
          static std::atomic<int> n_dump{0};
          if (n_dump++ < 200) {
            static std::mutex mx; std::lock_guard<std::mutex> lk(mx);
            if (FILE* fo = std::fopen(dump, "a")) {
              std::fprintf(fo, "%d %d %.17g", st[j], int(L::n_in), dt);
              for (int c = 0; c < L::n_in; ++c) std::fprintf(fo, " %.17g", from_df(in[c*m + j]));
              std::fprintf(fo, "\n"); std::fclose(fo);
            }
          }
        }
        continue;
      }
      static const int inject = getenv("DAMASK_METAL_INJECT") ? atoi(getenv("DAMASK_METAL_INJECT")) : -1;  // QA: fault injection
      if (inject >= 0 && i % 64 == 0) {                                                             // corrupt the GPU output, just above tolerance
        df64* o = const_cast<df64*>(out);
        auto scale = [&](int c, double f) { o[c*m + j] = to_df(from_df(o[c*m + j])*f); };
        switch (inject) {
          case 0: for (int k = 0; k < 6; ++k) scale(L::S+k, 1.01); break;                           // stress
          case 1: for (int k = 0; k < (M::FLAT ? M::ND : M::N); ++k) scale(L::X+k, 1.002); break;  // first state block (resistance/density)
          case 2: for (int k = (M::FLAT || M::NBX == 0 ? M::ND/2 : (M::NBX-1)*M::N); k < (M::FLAT || M::NBX == 0 ? M::ND : M::NBX*M::N); ++k)
                    o[(L::X+k)*m + j] = to_df(from_df(o[(L::X+k)*m + j])*1.5 + 1e-3); break;      // last state block (shear)
          case 3: for (int k = 0; k < 9; ++k) scale(L::Lp+k, 1.05); break;
          case 4: scale(L::P, 1.0 + 1e-6); break;
          case 5: scale(L::Fe+4, 1.0 + 1e-7); break;
          case 6: scale(L::iFp+8, 1.0 + 1e-7); break;
        }
      }
      if (!no_verify && (!cp_verify<M>(prm, ph.num, dt, in, out, m, j) || (tg && !cp_tangent_ok(tg + j, m, ph.C_norm)))) {
        status[i] = CP_FAIL_VERIFY;
        continue;
      }
      double iFp[9], Fpo[9];
      for (int k = 0; k < 9; ++k) iFp[k] = from_df(out[(L::iFp+k)*m + j]);
      double d = cp_inv33(iFp, Fpo);                                                                // Fp = inv(invFp_new)/det^(1/3), FP64
      if (!(d > 0.0)) { status[i] = CP_FAIL_SINGULAR; continue; }
      double sc = std::cbrt(1.0/d);
      for (int k = 0; k < 9; ++k) {
        P[9*i+k]  = from_df(out[(L::P+k)*m + j]);
        Fe[9*i+k] = from_df(out[(L::Fe+k)*m + j]);
        Lp[9*i+k] = from_df(out[(L::Lp+k)*m + j]);
        Fp[9*i+k] = Fpo[k] / sc;
      }
      double sv[6];
      for (int k = 0; k < 6; ++k) sv[k] = from_df(out[(L::S+k)*m + j]);
      double* s = S + 9*i;                                                                          // Voigt -> symmetric (3,3)
      s[0] = sv[0]; s[4] = sv[1]; s[8] = sv[2];
      s[CP_IDX(1,2)] = s[CP_IDX(2,1)] = sv[3];
      s[CP_IDX(0,2)] = s[CP_IDX(2,0)] = sv[4];
      s[CP_IDX(0,1)] = s[CP_IDX(1,0)] = sv[5];
      for (int q = 0; q < NS; ++q) state[NS*i + q] = from_df(out[(mapped ? L::X + ph.smap[q] : cp_state_comp<M>(q, N, L::X))*m + j]);
      if (tg) for (int k = 0; k < 81; ++k) dPdF[81*i+k] = tg[k*m + j];
    }
  });
}

//> chunked, double-buffered execution: pack chunk k+1 while the GPU integrates chunk k;
//> per chunk one command buffer: integrate, then (optionally) tangent at the converged state
template<class M>
int cp_run(Context* ctx, const CPPhase& ph, int n, double dt,
           const double* F, const double* Fp0, const double* S0, const double* Lp_guess, const double* state0, const double* T,
           double* P, double* S, double* Fp, double* Fe, double* Lp, double* state, float* dPdF,
           int* status, int* iterations) {
  typedef cp_layout<M> L;
  const size_t chunk = ctx->cp_chunk;
  if (ctx->cp_buf_in < size_t(L::n_in) || ctx->cp_buf_out < size_t(L::n_out)) {
    ctx->cp_buf_in  = std::max(ctx->cp_buf_in,  size_t(L::n_in));
    ctx->cp_buf_out = std::max(ctx->cp_buf_out, size_t(L::n_out));
    for (int b = 0; b < 2; ++b) {
      ctx->cp_buf[b].in   = [ctx->device newBufferWithLength:chunk*ctx->cp_buf_in*sizeof(df64)  options:MTLResourceStorageModeShared];
      ctx->cp_buf[b].out  = [ctx->device newBufferWithLength:chunk*ctx->cp_buf_out*sizeof(df64) options:MTLResourceStorageModeShared];
      ctx->cp_buf[b].stat = [ctx->device newBufferWithLength:chunk*2*sizeof(int)         options:MTLResourceStorageModeShared];
      ctx->cp_buf[b].dPdF = [ctx->device newBufferWithLength:chunk*81*sizeof(float)      options:MTLResourceStorageModeShared];
    }
  }

  const size_t n_chunks = (size_t(n) + chunk - 1) / chunk;
  std::vector<id<MTLCommandBuffer>> cbs(n_chunks);
  double t_host = 0.0, t_exec = 0.0;
  auto t0 = Clock::now();
  auto finish = [&](size_t k) -> bool {
    [cbs[k] waitUntilCompleted];
    t_exec += cbs[k].GPUEndTime - cbs[k].GPUStartTime;
    if (cbs[k].status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr, "DAMASK_Metal: constitutive command buffer failed: %s\n",
                   cbs[k].error ? cbs[k].error.localizedDescription.UTF8String : "unknown");
      return false;
    }
    auto a = Clock::now();
    size_t off = k*chunk, m = std::min(chunk, size_t(n) - off);
    cp_unpack<M>(ctx->cp_buf[k%2], m, off, ph, dt, P, S, Fp, Fe, Lp, state, dPdF, status, iterations);
    t_host += seconds(a, Clock::now());
    return true;
  };
  auto fail = [&]() -> int {                                                                        // drain: the next call reuses cp_buf
    for (auto& cb : cbs) if (cb) [cb waitUntilCompleted];
    return 1;
  };
  for (size_t k = 0; k < n_chunks; ++k) {
    if (k >= 2 && !finish(k-2)) return fail();
    auto a = Clock::now();
    size_t off = k*chunk, m = std::min(chunk, size_t(n) - off);
    const CPBuffers& buf = ctx->cp_buf[k%2];
    cp_pack<M>(buf, m, off, ph, F, Fp0, S0, Lp_guess, state0, T);
    t_host += seconds(a, Clock::now());
    CPLaunch l;
    l.n = uint32_t(m); l.pad = 0; l.dt = to_df(dt);
    cbs[k] = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cbs[k] computeCommandEncoder];
    [enc setComputePipelineState:ph.integrate];
    [enc setBuffer:buf.in   offset:0 atIndex:0];
    [enc setBuffer:buf.out  offset:0 atIndex:1];
    [enc setBuffer:buf.stat offset:0 atIndex:2];
    [enc setBuffer:ph.param offset:0 atIndex:3];
    [enc setBytes:&ph.num length:sizeof(cp_numerics) atIndex:4];
    [enc setBytes:&l length:sizeof(l) atIndex:5];
    [enc setBuffer:buf.dPdF offset:0 atIndex:6];
    NSUInteger tg = std::min(ph.integrate.maxTotalThreadsPerThreadgroup, ctx->cp_tg);
    [enc dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    if (dPdF) {
      [enc setComputePipelineState:ph.tangent];
      tg = std::min(ph.tangent.maxTotalThreadsPerThreadgroup, ctx->cp_tg);
      [enc dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
    [enc endEncoding];
    [cbs[k] commit];
  }
  for (size_t k = n_chunks >= 2 ? n_chunks-2 : 0; k < n_chunks; ++k)
    if (!finish(k)) return fail();
  ctx->t_host = t_host;
  ctx->t_gpu  = seconds(t0, Clock::now()) - t_host;
  if (getenv("DAMASK_METAL_VERBOSE")) {
    int n_fail = 0, n_rej = 0, hist[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n; ++i) { n_fail += status[i] != CP_OK; n_rej += status[i] == CP_FAIL_VERIFY; hist[std::min(std::max(status[i],0),5)]++; }
    std::fprintf(stderr, "DAMASK_Metal: cp %d points, %d to CPU (%d rejected by certificate), GPU %.1f ms, host %.1f ms, GPU exec %.1f ms"
                 " [stress %d nan %d state %d singular %d]\n",
                 n, n_fail, n_rej, 1e3*ctx->t_gpu, 1e3*t_host, 1e3*t_exec, hist[1], hist[2], hist[3], hist[4]);
  }
  return 0;
}

//> padded kernel size for N slip systems (0: unsupported)
inline int cp_padded(int N) { for (int Np : {12, 24, 30, 48}) if (N <= Np) return Np; return 0; }

//> tdt kernel capacities (slip, twin systems, slip families); 0: unsupported
inline void cp_tdt_caps(int ns, int nt, int nf, int& NSL, int& NTW, int& NFAM) {
  NSL = NTW = NFAM = 0;
  for (int c : {12, 18, 24, 36}) if (ns <= c) { NSL = c; break; }
  for (int c : {6, 12})          if (nt <= c) { NTW = c; break; }
  for (int c : {3, 6})           if (nf <= c) { NFAM = c; break; }
}
#define CP_TDT_CASES(X) X(12,6,3) X(12,6,6) X(12,12,3) X(12,12,6) X(18,6,3) X(18,6,6) X(18,12,3) X(18,12,6) \
                        X(24,6,3) X(24,6,6) X(24,12,3) X(24,12,6) X(36,6,3) X(36,6,6) X(36,12,3) X(36,12,6)

MPSGraphExecutable* compile_fft(Context* ctx, bool inverse) {
  MPSGraph* g = [MPSGraph new];
  NSArray<NSNumber*>* shape_real = @[@9, @(ctx->n3), @(ctx->n2), @(ctx->n1)];
  NSArray<NSNumber*>* shape_four = @[@9, @(ctx->n3), @(ctx->n2), @(ctx->h)];
  MPSGraphFFTDescriptor* d = [MPSGraphFFTDescriptor descriptor];
  d.inverse     = inverse;
  d.scalingMode = MPSGraphFFTScalingModeNone;
  MPSGraphTensor *in, *out;
  if (!inverse) {
    in  = [g placeholderWithShape:shape_real dataType:MPSDataTypeFloat32 name:nil];
    out = [g realToHermiteanFFTWithTensor:in axes:@[@1, @2, @3] descriptor:d name:nil];
  } else {
    d.roundToOddHermitean = (ctx->n1 % 2 == 1);
    in  = [g placeholderWithShape:shape_four dataType:MPSDataTypeComplexFloat32 name:nil];
    out = [g HermiteanToRealFFTWithTensor:in axes:@[@1, @2, @3] descriptor:d name:nil];
  }
  MPSGraphShapedType* t = [[MPSGraphShapedType alloc] initWithShape:(inverse ? shape_four : shape_real)
                                                           dataType:(inverse ? MPSDataTypeComplexFloat32 : MPSDataTypeFloat32)];
  return [g compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:ctx->device]
                        feeds:@{in: t}
                targetTensors:@[out]
             targetOperations:nil
        compilationDescriptor:nil];
}

}


extern "C" {

void* C_Metal_new(int n1, int n2, int n3, const char* msl_source) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return nullptr;

    Context* ctx = new Context();
    // Without this, GPU work of a background (terminal) process waited up to 15 min to be scheduled
    // while the display was asleep (GPU execution itself took ~30 ms). The display may still sleep.
    ctx->activity = [[NSProcessInfo processInfo] beginActivityWithOptions:(NSActivityUserInitiated | NSActivityLatencyCritical)
                                                                   reason:@"DAMASK grid solver on the GPU"];
    ctx->device = dev;
    ctx->queue  = [dev newCommandQueue];
    ctx->n1 = n1; ctx->n2 = n2; ctx->n3 = n3; ctx->h = n1/2 + 1;
    ctx->N = size_t(n1)*n2*n3;
    ctx->M = size_t(ctx->h)*n2*n3;
    std::snprintf(ctx->name, sizeof(ctx->name), "%s", dev.name.UTF8String);

    NSError* err = nil;
    id<MTLLibrary> lib = compile_msl(dev, msl_source, &err);
    if (!lib) {
      std::fprintf(stderr, "DAMASK_Metal: shader compilation failed:\n%s\n", err.localizedDescription.UTF8String);
      delete ctx;
      return nullptr;
    }
    ctx->gamma = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gamma_convolution"] error:&err];
    if (!ctx->gamma) {
      std::fprintf(stderr, "DAMASK_Metal: pipeline creation failed:\n%s\n", err.localizedDescription.UTF8String);
      delete ctx;
      return nullptr;
    }

    ctx->real    = [dev newBufferWithLength:9*ctx->N*sizeof(float)   options:MTLResourceStorageModeShared];
    ctx->fourier = [dev newBufferWithLength:9*ctx->M*2*sizeof(float) options:MTLResourceStorageModeShared];
    ctx->xi      = [dev newBufferWithLength:3*ctx->M*2*sizeof(float) options:MTLResourceStorageModeShared];
    if (!ctx->real || !ctx->fourier || !ctx->xi) { delete ctx; return nullptr; }
    std::memset(ctx->xi.contents, 0, ctx->xi.length);

    ctx->td_real    = [[MPSGraphTensorData alloc] initWithMTLBuffer:ctx->real
                                                              shape:@[@9, @(n3), @(n2), @(n1)]
                                                           dataType:MPSDataTypeFloat32];
    ctx->td_fourier = [[MPSGraphTensorData alloc] initWithMTLBuffer:ctx->fourier
                                                              shape:@[@9, @(n3), @(n2), @(ctx->h)]
                                                           dataType:MPSDataTypeComplexFloat32];
    ctx->fft_forth = compile_fft(ctx, false);
    ctx->fft_back  = compile_fft(ctx, true);
    if (!ctx->fft_forth || !ctx->fft_back) { delete ctx; return nullptr; }

    std::memset(&ctx->params, 0, sizeof(ctx->params));
    ctx->params.M = uint32_t(ctx->M);
    return ctx;
  }
}

void C_Metal_delete(void* p) {
  Context* ctx = static_cast<Context*>(p);
  if (ctx && ctx->cp_group) dispatch_group_wait(ctx->cp_group, DISPATCH_TIME_FOREVER);
  if (ctx && ctx->activity) [[NSProcessInfo processInfo] endActivity:ctx->activity];
  delete ctx;
}

void* C_Metal_new_embedded(int n1, int n2, int n3) {
#ifdef DAMASK_METAL_EMBEDDED
  return C_Metal_new(n1, n2, n3, kMetalSource);
#else
  return nullptr;
#endif
}

const char* C_Metal_device_name(void* p) {
  return static_cast<Context*>(p)->name;
}

void C_Metal_device_name_copy(void* p, char* buf, int len) {
  const char* name = static_cast<Context*>(p)->name;
  int n = int(std::strlen(name));
  for (int i = 0; i < len; ++i) buf[i] = i < n ? name[i] : ' ';
}

int C_Metal_set_xi(void* p, const double* xi1st) {
  Context* ctx = static_cast<Context*>(p);
  float* xi = static_cast<float*>(ctx->xi.contents);
  const size_t h = ctx->h, n2 = ctx->n2, n3 = ctx->n3;
  for (size_t y = 0; y < n2; ++y)
    for (size_t z = 0; z < n3; ++z)
      for (size_t x = 0; x < h; ++x) {
        size_t src = 3*(x + h*(z + n3*y));                                                          // xi1st(l,x,z,y)
        size_t w   = x + h*(y + n2*z);
        for (size_t l = 0; l < 3; ++l) {
          xi[2*(3*w+l)]   = float(xi1st[2*(src+l)]);
          xi[2*(3*w+l)+1] = float(xi1st[2*(src+l)+1]);
        }
      }
  return 0;
}

int C_Metal_set_C_ref(void* p, const double* c_ref) {
  Context* ctx = static_cast<Context*>(p);
  double c0 = 0.0;
  for (int k = 0; k < 81; ++k) c0 = std::max(c0, std::fabs(c_ref[k]));
  if (!(c0 > 0.0)) return 1;
  for (int k = 0; k < 81; ++k) ctx->params.C_hat[k] = float(c_ref[k]/c0);
  ctx->params.inv_c0 = float(1.0/c0);
  return 0;
}

int C_Metal_gamma_convolution(void* p, const double* field, const double* aim, double* out) {
  Context* ctx = static_cast<Context*>(p);
  @autoreleasepool {
    auto t0 = Clock::now();
    float* r = static_cast<float*>(ctx->real.contents);
    const size_t N = ctx->N;
    parallel_for(N, [&](size_t b, size_t e) {                                                       // (9,N) double -> [9][N] float
      for (size_t i = b; i < e; ++i)
        for (size_t c = 0; c < 9; ++c) r[c*N + i] = float(field[9*i + c]);
    });
    for (int c = 0; c < 9; ++c) ctx->params.aim[c] = float(aim[c]);
    auto t1 = Clock::now();

    MPSCommandBuffer* cb = [MPSCommandBuffer commandBufferFromCommandQueue:ctx->queue];
    [ctx->fft_forth encodeToCommandBuffer:cb inputsArray:@[ctx->td_real]    resultsArray:@[ctx->td_fourier] executionDescriptor:nil];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->gamma];
    [enc setBuffer:ctx->fourier offset:0 atIndex:0];
    [enc setBuffer:ctx->xi      offset:0 atIndex:1];
    [enc setBytes:&ctx->params length:sizeof(ctx->params) atIndex:2];
    NSUInteger tg = std::min<NSUInteger>(ctx->gamma.maxTotalThreadsPerThreadgroup, 256);
    [enc dispatchThreads:MTLSizeMake(ctx->M, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [ctx->fft_back  encodeToCommandBuffer:cb inputsArray:@[ctx->td_fourier] resultsArray:@[ctx->td_real]    executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr, "DAMASK_Metal: command buffer failed: %s\n",
                   cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
      return 1;
    }
    auto t2 = Clock::now();

    parallel_for(N, [&](size_t b, size_t e) {                                                       // [9][N] float -> (9,N) double
      for (size_t i = b; i < e; ++i)
        for (size_t c = 0; c < 9; ++c) out[9*i + c] = double(r[c*N + i]);
    });
    auto t3 = Clock::now();
    ctx->t_gpu  = seconds(t1, t2);
    ctx->t_host = seconds(t0, t1) + seconds(t2, t3);
  }
  return 0;
}

int C_Metal_cp_set_source(void* p, const char* msl_source) {
  @autoreleasepool {
    return cp_init(static_cast<Context*>(p), msl_source) ? 0 : 1;
  }
}

int C_Metal_cp_add_phase_model(void* p, int model, int N_sl, const double* C66,
                               const double* P_sl, const double* P_nS_pos, const double* P_nS_neg,
                               const double* h_sl_sl, const double* vecs, const double* scalars,
                               const double* numerics) {
  Context* ctx = static_cast<Context*>(p);
  @autoreleasepool {
    if (!cp_init_embedded(ctx)) return -1;
    const int Np = cp_padded(N_sl);
    if (Np == 0 || N_sl <= 0) return -2;
    CPPhase ph;
    ph.model = model; ph.N = N_sl; ph.Np = Np;
    const char* policy = model == 1 ? "cp_model_pheno" : model == 2 ? "cp_model_kineh" : model == 3 ? "cp_model_dtung"
                       : model == 4 ? "cp_model_phenotw" : nullptr;
    if (!policy) return -2;
    if (!cp_pipelines(ctx, std::string(policy) + "<" + std::to_string(Np) + ">", ph)) return -3;

    // common: stiffness and Schmid tensors (padded systems: zero), interaction matrix with padded stride
    auto fill_common = [&](auto* prm) {
      std::memset(prm, 0, sizeof(*prm));
      for (int k = 0; k < 36; ++k) prm->C66[k] = cp_to_pair(C66[k]);
      for (int k = 0; k < 9*N_sl; ++k) {
        prm->P_sl[k] = cp_to_pair(P_sl[k]); prm->P_nS_pos[k] = cp_to_pair(P_nS_pos[k]); prm->P_nS_neg[k] = cp_to_pair(P_nS_neg[k]);
      }
      for (int i = 0; i < N_sl; ++i) for (int j = 0; j < N_sl; ++j) prm->h_sl_sl[i+Np*j] = cp_to_pair(h_sl_sl[i+N_sl*j]);
      prm->atol_xi = float(scalars[0]); prm->atol_gamma = float(scalars[1]); prm->N_sl = N_sl;
    };
    // per-system vectors: actual values, padded systems get pad (inert, no division by zero)
    auto vec = [&](cp_pair* dst, int k, double pad) {
      for (int i = 0; i < Np; ++i) dst[i] = cp_to_pair(i < N_sl ? vecs[k*N_sl + i] : pad);
    };
    if (model == 1) {
      ph.param = [ctx->device newBufferWithLength:sizeof(cp_pheno_param) options:MTLResourceStorageModeShared];
      cp_pheno_param* prm = static_cast<cp_pheno_param*>(ph.param.contents);
      fill_common(prm);
      vec(prm->dot_gamma_0, 0, 0.0); vec(prm->n, 1, 1.0); vec(prm->a, 2, 1.0); vec(prm->h_0, 3, 0.0); vec(prm->xi_inf, 4, 1.0);
    } else if (model == 4) {
      ph.param = [ctx->device newBufferWithLength:sizeof(cp_phenotw_param) options:MTLResourceStorageModeShared];
      cp_phenotw_param* prm = static_cast<cp_phenotw_param*>(ph.param.contents);
      fill_common(prm);
      vec(prm->dot_gamma_0, 0, 0.0); vec(prm->n, 1, 1.0);      vec(prm->a, 2, 1.0);       vec(prm->h_0, 3, 0.0);
      vec(prm->xi_inf, 4, 1.0);      vec(prm->c_1, 5, 0.0);    vec(prm->c_2, 6, 1.0);     vec(prm->f_sat, 7, 0.0);
      vec(prm->h_0_tw_sl, 8, 0.0);   vec(prm->h_0_tw_tw, 9, 0.0); vec(prm->c_3, 10, 1.0); vec(prm->c_4, 11, 0.0);
      vec(prm->gamma_char, 12, 1.0);
      prm->N_slip = int(scalars[2]);
      prm->N_twin = N_sl - prm->N_slip;
    } else if (model == 3) {
      ph.param = [ctx->device newBufferWithLength:sizeof(cp_dtung_param) options:MTLResourceStorageModeShared];
      cp_dtung_param* prm = static_cast<cp_dtung_param*>(ph.param.contents);
      fill_common(prm);
      vec(prm->b_sl, 0, 1.0);  vec(prm->Q_s, 1, 0.0);   vec(prm->i_sl, 2, 1.0);  vec(prm->tau_P, 3, 1.0);
      vec(prm->p, 4, 1.0);     vec(prm->q, 5, 1.0);     vec(prm->h, 6, 0.0);     vec(prm->w, 7, 0.0);
      vec(prm->omega, 8, 1.0); vec(prm->B, 9, 1.0);     vec(prm->d_caron, 10, 0.0); vec(prm->f_at, 11, 0.0);
      const double* fp = vecs + 12*size_t(N_sl);                                                    // forestProjection (N_sl,N_sl)
      for (int i = 0; i < N_sl; ++i) for (int j = 0; j < N_sl; ++j) prm->forest[i+Np*j] = cp_to_pair(fp[i+N_sl*j]);
      prm->D = cp_to_pair(scalars[2]); prm->D_0 = cp_to_pair(scalars[3]); prm->Q_cl = cp_to_pair(scalars[4]);
      prm->mu = cp_to_pair(scalars[5]); prm->nu = cp_to_pair(scalars[6]); prm->k_B = cp_to_pair(scalars[7]);
      prm->dipoleformation = scalars[8] != 0.0;
    } else {
      ph.param = [ctx->device newBufferWithLength:sizeof(cp_kineh_param) options:MTLResourceStorageModeShared];
      cp_kineh_param* prm = static_cast<cp_kineh_param*>(ph.param.contents);
      fill_common(prm);
      vec(prm->dot_gamma_0, 0, 0.0); vec(prm->n, 1, 1.0); vec(prm->h_0_xi, 2, 0.0); vec(prm->h_0_chi, 3, 0.0);
      vec(prm->h_inf_xi, 4, 0.0); vec(prm->h_inf_chi, 5, 0.0); vec(prm->xi_inf, 6, 1.0); vec(prm->chi_inf, 7, 1.0);
    }
    cp_set_numerics(ph, C66, numerics);
    if (getenv("DAMASK_METAL_VERBOSE"))
      std::fprintf(stderr, "DAMASK_Metal: phase model %s, %d slip systems (kernel %d), threadgroup %lu\n", policy, N_sl, Np,
                   (unsigned long)ph.integrate.maxTotalThreadsPerThreadgroup);
    return cp_register_phase(ctx, ph);
  }
}

//> tdt (see phase_mechanical_plastic_tdt.f90, plastic_tdt_GPUadd for the blob layout)
int C_Metal_cp_add_phase_tdt(void* p, const int* dims, const double* C66, const double* blob,
                             const double* atols, const double* numerics) {
  Context* ctx = static_cast<Context*>(p);
  @autoreleasepool {
    if (!cp_init_embedded(ctx)) return -1;
    const int ns = dims[0], nt = dims[1], nf = dims[2];
    int NSL, NTW, NFAM;
    cp_tdt_caps(ns, nt, nf, NSL, NTW, NFAM);
    if (NSL == 0 || NTW == 0 || NFAM == 0 || ns <= 0 || nt <= 0 || nf <= 0) return -2;
    CPPhase ph;
    ph.model = 5; ph.N = ns; ph.Np = NSL; ph.Np_tw = NTW; ph.Np_fam = NFAM;
    std::string type = "cp_model_tdt<" + std::to_string(NSL) + "," + std::to_string(NTW) + "," + std::to_string(NFAM) + ">";
    if (!cp_pipelines(ctx, type, ph)) return -3;

    if (const char* cap = getenv("DAMASK_METAL_CAPTURE"))                                           // diagnostics: phase for replay (bench_cp_replay)
      if (FILE* fo = std::fopen((std::string(cap) + ".phase").c_str(), "wb")) {
        const int nb = 9*ns + 9*ns*nt + 9*nt + 36*(1+nt) + 6*ns + 3*nf + 7*nt + ns*ns + 2*ns*nt + nt*nt + 3;
        std::fwrite(dims, sizeof(int), 4, fo); std::fwrite(&nb, sizeof(int), 1, fo);
        std::fwrite(C66, sizeof(double), 36, fo); std::fwrite(blob, sizeof(double), nb, fo);
        std::fwrite(atols, sizeof(double), 3, fo); std::fwrite(numerics, sizeof(double), 7, fo); std::fclose(fo);
      }
    ph.param = [ctx->device newBufferWithLength:sizeof(cp_tdt_param) options:MTLResourceStorageModeShared];
    cp_tdt_pack(dims, C66, blob, atols, NSL, NTW, NFAM, *static_cast<cp_tdt_param*>(ph.param.contents), ph.smap, ph.sinit);
    if (const char* dump = getenv("DAMASK_METAL_DUMP"))                                             // diagnostics: parameters for replay
      if (FILE* fo = std::fopen((std::string(dump) + ".param").c_str(), "wb")) {
        std::fwrite(ph.param.contents, sizeof(cp_tdt_param), 1, fo); std::fclose(fo);
      }

    cp_set_numerics(ph, C66, numerics);
    if (getenv("DAMASK_METAL_VERBOSE"))
      std::fprintf(stderr, "DAMASK_Metal: phase model %s (%d slip, %d twin systems, %d slip families)\n", type.c_str(), ns, nt, nf);
    return cp_register_phase(ctx, ph);
  }
}

//> twintrip (see phase_mechanical_plastic_twintrip.f90, plastic_twintrip_GPUadd for dims and the blob layout)
int C_Metal_cp_add_phase_twintrip(void* p, const int* dims, int n_blob, const double* C66, const double* blob,
                                  const double* atol, const double* numerics) {
  Context* ctx = static_cast<Context*>(p);
  @autoreleasepool {
    if (!cp_init_embedded(ctx)) return -1;
    CPPhase ph;
    ph.model = 6; ph.N = 12; ph.Np = 12;
    for (int k = 0; k < 9; ++k) ph.tt[k] = dims[k];
    ph.n_state = dims[10] + 60 + dims[0] + dims[1] + dims[2];
    bool known = false;
#define CP_TT_KNOWN(q1,q2,q3,q4,q5,q6,q7,q8,q9) known = known || (dims[0]==q1 && dims[1]==q2 && dims[2]==q3 && dims[3]==q4 && dims[4]==q5 \
                                                      && dims[5]==q6 && dims[6]==q7 && dims[7]==q8 && dims[8]==q9);
    CP_TT_CASES(CP_TT_KNOWN)
#undef CP_TT_KNOWN
    if (!known || dims[10] > CP_TT_NDMAX) return -2;
    std::string type = "cp_model_twintrip<";
    for (int k = 0; k < 9; ++k) type += std::to_string(dims[k]) + (k < 8 ? "," : ">");
    if (!cp_pipelines(ctx, type, ph)) return -3;

    if (const char* cap = getenv("DAMASK_METAL_CAPTURE"))                                           // diagnostics: phase for replay (bench_cp_replay)
      if (FILE* fo = std::fopen((std::string(cap) + ".ttphase").c_str(), "wb")) {
        const int model = 6;
        std::fwrite(&model, sizeof(int), 1, fo); std::fwrite(dims, sizeof(int), 11, fo); std::fwrite(&n_blob, sizeof(int), 1, fo);
        std::fwrite(C66, sizeof(double), 36, fo); std::fwrite(blob, sizeof(double), n_blob, fo);
        std::fwrite(atol, sizeof(double), dims[10], fo); std::fwrite(numerics, sizeof(double), 7, fo); std::fclose(fo);
      }
    ph.param = [ctx->device newBufferWithLength:sizeof(cp_twintrip_param) options:MTLResourceStorageModeShared];
    cp_twintrip_pack(dims, C66, blob, atol, *static_cast<cp_twintrip_param*>(ph.param.contents));

    cp_set_numerics(ph, C66, numerics);
    if (getenv("DAMASK_METAL_VERBOSE"))
      std::fprintf(stderr, "DAMASK_Metal: phase model %s (state %d)\n", type.c_str(), dims[10]);
    return cp_register_phase(ctx, ph);
  }
}

int C_Metal_cp_add_phase(void* p, int N_sl, const double* C66,
                         const double* P_sl, const double* P_nS_pos, const double* P_nS_neg,
                         const double* h_sl_sl, const double* dot_gamma_0, const double* n,
                         const double* a, const double* h_0, const double* xi_inf,
                         double atol_xi, double atol_gamma, const double* numerics) {
  std::vector<double> v(5*size_t(N_sl));
  for (int i = 0; i < N_sl; ++i) {
    v[i] = dot_gamma_0[i]; v[N_sl+i] = n[i]; v[2*N_sl+i] = a[i]; v[3*N_sl+i] = h_0[i]; v[4*N_sl+i] = xi_inf[i];
  }
  const double sc[2] = {atol_xi, atol_gamma};
  return C_Metal_cp_add_phase_model(p, 1, N_sl, C66, P_sl, P_nS_pos, P_nS_neg, h_sl_sl, v.data(), sc, numerics);
}

int C_Metal_cp_integrate(void* p, int phase, int n, double dt,
                         const double* F, const double* Fp0, const double* S0, const double* Lp_guess,
                         const double* state0, const double* T,
                         double* P, double* S, double* Fp, double* Fe, double* Lp, double* state, float* dPdF,
                         int* status, int* iterations) {
  Context* ctx = static_cast<Context*>(p);
  if (phase < 0 || phase >= int(ctx->cp_phases.size())) return 1;
  if (n <= 0) return 0;
  const CPPhase& ph = ctx->cp_phases[phase];
  if (const char* cap = getenv("DAMASK_METAL_CAPTURE")) {                                           // diagnostics: inputs of call #DAMASK_METAL_CAPTURE_CALL, then exit
    static int n_call = 0;
    const int at = getenv("DAMASK_METAL_CAPTURE_CALL") ? atoi(getenv("DAMASK_METAL_CAPTURE_CALL")) : 100;
    if (++n_call == at) {
      if (FILE* fo = std::fopen((std::string(cap) + ".batch").c_str(), "wb")) {
        const int NS = ph.smap.empty() ? ph.n_state : int(ph.smap.size());
        std::fwrite(&n, sizeof(int), 1, fo); std::fwrite(&NS, sizeof(int), 1, fo); std::fwrite(&dt, sizeof(double), 1, fo);
        for (const double* a : {F, Fp0, S0, Lp_guess}) std::fwrite(a, sizeof(double), 9*size_t(n), fo);
        std::fwrite(state0, sizeof(double), size_t(NS)*n, fo);
        std::fwrite(T, sizeof(double), size_t(n), fo); std::fclose(fo);
      }
      std::fprintf(stderr, "DAMASK_Metal: captured call %d (%d points) to %s.batch\n", at, n, cap);
      std::exit(0);
    }
  }
  @autoreleasepool {
#define CP_CASE(MODEL, POLICY, NP)                                                                  \
    if (ph.model == MODEL && ph.Np == NP)                                                           \
      return cp_run<POLICY<NP>>(ctx, ph, n, dt, F, Fp0, S0, Lp_guess, state0, T, P, S, Fp, Fe, Lp, state, dPdF, status, iterations);
    CP_CASE(1, cp_model_pheno, 12) CP_CASE(1, cp_model_pheno, 24) CP_CASE(1, cp_model_pheno, 30) CP_CASE(1, cp_model_pheno, 48)
    CP_CASE(2, cp_model_kineh, 12) CP_CASE(2, cp_model_kineh, 24) CP_CASE(2, cp_model_kineh, 30) CP_CASE(2, cp_model_kineh, 48)
    CP_CASE(3, cp_model_dtung, 12) CP_CASE(3, cp_model_dtung, 24) CP_CASE(3, cp_model_dtung, 30) CP_CASE(3, cp_model_dtung, 48)
    CP_CASE(4, cp_model_phenotw, 12) CP_CASE(4, cp_model_phenotw, 24) CP_CASE(4, cp_model_phenotw, 30) CP_CASE(4, cp_model_phenotw, 48)
#undef CP_CASE
#define CP_TDT_CASE(A, B, C)                                                                        \
    if (ph.model == 5 && ph.Np == A && ph.Np_tw == B && ph.Np_fam == C)                              \
      return cp_run<cp_model_tdt<A,B,C>>(ctx, ph, n, dt, F, Fp0, S0, Lp_guess, state0, T, P, S, Fp, Fe, Lp, state, dPdF, status, iterations);
    CP_TDT_CASES(CP_TDT_CASE)
#undef CP_TDT_CASE
#define CP_TT_CASE(q1,q2,q3,q4,q5,q6,q7,q8,q9)                                                               \
    if (ph.model == 6 && ph.tt[0]==q1 && ph.tt[1]==q2 && ph.tt[2]==q3 && ph.tt[3]==q4 && ph.tt[4]==q5     \
        && ph.tt[5]==q6 && ph.tt[6]==q7 && ph.tt[7]==q8 && ph.tt[8]==q9)                                 \
      return cp_run<cp_model_twintrip<q1,q2,q3,q4,q5,q6,q7,q8,q9>>(ctx, ph, n, dt, F, Fp0, S0, Lp_guess, state0, T, P, S, Fp, Fe, Lp, state, dPdF, status, iterations);
    CP_TT_CASES(CP_TT_CASE)
#undef CP_TT_CASE
  }
  return 1;
}

int C_Metal_cp_submit(void* p, int phase, int n, double dt,
                      const double* F, const double* Fp0, const double* S0, const double* Lp_guess,
                      const double* state0, const double* T,
                      double* P, double* S, double* Fp, double* Fe, double* Lp, double* state, float* dPdF,
                      int* status, int* iterations) {
  Context* ctx = static_cast<Context*>(p);
  if (phase < 0 || phase >= int(ctx->cp_phases.size())) return 1;
  if (!ctx->cp_queue) {
    ctx->cp_queue = dispatch_queue_create("DAMASK_Metal.cp", DISPATCH_QUEUE_SERIAL);
    ctx->cp_group = dispatch_group_create();
  }
  ctx->cp_job_rc[phase] = -1;
  auto t0 = Clock::now();
  dispatch_group_async(ctx->cp_group, ctx->cp_queue, ^{
    int rc = C_Metal_cp_integrate(p, phase, n, dt, F, Fp0, S0, Lp_guess, state0, T, P, S, Fp, Fe, Lp, state, dPdF,
                                  status, iterations);
    ctx->cp_job_time[phase] = seconds(t0, Clock::now());
    ctx->cp_job_rc[phase] = rc;
  });
  return 0;
}

int C_Metal_cp_wait(void* p, int phase, double* t_wall) {
  Context* ctx = static_cast<Context*>(p);
  if (ctx->cp_group) dispatch_group_wait(ctx->cp_group, DISPATCH_TIME_FOREVER);
  if (phase < 0 || phase >= int(ctx->cp_job_rc.size())) return 1;
  if (t_wall) *t_wall = ctx->cp_job_time[phase];
  return ctx->cp_job_rc[phase];
}

void C_Metal_cp_set_precision(int mode) { g_cp_prec = mode; }

void C_Metal_last_timing(void* p, double* t_gpu, double* t_host) {
  Context* ctx = static_cast<Context*>(p);
  *t_gpu  = ctx->t_gpu;
  *t_host = ctx->t_host;
}

}
