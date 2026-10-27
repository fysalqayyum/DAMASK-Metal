// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief C interface of the Metal (Apple GPU) backend of the grid solver.
//> @details All arrays are Fortran-ordered doubles; conversion to FP32 happens inside.
//> Functions returning int return 0 on success.
//--------------------------------------------------------------------------------------------------
#ifndef DAMASK_METAL_H
#define DAMASK_METAL_H

#ifdef __cplusplus
extern "C" {
#endif

//> create the backend for a single-rank grid of n1 x n2 x n3 cells; NULL if no GPU is available.
//> msl_source: Metal Shading Language source (embedded at build time), compiled at runtime
void* C_Metal_new(int n1, int n2, int n3, const char* msl_source);
void* C_Metal_new_embedded(int n1, int n2, int n3);                                                 // shaders embedded at build time
void  C_Metal_delete(void* ctx);
const char* C_Metal_device_name(void* ctx);
void  C_Metal_device_name_copy(void* ctx, char* buf, int len);                                      // blank padded (Fortran string)

//> wave vectors: DAMASK's xi1st, complex(3, n1/2+1, n3, n2) (transposed Fourier layout of FFTW)
int C_Metal_set_xi(void* ctx, const double* xi1st);

//> reference stiffness as stored by utilities_updateGamma (C/wgt), real(3,3,3,3)
int C_Metal_set_C_ref(void* ctx, const double* c_ref);

//> constitutive model ids (model of C_Metal_cp_add_phase_model; Fortran: METAL_MODEL_* in Metal.f90)
enum {
  CP_MODEL_PHENO    = 1,                                                                            // phenopowerlaw, slip only
  CP_MODEL_KINEH    = 2,                                                                            // kinehardening
  CP_MODEL_DTUNG    = 3,                                                                            // dislotungsten
  CP_MODEL_PHENOTW  = 4,                                                                            // phenopowerlaw with twinning
  CP_MODEL_TDT      = 5,                                                                            // tdt (C_Metal_cp_add_phase_tdt)
  CP_MODEL_TWINTRIP = 6                                                                             // twintrip (C_Metal_cp_add_phase_twintrip)
};

//> out = Gamma * field with average aim, field/out real(3,3,n1,n2,n3) (may alias)
int C_Metal_gamma_convolution(void* ctx, const double* field, const double* aim, double* out);

//> compile the constitutive shaders from the given source instead of the embedded one (tests)
int C_Metal_cp_set_source(void* ctx, const char* msl_source);

//> constitutive update on the GPU: Hooke elasticity + phenopowerlaw slip (no twinning, Schmid law),
//> Fi = I, no eigenstrain; FPI state integration (cp_core.h, cp_model_pheno). N_sl <= 48 (padded to
//> 12, 24, 30 or 48). C66(6,6), P_sl/P_nS_pos/P_nS_neg(3,3,N_sl), h_sl_sl(N_sl,N_sl), per-system
//> vectors (N_sl); h_0 includes the factor (1+c_1*0**c_2) of DAMASK 3.1 (no twins).
//> numerics: rtol_S, atol_S, rtol_state, r_linesearch, trust_S, N_iter_stress, N_iter_state.
//> Returns a handle >= 0 for C_Metal_cp_integrate, < 0 on failure.
int C_Metal_cp_add_phase(void* ctx, int N_sl, const double* C66,
                         const double* P_sl, const double* P_nS_pos, const double* P_nS_neg,
                         const double* h_sl_sl, const double* dot_gamma_0, const double* n,
                         const double* a, const double* h_0, const double* xi_inf,
                         double atol_xi, double atol_gamma, const double* numerics);

//> generic form: model 1 = phenopowerlaw, vecs = [dot_gamma_0, n, a, h_0, xi_inf] (5*N_sl);
//>               model 2 = kinehardening, vecs = [dot_gamma_0, n, h_0_xi, h_0_chi, h_inf_xi, h_inf_chi,
//>                                                xi_inf, chi_inf] (8*N_sl);
//>               model 3 = dislotungsten, vecs = [b_sl, Q_s, i_sl, tau_Peierls, p, q, h, w, omega, B,
//>                                                d_caron, f_at] (12*N_sl) followed by forestProjection (N_sl,N_sl);
//>                                       scalars = [atol_rho, atol_gamma, D, D_0, Q_cl, mu, nu, k_B, dipoleformation]
//>               model 4 = phenopowerlaw with twins: systems = [slip, twin] (N_sl = total), vecs (13*N) =
//>                         [dot_gamma_0, n, a, h_0_sl-sl, xi_inf, c_1, c_2, f_sat_sl-tw, h_0_tw-sl, h_0_tw-tw,
//>                          c_3, c_4, gamma_char] (slip/twin-only entries unused), h_sl_sl = combined (N,N)
//>                         [h_sl_sl h_sl_tw; h_tw_sl h_tw_tw], scalars = [atol_xi, atol_gamma, N_slip]
//> scalars = [atol_xi, atol_gamma] (CP_MODEL_PHENO, CP_MODEL_KINEH). Any N_sl <= 48 (padded to 12/24/30/48 internally).
//> For models with a dependent state (CP_MODEL_DTUNG), state0/state of C_Metal_cp_integrate hold DAMASK's state followed
//> by the dependent state ([tau_pass, Lambda_sl]).
int C_Metal_cp_add_phase_model(void* ctx, int model, int N_sl, const double* C66,
                               const double* P_sl, const double* P_nS_pos, const double* P_nS_neg,
                               const double* h_sl_sl, const double* vecs, const double* scalars,
                               const double* numerics);

//> composite twinning model tdt (phase_mechanical_plastic_tdt.f90, cp_model_tdt in cp_models_composite.h):
//> dims = [N_slip, N_twin, N_slip_families, Reuss(1)/Voigt(0)], C66 = parent stiffness, blob as assembled
//> by plastic_tdt_GPUadd (Schmid tensors of parent and twins, compliances/stiffnesses, per-system and
//> per-family parameters, scaled interaction matrices, f_n, f_sat, f_min, c_1, c_2, f_sat_sl-tw (per slip
//> system); n_blob values, which must be
//> exactly what dims imply, else < 0), atols = [xi, gamma, f].
//> Up to 36 slip systems, 12 twin systems, 6 slip families. state0/state hold DAMASK's state followed by
//> the homogenized C66 (36).
int C_Metal_cp_add_phase_tdt(void* ctx, const int* dims, int n_blob, const double* C66, const double* blob,
                             const double* atols, const double* numerics);

//> add a twintrip phase (phase_mechanical_plastic_twintrip.f90): dims = [N_tw, N_tr, alpha', slip systems
//> and families per plane of the twin, epsilon and alpha' regions, Reuss, sizeState], blob (n_blob values,
//> exactly what dims imply, else < 0) as packed by plastic_twintrip_GPUadd, atol (sizeState). Fi = (1 + Delta_V sum(f_a))^(1/3) I of the
//> returned state (Fe is returned as F invFp invFi). state0/state hold DAMASK's state followed by
//> [C66, Lambda_sl, Lambda_tw, Lambda_tr, tau_pass, (j^2)].
int C_Metal_cp_add_phase_twintrip(void* ctx, const int* dims, int n_blob, const double* C66, const double* blob,
                                  const double* atol, const double* numerics);

//> integrate n points of one phase over dt. Inputs: F, Fp0, S0, Lp_guess (3,3,n), state0 (n_state,n)
//> as DAMASK's plasticState%state0, T (n) temperature or NULL. Outputs (written only where status == 0):
//> P, S, Fp, Fe, Lp (3,3,n), state (n_state,n), dPdF (3,3,3,3,n) in float (DAMASK's analytic tangent;
//> skipped if NULL). status(n): 0 = converged and certified, otherwise the point needs the CPU path.
//> n_state must equal the state stride of the phase (DAMASK state + dependent state), else nonzero is
//> returned and nothing is read or written.
//> iterations(n) may be NULL.
int C_Metal_cp_integrate(void* ctx, int phase, int n, int n_state, double dt,
                         const double* F, const double* Fp0, const double* S0, const double* Lp_guess,
                         const double* state0, const double* T,
                         double* P, double* S, double* Fp, double* Fe, double* Lp, double* state, float* dPdF,
                         int* status, int* iterations);

//> asynchronous form of C_Metal_cp_integrate: queued (one job at a time, in submission order) and
//> returns at once; all arrays must stay valid until C_Metal_cp_wait. C_Metal_cp_wait waits for all
//> queued jobs and returns the result of the job of the given phase, t_wall (may be NULL): its time
//> from submission to completion [s].
int C_Metal_cp_submit(void* ctx, int phase, int n, int n_state, double dt,
                      const double* F, const double* Fp0, const double* S0, const double* Lp_guess,
                      const double* state0, const double* T,
                      double* P, double* S, double* Fp, double* Fe, double* Lp, double* state, float* dPdF,
                      int* status, int* iterations);
int C_Metal_cp_wait(void* ctx, int phase, double* t_wall);

//> precision of the constitutive update of phases added afterwards (DAMASK_METAL_PRECISION overrides):
//> 0 df64, 1 mixed (FP32 pre-solve, df64 polish to eps_rel_S), 2 fast (FP32 kinetics and stress solve,
//> df64 state accumulation and kinematics)
void C_Metal_cp_set_precision(int mode);

//> timings of the last call [s]: GPU execution and host-side conversion
void C_Metal_last_timing(void* ctx, double* t_gpu, double* t_host);

#ifdef __cplusplus
}
#endif

#endif
