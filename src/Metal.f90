! SPDX-License-Identifier: AGPL-3.0-or-later
!--------------------------------------------------------------------------------------------------
!> @brief Interface to the Apple GPU (Metal) backend of the grid solver (src/grid/metal).
!> @details Single MPI rank only. All arrays are passed as contiguous FP64 (Fortran order);
!> the backend computes in FP32 (with FP64-equivalent residuals where accuracy demands it).
!> Shared by the spectral operators (spectral_utilities) and the constitutive update (phase), hence
!> located in src/; only active for the grid solver.
!--------------------------------------------------------------------------------------------------
module Metal
  use, intrinsic :: ISO_C_binding

  implicit none(type,external)
  private

#if defined(METAL) && defined(GRID)
  type(C_PTR), public :: &
    Metal_ctx = C_NULL_PTR                                                                          !< backend, created by spectral_utilities_init
  integer, public :: &
    Metal_nState = -1                                                                               !< max. state iterations on the GPU (points needing more: CPU); -1: per model
  logical, public :: &
    Metal_constitutive = .false., &                                                                 !< constitutive update on GPU
    Metal_tangent      = .false.                                                                    !< tangent dPdF of GPU-integrated points on GPU
  real(C_DOUBLE), public :: &
    Metal_rtol_S = 1.0e-10_C_DOUBLE, &                                                              !< relative tolerance of the GPU stress residual
    Metal_split  = -1.0_C_DOUBLE                                                                    !< share of the GPU-capable cells integrated on the GPU while the CPU does the rest (< 0: automatic)

  interface

    function Metal_new(n1,n2,n3) bind(C, name='C_Metal_new_embedded') result(ctx)
      import :: C_INT, C_PTR
      integer(C_INT), value :: n1, n2, n3
      type(C_PTR) :: ctx
    end function Metal_new

    subroutine Metal_delete(ctx) bind(C, name='C_Metal_delete')
      import :: C_PTR
      type(C_PTR), value :: ctx
    end subroutine Metal_delete

    subroutine Metal_deviceName(ctx,buf,len) bind(C, name='C_Metal_device_name_copy')
      import :: C_PTR, C_CHAR, C_INT
      type(C_PTR), value :: ctx
      character(kind=C_CHAR), dimension(*) :: buf
      integer(C_INT), value :: len
    end subroutine Metal_deviceName

    function Metal_setXi(ctx,xi1st) bind(C, name='C_Metal_set_xi') result(stat)
      import :: C_PTR, C_INT, C_DOUBLE_COMPLEX
      type(C_PTR), value :: ctx
      complex(C_DOUBLE_COMPLEX), dimension(*), intent(in) :: xi1st
      integer(C_INT) :: stat
    end function Metal_setXi

    function Metal_setCRef(ctx,C_ref) bind(C, name='C_Metal_set_C_ref') result(stat)
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value :: ctx
      real(C_DOUBLE), dimension(*), intent(in) :: C_ref
      integer(C_INT) :: stat
    end function Metal_setCRef

    function Metal_GammaConvolution(ctx,field,aim,gammaField) bind(C, name='C_Metal_gamma_convolution') result(stat)
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value :: ctx
      real(C_DOUBLE), dimension(*), intent(in)  :: field, aim
      real(C_DOUBLE), dimension(*), intent(out) :: gammaField
      integer(C_INT) :: stat
    end function Metal_GammaConvolution

    function Metal_cpAddPhase(ctx,N_sl,C66,P_sl,P_nS_pos,P_nS_neg,h_sl_sl,dot_gamma_0,n,a,h_0,xi_inf, &
                              atol_xi,atol_gamma,numerics) bind(C, name='C_Metal_cp_add_phase') result(handle)
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value :: ctx
      integer(C_INT), value :: N_sl
      real(C_DOUBLE), dimension(*), intent(in) :: C66, P_sl, P_nS_pos, P_nS_neg, h_sl_sl, &
                                                  dot_gamma_0, n, a, h_0, xi_inf, numerics
      real(C_DOUBLE), value :: atol_xi, atol_gamma
      integer(C_INT) :: handle
    end function Metal_cpAddPhase

    function Metal_cpAddPhaseModel(ctx,model,N_sl,C66,P_sl,P_nS_pos,P_nS_neg,h_sl_sl,vecs,scalars,numerics) &
                                   bind(C, name='C_Metal_cp_add_phase_model') result(handle)
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value :: ctx
      integer(C_INT), value :: model, N_sl
      real(C_DOUBLE), dimension(*), intent(in) :: C66, P_sl, P_nS_pos, P_nS_neg, h_sl_sl, vecs, scalars, numerics
      integer(C_INT) :: handle
    end function Metal_cpAddPhaseModel

    function Metal_cpAddPhaseTdt(ctx,dims,C66,blob,atols,numerics) &
                                 bind(C, name='C_Metal_cp_add_phase_tdt') result(handle)
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value :: ctx
      integer(C_INT), dimension(*), intent(in) :: dims
      real(C_DOUBLE), dimension(*), intent(in) :: C66, blob, atols, numerics
      integer(C_INT) :: handle
    end function Metal_cpAddPhaseTdt

    function Metal_cpAddPhaseTwintrip(ctx,dims,n_blob,C66,blob,atol,numerics) &
                                      bind(C, name='C_Metal_cp_add_phase_twintrip') result(handle)
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value :: ctx
      integer(C_INT), dimension(*), intent(in) :: dims
      integer(C_INT), value :: n_blob
      real(C_DOUBLE), dimension(*), intent(in) :: C66, blob, atol, numerics
      integer(C_INT) :: handle
    end function Metal_cpAddPhaseTwintrip

    function Metal_cpSubmit(ctx,handle,n,dt,F,Fp0,S0,Lp_guess,state0,T,P,S,Fp,Fe,Lp,state,dPdF,status,iterations) &
                            bind(C, name='C_Metal_cp_submit') result(stat)
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value :: ctx
      integer(C_INT), value :: handle, n
      real(C_DOUBLE), value :: dt
      real(C_DOUBLE), dimension(*), intent(in),    asynchronous :: F, Fp0, S0, Lp_guess, state0
      type(C_PTR), value :: T
      real(C_DOUBLE), dimension(*), intent(inout), asynchronous :: P, S, Fp, Fe, Lp, state
      type(C_PTR), value :: dPdF
      integer(C_INT), dimension(*), intent(inout), asynchronous :: status, iterations
      integer(C_INT) :: stat
    end function Metal_cpSubmit

    function Metal_cpWait(ctx,handle,t_wall) bind(C, name='C_Metal_cp_wait') result(stat)
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value :: ctx
      integer(C_INT), value :: handle
      real(C_DOUBLE), intent(out) :: t_wall
      integer(C_INT) :: stat
    end function Metal_cpWait

    subroutine Metal_cpSetPrecision(mode) bind(C, name='C_Metal_cp_set_precision')
      import :: C_INT
      integer(C_INT), value :: mode
    end subroutine Metal_cpSetPrecision

  end interface

  public :: &
    Metal_new, &
    Metal_delete, &
    Metal_deviceName, &
    Metal_setXi, &
    Metal_setCRef, &
    Metal_GammaConvolution, &
    Metal_cpAddPhase, &
    Metal_cpAddPhaseModel, &
    Metal_cpAddPhaseTdt, &
    Metal_cpAddPhaseTwintrip, &
    Metal_cpSubmit, &
    Metal_cpWait, &
    Metal_cpSetPrecision
#endif

end module Metal
