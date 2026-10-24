! SPDX-License-Identifier: AGPL-3.0-or-later
!----------------------------------------------------------------------------------------------------
!> @brief internal microstructure state for all plasticity constitutive models
!----------------------------------------------------------------------------------------------------
submodule(phase) mechanical

  type(tTensorContainer), dimension(:), allocatable :: &
    ! current value
    phase_mechanical_Fe, &
    phase_mechanical_Fi, &
    phase_mechanical_Fp, &
    phase_mechanical_F, &
    phase_mechanical_Li, &
    phase_mechanical_Lp, &
    phase_mechanical_S, &
    phase_mechanical_P, &
    ! converged value at end of last solver increment
    phase_mechanical_Fi0, &
    phase_mechanical_Fp0, &
    phase_mechanical_F0, &
    phase_mechanical_Li0, &
    phase_mechanical_Lp0, &
    phase_mechanical_S0

#if defined(METAL) && defined(GRID)
  integer, dimension(:), allocatable :: &
    GPU_handle                                                                                      !< GPU constitutive model of each phase (< 0: CPU only)
  real(pREAL), dimension(:,:,:), allocatable :: &
    GPU_C66                                                                                         !< stiffness uploaded to the GPU
  type :: tGPUtangent
    real(C_FLOAT), dimension(:,:,:,:,:), allocatable :: dPdF                                        !< analytic tangent computed on the GPU
    logical,       dimension(:),         allocatable :: valid                                       !< dPdF belongs to the current state of the entry
  end type tGPUtangent
  type(tGPUtangent), dimension(:), allocatable :: GPU_tangent
  type :: tGPUwork                                                                                  !< batch buffers, kept across calls (avoids page faults)
    real(pREAL), dimension(:,:,:), allocatable :: F, Fp0, S0, Lp_guess, P, S, Fp, Fe, Lp
    real(pREAL), dimension(:,:),   allocatable :: state0, state
    integer(C_INT), dimension(:),  allocatable :: stat, iterations
    real(C_FLOAT), dimension(:,:,:,:,:), allocatable :: dPdF
    real(C_DOUBLE), dimension(:), allocatable :: T
    integer, dimension(:), allocatable :: ce_list
    logical :: submitted = .false.                                                                  !< queued on the GPU, not yet unpacked
    integer :: n_CPU = 0                                                                            !< GPU-capable cells left to the CPU (split)
    real(pREAL) :: share = 0.8_pREAL                                                                !< share of the GPU-capable cells on the GPU (automatic split)
    real(pREAL) :: dt = 0.0_pREAL                                                                   !< time step of the submitted batch
  end type tGPUwork
  integer, dimension(:), allocatable :: GPU_ndep                                                    !< size of the dependent state passed along with the state
  type(tGPUwork), dimension(:), allocatable, target :: GPU_work
  logical :: GPU_tangent_verify = .false.                                                           !< diagnostics: compare GPU and CPU tangent (env DAMASK_METAL_VERIFY_TANGENT)
  logical :: GPU_verbose = .false.                                                                  !< diagnostics: split timings (env DAMASK_METAL_VERBOSE)
  real(pREAL) :: GPU_tangent_maxdiff = 0.0_pREAL
#endif


  interface

    module subroutine eigen_init(phases)
      type(tDict), pointer :: phases
    end subroutine eigen_init

#if defined(METAL) && defined(GRID)
    module function eigen_active(ph) result(active)
      integer, intent(in) :: ph
      logical :: active
    end function eigen_active

    module function plastic_phenopowerlaw_GPUadd(ph,C66,numerics) result(handle)
      integer, intent(in) :: ph
      real(pREAL), dimension(6,6), intent(in) :: C66
      real(pREAL), dimension(7), intent(in) :: numerics
      integer :: handle
    end function plastic_phenopowerlaw_GPUadd

    module function plastic_kinehardening_GPUadd(ph,C66,numerics) result(handle)
      integer, intent(in) :: ph
      real(pREAL), dimension(6,6), intent(in) :: C66
      real(pREAL), dimension(7), intent(in) :: numerics
      integer :: handle
    end function plastic_kinehardening_GPUadd

    module function plastic_dislotungsten_GPUadd(ph,C66,numerics) result(handle)
      integer, intent(in) :: ph
      real(pREAL), dimension(6,6), intent(in) :: C66
      real(pREAL), dimension(7), intent(in) :: numerics
      integer :: handle
    end function plastic_dislotungsten_GPUadd

    module function plastic_tdt_GPUadd(ph,C66,numerics) result(handle)
      integer, intent(in) :: ph
      real(pREAL), dimension(6,6), intent(in) :: C66
      real(pREAL), dimension(7), intent(in) :: numerics
      integer :: handle
    end function plastic_tdt_GPUadd

    module function plastic_twintrip_GPUadd(ph,C66,numerics,ndep) result(handle)
      integer, intent(in) :: ph
      real(pREAL), dimension(6,6), intent(in) :: C66
      real(pREAL), dimension(7), intent(in) :: numerics
      integer, intent(out) :: ndep
      integer :: handle
    end function plastic_twintrip_GPUadd

    module subroutine plastic_twintrip_GPUdependent(ph,en,D,set)
      integer, intent(in) :: ph, en
      real(pREAL), dimension(:), intent(inout) :: D
      logical, intent(in) :: set
    end subroutine plastic_twintrip_GPUdependent

    module function plastic_twintrip_GPUj(ph,en) result(j)
      integer, intent(in) :: ph, en
      real(pREAL) :: j
    end function plastic_twintrip_GPUj


    module subroutine plastic_dislotungsten_GPUdependent(ph,en,D,set)
      integer, intent(in) :: ph, en
      real(pREAL), dimension(:), intent(inout) :: D                                                  !< [tau_pass, Lambda_sl]
      logical, intent(in) :: set
    end subroutine plastic_dislotungsten_GPUdependent
#endif

    module subroutine elastic_init(phases)
      type(tDict), pointer :: phases
    end subroutine elastic_init

    module subroutine plastic_init
    end subroutine plastic_init

    module subroutine phase_hooke_SandItsTangents(S,dS_dFe,dS_dFi,Fe,Fi,ph,en)
      integer, intent(in) :: &
        ph, &
        en
      real(pREAL),   intent(in),  dimension(3,3) :: &
        Fe, &                                                                                       !< elastic deformation gradient
        Fi                                                                                          !< intermediate deformation gradient
      real(pREAL),   intent(out), dimension(3,3) :: &
        S                                                                                           !< 2nd Piola-Kirchhoff stress tensor in lattice configuration
      real(pREAL),   intent(out), dimension(3,3,3,3) :: &
        dS_dFe, &                                                                                   !< derivative of 2nd P-K stress with respect to elastic deformation gradient
        dS_dFi                                                                                      !< derivative of 2nd P-K stress with respect to intermediate deformation gradient
    end subroutine phase_hooke_SandItsTangents

    module subroutine plastic_isotropic_LiAndItsTangent(Li,dLi_dMi,Mi,ph,en)
      real(pREAL), dimension(3,3),     intent(out) :: &
        Li                                                                                          !< inleastic velocity gradient
      real(pREAL), dimension(3,3,3,3), intent(out)  :: &
        dLi_dMi                                                                                     !< derivative of Li with respect to Mandel stress
      real(pREAL), dimension(3,3),     intent(in) :: &
        Mi                                                                                          !< Mandel stress
      integer,                         intent(in) :: &
        ph, &
        en
    end subroutine plastic_isotropic_LiAndItsTangent

    module subroutine twintrip_LiAndItsTangent(Li,dLi_dMi,detFi,detFi0,Delta_t,ph,en)
      real(pREAL), dimension(3,3),     intent(out) :: Li
      real(pREAL), dimension(3,3,3,3), intent(out) :: dLi_dMi
      real(pREAL),                     intent(in)  :: detFi, detFi0, Delta_t
      integer,                         intent(in)  :: ph, en
    end subroutine twintrip_LiAndItsTangent

    module function plastic_statePreconditioner(subdt,ph,en) result(pc)
      integer, intent(in) :: &
        ph, &
        en
      real(pREAL),  intent(in) :: &
        subdt
      real(pREAL), dimension(plasticState(ph)%sizeDotState) :: &
        pc
    end function plastic_statePreconditioner

    module function plastic_dotState(subdt,ph,en) result(dotState)
      integer, intent(in) :: &
        ph, &
        en
      real(pREAL),  intent(in) :: &
        subdt                                                                                           !< timestep
      real(pREAL), dimension(plasticState(ph)%sizeDotState) :: &
        dotState
    end function plastic_dotState

    module function plastic_deltaState(ph, en) result(status)
      integer, intent(in) :: &
        ph, &
        en
      integer(kind(STATUS_OK)) :: &
        status
    end function plastic_deltaState

    module subroutine phase_LiAndItsTangents(Li, dLi_dS, dLi_dFi, &
                                             S, Fi, ph,en, Delta_t,invFi0)
      integer, intent(in) :: &
        ph,en
      real(pREAL),   intent(in), optional :: &
        Delta_t                                                                                         !< time increment (state-defined eigenstrains)
      real(pREAL),   intent(in), optional, dimension(3,3) :: &
        invFi0                                                                                          !< inverse of Fi at the start of the increment
      real(pREAL),   intent(in),  dimension(3,3) :: &
        S                                                                                               !< 2nd Piola-Kirchhoff stress
      real(pREAL),   intent(in),  dimension(3,3) :: &
        Fi                                                                                              !< intermediate deformation gradient
      real(pREAL),   intent(out), dimension(3,3) :: &
        Li                                                                                              !< intermediate velocity gradient
      real(pREAL),   intent(out), dimension(3,3,3,3) :: &
        dLi_dS, &                                                                                       !< derivative of Li with respect to S
        dLi_dFi

    end subroutine phase_LiAndItsTangents


    module subroutine plastic_LpAndItsTangents(Lp, dLp_dS, dLp_dFi, &
                                               S, Fi, Delta_t, ph,en)
      integer, intent(in) :: &
        ph,en
      real(pREAL),   intent(in) :: &
        Delta_t                                                                                         !< time step (models with closed-form state updates)
      real(pREAL),   intent(in),  dimension(3,3) :: &
        S, &                                                                                            !< 2nd Piola-Kirchhoff stress
        Fi                                                                                              !< intermediate deformation gradient
      real(pREAL),   intent(out), dimension(3,3) :: &
        Lp                                                                                              !< plastic velocity gradient
      real(pREAL),   intent(out), dimension(3,3,3,3) :: &
        dLp_dS, &
        dLp_dFi                                                                                         !< derivative of Lp with respect to Fi
    end subroutine plastic_LpAndItsTangents


    module subroutine plastic_isotropic_result(ph,group)
      integer,          intent(in) :: ph
      character(len=*), intent(in) :: group
    end subroutine plastic_isotropic_result

    module subroutine plastic_phenopowerlaw_result(ph,group)
      integer,          intent(in) :: ph
      character(len=*), intent(in) :: group
    end subroutine plastic_phenopowerlaw_result

    module subroutine plastic_kinehardening_result(ph,group)
      integer,          intent(in) :: ph
      character(len=*), intent(in) :: group
    end subroutine plastic_kinehardening_result

    module subroutine plastic_dislotwin_result(ph,group)
      integer,          intent(in) :: ph
      character(len=*), intent(in) :: group
    end subroutine plastic_dislotwin_result

    module subroutine plastic_tdt_result(ph,group)
      integer,          intent(in) :: ph
      character(len=*), intent(in) :: group
    end subroutine plastic_tdt_result

    module subroutine plastic_twintrip_result(ph,group)
      integer,          intent(in) :: ph
      character(len=*), intent(in) :: group
    end subroutine plastic_twintrip_result

    module subroutine plastic_dislotungsten_result(ph,group)
      integer,          intent(in) :: ph
      character(len=*), intent(in) :: group
    end subroutine plastic_dislotungsten_result

    module subroutine plastic_nonlocal_result(ph,group)
      integer,          intent(in) :: ph
      character(len=*), intent(in) :: group
    end subroutine plastic_nonlocal_result

    module function plastic_tdt_homogenizedC(ph,en) result(homogenizedC)
      real(pREAL), dimension(6,6) :: homogenizedC
      integer,     intent(in) :: ph, en
    end function plastic_tdt_homogenizedC

    module function plastic_twintrip_homogenizedC(ph,en) result(homogenizedC)
      real(pREAL), dimension(6,6) :: homogenizedC
      integer,     intent(in) :: ph, en
    end function plastic_twintrip_homogenizedC

    module function plastic_dislotwin_homogenizedC(ph,en) result(homogenizedC)
      real(pREAL), dimension(6,6) :: homogenizedC
      integer,     intent(in) :: ph,en
    end function plastic_dislotwin_homogenizedC

    pure module function elastic_C66(ph,en) result(C66)
      real(pREAL), dimension(6,6) :: C66
      integer,     intent(in) :: ph, en
    end function elastic_C66

    pure module function elastic_mu(ph,en,isotropic_bound) result(mu)
      real(pREAL) :: mu
      integer, intent(in) :: ph, en
      character(len=*), intent(in) :: isotropic_bound
    end function elastic_mu

    pure module function elastic_nu(ph,en,isotropic_bound) result(nu)
      real(pREAL) :: nu
      integer, intent(in) :: ph, en
      character(len=*), intent(in) :: isotropic_bound
    end function elastic_nu

  end interface

  type :: tOutput                                                                                   !< requested output (per phase)
    character(len=pSTRLEN), allocatable, dimension(:) :: &
      label
  end type tOutput
  type(tOutput), allocatable, dimension(:) :: output_mechanical

  procedure(integrateStateFPI), pointer :: integrateState

contains


!--------------------------------------------------------------------------------------------------
!> @brief Initialize mechanical field related constitutive models
!> @details Initialize elasticity, plasticity and stiffness degradation models.
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_init(phases, num_mech)

  type(tDict), pointer :: &
    phases, &
    num_mech

  integer :: &
    ce, &
    co, &
    ma, &
    ph, &
    en, &
    Nmembers
  type(tDict), pointer :: &
    phase, &
    mech, &
    num_mech_plastic, &
    num_mech_eigen
  character(len=:), allocatable :: extmsg


  print'(/,1x,a)', '<<<+-  phase:mechanical init  -+>>>'

!-------------------------------------------------------------------------------------------------
  allocate(output_mechanical(size(phases)))

  allocate(phase_mechanical_Fe(size(phases)))
  allocate(phase_mechanical_Fi(size(phases)))
  allocate(phase_mechanical_Fi0(size(phases)))
  allocate(phase_mechanical_Fp(size(phases)))
  allocate(phase_mechanical_Fp0(size(phases)))
  allocate(phase_mechanical_F(size(phases)))
  allocate(phase_mechanical_F0(size(phases)))
  allocate(phase_mechanical_Li(size(phases)))
  allocate(phase_mechanical_Li0(size(phases)))
  allocate(phase_mechanical_Lp(size(phases)))
  allocate(phase_mechanical_Lp0(size(phases)))
  allocate(phase_mechanical_S(size(phases)))
  allocate(phase_mechanical_P(size(phases)))
  allocate(phase_mechanical_S0(size(phases)))

  do ph = 1, size(phases)
    Nmembers = count(material_ID_phase == ph)

    allocate(phase_mechanical_Fe(ph)%data(3,3,Nmembers))
    allocate(phase_mechanical_Fi(ph)%data(3,3,Nmembers))
    allocate(phase_mechanical_Fp(ph)%data(3,3,Nmembers))
    allocate(phase_mechanical_F(ph)%data(3,3,Nmembers))
    allocate(phase_mechanical_Li(ph)%data(3,3,Nmembers),source=0.0_pREAL)
    allocate(phase_mechanical_Li0(ph)%data(3,3,Nmembers),source=0.0_pREAL)
    allocate(phase_mechanical_Lp(ph)%data(3,3,Nmembers),source=0.0_pREAL)
    allocate(phase_mechanical_Lp0(ph)%data(3,3,Nmembers),source=0.0_pREAL)
    allocate(phase_mechanical_S(ph)%data(3,3,Nmembers),source=0.0_pREAL)
    allocate(phase_mechanical_P(ph)%data(3,3,Nmembers),source=0.0_pREAL)
    allocate(phase_mechanical_S0(ph)%data(3,3,Nmembers),source=0.0_pREAL)

    phase => phases%get_dict(ph)
    mech  => phase%get_dict('mechanical')
#if defined(__GFORTRAN__)
    output_mechanical(ph)%label = output_as1dStr(mech)
#else
    output_mechanical(ph)%label = mech%get_as1dStr('output',defaultVal=emptyStrArray)
#endif
  end do

  do ce = 1, size(material_ID_phase,2)
    ma = discretization_materialAt((ce-1)/discretization_nIPs+1)
    do co = 1,homogenization_Nconstituents(material_ID_homogenization(ce))
      ph = material_ID_phase(co,ce)
      en = material_entry_phase(co,ce)
      phase_mechanical_F(ph)%data(1:3,1:3,en)  = math_I3
      phase_mechanical_Fp(ph)%data(1:3,1:3,en) = material_O_0(ma)%data(co)%asMatrix()              ! Fp reflects initial orientation (see 10.1016/j.actamat.2006.01.005)
      phase_mechanical_Fe(ph)%data(1:3,1:3,en) = matmul(material_V_e_0(ma)%data(1:3,1:3,co), &
                                                        transpose(phase_mechanical_Fp(ph)%data(1:3,1:3,en)))
      phase_mechanical_Fi(ph)%data(1:3,1:3,en) = material_O_0(ma)%data(co)%rotate(math_inv33(material_V_e_0(ma)%data(1:3,1:3,co)))
    end do
  end do

  do ph = 1, size(phases)
    phase_mechanical_F0(ph)%data  = phase_mechanical_F(ph)%data
    phase_mechanical_Fp0(ph)%data = phase_mechanical_Fp(ph)%data
    phase_mechanical_Fi0(ph)%data = phase_mechanical_Fi(ph)%data
  end do


  call elastic_init(phases)
  allocate(plasticState(size(phases)))
  allocate(mechanical_plasticity_type(size(phases)),source = UNDEFINED)
  call plastic_init()
  do ph = 1,size(phases)
    plasticState(ph)%state0 = plasticState(ph)%state
  end do
  call eigen_init(phases)

  num_mech_plastic => num_mech%get_dict('plastic', defaultVal=emptyDict)
  num_mech_eigen   => num_mech%get_dict('eigen',   defaultVal=emptyDict)

  num%stepMinCryst           = num_mech%get_asReal         ('r_cutback_min',      defaultVal=1.0e-3_pREAL)
  num%stepSizeCryst          = num_mech%get_asReal         ('r_cutback',          defaultVal=0.25_pREAL)
  num%stepIncreaseCryst      = num_mech%get_asReal         ('r_increase',         defaultVal=1.5_pREAL)
  num%rtol_crystalliteState  = num_mech%get_asReal         ('eps_rel_state',      defaultVal=1.0e-6_pREAL)
  num%nState                 = num_mech%get_asInt          ('N_iter_state_max',   defaultVal=20)
  num%nStress_Lp             = num_mech_plastic%get_asInt  ('N_iter_Lp_max',      defaultVal=40)
  num%stepSizeLp             = num_mech_plastic%get_asReal ('r_linesearch_Lp',    defaultVal=0.5_pREAL)
  num%rtol_Lp                = num_mech_plastic%get_asReal ('eps_rel_Lp',         defaultVal=1.0e-6_pREAL)
  num%atol_Lp                = num_mech_plastic%get_asReal ('eps_abs_Lp',         defaultVal=1.0e-8_pREAL)
  num%iJacoLpresiduum        = num_mech_plastic%get_asInt  ('f_update_jacobi_Lp', defaultVal=1)
  num%nStress_Li             = num_mech_eigen%get_asInt    ('N_iter_Li_max',      defaultVal=40)
  num%stepSizeLi             = num_mech_eigen%get_asReal   ('r_linesearch_Li',    defaultVal=0.5_pREAL)
  num%rtol_Li                = num_mech_eigen%get_asReal   ('eps_rel_Li',         defaultVal=num%rtol_Lp)
  num%atol_Li                = num_mech_eigen%get_asReal   ('eps_abs_Li',         defaultVal=num%atol_Lp)
  num%iJacoLiresiduum        = num_mech_eigen%get_asInt    ('f_update_jacobi_Li', defaultVal=num%iJacoLpresiduum)

  extmsg = ''
  if (num%stepMinCryst   <= 0.0_pREAL)     extmsg = trim(extmsg)//' r_cutback_min'
  if (num%stepSizeCryst  <= 0.0_pREAL)     extmsg = trim(extmsg)//' r_cutback'
  if (num%stepIncreaseCryst <= 0.0_pREAL)  extmsg = trim(extmsg)//' r_increase'
  if (num%stepSizeLp <= 0.0_pREAL)         extmsg = trim(extmsg)//' r_linesearch_Lp'
  if (num%stepSizeLi <= 0.0_pREAL)         extmsg = trim(extmsg)//' r_linesearch_Li'
  if (num%rtol_Lp <= 0.0_pREAL)            extmsg = trim(extmsg)//' epl_rel_Lp'
  if (num%atol_Lp <= 0.0_pREAL)            extmsg = trim(extmsg)//' eps_abs_Lp'
  if (num%rtol_Li <= 0.0_pREAL)            extmsg = trim(extmsg)//' eps_rel_Li'
  if (num%atol_Li <= 0.0_pREAL)            extmsg = trim(extmsg)//' eps_abs_Li'
  if (num%iJacoLpresiduum < 1)             extmsg = trim(extmsg)//' f_update_jacobi_Lp'
  if (num%iJacoLiresiduum < 1)             extmsg = trim(extmsg)//' f_update_jacobi_Li'
  if (num%nState  < 1)                     extmsg = trim(extmsg)//' N_iter_state_max'
  if (num%nStress_Lp < 1)                  extmsg = trim(extmsg)//' N_iter_Lp_max'
  if (num%nStress_Li < 1)                  extmsg = trim(extmsg)//' N_iter_Li_max'

  if (extmsg /= '') call IO_error(301,ext_msg=trim(extmsg))

  select case(num_mech_plastic%get_asStr('integrator_state',defaultVal='FPI'))

    case('FPI')
      integrateState => integrateStateFPI

    case('Euler')
      integrateState => integrateStateEuler

    case('Euler_adaptive')
      integrateState => integrateStateEulerAdaptive

    case('RK4')
      integrateState => integrateStateRK4

    case('RKCK45')
      integrateState => integrateStateRKCK45

    case default
     call IO_error(301,ext_msg='integrator')

  end select


end subroutine mechanical_init


module subroutine mechanical_result(group,ph)

  character(len=*), intent(in) :: group
  integer,          intent(in) :: ph


  call results(group,ph)

  select case(mechanical_plasticity_type(ph))

    case(MECHANICAL_PLASTICITY_ISOTROPIC)
      call plastic_isotropic_result(ph,group//'mechanical/')

    case(MECHANICAL_PLASTICITY_PHENOPOWERLAW)
      call plastic_phenopowerlaw_result(ph,group//'mechanical/')

    case(MECHANICAL_PLASTICITY_KINEHARDENING)
      call plastic_kinehardening_result(ph,group//'mechanical/')

    case(MECHANICAL_PLASTICITY_DISLOTWIN)
      call plastic_dislotwin_result(ph,group//'mechanical/')

    case(MECHANICAL_PLASTICITY_DISLOTUNGSTEN)
      call plastic_dislotungsten_result(ph,group//'mechanical/')

    case(MECHANICAL_PLASTICITY_TDT)
      call plastic_tdt_result(ph,group//'mechanical/')

    case(MECHANICAL_PLASTICITY_TWINTRIP)
      call plastic_twintrip_result(ph,group//'mechanical/')

    case(MECHANICAL_PLASTICITY_NONLOCAL)
      call plastic_nonlocal_result(ph,group//'mechanical/')

  end select


end subroutine mechanical_result


!--------------------------------------------------------------------------------------------------
!> @brief calculation of stress (P) with time integration based on a residuum in Lp and
!> intermediate acceleration of the Newton-Raphson correction
!--------------------------------------------------------------------------------------------------
function integrateStress(F,Fp0,Fi0,Delta_t,ph,en) result(status)

  real(pREAL), dimension(3,3), intent(in) :: F,Fp0,Fi0
  real(pREAL),                 intent(in) :: Delta_t
  integer, intent(in) :: ph, en
  integer(kind(STATUS_OK)) :: status

  real(pREAL), dimension(3,3)::       Fp_new, &                                                     ! plastic deformation gradient at end of timestep
                                      invFp_new, &                                                  ! inverse of Fp_new
                                      invFp_current, &                                              ! inverse of Fp_current
                                      Lpguess, &                                                    ! current guess for plastic velocity gradient
                                      Lpguess_old, &                                                ! known last good guess for plastic velocity gradient
                                      Lp_constitutive, &                                            ! plastic velocity gradient resulting from constitutive law
                                      residuumLp, &                                                 ! current residuum of plastic velocity gradient
                                      deltaLp, &                                                    ! direction of next guess
                                      Fi_new, &                                                     ! gradient of intermediate deformation stages
                                      invFi_new, &
                                      invFi_current, &                                              ! inverse of Fi_current
                                      Liguess, &                                                    ! current guess for intermediate velocity gradient
                                      Liguess_old, &                                                ! known last good guess for intermediate velocity gradient
                                      Li_constitutive, &                                            ! intermediate velocity gradient resulting from constitutive law
                                      residuumLi, &                                                 ! current residuum of intermediate velocity gradient
                                      deltaLi, &                                                    ! direction of next guess
                                      Fe, &                                                         ! elastic deformation gradient
                                      S, &                                                          ! 2nd Piola-Kirchhoff Stress in plastic (lattice) configuration
                                      A, &
                                      B, &
                                      temp_33
  real(pREAL), dimension(9) ::        temp_9                                                        ! needed for matrix inversion by LAPACK
  integer,     dimension(9) ::        devNull_9                                                     ! needed for matrix inversion by LAPACK
  real(pREAL), dimension(9,9) ::      dRLp_dLp, &                                                   ! partial derivative of residuum (Jacobian for Newton-Raphson scheme)
                                      dRLi_dLi                                                      ! partial derivative of residuumI (Jacobian for Newton-Raphson scheme)
  real(pREAL), dimension(3,3,3,3)::   dS_dFe, &                                                     ! partial derivative of 2nd Piola-Kirchhoff stress
                                      dS_dFi, &
                                      dFe_dLp, &                                                    ! partial derivative of elastic deformation gradient
                                      dFe_dLi, &
                                      dFi_dLi, &
                                      dLp_dFi, &
                                      dLi_dFi, &
                                      dLp_dS, &
                                      dLi_dS
  real(pREAL)                         steplengthLp, &
                                      steplengthLi, &
                                      residuumLi_old_norm, &                                        ! last residuum of intermediate velocity gradient
                                      residuumLp_old_norm, &                                        ! last residuum of plastic velocity gradient
                                      atol_Lp, &
                                      atol_Li
  integer                             NiterationStressLp, &                                         ! number of stress integrations
                                      NiterationStressLi, &                                         ! number of inner stress integrations
                                      ierr, &                                                       ! error indicator for LAPACK
                                      o, &
                                      p, &
                                      jacoCounterLp, &
                                      jacoCounterLi                                                 ! counters to check for Jacobian update
  logical :: error


  status = STATUS_FAIL_PHASE_MECHANICAL_STRESS
  call plastic_dependentState(ph,en)

  Lpguess = phase_mechanical_Lp(ph)%data(1:3,1:3,en)                                                ! take as first guess
  Liguess = phase_mechanical_Li(ph)%data(1:3,1:3,en)                                                ! take as first guess

  call math_invert33(invFp_current,error=error,A=Fp0)
  if (error) return ! error
  call math_invert33(invFi_current,error=error,A=Fi0)
  if (error) return ! error

  A = matmul(F,invFp_current)                                                                       ! intermediate tensor needed later to calculate dFe_dLp

  jacoCounterLi = 0
  steplengthLi  = 1.0_pREAL
  Liguess_old   = Liguess
  residuumLi_old_norm = huge(1.0_pREAL)

  NiterationStressLi = 0
  LiLoop: do
    NiterationStressLi = NiterationStressLi + 1
    if (NiterationStressLi>num%nStress_Li) return ! error

    invFi_new = matmul(invFi_current,math_I3 - Delta_t*Liguess)
    Fi_new    = math_inv33(invFi_new)

    jacoCounterLp = 0
    steplengthLp  = 1.0_pREAL
    Lpguess_old   = Lpguess
    residuumLp_old_norm = huge(1.0_pREAL)

    NiterationStressLp = 0
    LpLoop: do
      NiterationStressLp = NiterationStressLp + 1
      if (NiterationStressLp>num%nStress_Lp) then
        status = STATUS_FAIL_PHASE_MECHANICAL_STRESS_LP_MAXIT
        return
      endif
      B  = math_I3 - Delta_t*Lpguess
      Fe = matmul(matmul(A,B), invFi_new)
      call phase_hooke_SandItsTangents(S, dS_dFe, dS_dFi, &
                                       Fe, Fi_new, ph, en)

      call plastic_LpAndItsTangents(Lp_constitutive, dLp_dS, dLp_dFi, &
                                    S, Fi_new, Delta_t, ph,en)

      !* update current residuum and check for convergence of loop
      atol_Lp = max(num%rtol_Lp * max(norm2(Lpguess),norm2(Lp_constitutive)), &                     ! absolute tolerance from largest acceptable relative error
                    num%atol_Lp)                                                                    ! minimum lower cutoff
      residuumLp = Lpguess - Lp_constitutive

      if (any(IEEE_is_NaN(residuumLp))) then
        status = STATUS_FAIL_PHASE_MECHANICAL_STRESS_LP_NAN
        return
      elseif (norm2(residuumLp) < atol_Lp) then                                                     ! converged if below absolute tolerance
        exit LpLoop
      elseif (norm2(residuumLp) < residuumLp_old_norm) then                                         ! not converged, but improved norm of residuum...
        residuumLp_old_norm = norm2(residuumLp)                                                     ! ...remember old values and...
        Lpguess_old = Lpguess
        steplengthLp = 1.0_pREAL                                                                    ! ...proceed with normal step length (calculate new search direction)
      else                                                                                          ! not converged and residuum not improved...
        steplengthLp = num%stepSizeLp * steplengthLp                                                ! ...try with smaller step length in same direction
        Lpguess      = Lpguess_old &
                     + deltaLp * stepLengthLp
        cycle LpLoop
      end if

      calculateJacobiLp: if (mod(jacoCounterLp, num%iJacoLpresiduum) == 0) then
        jacoCounterLp = jacoCounterLp + 1

        do o=1,3; do p=1,3
          dFe_dLp(o,1:3,p,1:3) = - Delta_t * A(o,p)*transpose(invFi_new)                            ! dFe_dLp(i,j,k,l) = -Delta_t * A(i,k) invFi(l,j)
        end do; end do
        dRLp_dLp = math_eye(9) &
                 - math_3333to99(math_mul3333xx3333(math_mul3333xx3333(dLp_dS,dS_dFe),dFe_dLp))
        temp_9 = math_33to9(residuumLp)
        call dgesv(9,1,dRLp_dLp,9,devNull_9,temp_9,9,ierr)                                          ! solve dRLp/dLp * delta Lp = -res for delta Lp
        if (ierr /= 0) then
          status = STATUS_FAIL_PHASE_MECHANICAL_STRESS_LP_DGESV
          return
        end if
        deltaLp = - math_9to33(temp_9)
      end if calculateJacobiLp

      Lpguess = Lpguess &
              + deltaLp * steplengthLp
    end do LpLoop

    call phase_LiAndItsTangents(Li_constitutive, dLi_dS, dLi_dFi, &
                                S, Fi_new, ph,en, Delta_t,invFi_current)

    !* update current residuum and check for convergence of loop
    atol_Li = max(num%rtol_Li * max(norm2(Liguess),norm2(Li_constitutive)), &                       ! absolute tolerance from largest acceptable relative error
                  num%atol_Li)                                                                      ! minimum lower cutoff
    residuumLi = Liguess - Li_constitutive
    if (any(IEEE_is_NaN(residuumLi))) then
      return ! error
    elseif (norm2(residuumLi) < atol_Li) then                                                       ! converged if below absolute tolerance
      exit LiLoop
    elseif (norm2(residuumLi) < residuumLi_old_norm) then                                           ! not converged, but improved norm of residuum ...
      residuumLi_old_norm = norm2(residuumLi)                                                       ! ...remember old values and...
      Liguess_old = Liguess
      steplengthLi = 1.0_pREAL                                                                      ! ...proceed with normal step length (calculate new search direction)
    else                                                                                            ! not converged and residuum not improved...
      steplengthLi = num%stepSizeLi * steplengthLi                                                  ! ...try with smaller step length in same direction
      Liguess      = Liguess_old &
                   + deltaLi * steplengthLi
      cycle LiLoop
    end if

    calculateJacobiLi: if (mod(jacoCounterLi, num%iJacoLiresiduum) == 0) then
      jacoCounterLi = jacoCounterLi + 1

      temp_33 = matmul(matmul(A,B),invFi_current)
      do o=1,3; do p=1,3
        dFe_dLi(1:3,o,1:3,p) = -Delta_t*math_I3(o,p)*temp_33                                        ! dFe_dLp(i,j,k,l) = -Delta_t * A(i,k) invFi(l,j)
        dFi_dLi(1:3,o,1:3,p) = -Delta_t*math_I3(o,p)*invFi_current
      end do; end do
      do o=1,3; do p=1,3
        dFi_dLi(1:3,1:3,o,p) = matmul(matmul(Fi_new,dFi_dLi(1:3,1:3,o,p)),Fi_new)
      end do; end do
      dRLi_dLi  = math_eye(9) &
                - math_3333to99(math_mul3333xx3333(dLi_dS,  math_mul3333xx3333(dS_dFe, dFe_dLi) &
                                                          + math_mul3333xx3333(dS_dFi, dFi_dLi)))  &
                - math_3333to99(math_mul3333xx3333(dLi_dFi, dFi_dLi))
      temp_9 = math_33to9(residuumLi)
      call dgesv(9,1,dRLi_dLi,9,devNull_9,temp_9,9,ierr)                                            ! solve dRLi/dLp * delta Li = -res for delta Li
      if (ierr /= 0) return ! error
      deltaLi = - math_9to33(temp_9)
    end if calculateJacobiLi

    Liguess = Liguess &
            + deltaLi * steplengthLi
  end do LiLoop

  invFp_new = matmul(invFp_current,B)
  call math_invert33(Fp_new,error=error,A=invFp_new)
  if (error) return ! error

  phase_mechanical_P(ph)%data(1:3,1:3,en)  = matmul(matmul(F,invFp_new),matmul(S,transpose(invFp_new)))
  phase_mechanical_S(ph)%data(1:3,1:3,en)  = S
  phase_mechanical_Lp(ph)%data(1:3,1:3,en) = Lpguess
  phase_mechanical_Li(ph)%data(1:3,1:3,en) = Liguess
  phase_mechanical_Fp(ph)%data(1:3,1:3,en) = Fp_new / math_det33(Fp_new)**(1.0_pREAL/3.0_pREAL)    ! regularize
  phase_mechanical_Fi(ph)%data(1:3,1:3,en) = Fi_new
  phase_mechanical_Fe(ph)%data(1:3,1:3,en) = matmul(matmul(F,invFp_new),invFi_new)
  status = STATUS_OK

end function integrateStress


!--------------------------------------------------------------------------------------------------
!> @brief integrate stress, state with adaptive 1st order explicit Euler method
!> using Fixed Point Iteration to adapt the stepsize
!--------------------------------------------------------------------------------------------------
function integrateStateFPI(F_0,F,Fp0,Fi0,state0,Delta_t,ph,en) result(status)

  real(pREAL), intent(in),dimension(3,3) :: F_0,F,Fp0,Fi0
  real(pREAL), intent(in),dimension(:)   :: state0
  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: &
    ph, &
    en
  integer(kind(STATUS_OK)) :: status

  integer :: &
    NiterationState, &                                                                              !< number of iterations in state loop
    sizeDotState
  real(pREAL) :: &
    zeta
  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: &
    r, &                                                                                            ! state residuum
    dotState
  real(pREAL), dimension(plasticState(ph)%sizeDotState,2) :: &
    dotState_last


  status = STATUS_FAIL_PHASE_MECHANICAL_STATE

  dotState = plastic_dotState(Delta_t,ph,en)
  if (any(IEEE_is_NaN(dotState))) return

  sizeDotState = plasticState(ph)%sizeDotState
  plasticState(ph)%state(1:sizeDotState,en) = state0 + dotState * Delta_t

  iteration: do NiterationState = 1, num%nState

    dotState_last(1:sizeDotState,2) = merge(dotState_last(1:sizeDotState,1),0.0_pREAL, nIterationState > 1)
    dotState_last(1:sizeDotState,1) = dotState

    status = integrateStress(F,Fp0,Fi0,Delta_t,ph,en)
    if (status /= STATUS_OK) exit iteration

    dotState = plastic_dotState(Delta_t,ph,en)
    if (any(IEEE_is_NaN(dotState))) exit iteration

    zeta = damper(dotState,dotState_last(1:sizeDotState,1),dotState_last(1:sizeDotState,2))
    dotState = dotState * zeta &
             + dotState_last(1:sizeDotState,1) * (1.0_pREAL - zeta)
    r = plasticState(ph)%state(1:sizeDotState,en) &
      - state0 &
      - dotState * Delta_t
    if (mechanical_plasticity_type(ph) == MECHANICAL_PLASTICITY_TDT) then                           ! Jacobi-Newton: same fixed point, stiff self-coupled state
      plasticState(ph)%state(1:sizeDotState,en) = plasticState(ph)%state(1:sizeDotState,en) &
                                                - r*plastic_statePreconditioner(Delta_t,ph,en)
    else
      plasticState(ph)%state(1:sizeDotState,en) = plasticState(ph)%state(1:sizeDotState,en) - r
    end if

    if (converged(r,plasticState(ph)%state(1:sizeDotState,en),plasticState(ph)%atol(1:sizeDotState))) then
      status = plastic_deltaState(ph,en)
      exit iteration
    end if

  end do iteration


  contains

  !--------------------------------------------------------------------------------------------------
  !> @brief calculate the damping for correction of state and dot state
  !--------------------------------------------------------------------------------------------------
  real(pREAL) pure function damper(omega_0,omega_1,omega_2)
#ifndef __GFORTRAN__
    use prec
    import, none
#endif

    real(pREAL), dimension(:), intent(in) :: &
      omega_0, omega_1, omega_2

    real(pREAL) :: dot_prod12, dot_prod22


    dot_prod12 = dot_product(omega_0-omega_1, omega_1-omega_2)
    dot_prod22 = dot_product(omega_1-omega_2, omega_1-omega_2)

    if (min(dot_product(omega_0,omega_1),dot_prod12) < 0.0_pREAL .and. dot_prod22 > 0.0_pREAL) then
      damper = 0.75_pREAL + 0.25_pREAL * tanh(2.0_pREAL + 4.0_pREAL * dot_prod12 / dot_prod22)
    else
      damper = 1.0_pREAL
    end if

  end function damper

end function integrateStateFPI


!--------------------------------------------------------------------------------------------------
!> @brief integrate state with 1st order explicit Euler method
!--------------------------------------------------------------------------------------------------
function integrateStateEuler(F_0,F,Fp0,Fi0,state0,Delta_t,ph,en) result(status)

  real(pREAL), intent(in),dimension(3,3) :: F_0,F,Fp0,Fi0
  real(pREAL), intent(in),dimension(:)   :: state0
  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: &
    ph, &
    en                                                                                               !< grain index in grain loop
  integer(kind(STATUS_OK)) :: &
    status

  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: &
    dotState
  integer :: &
    sizeDotState


  status = STATUS_FAIL_PHASE_MECHANICAL_STATE

  dotState = plastic_dotState(Delta_t,ph,en)
  if (any(IEEE_is_NaN(dotState))) return

  sizeDotState = plasticState(ph)%sizeDotState
  plasticState(ph)%state(1:sizeDotState,en) = state0 + dotState*Delta_t

  status = plastic_deltaState(ph,en)
  if (status /= STATUS_OK) return

  status = integrateStress(F,Fp0,Fi0,Delta_t,ph,en)

end function integrateStateEuler


!--------------------------------------------------------------------------------------------------
!> @brief integrate stress, state with 1st order Euler method with adaptive step size
!--------------------------------------------------------------------------------------------------
function integrateStateEulerAdaptive(F_0,F,Fp0,Fi0,state0,Delta_t,ph,en) result(status)

  real(pREAL), intent(in),dimension(3,3) :: F_0,F,Fp0,Fi0
  real(pREAL), intent(in),dimension(:)   :: state0
  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: &
    ph, &
    en
  integer(kind(STATUS_OK)) :: &
    status

  integer :: &
    sizeDotState
  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: &
    r, &
    dotState


  status = STATUS_FAIL_PHASE_MECHANICAL_STATE

  dotState = plastic_dotState(Delta_t,ph,en)
  if (any(IEEE_is_NaN(dotState))) return

  sizeDotState = plasticState(ph)%sizeDotState

  r = - dotState * 0.5_pREAL * Delta_t
  plasticState(ph)%state(1:sizeDotState,en) = state0 + dotState*Delta_t

  status = plastic_deltaState(ph,en)
  if (status /= STATUS_OK) return

  status = integrateStress(F,Fp0,Fi0,Delta_t,ph,en)
  if (status /= STATUS_OK) return

  dotState = plastic_dotState(Delta_t,ph,en)
  ! ToDo: MD: need to set status to failed
  if (any(IEEE_is_NaN(dotState))) return

  status = merge(STATUS_OK, &
                 STATUS_FAIL_PHASE_MECHANICAL_STATE, &
                 converged(r + 0.5_pREAL * dotState * Delta_t, &
                           plasticState(ph)%state(1:sizeDotState,en), &
                           plasticState(ph)%atol(1:sizeDotState)))

end function integrateStateEulerAdaptive


!---------------------------------------------------------------------------------------------------
!> @brief Integrate state (including stress integration) with the classic Runge Kutta method
!---------------------------------------------------------------------------------------------------
function integrateStateRK4(F_0,F,Fp0,Fi0,state0,Delta_t,ph,en) result(status)

  real(pREAL), intent(in),dimension(3,3) :: F_0,F,Fp0,Fi0
  real(pREAL), intent(in),dimension(:)   :: state0
  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: ph, en
  integer(kind(STATUS_OK)) :: status

  real(pREAL), dimension(3,3), parameter :: &
    A = reshape([&
      0.5_pREAL, 0.0_pREAL, 0.0_pREAL, &
      0.0_pREAL, 0.5_pREAL, 0.0_pREAL, &
      0.0_pREAL, 0.0_pREAL, 1.0_pREAL],&
      shape(A))
  real(pREAL), dimension(3), parameter :: &
    C = [0.5_pREAL, 0.5_pREAL, 1.0_pREAL]
  real(pREAL), dimension(4), parameter :: &
    B = [6.0_pREAL, 3.0_pREAL, 3.0_pREAL, 6.0_pREAL]**(-1)


  status = integrateStateRK(F_0,F,Fp0,Fi0,state0,Delta_t,ph,en,A,B,C)

end function integrateStateRK4


!---------------------------------------------------------------------------------------------------
!> @brief Integrate state (including stress integration) with the Cash-Carp method
!---------------------------------------------------------------------------------------------------
function integrateStateRKCK45(F_0,F,Fp0,Fi0,state0,Delta_t,ph,en) result(status)

  real(pREAL), intent(in),dimension(3,3) :: F_0,F,Fp0,Fi0
  real(pREAL), intent(in),dimension(:)   :: state0
  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: ph, en
  integer(kind(STATUS_OK)) :: status

  real(pREAL), dimension(5,5), parameter :: &
    A = reshape([&
      1._pREAL/5._pREAL,       .0_pREAL,             .0_pREAL,               .0_pREAL,                  .0_pREAL, &
      3._pREAL/40._pREAL,      9._pREAL/40._pREAL,   .0_pREAL,               .0_pREAL,                  .0_pREAL, &
      3_pREAL/10._pREAL,       -9._pREAL/10._pREAL,  6._pREAL/5._pREAL,      .0_pREAL,                  .0_pREAL, &
      -11._pREAL/54._pREAL,    5._pREAL/2._pREAL,    -70.0_pREAL/27.0_pREAL, 35.0_pREAL/27.0_pREAL,     .0_pREAL, &
      1631._pREAL/55296._pREAL,175._pREAL/512._pREAL,575._pREAL/13824._pREAL,44275._pREAL/110592._pREAL,253._pREAL/4096._pREAL],&
      shape(A))
  real(pREAL), dimension(5), parameter :: &
    C = [0.2_pREAL, 0.3_pREAL, 0.6_pREAL, 1.0_pREAL, 0.875_pREAL]
  real(pREAL), dimension(6), parameter :: &
    B = &
      [37.0_pREAL/378.0_pREAL, .0_pREAL, 250.0_pREAL/621.0_pREAL, &
      125.0_pREAL/594.0_pREAL, .0_pREAL, 512.0_pREAL/1771.0_pREAL], &
    DB = B - &
      [2825.0_pREAL/27648.0_pREAL,    .0_pREAL,                18575.0_pREAL/48384.0_pREAL,&
      13525.0_pREAL/55296.0_pREAL, 277.0_pREAL/14336.0_pREAL,  1._pREAL/4._pREAL]


  status = integrateStateRK(F_0,F,Fp0,Fi0,state0,Delta_t,ph,en,A,B,C,DB)

end function integrateStateRKCK45


!--------------------------------------------------------------------------------------------------
!> @brief Integrate state (including stress integration) with an explicit Runge-Kutta method or an
!! embedded explicit Runge-Kutta method
!--------------------------------------------------------------------------------------------------
function integrateStateRK(F_0,F,Fp0,Fi0,state0,Delta_t,ph,en,A,B,C,DB) result(status)

  real(pREAL), intent(in),dimension(3,3) :: F_0,F,Fp0,Fi0
  real(pREAL), intent(in),dimension(:)   :: state0
  real(pREAL), intent(in) :: Delta_t
  real(pREAL), dimension(:,:), intent(in) :: A
  real(pREAL), dimension(:),   intent(in) :: B, C
  real(pREAL), dimension(:),   intent(in), optional :: DB
  integer, intent(in) :: &
    ph, &
    en
  integer(kind(STATUS_OK)) :: status

  integer :: &
    stage, &                                                                                        ! stage index in integration stage loop
    n, &
    sizeDotState
  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: &
    dotState
  real(pREAL), dimension(plasticState(ph)%sizeDotState,size(B)) :: &
    plastic_RKdotState


  status = STATUS_FAIL_PHASE_MECHANICAL_STATE
  sizeDotState = plasticState(ph)%sizeDotState

  dotState = plastic_dotState(Delta_t,ph,en)                                                        !MD: Delta_t is only used for nonlocal
  if (any(IEEE_is_NaN(dotState))) return

  do stage = 1, size(A,1)

    plastic_RKdotState(1:sizeDotState,stage) = dotState

    dotState = A(1,stage) * plastic_RKdotState(1:sizeDotState,1)
    do n = 2, stage
      dotState = dotState &
               + A(n,stage)*plastic_RKdotState(1:sizeDotState,n)
    end do

    plasticState(ph)%state(1:sizeDotState,en) = state0 &
                                              + dotState*Delta_t

    status = integrateStress(F_0+(F-F_0)*C(stage),Fp0,Fi0,Delta_t*C(stage), ph,en)
    if (status /= STATUS_OK) return

    dotState = plastic_dotState(Delta_t*C(stage), ph,en)                                            !MD: Delta_t is only used for nonlocal and it's unclear whether scaling by C makes sense
    ! ToDo: MD: need to set status to failed
    if (any(IEEE_is_NaN(dotState))) exit

  end do
  if (status /= STATUS_OK) return


  plastic_RKdotState(1:sizeDotState,size(B)) = dotState
  plasticState(ph)%state(1:sizeDotState,en) = state0 &
                                            + matmul(plastic_RKdotState,B)*Delta_t

  if (present(DB)) &
    status = merge(STATUS_OK, &
                   STATUS_FAIL_PHASE_MECHANICAL_STATE, &
                   converged(matmul(plastic_RKdotState(1:sizeDotState,1:size(DB)),DB) * Delta_t, &
                             plasticState(ph)%state(1:sizeDotState,en), &
                             plasticState(ph)%atol(1:sizeDotState)))

  if (status /= STATUS_OK) return

  status = plastic_deltaState(ph,en)
  if (status /= STATUS_OK) return

  status = integrateStress(F,Fp0,Fi0,Delta_t,ph,en)

end function integrateStateRK


!--------------------------------------------------------------------------------------------------
!> @brief Write mechanical results to HDF5 output file.
!--------------------------------------------------------------------------------------------------
subroutine results(group,ph)

  character(len=*), intent(in) :: group
  integer,          intent(in) :: ph

  integer :: ou


  call result_closeGroup(result_addGroup(group//'/mechanical'))

  do ou = 1, size(output_mechanical(ph)%label)

    select case (output_mechanical(ph)%label(ou))
      case('F')
        call result_writeDataset(phase_mechanical_F(ph)%data,group//'/mechanical/','F',&
                                 'deformation gradient','1')
      case('F_e')
        call result_writeDataset(phase_mechanical_Fe(ph)%data,group//'/mechanical/','F_e',&
                                 'elastic deformation gradient','1')
      case('F_p')
        call result_writeDataset(phase_mechanical_Fp(ph)%data,group//'/mechanical/','F_p', &
                                 'plastic deformation gradient','1')
      case('F_i')
        call result_writeDataset(phase_mechanical_Fi(ph)%data,group//'/mechanical/','F_i', &
                                 'inelastic deformation gradient','1')
      case('L_p')
        call result_writeDataset(phase_mechanical_Lp(ph)%data,group//'/mechanical/','L_p', &
                                 'plastic velocity gradient','1/s')
      case('L_i')
        call result_writeDataset(phase_mechanical_Li(ph)%data,group//'/mechanical/','L_i', &
                                 'inelastic velocity gradient','1/s')
      case('P')
        call result_writeDataset(phase_mechanical_P(ph)%data,group//'/mechanical/','P', &
                                 'first Piola-Kirchhoff stress','Pa')
      case('S')
        call result_writeDataset(phase_mechanical_S(ph)%data,group//'/mechanical/','S', &
                                 'second Piola-Kirchhoff stress','Pa')
      case('O')
        call result_writeDataset(to_quaternion(phase_O(ph)%data),group//'/mechanical','O', &
                                 'crystal orientation as quaternion q_0 (q_1 q_2 q_3)','1')
        call result_addAttribute('lattice',phase_lattice(ph),group//'/mechanical/O')
        if (any(phase_lattice(ph) == ['hP', 'tI'])) &
          call result_addAttribute('c/a',phase_cOverA(ph),group//'/mechanical/O')
    end select
  end do


  contains

!--------------------------------------------------------------------------------------------------
!> @brief Convert orientation array to quaternion array.
!--------------------------------------------------------------------------------------------------
  function to_quaternion(dataset)
#ifndef __GFORTRAN__
    use prec
    use rotations
    import, none
#endif

    type(tRotation), dimension(:), intent(in) :: dataset
    real(pREAL), dimension(4,size(dataset,1)) :: to_quaternion

    integer :: i


    do i = 1, size(dataset,1)
      to_quaternion(:,i) = dataset(i)%asQuaternion()
    end do

 end function to_quaternion

end subroutine results


!--------------------------------------------------------------------------------------------------
!> @brief Forward data after successful increment.
! ToDo: Any guessing for the current states possible?
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_forward()

  integer :: ph


  do ph = 1, size(plasticState)
    phase_mechanical_Fi0(ph) = phase_mechanical_Fi(ph)
    phase_mechanical_Fp0(ph) = phase_mechanical_Fp(ph)
    phase_mechanical_F0(ph)  = phase_mechanical_F(ph)
    phase_mechanical_Li0(ph) = phase_mechanical_Li(ph)
    phase_mechanical_Lp0(ph) = phase_mechanical_Lp(ph)
    phase_mechanical_S0(ph)  = phase_mechanical_S(ph)
    plasticState(ph)%state0 = plasticState(ph)%state
  end do

end subroutine mechanical_forward


!--------------------------------------------------------------------------------------------------
!> @brief calculate stress (P)
!--------------------------------------------------------------------------------------------------
module function phase_mechanical_constitutive(Delta_t,co,ce) result(status)

  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: &
    co, &
    ce
  integer(kind(STATUS_OK)) :: status

  real(pREAL) :: &
    formerStep
  integer :: &
    ph, en, sizeDotState
  logical :: todo
  real(pREAL) :: stepFrac,step
  real(pREAL), dimension(3,3) :: &
    Fp0, &
    Fi0, &
    Lp0, &
    Li0, &
    F0, &
    F
  real(pREAL), dimension(plasticState(material_ID_phase(co,ce))%sizeState) :: state0


  ph = material_ID_phase(co,ce)
  en = material_entry_phase(co,ce)
#if defined(METAL) && defined(GRID)
  if (allocated(GPU_tangent)) then
    if (allocated(GPU_tangent(ph)%valid)) GPU_tangent(ph)%valid(en) = .false.                       ! CPU result: tangent computed by phase_mechanical_dPdF
  end if
#endif

  state0 = plasticState(ph)%state0(:,en)
  Li0 = phase_mechanical_Li0(ph)%data(1:3,1:3,en)
  Lp0 = phase_mechanical_Lp0(ph)%data(1:3,1:3,en)
  Fp0 = phase_mechanical_Fp0(ph)%data(1:3,1:3,en)
  Fi0 = phase_mechanical_Fi0(ph)%data(1:3,1:3,en)
  F0  = phase_mechanical_F0(ph)%data(1:3,1:3,en)
  stepFrac = 0.0_pREAL
  todo = .true.
  step = 1.0_pREAL/num%stepSizeCryst                                                                ! pretend failed step of 1/stepSizeCryst
  status = STATUS_ITERATING

  todo = .true.
  cutbackLooping: do while (todo)

    if (status == STATUS_OK) then
      formerStep = step
      stepFrac = stepFrac + step
      step = min(1.0_pREAL - stepFrac, num%stepIncreaseCryst * step)

      todo = step > 0.0_pREAL                        ! still time left to integrate on?

      if (todo) then
        F0  = F
        Lp0 = phase_mechanical_Lp(ph)%data(1:3,1:3,en)
        Li0 = phase_mechanical_Li(ph)%data(1:3,1:3,en)
        Fp0 = phase_mechanical_Fp(ph)%data(1:3,1:3,en)
        Fi0 = phase_mechanical_Fi(ph)%data(1:3,1:3,en)
        state0 = plasticState(ph)%state(:,en)
      end if
!--------------------------------------------------------------------------------------------------
!  cut back (reduced time and restore)
    else
      step = num%stepSizeCryst * step
      phase_mechanical_Fp(ph)%data(1:3,1:3,en) = Fp0
      phase_mechanical_Fi(ph)%data(1:3,1:3,en) = Fi0
      phase_mechanical_S(ph)%data(1:3,1:3,en) = phase_mechanical_S0(ph)%data(1:3,1:3,en)
      if (step < 1.0_pREAL) then                                                                    ! actual (not initial) cutback
        phase_mechanical_Lp(ph)%data(1:3,1:3,en) = Lp0
        phase_mechanical_Li(ph)%data(1:3,1:3,en) = Li0
      end if
      plasticState(ph)%state(:,en) = state0
      todo = step > num%stepMinCryst                                                                ! still on track or already done (beyond repair)
    end if

!--------------------------------------------------------------------------------------------------
!  prepare for integration
    if (todo) then
      sizeDotState = plasticState(ph)%sizeDotState
      F = F0 &
        + step * (phase_mechanical_F(ph)%data(1:3,1:3,en) - phase_mechanical_F0(ph)%data(1:3,1:3,en))
      status = integrateState(F0,F,Fp0,Fi0,state0(1:sizeDotState),step * Delta_t,ph,en)
    end if

  end do cutbackLooping

end function phase_mechanical_constitutive


!--------------------------------------------------------------------------------------------------
!> @brief Restore data after homog cutback.
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_restore(ce)

  integer, intent(in) :: ce

  integer :: &
    co, ph, en


  do co = 1,homogenization_Nconstituents(material_ID_homogenization(ce))
    ph = material_ID_phase(co,ce)
    en = material_entry_phase(co,ce)

    phase_mechanical_Fp(ph)%data(1:3,1:3,en) = phase_mechanical_Fp0(ph)%data(1:3,1:3,en)
    phase_mechanical_Fi(ph)%data(1:3,1:3,en) = phase_mechanical_Fi0(ph)%data(1:3,1:3,en)
    phase_mechanical_S(ph)%data(1:3,1:3,en)  = phase_mechanical_S0(ph)%data(1:3,1:3,en)

    plasticState(ph)%state(:,en) = plasticState(ph)%State0(:,en)
  end do

end subroutine mechanical_restore


!--------------------------------------------------------------------------------------------------
!> @brief Calculate tangent (dPdF).
!--------------------------------------------------------------------------------------------------
module function phase_mechanical_dPdF(Delta_t,co,ce) result(dPdF)

  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: &
    co, &                                                                                            !< counter in constituent loop
    ce
  real(pREAL), dimension(3,3,3,3) :: dPdF

  integer :: &
    o, &
    p, ph, en
  real(pREAL), dimension(3,3)     ::   devNull, &
                                       invSubFp0,invSubFi0,invFp,invFi, &
                                       temp_33_1, temp_33_2, temp_33_3
  real(pREAL), dimension(3,3,3,3) ::   dSdFe, &
                                       dSdF, &
                                       dSdFi, &
                                       dLidS, &                                                     ! tangent in lattice configuration
                                       dLidFi, &
                                       dLpdS, &
                                       dLpdFi, &
                                       dFidS, &
                                       dFpinvdF, &
                                       rhs_3333, &
                                       lhs_3333, &
                                       temp_3333
  real(pREAL), dimension(9,9)::        temp_99
  logical :: error
#if defined(METAL) && defined(GRID)
  real(pREAL), dimension(3,3,3,3) :: dPdF_GPU
  logical :: compare_GPU
#endif


  ph = material_ID_phase(co,ce)
  en = material_entry_phase(co,ce)
#if defined(METAL) && defined(GRID)
  compare_GPU = .false.
  if (allocated(GPU_tangent)) then
    if (allocated(GPU_tangent(ph)%valid)) then
      if (GPU_tangent(ph)%valid(en)) then
        dPdF = real(GPU_tangent(ph)%dPdF(1:3,1:3,1:3,1:3,en),pREAL)
        if (.not. GPU_tangent_verify) return
        dPdF_GPU = dPdF                                                                             ! diagnostics: also compute on CPU, compare at the end
        compare_GPU = .true.
      end if
    end if
  end if
#endif

  call phase_hooke_SandItsTangents(devNull,dSdFe,dSdFi, &
                                   phase_mechanical_Fe(ph)%data(1:3,1:3,en), &
                                   phase_mechanical_Fi(ph)%data(1:3,1:3,en),ph,en)
  call phase_LiAndItsTangents(devNull,dLidS,dLidFi, &
                              phase_mechanical_S(ph)%data(1:3,1:3,en), &
                              phase_mechanical_Fi(ph)%data(1:3,1:3,en), &
                              ph,en)

  invFp = math_inv33(phase_mechanical_Fp(ph)%data(1:3,1:3,en))
  invFi = math_inv33(phase_mechanical_Fi(ph)%data(1:3,1:3,en))
  invSubFp0 = math_inv33(phase_mechanical_Fp0(ph)%data(1:3,1:3,en))
  invSubFi0 = math_inv33(phase_mechanical_Fi0(ph)%data(1:3,1:3,en))

  if (sum(abs(dLidS)) < tol_math_check) then
    dFidS = 0.0_pREAL
  else
    lhs_3333 = 0.0_pREAL; rhs_3333 = 0.0_pREAL
    do o=1,3; do p=1,3
      lhs_3333(1:3,1:3,o,p) = lhs_3333(1:3,1:3,o,p) &
                            + matmul(invSubFi0,dLidFi(1:3,1:3,o,p)) * Delta_t
      lhs_3333(1:3,o,1:3,p) = lhs_3333(1:3,o,1:3,p) &
                            + invFi*invFi(p,o)
      rhs_3333(1:3,1:3,o,p) = rhs_3333(1:3,1:3,o,p) &
                            - matmul(invSubFi0,dLidS(1:3,1:3,o,p)) * Delta_t
    end do; end do
    call math_invert(temp_99,error,math_3333to99(lhs_3333))
    if (error) then
      call IO_warning(600,'matrix inversion during analytic tangent calculation', IO_EOL, &
                          'at entry', en, 'of phase', ph, 'on MPI rank', worldrank, emph = [4,6,8])
      dFidS = 0.0_pREAL
    else
      dFidS = math_mul3333xx3333(math_99to3333(temp_99),rhs_3333)
    end if
    dLidS = math_mul3333xx3333(dLidFi,dFidS) + dLidS
  end if

  call plastic_LpAndItsTangents(devNull,dLpdS,dLpdFi, &
                                phase_mechanical_S(ph)%data(1:3,1:3,en), &
                                phase_mechanical_Fi(ph)%data(1:3,1:3,en),Delta_t,ph,en)
  dLpdS = math_mul3333xx3333(dLpdFi,dFidS) + dLpdS

!--------------------------------------------------------------------------------------------------
! calculate dSdF
  temp_33_1 = transpose(matmul(invFp,invFi))
  temp_33_2 = matmul(phase_mechanical_F(ph)%data(1:3,1:3,en),invSubFp0)
  temp_33_3 = matmul(matmul(phase_mechanical_F(ph)%data(1:3,1:3,en),invFp), invSubFi0)

  do o=1,3; do p=1,3
    rhs_3333(p,o,1:3,1:3)  = matmul(dSdFe(p,o,1:3,1:3),temp_33_1)
    temp_3333(1:3,1:3,p,o) = matmul(matmul(temp_33_2,dLpdS(1:3,1:3,p,o)), invFi) &
                           + matmul(temp_33_3,dLidS(1:3,1:3,p,o))
  end do; end do
  lhs_3333 = math_mul3333xx3333(dSdFe,temp_3333) * Delta_t &
           + math_mul3333xx3333(dSdFi,dFidS)

  call math_invert(temp_99,error,math_eye(9)+math_3333to99(lhs_3333))
  if (error) then
    call IO_warning(600,'matrix inversion during analytic tangent calculation', IO_EOL, &
                        'at entry', en, 'of phase', ph, 'on MPI rank', worldrank, emph = [4,6,8])
    dSdF = rhs_3333
  else
    dSdF = math_mul3333xx3333(math_99to3333(temp_99),rhs_3333)
  end if

!--------------------------------------------------------------------------------------------------
! calculate dFpinvdF
  temp_3333 = math_mul3333xx3333(dLpdS,dSdF)
  do o=1,3; do p=1,3
    dFpinvdF(1:3,1:3,p,o) = - matmul(invSubFp0, matmul(temp_3333(1:3,1:3,p,o),invFi)) * Delta_t
  end do; end do

!--------------------------------------------------------------------------------------------------
! assemble dPdF
  temp_33_1 = matmul(phase_mechanical_S(ph)%data(1:3,1:3,en),transpose(invFp))
  temp_33_2 = matmul(phase_mechanical_F(ph)%data(1:3,1:3,en),invFp)
  temp_33_3 = matmul(temp_33_2,phase_mechanical_S(ph)%data(1:3,1:3,en))

  dPdF = 0.0_pREAL
  do p=1,3
    dPdF(p,1:3,p,1:3) = transpose(matmul(invFp,temp_33_1))
  end do
  do o=1,3; do p=1,3
    dPdF(1:3,1:3,p,o) = dPdF(1:3,1:3,p,o) &
                      + matmul(matmul(phase_mechanical_F(ph)%data(1:3,1:3,en),dFpinvdF(1:3,1:3,p,o)),temp_33_1) &
                      + matmul(matmul(temp_33_2,dSdF(1:3,1:3,p,o)),transpose(invFp)) &
                      + matmul(temp_33_3,transpose(dFpinvdF(1:3,1:3,p,o)))
  end do; end do

#if defined(METAL) && defined(GRID)
  if (compare_GPU) then
    !$OMP CRITICAL (GPU_tangent_compare)
    GPU_tangent_maxdiff = max(GPU_tangent_maxdiff, norm2(dPdF_GPU-dPdF)/norm2(dPdF))
    !$OMP END CRITICAL (GPU_tangent_compare)
  end if
#endif

end function phase_mechanical_dPdF


!--------------------------------------------------------------------------------------------------
!< @brief Write restart information to file.
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_restartWrite(groupHandle,ph)

  integer(HID_T), intent(in) :: groupHandle
  integer, intent(in) :: ph


  call HDF5_write(plasticState(ph)%state,groupHandle,'omega_plastic')
  call HDF5_write(phase_mechanical_S(ph)%data,groupHandle,'S')
  call HDF5_write(phase_mechanical_F(ph)%data,groupHandle,'F')
  call HDF5_write(phase_mechanical_Fp(ph)%data,groupHandle,'F_p')
  call HDF5_write(phase_mechanical_Fi(ph)%data,groupHandle,'F_i')
  call HDF5_write(phase_mechanical_Lp(ph)%data,groupHandle,'L_p')
  call HDF5_write(phase_mechanical_Li(ph)%data,groupHandle,'L_i')

end subroutine mechanical_restartWrite


!--------------------------------------------------------------------------------------------------
!< @brief Read restart information from file.
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_restartRead(groupHandle,ph)

  integer(HID_T), intent(in) :: groupHandle
  integer, intent(in) :: ph


  call HDF5_read(plasticState(ph)%state0,groupHandle,'omega_plastic')
  call HDF5_read(phase_mechanical_S0(ph)%data,groupHandle,'S')
  call HDF5_read(phase_mechanical_F0(ph)%data,groupHandle,'F')
  call HDF5_read(phase_mechanical_Fp0(ph)%data,groupHandle,'F_p')
  call HDF5_read(phase_mechanical_Fi0(ph)%data,groupHandle,'F_i')
  call HDF5_read(phase_mechanical_Lp0(ph)%data,groupHandle,'L_p')
  call HDF5_read(phase_mechanical_Li0(ph)%data,groupHandle,'L_i')

end subroutine mechanical_restartRead


!--------------------------------------------------------------------------------------------------
!< @brief Get second Piola-Kirchhoff stress (for use by non-mech physics).
!--------------------------------------------------------------------------------------------------
module function mechanical_S(ph,en) result(S)

  integer, intent(in) :: ph,en
  real(pREAL), dimension(3,3) :: S


  S = phase_mechanical_S(ph)%data(1:3,1:3,en)

end function mechanical_S


!--------------------------------------------------------------------------------------------------
!< @brief Get plastic velocity gradient (for use by non-mech physics).
!--------------------------------------------------------------------------------------------------
module function mechanical_L_p(ph,en) result(L_p)

  integer, intent(in) :: ph,en
  real(pREAL), dimension(3,3) :: L_p


  L_p = phase_mechanical_Lp(ph)%data(1:3,1:3,en)

end function mechanical_L_p


!--------------------------------------------------------------------------------------------------
!< @brief Get elastic deformation gradient (for use by non-mech physics).
!--------------------------------------------------------------------------------------------------
module function mechanical_F_e(ph,en) result(F_e)

  integer, intent(in) :: ph,en
  real(pREAL), dimension(3,3) :: F_e


  F_e = phase_mechanical_Fe(ph)%data(1:3,1:3,en)

end function mechanical_F_e


!--------------------------------------------------------------------------------------------------
!< @brief Get eigen deformation gradient (for use by non-mech physics).
!--------------------------------------------------------------------------------------------------
module function mechanical_F_i(ph,en) result(F_i)

  integer, intent(in) :: ph,en
  real(pREAL), dimension(3,3) :: F_i


  F_i = phase_mechanical_Fi(ph)%data(1:3,1:3,en)

end function mechanical_F_i


!--------------------------------------------------------------------------------------------------
!< @brief Get first Piola-Kirchhoff stress (for use by homogenization).
!--------------------------------------------------------------------------------------------------
module function phase_P(co,ce) result(P)

  integer, intent(in) :: co, ce
  real(pREAL), dimension(3,3) :: P


  P = phase_mechanical_P(material_ID_phase(co,ce))%data(1:3,1:3,material_entry_phase(co,ce))

end function phase_P


!--------------------------------------------------------------------------------------------------
!< @brief Get deformation gradient (for use by homogenization).
!--------------------------------------------------------------------------------------------------
module function phase_F(co,ce) result(F)

  integer, intent(in) :: co, ce
  real(pREAL), dimension(3,3) :: F


  F = phase_mechanical_F(material_ID_phase(co,ce))%data(1:3,1:3,material_entry_phase(co,ce))

end function phase_F


!--------------------------------------------------------------------------------------------------
!< @brief Set deformation gradient (for use by homogenization).
!--------------------------------------------------------------------------------------------------
module subroutine phase_set_F(F,co,ce)

  real(pREAL), dimension(3,3), intent(in) :: F
  integer, intent(in) :: co, ce


  phase_mechanical_F(material_ID_phase(co,ce))%data(1:3,1:3,material_entry_phase(co,ce)) = F

end subroutine phase_set_F


#if defined(METAL) && defined(GRID)
!--------------------------------------------------------------------------------------------------
!> @brief Constitutive update of a batch of cells on the GPU (queued; see
!> phase_mechanical_constitutive_GPU_finish).
!> @details Expects single-constituent cells that are restored and partitioned (F set). Only
!> entries the GPU kernel reproduces exactly are processed: supported plasticity models, Hooke
!> elasticity with the stiffness uploaded at setup, Fi0 = I, no eigenstrain. A share of these cells
!> (GPU: split, automatic: balanced by the measured times) is left to the CPU, which integrates them
!> while the GPU runs. pending(ce) = .true. if the cell was queued on the GPU. All other cells, and
!> those the GPU does not integrate, need phase_mechanical_constitutive (CPU, incl. cutback), which
!> gives the same robustness as without GPU.
!--------------------------------------------------------------------------------------------------
module subroutine phase_mechanical_constitutive_GPU(Delta_t,ce_start,ce_end,eligible,pending)

  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: ce_start, ce_end
  logical, dimension(ce_start:ce_end), intent(in)  :: eligible
  logical, dimension(ce_start:ce_end), intent(out) :: pending

  integer :: ph, en, ce, i, n, N_st, N_s, n_take
  logical, dimension(ce_start:ce_end) :: take
  type(C_PTR) :: dPdF_ptr
  real(pREAL) :: alpha


  pending = .false.
  if (.not. (Metal_constitutive .and. c_associated(Metal_ctx))) return
  if (.not. allocated(GPU_handle)) call GPU_setup()
  if (GPU_tangent_verify) then
    print'(1x,a,es9.2)', 'GPU vs CPU tangent (previous call), max relative difference: ', GPU_tangent_maxdiff
    GPU_tangent_maxdiff = 0.0_pREAL
  end if

  do ph = 1, size(GPU_handle)
    if (GPU_handle(ph) < 0) cycle

    !$OMP PARALLEL DO PRIVATE(en)
    do ce = ce_start, ce_end
      take(ce) = eligible(ce)
      if (take(ce)) take(ce) = material_ID_phase(1,ce) == ph
      if (take(ce)) then
        en = material_entry_phase(1,ce)
        if (mechanical_plasticity_type(ph) == MECHANICAL_PLASTICITY_TWINTRIP) then                    ! isotropic eigenstrain Fi = j(state) I
          take(ce) = all(dEq0(phase_mechanical_Fi0(ph)%data(1:3,1:3,en) &
                              - phase_mechanical_Fi0(ph)%data(1,1,en)*math_I3,1.0e-10_pREAL))
        else
          take(ce) = all(dEq0(phase_mechanical_Fi0(ph)%data(1:3,1:3,en) - math_I3,1.0e-10_pREAL))        ! O.inv(V_e).O^T = I up to rounding
        end if
        if (take(ce)) then
          if (any(mechanical_plasticity_type(ph) == [MECHANICAL_PLASTICITY_TDT, &
                                                     MECHANICAL_PLASTICITY_TWINTRIP])) then           ! stiffness from the state: parent as uploaded
            take(ce) = all(dEq(elastic_C66(ph,en),GPU_C66(1:6,1:6,ph)))
          else
            take(ce) = all(dEq(phase_damage_C66(phase_homogenizedC66(ph,en),ph,en),GPU_C66(1:6,1:6,ph)))
          end if
        end if
      end if
    end do
    !$OMP END PARALLEL DO
    n_take = count(take)
    if (n_take == 0) cycle
    alpha = merge(Metal_split,GPU_work(ph)%share,Metal_split >= 0.0_pREAL)
    if (alpha < 1.0_pREAL) then                                                                     ! CPU share spread evenly (balanced OpenMP loop)
      i = 0
      do ce = ce_start, ce_end
        if (.not. take(ce)) cycle
        i = i + 1
        take(ce) = floor(real(i,pREAL)*alpha) > floor(real(i-1,pREAL)*alpha)
      end do
    end if
    n = count(take)
    GPU_work(ph)%n_CPU = n_take - n
    if (n == 0) cycle

    N_s  = plasticState(ph)%sizeState
    N_st = N_s + GPU_ndep(ph)                                                                       ! state followed by dependent state
    associate(w => GPU_work(ph))
    if (allocated(w%ce_list)) then
      if (size(w%ce_list) /= n) deallocate(w%ce_list,w%F,w%Fp0,w%S0,w%Lp_guess,w%state0, &
                                           w%P,w%S,w%Fp,w%Fe,w%Lp,w%state,w%stat,w%iterations,w%dPdF,w%T)
    end if
    if (.not. allocated(w%ce_list)) then
      allocate(w%ce_list(n),w%F(3,3,n),w%Fp0(3,3,n),w%S0(3,3,n),w%Lp_guess(3,3,n),w%state0(N_st,n))
      allocate(w%P(3,3,n),w%S(3,3,n),w%Fp(3,3,n),w%Fe(3,3,n),w%Lp(3,3,n),w%state(N_st,n),w%stat(n),w%iterations(n))
      allocate(w%dPdF(3,3,3,3,merge(n,1,Metal_tangent)),w%T(n))
    end if
    w%ce_list = pack([(ce,ce=ce_start,ce_end)],take)
    dPdF_ptr = merge(c_loc(w%dPdF),C_NULL_PTR,Metal_tangent)
    associate(ce_list => w%ce_list, F => w%F, Fp0 => w%Fp0, S0 => w%S0, Lp_guess => w%Lp_guess, &
              state0 => w%state0, P => w%P, S => w%S, Fp => w%Fp, Fe => w%Fe, Lp => w%Lp, &
              state => w%state, stat => w%stat, iterations => w%iterations, dPdF => w%dPdF, Temp => w%T)

    !$OMP PARALLEL DO PRIVATE(en)
    do i = 1, n
      en = material_entry_phase(1,ce_list(i))
      F(1:3,1:3,i)        = phase_mechanical_F(ph)%data(1:3,1:3,en)
      Fp0(1:3,1:3,i)      = phase_mechanical_Fp0(ph)%data(1:3,1:3,en)
      S0(1:3,1:3,i)       = phase_mechanical_S0(ph)%data(1:3,1:3,en)                                ! = S after restore (initial dotState)
      Lp_guess(1:3,1:3,i) = phase_mechanical_Lp(ph)%data(1:3,1:3,en)                                ! as integrateStress
      state0(1:N_s,i)     = plasticState(ph)%state0(1:N_s,en)
      if (GPU_ndep(ph) > 0) call GPU_dependent(ph,en,state0(1:N_s,i),state0(N_s+1:N_st,i),.false.)   ! as stored (first predictor)
      Temp(i)             = thermal_T(ph,en)
    end do
    !$OMP END PARALLEL DO

    w%dt = Delta_t
    w%submitted = Metal_cpSubmit(Metal_ctx,int(GPU_handle(ph),C_INT),int(n,C_INT),Delta_t, &
                                 F,Fp0,S0,Lp_guess,state0,c_loc(Temp),P,S,Fp,Fe,Lp,state,dPdF_ptr,stat,iterations) == 0
    if (w%submitted) pending(ce_list) = .true.

    end associate
    end associate
  end do

  contains

  !------------------------------------------------------------------------------------------------
  !> @brief Upload the models of all GPU-capable phases (first call).
  !------------------------------------------------------------------------------------------------
  subroutine GPU_setup()

    integer :: ph
    real(pREAL), dimension(7) :: numerics


    block
      integer :: l
      call get_environment_variable('DAMASK_METAL_VERIFY_TANGENT',length=l)
      GPU_tangent_verify = l > 0
      call get_environment_variable('DAMASK_METAL_VERBOSE',length=l)
      GPU_verbose = l > 0
    end block
    numerics = [Metal_rtol_S, 1.0e-6_pREAL, num%rtol_crystalliteState, num%stepSizeLp, 0.5_pREAL, &
                real(num%nStress_Lp,pREAL), real(min(num%nState,Metal_nState),pREAL)]                ! slow points: CPU (the batch waits for the slowest)
    if (Metal_nState < 0) numerics(7) = real(min(num%nState,14),pREAL)                             ! default; tdt, twintrip: as the CPU (twin nucleation)
    allocate(GPU_handle(size(mechanical_plasticity_type)), source=-1)
    allocate(GPU_C66(6,6,size(GPU_handle)), source=0.0_pREAL)
    allocate(GPU_tangent(size(GPU_handle)))
    allocate(GPU_work(size(GPU_handle)))
    allocate(GPU_ndep(size(GPU_handle)), source=0)

    do ph = 1, size(GPU_handle)
      if (all(mechanical_plasticity_type(ph) /= [MECHANICAL_PLASTICITY_PHENOPOWERLAW, &
                                                 MECHANICAL_PLASTICITY_KINEHARDENING, &
                                                 MECHANICAL_PLASTICITY_DISLOTUNGSTEN, &
                                                 MECHANICAL_PLASTICITY_TDT, &
                                                 MECHANICAL_PLASTICITY_TWINTRIP])) cycle
      if (eigen_active(ph) .or. damage_type(ph) /= UNDEFINED) cycle
      if (size(phase_mechanical_F(ph)%data,3) == 0) cycle
      GPU_C66(1:6,1:6,ph) = phase_damage_C66(phase_homogenizedC66(ph,1),ph,1)
      select case (mechanical_plasticity_type(ph))
        case (MECHANICAL_PLASTICITY_PHENOPOWERLAW)
          GPU_handle(ph) = plastic_phenopowerlaw_GPUadd(ph,GPU_C66(1:6,1:6,ph),numerics)
        case (MECHANICAL_PLASTICITY_KINEHARDENING)
          GPU_handle(ph) = plastic_kinehardening_GPUadd(ph,GPU_C66(1:6,1:6,ph),numerics)
        case (MECHANICAL_PLASTICITY_DISLOTUNGSTEN)
          GPU_handle(ph) = plastic_dislotungsten_GPUadd(ph,GPU_C66(1:6,1:6,ph),numerics)
          GPU_ndep(ph) = 2*plasticState(ph)%sizeState/3                                              ! [tau_pass, Lambda_sl], 2 x N_sl
        case (MECHANICAL_PLASTICITY_TDT)
          GPU_C66(1:6,1:6,ph) = elastic_C66(ph,1)                                                   ! parent; homogenized on the GPU
          GPU_handle(ph) = plastic_tdt_GPUadd(ph,GPU_C66(1:6,1:6,ph), &
                                              [numerics(1:6),merge(real(num%nState,pREAL),numerics(7),Metal_nState < 0)])
          GPU_ndep(ph) = 36                                                                         ! homogenized C66
        case (MECHANICAL_PLASTICITY_TWINTRIP)
          GPU_C66(1:6,1:6,ph) = elastic_C66(ph,1)                                                   ! austenite; homogenized on the GPU
          GPU_handle(ph) = plastic_twintrip_GPUadd(ph,GPU_C66(1:6,1:6,ph), &
                                                   [numerics(1:6),merge(real(num%nState,pREAL),numerics(7),Metal_nState < 0)], &
                                                   GPU_ndep(ph))
      end select
      if (GPU_handle(ph) >= 0 .and. Metal_tangent) then
        allocate(GPU_tangent(ph)%dPdF(3,3,3,3,size(phase_mechanical_F(ph)%data,3)))
        allocate(GPU_tangent(ph)%valid(size(phase_mechanical_F(ph)%data,3)),source=.false.)
      end if
    end do

  end subroutine GPU_setup


end subroutine phase_mechanical_constitutive_GPU


!--------------------------------------------------------------------------------------------------
!> @brief Wait for the GPU batch of phase_mechanical_constitutive_GPU and store its results.
!> @details ok(ce) = .true. if the cell was integrated (P, S, Fp, Fe, Lp, state written). t_CPU: wall
!> time of the CPU work done meanwhile on n_CPU cells, used to balance the automatic split.
!--------------------------------------------------------------------------------------------------
module subroutine phase_mechanical_constitutive_GPU_finish(t_CPU,n_CPU,ce_start,ce_end,pending,ok)

  real(pREAL), intent(in) :: t_CPU
  integer, intent(in) :: n_CPU, ce_start, ce_end
  logical, dimension(ce_start:ce_end), intent(in)  :: pending
  logical, dimension(ce_start:ce_end), intent(out) :: ok

  integer :: ph, en, i, n, N_s, N_st
  real(C_DOUBLE) :: t_GPU
  real(pREAL) :: j


  ok = .false.
  if (.not. allocated(GPU_handle)) return

  do ph = 1, size(GPU_handle)
    associate(w => GPU_work(ph))
    if (.not. w%submitted) cycle
    w%submitted = .false.
    n    = size(w%ce_list)
    N_s  = plasticState(ph)%sizeState
    N_st = N_s + GPU_ndep(ph)
    if (Metal_cpWait(Metal_ctx,int(GPU_handle(ph),C_INT),t_GPU) == 0) then
      !$OMP PARALLEL DO PRIVATE(en,j)
      do i = 1, n
        en = material_entry_phase(1,w%ce_list(i))
        if (allocated(GPU_tangent(ph)%valid)) GPU_tangent(ph)%valid(en) = w%stat(i) == 0
        if (w%stat(i) /= 0) cycle
        if (Metal_tangent) GPU_tangent(ph)%dPdF(1:3,1:3,1:3,1:3,en) = w%dPdF(1:3,1:3,1:3,1:3,i)
        phase_mechanical_P(ph)%data(1:3,1:3,en)  = w%P(1:3,1:3,i)
        phase_mechanical_S(ph)%data(1:3,1:3,en)  = w%S(1:3,1:3,i)
        phase_mechanical_Fp(ph)%data(1:3,1:3,en) = w%Fp(1:3,1:3,i)
        phase_mechanical_Fe(ph)%data(1:3,1:3,en) = w%Fe(1:3,1:3,i)
        phase_mechanical_Lp(ph)%data(1:3,1:3,en) = w%Lp(1:3,1:3,i)
        plasticState(ph)%state(1:N_s,en)         = w%state(1:N_s,i)
        if (GPU_ndep(ph) > 0) call GPU_dependent(ph,en,w%state(1:N_s,i),w%state(N_s+1:N_st,i),.true.) ! of the last stress solve
        if (mechanical_plasticity_type(ph) == MECHANICAL_PLASTICITY_TWINTRIP) then                  ! converged Li loop: Fi = j I
          j = plastic_twintrip_GPUj(ph,en)
          phase_mechanical_Fi(ph)%data(1:3,1:3,en) = j*math_I3
          if (w%dt > 0.0_pREAL) &
            phase_mechanical_Li(ph)%data(1:3,1:3,en) = math_I3*(1.0_pREAL - phase_mechanical_Fi0(ph)%data(1,1,en)/j)/w%dt
        end if
        ok(w%ce_list(i)) = .true.
      end do
      !$OMP END PARALLEL DO

      if (GPU_verbose) print'(1x,a,i0,a,i0,a,i0,a,f8.2,a,f8.2,a,f6.3)', 'GPU split phase ',ph,': ',n,' GPU, ', &
                             n_CPU,' CPU cells; t_GPU ',1.0e3_pREAL*t_GPU,' ms, t_CPU ',1.0e3_pREAL*t_CPU,' ms, share ',w%share
      if (Metal_split < 0.0_pREAL .and. t_CPU > 0.0_pREAL .and. t_GPU > 0.0_C_DOUBLE) &           ! balance the wall times (GPU time is not
        w%share = min(max(w%share*sqrt(t_CPU/real(t_GPU,pREAL)),0.02_pREAL),0.95_pREAL)             ! proportional to the points: latency bound)
    end if
    end associate
  end do

end subroutine phase_mechanical_constitutive_GPU_finish


!--------------------------------------------------------------------------------------------------
!> @brief Get (set = .false.) or set the dependent state of an entry (models that have one).
!--------------------------------------------------------------------------------------------------
subroutine GPU_dependent(ph,en,x,D,set)

  integer, intent(in) :: ph, en
  real(pREAL), dimension(:), intent(in) :: x                                                        ! state
  real(pREAL), dimension(:), intent(inout) :: D
  logical, intent(in) :: set


  select case (mechanical_plasticity_type(ph))
    case (MECHANICAL_PLASTICITY_DISLOTUNGSTEN)
      call plastic_dislotungsten_GPUdependent(ph,en,D,set)
    case (MECHANICAL_PLASTICITY_TDT)
      if (.not. set) D = 0.0_pREAL                                                                  ! homogenized stiffness: computed on the GPU from the state
    case (MECHANICAL_PLASTICITY_TWINTRIP)
      call plastic_twintrip_GPUdependent(ph,en,D,set)
  end select

end subroutine GPU_dependent
#endif

end submodule mechanical
