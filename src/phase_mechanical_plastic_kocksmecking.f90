! SPDX-License-Identifier: AGPL-3.0-or-later
!--------------------------------------------------------------------------------------------------
!> @brief Crystal plasticity with a power-law flow rule, a Taylor (Franciosi) slip resistance and
!> Kocks-Mecking evolution of the dislocation density per slip system (experimental, DAMASK-Metal).
!> @details Equations as in M.A. Charpagne et al., Acta Materialia 215:117037, 2021, Eqs. (4)-(6),
!> and J.M. Hestroffer et al., J. Mech. Phys. Solids 196:106023, 2025, Eqs. (4), (5) and (7):
!>   dot_gamma_a = dot_gamma_0 |tau_a/g_a|^n sgn(tau_a)
!>   g_a         = g_0 + chi mu b sqrt(sum_b h_ab rho_b)
!>   dot_rho_a   = 1/b (sqrt(sum_b h_ab rho_b)/K - 2 y_c rho_a) |dot_gamma_a|
!> Kinematics, non-Schmid treatment and kinetics as for slip in phenopowerlaw, with the resistance
!> xi replaced by g (shear rate zero where the larger of the two resolved stresses is not positive).
!> The slip resistance g is computed from the current state wherever it is needed (no stored
!> dependent state), so that the CPU and the GPU (cp_model_km in src/grid/metal/cp_core.h)
!> evaluate the same function of the state.
!--------------------------------------------------------------------------------------------------
submodule(phase:plastic) kocksmecking

  type :: tParameters
    real(pREAL) :: &
      chi = 1.0_pREAL, &                                                                            !< statistical dislocation interaction coefficient
      mu  = 1.0_pREAL                                                                               !< shear modulus of the Taylor term [Pa]
    real(pREAL),               allocatable, dimension(:) :: &
      dot_gamma_0_sl, &                                                                             !< reference shear rate [1/s]
      n_sl, &                                                                                       !< stress exponent
      g_0_sl, &                                                                                     !< structure-independent slip resistance [Pa]
      b_sl, &                                                                                       !< magnitude of the Burgers vector [m]
      K, &                                                                                          !< Kocks-Mecking storage coefficient [1]
      y_c                                                                                           !< critical annihilation distance [m]
    real(pREAL),               allocatable, dimension(:,:) :: &
      h_sl_sl                                                                                       !< dislocation interaction a_ab
    real(pREAL),               allocatable, dimension(:,:,:) :: &
      P_sl, &
      P_nS_pos, &
      P_nS_neg
    integer :: &
      sum_N_sl                                                                                      !< total number of active slip systems
    character(len=pSTRLEN),    allocatable, dimension(:) :: &
      output
    character(len=:),          allocatable, dimension(:) :: &
      systems_sl
  end type tParameters

  type :: tIndexDotState
    integer, dimension(2) :: &
      rho_sl, &
      gamma_sl
  end type tIndexDotState

  type :: tKocksMeckingState
    real(pREAL), pointer, dimension(:,:) :: &
      rho_sl, &
      gamma_sl
  end type tKocksMeckingState

!--------------------------------------------------------------------------------------------------
! containers for parameters and state
  type(tParameters),        allocatable, dimension(:) :: param
  type(tIndexDotState),     allocatable, dimension(:) :: indexDotState
  type(tKocksMeckingState), allocatable, dimension(:) :: state

contains


!--------------------------------------------------------------------------------------------------
!> @brief Perform module initialization.
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function plastic_kocksmecking_init() result(myPlasticity)

  logical, dimension(:), allocatable :: myPlasticity
  integer :: &
    ph, &
    Nmembers, &
    sizeState, sizeDotState, &
    startIndex, endIndex
  integer,     dimension(:), allocatable :: &
    N_sl                                                                                            !< number of slip systems per family
  real(pREAL), dimension(:), allocatable :: &
    rho_0                                                                                           !< initial dislocation density per system
  real(pREAL), dimension(:,:), allocatable :: &
    a_nS                                                                                            !< non-Schmid coefficients
  character(len=:), allocatable :: &
    refs, &
    extmsg
  type(tDict), pointer :: &
    phases, &
    phase, &
    mech, &
    pl


  myPlasticity = plastic_active('kocksmecking')
  if (count(myPlasticity) == 0) return

  print'(/,1x,a)', '<<<+-  phase:mechanical:plastic:kocksmecking init  -+>>>'
  print'(/,1x,a)', 'experimental (DAMASK-Metal); equations as in'
  print'(  1x,a)', 'M.A. Charpagne et al., Acta Materialia 215:117037, 2021'
  print'(  1x,a)', 'https://doi.org/10.1016/j.actamat.2021.117037'

  print'(/,1x,a,1x,i0)', '# phases:',count(myPlasticity); flush(IO_STDOUT)


  phases => config_material%get_dict('phase')
  allocate(param(size(phases)))
  allocate(indexDotState(size(phases)))
  allocate(state(size(phases)))
  extmsg = ''

  do ph = 1, size(phases)
    if (.not. myPlasticity(ph)) cycle

    associate(prm => param(ph), &
              stt => state(ph), &
              idx_dot => indexDotState(ph))

    phase => phases%get_dict(ph)
    mech => phase%get_dict('mechanical')
    pl => mech%get_dict('plastic')

    print'(/,1x,a,1x,i0,a)', 'phase',ph,': '//phases%key(ph)
    refs = config_listReferences(pl,indent=3)
    if (len(refs) > 0) print'(/,1x,a)', refs

#if defined (__GFORTRAN__)
    prm%output = output_as1dStr(pl)
#else
    prm%output = pl%get_as1dStr('output',defaultVal=emptyStrArray)
#endif

!--------------------------------------------------------------------------------------------------
! slip related parameters
    N_sl = pl%get_as1dInt('N_sl',defaultVal=emptyIntArray)
    prm%sum_N_sl = sum(abs(N_sl))
    slipActive: if (prm%sum_N_sl > 0) then
      prm%P_sl = crystal_SchmidMatrix_slip(N_sl,phase_lattice(ph),phase_cOverA(ph))
      prm%systems_sl = crystal_labels_slip(N_sl,phase_lattice(ph))

      a_nS = pl%get_as2dReal('a_non-Schmid',defaultVal=reshape(emptyRealArray,[0,0]))
      prm%P_nS_pos = crystal_SchmidMatrix_slip(N_sl,phase_lattice(ph),phase_cOverA(ph), &
                                               nonSchmidCoefficients=a_nS,sense=+1)
      prm%P_nS_neg = crystal_SchmidMatrix_slip(N_sl,phase_lattice(ph),phase_cOverA(ph), &
                                               nonSchmidCoefficients=a_nS,sense=-1)

      prm%h_sl_sl = crystal_interaction_SlipBySlip(N_sl,pl%get_as1dReal('h_sl-sl'),phase_lattice(ph))

      prm%dot_gamma_0_sl = math_expand(pl%get_as1dReal('dot_gamma_0_sl',requiredSize=size(N_sl)),N_sl)
      prm%n_sl           = math_expand(pl%get_as1dReal('n_sl',          requiredSize=size(N_sl)),N_sl)
      prm%g_0_sl         = math_expand(pl%get_as1dReal('g_0_sl',        requiredSize=size(N_sl)),N_sl)
      prm%b_sl           = math_expand(pl%get_as1dReal('b_sl',          requiredSize=size(N_sl)),N_sl)
      rho_0              = math_expand(pl%get_as1dReal('rho_0',         requiredSize=size(N_sl)),N_sl)
      prm%K              = math_expand(pl%get_as1dReal('K',             requiredSize=size(N_sl)),N_sl)
      prm%y_c            = math_expand(pl%get_as1dReal('y_c',           requiredSize=size(N_sl)),N_sl)
      prm%chi            = pl%get_asReal('chi',defaultVal=1.0_pREAL)
      prm%mu             = pl%get_asReal('mu')

      ! sanity checks
      if (any(prm%dot_gamma_0_sl <= 0.0_pREAL)) extmsg = trim(extmsg)//' dot_gamma_0_sl'
      if (any(prm%n_sl           <= 0.0_pREAL)) extmsg = trim(extmsg)//' n_sl'
      if (any(prm%g_0_sl         <  0.0_pREAL)) extmsg = trim(extmsg)//' g_0_sl'
      if (any(prm%b_sl           <= 0.0_pREAL)) extmsg = trim(extmsg)//' b_sl'
      if (any(rho_0              <  0.0_pREAL)) extmsg = trim(extmsg)//' rho_0'
      if (any(prm%K              <= 0.0_pREAL)) extmsg = trim(extmsg)//' K'
      if (any(prm%y_c            <  0.0_pREAL)) extmsg = trim(extmsg)//' y_c'
      if (    prm%chi            <  0.0_pREAL)  extmsg = trim(extmsg)//' chi'
      if (    prm%mu             <= 0.0_pREAL)  extmsg = trim(extmsg)//' mu'
      if (any(prm%h_sl_sl        <  0.0_pREAL)) extmsg = trim(extmsg)//' h_sl-sl'
      if (any(prm%g_0_sl + prm%chi*prm%mu*prm%b_sl*sqrt(max(matmul(prm%h_sl_sl,rho_0),0.0_pREAL)) &
              <= 0.0_pREAL))                    extmsg = trim(extmsg)//' g_0_sl or rho_0 (zero slip resistance)'

    else slipActive
      rho_0 = emptyRealArray
      allocate(prm%dot_gamma_0_sl,prm%n_sl,prm%g_0_sl,prm%b_sl,prm%K,prm%y_c,source=emptyRealArray)
      allocate(prm%h_sl_sl(0,0))
    end if slipActive

!--------------------------------------------------------------------------------------------------
! allocate state arrays
    Nmembers = count(material_ID_phase == ph)
    sizeDotState = size(['rho_sl  ','gamma_sl']) * prm%sum_N_sl
    sizeState = sizeDotState

    call phase_allocateState(plasticState(ph),Nmembers,sizeState,sizeDotState,0)
    deallocate(plasticState(ph)%dotState) ! ToDo: remove dotState completely

!--------------------------------------------------------------------------------------------------
! state aliases and initialization
    startIndex = 1
    endIndex   = prm%sum_N_sl
    idx_dot%rho_sl = [startIndex,endIndex]
    stt%rho_sl => plasticState(ph)%state(startIndex:endIndex,:)
    stt%rho_sl = spread(rho_0,2,Nmembers)
    plasticState(ph)%atol(startIndex:endIndex) = pl%get_asReal('atol_rho',defaultVal=1.0_pREAL)
    if (any(plasticState(ph)%atol(startIndex:endIndex) < 0.0_pREAL)) extmsg = trim(extmsg)//' atol_rho'

    startIndex = endIndex + 1
    endIndex   = endIndex + prm%sum_N_sl
    idx_dot%gamma_sl = [startIndex,endIndex]
    stt%gamma_sl => plasticState(ph)%state(startIndex:endIndex,:)
    plasticState(ph)%atol(startIndex:endIndex) = pl%get_asReal('atol_gamma',defaultVal=1.0e-6_pREAL)
    if (any(plasticState(ph)%atol(startIndex:endIndex) < 0.0_pREAL)) extmsg = trim(extmsg)//' atol_gamma'

    end associate

!--------------------------------------------------------------------------------------------------
!  exit if any parameter is out of range
    if (extmsg /= '') call IO_error(211,ext_msg=trim(extmsg))

  end do

end function plastic_kocksmecking_init


!--------------------------------------------------------------------------------------------------
!> @brief Calculate plastic velocity gradient and its tangent.
!--------------------------------------------------------------------------------------------------
pure module subroutine kocksmecking_LpAndItsTangent(Lp,dLp_dMp,Mp,ph,en)

  real(pREAL), dimension(3,3),     intent(out) :: &
    Lp                                                                                              !< plastic velocity gradient
  real(pREAL), dimension(3,3,3,3), intent(out) :: &
    dLp_dMp                                                                                         !< derivative of Lp with respect to the Mandel stress
  real(pREAL), dimension(3,3),     intent(in) :: &
    Mp                                                                                              !< Mandel stress
  integer,                         intent(in) :: &
    ph, &
    en

  integer :: &
    i,k,l,m,n
  real(pREAL), dimension(param(ph)%sum_N_sl) :: &
    dot_gamma_sl,ddot_gamma_dtau_sl
  real(pREAL), dimension(3,3,param(ph)%sum_N_sl) :: &
    P_nS


  Lp = 0.0_pREAL
  dLp_dMp = 0.0_pREAL

  associate(prm => param(ph))

    call kinetics_sl(Mp,ph,en,dot_gamma_sl,ddot_gamma_dtau_sl)
    P_nS = merge(prm%P_nS_pos,prm%P_nS_neg, spread(spread(dot_gamma_sl,1,3),2,3)>0.0_pREAL)         ! faster than 'merge' in loop
    do i = 1, prm%sum_N_sl
      Lp = Lp + dot_gamma_sl(i)*prm%P_sl(1:3,1:3,i)
      forall (k=1:3,l=1:3,m=1:3,n=1:3) &
        dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) &
                         + ddot_gamma_dtau_sl(i) * prm%P_sl(k,l,i) * P_nS(m,n,i)
    end do

  end associate

end subroutine kocksmecking_LpAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief Calculate the rate of change of microstructure.
!--------------------------------------------------------------------------------------------------
module function kocksmecking_dotState(Mp,ph,en) result(dotState)

  real(pREAL), dimension(3,3),  intent(in) :: &
    Mp                                                                                              !< Mandel stress
  integer,                      intent(in) :: &
    ph, &
    en
  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: &
    dotState


  associate(prm => param(ph), stt => state(ph), &
            dot_rho_sl   => dotState(indexDotState(ph)%rho_sl(1):indexDotState(ph)%rho_sl(2)), &
            dot_gamma_sl => dotState(indexDotState(ph)%gamma_sl(1):indexDotState(ph)%gamma_sl(2)))

    call kinetics_sl(Mp,ph,en,dot_gamma_sl)
    dot_gamma_sl = abs(dot_gamma_sl)

    dot_rho_sl = (sqrt(matmul(prm%h_sl_sl,stt%rho_sl(:,en)))/prm%K &
                 - 2.0_pREAL*prm%y_c*stt%rho_sl(:,en)) * dot_gamma_sl / prm%b_sl

  end associate

end function kocksmecking_dotState


!--------------------------------------------------------------------------------------------------
!> @brief Write results to HDF5 output file.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_kocksmecking_result(ph,group)

  integer,          intent(in) :: ph
  character(len=*), intent(in) :: group

  integer :: ou, en
  real(pREAL), dimension(:,:), allocatable :: g


  associate(prm => param(ph), stt => state(ph))

    do ou = 1,size(prm%output)

      select case(trim(prm%output(ou)))

        case('rho_sl')
          call result_writeDataset(stt%rho_sl,group,trim(prm%output(ou)), &
                                   'dislocation density','1/m²',prm%systems_sl)
        case('g_sl')
          allocate(g(prm%sum_N_sl,size(stt%rho_sl,2)))
          do en = 1, size(stt%rho_sl,2)
            g(:,en) = slipResistance(ph,en)
          end do
          call result_writeDataset(g,group,trim(prm%output(ou)), &
                                   'resistance against plastic slip','Pa',prm%systems_sl)
          deallocate(g)
        case('gamma_sl')
          call result_writeDataset(stt%gamma_sl,group,trim(prm%output(ou)), &
                                   'plastic shear','1',prm%systems_sl)
      end select

    end do

  end associate

end subroutine plastic_kocksmecking_result


!--------------------------------------------------------------------------------------------------
!> @brief Slip resistance g = g_0 + chi mu b sqrt(h_sl_sl rho) of the current state.
!--------------------------------------------------------------------------------------------------
pure function slipResistance(ph,en) result(g)

  integer, intent(in) :: &
    ph, &
    en
  real(pREAL), dimension(param(ph)%sum_N_sl) :: &
    g


  associate(prm => param(ph), stt => state(ph))
    g = prm%g_0_sl + prm%chi*prm%mu*prm%b_sl*sqrt(matmul(prm%h_sl_sl,stt%rho_sl(:,en)))
  end associate

end function slipResistance


!--------------------------------------------------------------------------------------------------
!> @brief Calculate shear rates on slip systems and their derivatives with respect to resolved
!         stress.
!> @details Sign of dot_gamma_sl conveys sense of shear. Derivatives are calculated only optionally,
! hence, contrary to common convention, the result (i.e. intent(out)) variables are put at the end.
!--------------------------------------------------------------------------------------------------
pure subroutine kinetics_sl(Mp,ph,en, &
                            dot_gamma_sl,ddot_gamma_dtau_sl)

  real(pREAL), dimension(3,3),                           intent(in) :: &
    Mp                                                                                              !< Mandel stress
  integer,                                               intent(in) :: &
    ph, &
    en

  real(pREAL), dimension(param(ph)%sum_N_sl),           intent(out) :: &
    dot_gamma_sl
  real(pREAL), dimension(param(ph)%sum_N_sl), optional, intent(out) :: &
    ddot_gamma_dtau_sl

  real(pREAL), dimension(param(ph)%sum_N_sl) :: &
    tau_sl_pos, &
    tau_sl_neg, &
    tau_sl_max, &
    g
  integer :: i


  associate(prm => param(ph))

    tau_sl_pos = [(math_tensordot(Mp,prm%P_nS_pos(1:3,1:3,i)),i=1,prm%sum_N_sl)]
    tau_sl_neg = [(math_tensordot(Mp,prm%P_nS_neg(1:3,1:3,i)),i=1,prm%sum_N_sl)]
    tau_sl_max = max(tau_sl_pos,tau_sl_neg)
    g = slipResistance(ph,en)

    where(tau_sl_max > 0.0_pREAL)
      dot_gamma_sl = merge(+1.0_pREAL,-1.0_pREAL, tau_sl_pos>tau_sl_neg) &
                   * prm%dot_gamma_0_sl &
                   * (tau_sl_max/g)**prm%n_sl
    else where
      dot_gamma_sl = 0.0_pREAL
    end where

    if (present(ddot_gamma_dtau_sl)) then
      where(dNeq0(dot_gamma_sl))
        ddot_gamma_dtau_sl = dot_gamma_sl*prm%n_sl/tau_sl_max
      else where
        ddot_gamma_dtau_sl = 0.0_pREAL
      end where
    end if

  end associate

end subroutine kinetics_sl

#if defined(METAL) && defined(GRID)
!--------------------------------------------------------------------------------------------------
!> @brief Upload the model of a phase to the GPU constitutive kernel (cp_model_km in
!> src/grid/metal/cp_core.h). Up to 48 slip systems, non-Schmid law allowed.
!> Returns the GPU handle or a negative number if the phase stays on the CPU.
!--------------------------------------------------------------------------------------------------
module function plastic_kocksmecking_GPUadd(ph,C66,numerics) result(handle)

  integer, intent(in) :: ph
  real(pREAL), dimension(6,6), intent(in) :: C66
  real(pREAL), dimension(7), intent(in) :: numerics
  integer :: handle

  character(len=:), allocatable :: reason


  handle = -1
  associate(prm => param(ph))
    if (prm%sum_N_sl == 0 .or. prm%sum_N_sl > 48) then
      reason = 'number of slip systems'
    else
      handle = Metal_cpAddPhaseModel(Metal_ctx,METAL_MODEL_KOCKSMECKING,int(prm%sum_N_sl,C_INT),C66, &
                                     prm%P_sl,prm%P_nS_pos,prm%P_nS_neg,prm%h_sl_sl, &
                                     [prm%dot_gamma_0_sl,prm%n_sl,prm%g_0_sl,prm%b_sl,prm%K,prm%y_c], &
                                     [plasticState(ph)%atol(indexDotState(ph)%rho_sl(1)), &
                                      plasticState(ph)%atol(indexDotState(ph)%gamma_sl(1)), &
                                      prm%chi,prm%mu],numerics)
      reason = 'GPU initialization failed'
    end if
  end associate

  if (handle >= 0) then
    print'(/,1x,a,i0,a)', 'phase ',ph,': kocksmecking constitutive update on GPU'
  else
    print'(/,1x,a,i0,a)', 'phase ',ph,': kocksmecking constitutive update on CPU ('//reason//')'
  end if
  flush(IO_STDOUT)

end function plastic_kocksmecking_GPUadd
#endif

end submodule kocksmecking
