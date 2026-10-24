! SPDX-License-Identifier: AGPL-3.0-or-later
!--------------------------------------------------------------------------------------------------
!> @brief  Composite twinning model 'tdt': twinning, detwinning and retwinning with slip inside twins
!> @details Every material point is a composite of the parent (volume fraction w_0 = 1 - sum(f)) and
!> one region per twin system k (volume fraction f_k, lattice rotated by Q_k = 180 deg about the twin
!> plane normal). All regions see the same Mandel stress (iso-stress, Kalidindi 1998):
!>   Lp = w_0 sum_i gd_i P_i + sum_k f_k sum_i gd_i^(k) Q_k P_i Q_k^T + sum_k gamma_k fd_k P_tw,k
!> slip: phenomenological power law, own resistances in the parent (per system) and in each twin
!> (per slip family, Voce). Twin volume fractions evolve in both directions (TDT, Wang et al. 2013):
!>   fd_k = gd_0/gamma_k * [ (1 - sum(f)/f_sat) <tau_k/xi+_k>^n - f_k <-tau_k/xi-_k>^n ]
!>   xi+ = xi_tw + (xi_nuc - xi_0_tw) exp(-f_k/f_n)        nucleation -> growth
!>   xi- = xi_dtw + (xi_retw - xi_0_dtw) exp(-w_0/f_n)     shrinkage -> parent re-nucleation
!>   tau_k = Mp:P_tw,k - H_chi f_k w_0                     (optional twin back stress)
!> The twin volume fractions are advanced in closed form (backward Euler over the step, f0 = f at its
!> start, kept as delta state), which keeps 0 <= f_k and sum(f) <= f_sat and removes the stiffness of
!> saturation and complete detwinning that stops fixed-point state integration:
!>   shrinkage (tau_k < 0): fd_k = -lambda_k f0_k / (1 + lambda_k dt)
!>   growth    (tau_k > 0): fd_k = a_k R / (f_sat + A dt), A = sum a, R = f_sat - F_end(others) - F0(growing)
!> Nucleation threshold and back stress use f0, so tau_k depends on the stress only.
!> Elasticity: Reuss (default, consistent with iso-stress) or Voigt average of the rotated stiffnesses.
!--------------------------------------------------------------------------------------------------
submodule(phase:plastic) tdt

  real(pREAL), parameter :: F_MIN = 1.0e-12_pREAL                                                  !< twins with f <= F_MIN carry no slip

  type :: tParameters
    real(pREAL), allocatable, dimension(:) :: &
      dot_gamma_0_sl, n_sl, a_sl, h_0_sl_sl, xi_inf_sl, h_0_sl_tw, &                               !< per slip system
      xi_0_sl_tw, xi_inf_sl_tw, h_0_sl_tw_f, a_sl_tw, &                                             !< per slip family: slip inside twins
      dot_gamma_0_tw, n_tw, gamma_char, dxi_nuc, dxi_retw, H_chi, h_0_tw_tw, h_0_tw_sl, h_0_dtw     !< per twin system
    real(pREAL), allocatable, dimension(:,:) :: &
      h_sl_sl, h_sl_tw, h_tw_sl, h_tw_tw
    real(pREAL), allocatable, dimension(:,:,:) :: &
      P_sl, P_tw, Q_tw
    real(pREAL), allocatable, dimension(:,:,:,:) :: &
      P_sl_tw                                                                                       !< (3,3,N_sl,N_tw): Q_k P_i Q_k^T
    real(pREAL) :: &
      f_n, f_sat
    integer :: &
      sum_N_sl, sum_N_tw, N_fam
    integer, allocatable, dimension(:) :: &
      fam                                                                                           !< slip family (1..N_fam) of each slip system
    integer, allocatable, dimension(:) :: &
      N_tw_fam
    logical :: &
      Reuss
    character(len=pSTRLEN), allocatable, dimension(:) :: &
      output
    character(len=:), allocatable, dimension(:) :: &
      systems_sl, systems_tw
  end type tParameters

  type :: tTdtState
    real(pREAL), pointer, dimension(:,:) :: &
      xi_sl, gamma_sl, f_tw, xi_tw, xi_dtw, xi_sl_tw, gamma_sl_tw, &
      f_tw_0                                                                                        !< f at the start of the step (delta state)
  end type tTdtState

  type :: tTdtCache                                                                                 !< last homogenized stiffness per entry (homogenizedC is called many times per state)
    real(pREAL), allocatable, dimension(:,:)   :: key                                               !< f_tw and parent C66 it belongs to
    real(pREAL), allocatable, dimension(:,:,:) :: C_bar
  end type tTdtCache

  type(tParameters), allocatable, dimension(:) :: param
  type(tTdtState),   allocatable, dimension(:) :: state, deltaState
  type(tTdtCache),   allocatable, dimension(:) :: cache

contains


!--------------------------------------------------------------------------------------------------
!> @brief Perform module initialization.
!--------------------------------------------------------------------------------------------------
module function plastic_tdt_init() result(myPlasticity)

  logical, dimension(:), allocatable :: myPlasticity
  integer :: ph, i, k, f, Nmembers, sizeState, s, e
  integer, dimension(:), allocatable :: N_sl, N_tw
  real(pREAL), dimension(:), allocatable :: xi_0_sl, xi_0_tw, xi_0_dtw, xi_nuc, xi_retw, x
  character(len=:), allocatable :: refs, extmsg, mixing
  type(tDict), pointer :: phases, phase, mech, pl


  myPlasticity = plastic_active('tdt')
  if (count(myPlasticity) == 0) return

  print'(/,1x,a)', '<<<+-  phase:mechanical:plastic:tdt init  -+>>>'
  print'(/,1x,a)', 'S.R. Kalidindi, Journal of the Mechanics and Physics of Solids 46:267-290, 1998'
  print'(  1x,a)', 'https://doi.org/10.1016/S0022-5096(97)00051-3'
  print'(/,1x,a)', 'J. Wang et al., International Journal of Plasticity 49:36-52, 2013 (TDT)'
  print'(  1x,a)', 'https://doi.org/10.1016/j.ijplas.2013.02.016'
  print'(/,1x,a,1x,i0)', '# phases:',count(myPlasticity); flush(IO_STDOUT)

  phases => config_material%get_dict('phase')
  allocate(param(size(phases)))
  allocate(state(size(phases)))
  allocate(cache(size(phases)))
  allocate(deltaState(size(phases)))
  extmsg = ''

  do ph = 1, size(phases)
    if (.not. myPlasticity(ph)) cycle

    associate(prm => param(ph), stt => state(ph), dlt => deltaState(ph))

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
! slip (parent and inside twins)
    N_sl = pl%get_as1dInt('N_sl')
    prm%sum_N_sl = sum(abs(N_sl))
    if (prm%sum_N_sl == 0) extmsg = trim(extmsg)//' N_sl'
    prm%P_sl = crystal_SchmidMatrix_slip(N_sl,phase_lattice(ph),phase_cOverA(ph))
    prm%systems_sl = crystal_labels_slip(N_sl,phase_lattice(ph))
    prm%h_sl_sl = crystal_interaction_SlipBySlip(N_sl,pl%get_as1dReal('h_sl-sl'),phase_lattice(ph))

    prm%dot_gamma_0_sl = math_expand(pl%get_as1dReal('dot_gamma_0_sl',requiredSize=size(N_sl)), N_sl)
    prm%n_sl           = math_expand(pl%get_as1dReal('n_sl',          requiredSize=size(N_sl)), N_sl)
    prm%a_sl           = math_expand(pl%get_as1dReal('a_sl',          requiredSize=size(N_sl)), N_sl)
    prm%h_0_sl_sl      = math_expand(pl%get_as1dReal('h_0_sl-sl',     requiredSize=size(N_sl)), N_sl)
    xi_0_sl            = math_expand(pl%get_as1dReal('xi_0_sl',       requiredSize=size(N_sl)), N_sl)
    prm%xi_inf_sl      = math_expand(pl%get_as1dReal('xi_inf_sl',     requiredSize=size(N_sl)), N_sl)

    prm%N_fam = count(N_sl > 0)                                                                     ! families with active systems
    allocate(prm%fam(prm%sum_N_sl))
    k = 0
    do f = 1, size(N_sl)
      if (N_sl(f) <= 0) cycle
      k = k + 1
      prm%fam(sum(N_sl(1:f-1))+1:sum(N_sl(1:f))) = k
    end do
    x = pl%get_as1dReal('xi_0_sl',requiredSize=size(N_sl))
    prm%xi_0_sl_tw   = pack(pl%get_as1dReal('xi_0_sl_tw',  requiredSize=size(N_sl),defaultVal=x),N_sl > 0)
    x = pl%get_as1dReal('xi_inf_sl',requiredSize=size(N_sl))
    prm%xi_inf_sl_tw = pack(pl%get_as1dReal('xi_inf_sl_tw',requiredSize=size(N_sl),defaultVal=x),N_sl > 0)
    x = pl%get_as1dReal('h_0_sl-sl',requiredSize=size(N_sl))
    prm%h_0_sl_tw_f  = pack(pl%get_as1dReal('h_0_sl_tw',   requiredSize=size(N_sl),defaultVal=x),N_sl > 0)
    prm%a_sl_tw      = pack(pl%get_as1dReal('a_sl',        requiredSize=size(N_sl)),N_sl > 0)

    if (any(prm%dot_gamma_0_sl <= 0.0_pREAL)) extmsg = trim(extmsg)//' dot_gamma_0_sl'
    if (any(prm%n_sl           <= 0.0_pREAL)) extmsg = trim(extmsg)//' n_sl'
    if (any(prm%a_sl           <= 0.0_pREAL)) extmsg = trim(extmsg)//' a_sl'
    if (any(xi_0_sl            <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_0_sl'
    if (any(prm%xi_inf_sl      <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_inf_sl'
    if (any(prm%xi_0_sl_tw     <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_0_sl_tw'
    if (any(prm%xi_inf_sl_tw   <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_inf_sl_tw'

!--------------------------------------------------------------------------------------------------
! twinning
    N_tw = pl%get_as1dInt('N_tw')
    prm%N_tw_fam = N_tw
    prm%sum_N_tw = sum(abs(N_tw))
    if (prm%sum_N_tw == 0) extmsg = trim(extmsg)//' N_tw'
    prm%P_tw = crystal_SchmidMatrix_twin(N_tw,phase_lattice(ph),phase_cOverA(ph))
    prm%Q_tw = crystal_rotation_twin(N_tw,phase_lattice(ph),phase_cOverA(ph))
    prm%systems_tw = crystal_labels_twin(N_tw,phase_lattice(ph))
    prm%gamma_char = abs(crystal_characteristicShear_twin(N_tw,phase_lattice(ph),phase_cOverA(ph)))
    allocate(prm%P_sl_tw(3,3,prm%sum_N_sl,prm%sum_N_tw))
    do k = 1, prm%sum_N_tw; do i = 1, prm%sum_N_sl
      prm%P_sl_tw(1:3,1:3,i,k) = matmul(matmul(prm%Q_tw(1:3,1:3,k),prm%P_sl(1:3,1:3,i)),transpose(prm%Q_tw(1:3,1:3,k)))
    end do; end do

    prm%h_tw_tw        = crystal_interaction_TwinByTwin(N_tw,pl%get_as1dReal('h_tw-tw'),phase_lattice(ph))
    prm%h_sl_tw        = crystal_interaction_SlipByTwin(N_sl,N_tw,pl%get_as1dReal('h_sl-tw'),phase_lattice(ph))
    prm%h_tw_sl        = crystal_interaction_TwinBySlip(N_tw,N_sl,pl%get_as1dReal('h_tw-sl'),phase_lattice(ph))

    prm%dot_gamma_0_tw = math_expand(pl%get_as1dReal('dot_gamma_0_tw',requiredSize=size(N_tw)), N_tw)
    prm%n_tw           = math_expand(pl%get_as1dReal('n_tw',          requiredSize=size(N_tw)), N_tw)
    xi_0_tw            = math_expand(pl%get_as1dReal('xi_0_tw',       requiredSize=size(N_tw)), N_tw)
    x                  = pl%get_as1dReal('xi_0_tw',requiredSize=size(N_tw))
    xi_nuc             = math_expand(pl%get_as1dReal('xi_nuc_tw',     requiredSize=size(N_tw),defaultVal=x), N_tw)
    xi_0_dtw           = math_expand(pl%get_as1dReal('xi_0_dtw',      requiredSize=size(N_tw),defaultVal=x), N_tw)
    x                  = pl%get_as1dReal('xi_nuc_tw',requiredSize=size(N_tw),defaultVal=x)
    xi_retw            = math_expand(pl%get_as1dReal('xi_retw',       requiredSize=size(N_tw),defaultVal=x), N_tw)
    prm%dxi_nuc        = xi_nuc - xi_0_tw
    prm%dxi_retw       = xi_retw - xi_0_dtw
    prm%H_chi          = math_expand(pl%get_as1dReal('H_chi',         requiredSize=size(N_tw), &
                                                                      defaultVal=misc_zeros(size(N_tw))), N_tw)
    prm%h_0_tw_tw      = math_expand(pl%get_as1dReal('h_0_tw-tw',     requiredSize=size(N_tw)), N_tw)
    prm%h_0_tw_sl      = math_expand(pl%get_as1dReal('h_0_tw-sl',     requiredSize=size(N_tw)), N_tw)
    prm%h_0_dtw        = math_expand(pl%get_as1dReal('h_0_dtw',       requiredSize=size(N_tw), &
                                                                      defaultVal=misc_zeros(size(N_tw))), N_tw)
    prm%h_0_sl_tw      = math_expand(pl%get_as1dReal('h_0_sl-tw',     requiredSize=size(N_sl)), N_sl)
    prm%f_n            = pl%get_asReal('f_n',  defaultVal=0.01_pREAL)
    prm%f_sat          = pl%get_asReal('f_sat',defaultVal=1.0_pREAL)
    mixing             = pl%get_asStr('elastic_mixing',defaultVal='Reuss')
    prm%Reuss          = mixing == 'Reuss'

    if (any(prm%dot_gamma_0_tw <= 0.0_pREAL)) extmsg = trim(extmsg)//' dot_gamma_0_tw'
    if (any(prm%n_tw           <= 0.0_pREAL)) extmsg = trim(extmsg)//' n_tw'
    if (any(xi_0_tw            <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_0_tw'
    if (any(xi_nuc             <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_nuc_tw'
    if (any(xi_0_dtw           <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_0_dtw'
    if (any(xi_retw            <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_retw'
    if (prm%f_n <= 0.0_pREAL)                 extmsg = trim(extmsg)//' f_n'
    if (prm%f_sat <= 0.0_pREAL .or. prm%f_sat > 1.0_pREAL) extmsg = trim(extmsg)//' f_sat'
    if (all(mixing /= ['Reuss','Voigt']))     extmsg = trim(extmsg)//' elastic_mixing'

!--------------------------------------------------------------------------------------------------
! state: xi_sl, gamma_sl, f_tw, xi_tw, xi_dtw, xi_sl_tw (N_fam x N_tw), gamma_sl_tw (dot state), f_tw_0 (delta state)
    Nmembers = count(material_ID_phase == ph)
    sizeState = 2*prm%sum_N_sl + (5+prm%N_fam)*prm%sum_N_tw
    call phase_allocateState(plasticState(ph),Nmembers,sizeState,sizeState-prm%sum_N_tw,prm%sum_N_tw)
    allocate(cache(ph)%key(prm%sum_N_tw+36,Nmembers),source=-1.0_pREAL)
    allocate(cache(ph)%C_bar(6,6,Nmembers),source=0.0_pREAL)
    deallocate(plasticState(ph)%dotState)

    s = 1; e = prm%sum_N_sl
    stt%xi_sl => plasticState(ph)%state(s:e,:)
    stt%xi_sl = spread(xi_0_sl, 2, Nmembers)
    plasticState(ph)%atol(s:e) = pl%get_asReal('atol_xi',defaultVal=1.0_pREAL)

    s = e + 1; e = e + prm%sum_N_sl
    stt%gamma_sl => plasticState(ph)%state(s:e,:)
    plasticState(ph)%atol(s:e) = pl%get_asReal('atol_gamma',defaultVal=1.0e-6_pREAL)

    s = e + 1; e = e + prm%sum_N_tw
    stt%f_tw => plasticState(ph)%state(s:e,:)
    plasticState(ph)%atol(s:e) = pl%get_asReal('atol_f',defaultVal=1.0e-6_pREAL)

    s = e + 1; e = e + prm%sum_N_tw
    stt%xi_tw => plasticState(ph)%state(s:e,:)
    stt%xi_tw = spread(xi_0_tw, 2, Nmembers)
    plasticState(ph)%atol(s:e) = pl%get_asReal('atol_xi',defaultVal=1.0_pREAL)

    s = e + 1; e = e + prm%sum_N_tw
    stt%xi_dtw => plasticState(ph)%state(s:e,:)
    stt%xi_dtw = spread(xi_0_dtw, 2, Nmembers)
    plasticState(ph)%atol(s:e) = pl%get_asReal('atol_xi',defaultVal=1.0_pREAL)

    s = e + 1; e = e + prm%N_fam*prm%sum_N_tw
    stt%xi_sl_tw => plasticState(ph)%state(s:e,:)
    stt%xi_sl_tw = spread([(prm%xi_0_sl_tw,k=1,prm%sum_N_tw)], 2, Nmembers)
    plasticState(ph)%atol(s:e) = pl%get_asReal('atol_xi',defaultVal=1.0_pREAL)

    s = e + 1; e = e + prm%sum_N_tw
    stt%gamma_sl_tw => plasticState(ph)%state(s:e,:)
    plasticState(ph)%atol(s:e) = pl%get_asReal('atol_gamma',defaultVal=1.0e-6_pREAL)

    s = e + 1; e = e + prm%sum_N_tw
    stt%f_tw_0 => plasticState(ph)%state(s:e,:)
    dlt%f_tw_0 => plasticState(ph)%deltaState(1:prm%sum_N_tw,:)
    plasticState(ph)%atol(s:e) = pl%get_asReal('atol_f',defaultVal=1.0e-6_pREAL)

    if (any(plasticState(ph)%atol < 0.0_pREAL)) extmsg = trim(extmsg)//' atol'

    end associate

    if (extmsg /= '') call IO_error(211,ext_msg=trim(extmsg))

  end do

end function plastic_tdt_init


!--------------------------------------------------------------------------------------------------
!> @brief Plastic velocity gradient and its tangent.
!--------------------------------------------------------------------------------------------------
pure module subroutine tdt_LpAndItsTangent(Lp,dLp_dMp,Mp,Delta_t,ph,en)

  real(pREAL), dimension(3,3),     intent(out) :: Lp
  real(pREAL), dimension(3,3,3,3), intent(out) :: dLp_dMp
  real(pREAL), dimension(3,3),     intent(in)  :: Mp
  real(pREAL),                     intent(in)  :: Delta_t
  integer,                         intent(in)  :: ph, en

  real(pREAL), dimension(param(ph)%sum_N_sl) :: gd
  real(pREAL), dimension(param(ph)%sum_N_tw) :: gr, fd


  call kinetics(Mp,Delta_t,ph,en,gd,gr,fd,Lp,dLp_dMp)

end subroutine tdt_LpAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief Rate of change of the state.
!--------------------------------------------------------------------------------------------------
module function tdt_dotState(Mp,Delta_t,ph,en) result(dotState)

  real(pREAL), dimension(3,3), intent(in) :: Mp
  real(pREAL),                 intent(in) :: Delta_t
  integer,                     intent(in) :: ph, en
  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: dotState

  real(pREAL), dimension(param(ph)%sum_N_sl) :: gd, x
  real(pREAL), dimension(param(ph)%sum_N_tw) :: gr, fd
  real(pREAL) :: y
  integer :: ns, nt, nf, k, f, o


  associate(prm => param(ph), stt => state(ph))

    ns = prm%sum_N_sl; nt = prm%sum_N_tw; nf = prm%N_fam
    call kinetics(Mp,Delta_t,ph,en,gd,gr,fd)

    x = 1.0_pREAL - stt%xi_sl(:,en)/prm%xi_inf_sl
    dotState(1:ns)                 = prm%h_0_sl_sl * sign(abs(x)**prm%a_sl,x) * matmul(prm%h_sl_sl,abs(gd)) &
                                   + prm%h_0_sl_tw * matmul(prm%h_sl_tw,prm%gamma_char*abs(fd))
    dotState(ns+1:2*ns)            = abs(gd)
    o = 2*ns
    dotState(o+1:o+nt)             = fd
    dotState(o+nt+1:o+2*nt)        = prm%h_0_tw_tw * matmul(prm%h_tw_tw,prm%gamma_char*max(fd,0.0_pREAL)) &
                                   + prm%h_0_tw_sl * matmul(prm%h_tw_sl,abs(gd))
    dotState(o+2*nt+1:o+3*nt)      = prm%h_0_dtw * gr
    o = o + 3*nt
    do k = 1, nt
      do f = 1, nf
        y = 1.0_pREAL - stt%xi_sl_tw(nf*(k-1)+f,en)/prm%xi_inf_sl_tw(f)
        dotState(o+nf*(k-1)+f) = prm%h_0_sl_tw_f(f) * sign(abs(y)**prm%a_sl_tw(f),y) * gr(k)
      end do
    end do
    o = o + nf*nt
    dotState(o+1:o+nt)             = gr

  end associate

end function tdt_dotState


!--------------------------------------------------------------------------------------------------
!> @brief Diagonal of d(dotState)/d(state) (<= 0) for the preconditioned state update
!> (Jacobi-Newton, see integrateStateFPI): slip resistances of the parent and inside twins.
!--------------------------------------------------------------------------------------------------
module function tdt_dotStateDiag(Mp,Delta_t,ph,en) result(Jd)

  real(pREAL), dimension(3,3), intent(in) :: Mp
  real(pREAL),                 intent(in) :: Delta_t
  integer,                     intent(in) :: ph, en
  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: Jd

  real(pREAL), dimension(param(ph)%sum_N_sl) :: gd
  real(pREAL), dimension(param(ph)%sum_N_tw) :: gr, fd
  real(pREAL), dimension(param(ph)%N_fam,param(ph)%sum_N_tw) :: grn
  integer :: ns, nt, nf, i, k, f, o


  associate(prm => param(ph), stt => state(ph))
    ns = prm%sum_N_sl; nt = prm%sum_N_tw; nf = prm%N_fam
    call kinetics(Mp,Delta_t,ph,en,gd,gr,fd,grn=grn)
    Jd = 0.0_pREAL
    do i = 1, ns
      Jd(i) = slope(prm%h_0_sl_sl(i),1.0_pREAL-stt%xi_sl(i,en)/prm%xi_inf_sl(i),prm%a_sl(i),prm%xi_inf_sl(i), &
                    dot_product(prm%h_sl_sl(i,:),abs(gd)),stt%xi_sl(i,en),prm%h_sl_sl(i,i)*prm%n_sl(i)*abs(gd(i)))
    end do
    o = 2*ns + 3*nt
    do k = 1, nt
      do f = 1, nf
        Jd(o+nf*(k-1)+f) = slope(prm%h_0_sl_tw_f(f),1.0_pREAL-stt%xi_sl_tw(nf*(k-1)+f,en)/prm%xi_inf_sl_tw(f), &
                                 prm%a_sl_tw(f),prm%xi_inf_sl_tw(f),gr(k),stt%xi_sl_tw(nf*(k-1)+f,en),grn(f,k))
      end do
    end do
  end associate

  contains

  !> d/dxi of h0 sign(|x|^a,x) G(xi), x = 1 - xi/xi_inf, dG/dxi = -nG/xi, clamped <= 0
  pure real(pREAL) function slope(h0,x,a,xi_inf,G,xi,nG)
    real(pREAL), intent(in) :: h0, x, a, xi_inf, G, xi, nG
    real(pREAL) :: l, dl
    l  = merge(abs(x)**a,0.0_pREAL,abs(x) > 0.0_pREAL)
    dl = merge(a*l/abs(x),0.0_pREAL,abs(x) > 0.0_pREAL)
    slope = min(h0*(-(dl/xi_inf)*G - sign(l,x)*nG/xi),0.0_pREAL)
  end function slope

end function tdt_dotStateDiag


!--------------------------------------------------------------------------------------------------
!> @brief Twin volume fractions at the start of the next step.
!--------------------------------------------------------------------------------------------------
module subroutine tdt_deltaState(ph,en)

  integer, intent(in) :: ph, en


  deltaState(ph)%f_tw_0(:,en) = state(ph)%f_tw(:,en) - state(ph)%f_tw_0(:,en)

end subroutine tdt_deltaState


!--------------------------------------------------------------------------------------------------
!> @brief Homogenized stiffness of parent and twins (Reuss or Voigt).
!--------------------------------------------------------------------------------------------------
module function plastic_tdt_homogenizedC(ph,en) result(homogenizedC)

  integer, intent(in) :: ph, en
  real(pREAL), dimension(6,6) :: homogenizedC

  real(pREAL), dimension(param(ph)%sum_N_tw+36) :: key


  key = [state(ph)%f_tw(:,en),reshape(elastic_C66(ph,en),[36])]
  associate(c => cache(ph))
    if (any(key /= c%key(:,en))) then                                                               ! tdt_C66 is a function of key only: same result
      c%key(:,en) = key
      c%C_bar(1:6,1:6,en) = tdt_C66(reshape(key(param(ph)%sum_N_tw+1:),[6,6]),key(1:param(ph)%sum_N_tw),ph)
    end if
    homogenizedC = c%C_bar(1:6,1:6,en)
  end associate

end function plastic_tdt_homogenizedC


!--------------------------------------------------------------------------------------------------
!> @brief Homogenized stiffness for given parent stiffness and twin volume fractions.
!--------------------------------------------------------------------------------------------------
function tdt_C66(C,f_tw,ph) result(Cbar)

  real(pREAL), dimension(6,6), intent(in) :: C
  real(pREAL), dimension(:),   intent(in) :: f_tw
  integer,                     intent(in) :: ph
  real(pREAL), dimension(6,6) :: Cbar

  real(pREAL), dimension(6,6,param(ph)%sum_N_tw) :: C_tw
  integer :: k


  associate(prm => param(ph))
    C_tw = crystal_C66_twin(prm%N_tw_fam,C,phase_lattice(ph),phase_cOverA(ph))
    if (prm%Reuss) then
      Cbar = (1.0_pREAL-sum(f_tw)) * inv66(C)
      do k = 1, prm%sum_N_tw
        Cbar = Cbar + f_tw(k) * inv66(C_tw(1:6,1:6,k))
      end do
      Cbar = inv66(Cbar)
    else
      Cbar = (1.0_pREAL-sum(f_tw)) * C
      do k = 1, prm%sum_N_tw
        Cbar = Cbar + f_tw(k) * C_tw(1:6,1:6,k)
      end do
    end if
  end associate

end function tdt_C66


!--------------------------------------------------------------------------------------------------
!> @brief Write results to HDF5 output file.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_tdt_result(ph,group)

  integer,          intent(in) :: ph
  character(len=*), intent(in) :: group

  integer :: ou


  associate(prm => param(ph), stt => state(ph))
    do ou = 1, size(prm%output)
      select case(trim(prm%output(ou)))
        case('xi_sl')
          call result_writeDataset(stt%xi_sl,group,'xi_sl','resistance against slip (parent)','Pa',prm%systems_sl)
        case('gamma_sl')
          call result_writeDataset(stt%gamma_sl,group,'gamma_sl','plastic shear (parent)','1',prm%systems_sl)
        case('f_tw')
          call result_writeDataset(stt%f_tw,group,'f_tw','twin volume fraction','m³/m³',prm%systems_tw)
        case('xi_tw')
          call result_writeDataset(stt%xi_tw,group,'xi_tw','resistance against twin growth','Pa',prm%systems_tw)
        case('xi_dtw')
          call result_writeDataset(stt%xi_dtw,group,'xi_dtw','resistance against twin shrinkage','Pa',prm%systems_tw)
        case('xi_sl_tw')
          call result_writeDataset(stt%xi_sl_tw,group,'xi_sl_tw', &
                                   'resistance against slip inside twins (slip family fastest)','Pa')
        case('gamma_sl_tw')
          call result_writeDataset(stt%gamma_sl_tw,group,'gamma_sl_tw','plastic shear inside twins','1',prm%systems_tw)
      end select
    end do
  end associate

end subroutine plastic_tdt_result


!--------------------------------------------------------------------------------------------------
!> @brief Shear rates of all regions, twin volume fraction rates and (optionally) Lp, dLp/dMp.
!> @details gd: parent slip rates (signed), gr: sum of |slip rates| inside each twin, fd: df/dt.
!--------------------------------------------------------------------------------------------------
pure subroutine kinetics(Mp,Delta_t,ph,en,gd,gr,fd,Lp,dLp_dMp,grn)

  real(pREAL), dimension(3,3), intent(in) :: Mp
  real(pREAL),                 intent(in) :: Delta_t
  integer,                     intent(in) :: ph, en
  real(pREAL), dimension(param(ph)%sum_N_sl), intent(out) :: gd
  real(pREAL), dimension(param(ph)%sum_N_tw), intent(out) :: gr, fd
  real(pREAL), dimension(3,3),     intent(out), optional :: Lp
  real(pREAL), dimension(3,3,3,3), intent(out), optional :: dLp_dMp
  real(pREAL), dimension(param(ph)%N_fam,param(ph)%sum_N_tw), intent(out), optional :: grn          !< sum n |gd| per slip family and twin

  real(pREAL) :: w0, w00, tau, g, dg, xi, c, f0, q, A, F_e, R, den, sh, lmax
  real(pREAL), dimension(param(ph)%sum_N_tw) :: tau_tw, lr, al, ga
  real(pREAL), dimension(3,3) :: u, v
  integer :: i, k, nf, kk, l, m, n
  logical :: deriv, grow, any_grow


  deriv = present(Lp)
  if (deriv) then
    Lp = 0.0_pREAL
    dLp_dMp = 0.0_pREAL
  end if

  associate(prm => param(ph), stt => state(ph))

    nf = prm%N_fam
    w0 = 1.0_pREAL - sum(stt%f_tw(:,en))
    if (present(grn)) grn = 0.0_pREAL

    ! parent
    do i = 1, prm%sum_N_sl
      tau = math_tensordot(Mp,prm%P_sl(1:3,1:3,i))
      gd(i) = sign(prm%dot_gamma_0_sl(i) * (abs(tau)/stt%xi_sl(i,en))**prm%n_sl(i), tau)
      if (deriv .and. dNeq0(gd(i))) then
        Lp = Lp + w0*gd(i)*prm%P_sl(1:3,1:3,i)
        dg = w0*gd(i)*prm%n_sl(i)/tau
        forall (kk=1:3,l=1:3,m=1:3,n=1:3) &
          dLp_dMp(kk,l,m,n) = dLp_dMp(kk,l,m,n) + dg*prm%P_sl(kk,l,i)*prm%P_sl(m,n,i)
      end if
    end do

    ! slip inside twins
    do k = 1, prm%sum_N_tw
      gr(k) = 0.0_pREAL
      if (stt%f_tw(k,en) <= F_MIN) cycle
      do i = 1, prm%sum_N_sl
        tau = math_tensordot(Mp,prm%P_sl_tw(1:3,1:3,i,k))
        g = sign(prm%dot_gamma_0_sl(i) * (abs(tau)/stt%xi_sl_tw(nf*(k-1)+prm%fam(i),en))**prm%n_sl(i), tau)
        gr(k) = gr(k) + abs(g)
        if (present(grn)) grn(prm%fam(i),k) = grn(prm%fam(i),k) + prm%n_sl(i)*abs(g)
        if (deriv .and. dNeq0(g)) then
          Lp = Lp + stt%f_tw(k,en)*g*prm%P_sl_tw(1:3,1:3,i,k)
          dg = stt%f_tw(k,en)*g*prm%n_sl(i)/tau
          forall (kk=1:3,l=1:3,m=1:3,n=1:3) &
            dLp_dMp(kk,l,m,n) = dLp_dMp(kk,l,m,n) + dg*prm%P_sl_tw(kk,l,i,k)*prm%P_sl_tw(m,n,i,k)
        end if
      end do
    end do

    ! twin volume fractions (closed-form backward Euler with f0 = f at the start of the step),
    ! evaluated with logarithms of the rates (no overflow for stresses far above the thresholds)
    w00 = 1.0_pREAL - sum(stt%f_tw_0(:,en))
    F_e = 0.0_pREAL; lmax = -huge(1.0_pREAL)
    al = 0.0_pREAL; ga = 0.0_pREAL; lr = 0.0_pREAL; fd = 0.0_pREAL
    any_grow = .false.
    do k = 1, prm%sum_N_tw
      f0 = stt%f_tw_0(k,en)
      tau_tw(k) = math_tensordot(Mp,prm%P_tw(1:3,1:3,k)) - prm%H_chi(k)*f0*w00
      c = log(prm%dot_gamma_0_tw(k)/prm%gamma_char(k))
      if (tau_tw(k) > 0.0_pREAL) then                                                               ! growth / nucleation
        xi = stt%xi_tw(k,en) + prm%dxi_nuc(k)*exp(-f0/prm%f_n)
        lr(k) = c + prm%n_tw(k)*log(tau_tw(k)/xi)
        lmax = max(lmax,lr(k))
        any_grow = .true.
        F_e = F_e + f0
      elseif (tau_tw(k) < 0.0_pREAL) then                                                           ! shrinkage / re-nucleation of the parent
        xi = stt%xi_dtw(k,en) + prm%dxi_retw(k)*exp(-w00/prm%f_n)
        lr(k) = c + prm%n_tw(k)*log(-tau_tw(k)/xi)
        q = merge(lr(k) + log(max(Delta_t,tiny(1.0_pREAL))),-huge(1.0_pREAL),Delta_t > 0.0_pREAL)    ! ln(lambda dt)
        if (q < 0.0_pREAL) then
          sh = 1.0_pREAL/(1.0_pREAL + exp(q))
          fd(k) = -max(f0,0.0_pREAL)*exp(lr(k))*sh
        else
          g = exp(-q)
          sh = g/(1.0_pREAL + g)
          fd(k) = -max(f0,0.0_pREAL)/Delta_t/(1.0_pREAL + g)
        end if
        al(k) = prm%n_tw(k)*fd(k)*sh/tau_tw(k)
        ga(k) = -Delta_t*al(k)
        F_e = F_e + f0 + fd(k)*Delta_t
      else
        F_e = F_e + f0
      end if
    end do
    R = prm%f_sat - F_e
    grow = any_grow .and. R > 0.0_pREAL
    u = 0.0_pREAL; v = 0.0_pREAL
    if (grow) then
      A = sum(exp(lr-lmax),mask=tau_tw > 0.0_pREAL)
      if (Delta_t > 0.0_pREAL) then
        q = lmax + log(Delta_t) + log(A)                                                            ! ln(A dt)
      else
        q = -huge(1.0_pREAL)
      end if
      if (.not. Delta_t > 0.0_pREAL) then                                                           ! initial evaluation
        den = R*exp(lmax)*A/prm%f_sat
      elseif (q < 0.0_pREAL) then
        den = R*exp(q)/Delta_t/(prm%f_sat + exp(q))
      else
        den = R/Delta_t/(prm%f_sat*exp(-q) + 1.0_pREAL)
      end if
      do k = 1, prm%sum_N_tw
        if (.not. tau_tw(k) > 0.0_pREAL) cycle
        fd(k) = den*exp(lr(k)-lmax)/A
        al(k) = prm%n_tw(k)*fd(k)/tau_tw(k)
        ga(k) = -Delta_t*al(k)
        u = u + prm%gamma_char(k)*fd(k)/R*prm%P_tw(1:3,1:3,k)
      end do
    else
      where(tau_tw > 0.0_pREAL) ga = 0.0_pREAL
    end if
    if (deriv) then
      do k = 1, prm%sum_N_tw
        if (dNeq0(fd(k))) Lp = Lp + prm%gamma_char(k)*fd(k)*prm%P_tw(1:3,1:3,k)
        if (dNeq0(al(k))) then
          dg = prm%gamma_char(k)*al(k)
          forall (kk=1:3,l=1:3,m=1:3,n=1:3) &
            dLp_dMp(kk,l,m,n) = dLp_dMp(kk,l,m,n) + dg*prm%P_tw(kk,l,k)*prm%P_tw(m,n,k)
        end if
        if (grow) v = v + ga(k)*prm%P_tw(1:3,1:3,k)
      end do
      if (grow) forall (kk=1:3,l=1:3,m=1:3,n=1:3) dLp_dMp(kk,l,m,n) = dLp_dMp(kk,l,m,n) + u(kk,l)*v(m,n)
    end if

  end associate

end subroutine kinetics


!--------------------------------------------------------------------------------------------------
!> @brief Inverse of a 6x6 matrix (Gauss-Jordan, partial pivoting); same algorithm as on the GPU.
!--------------------------------------------------------------------------------------------------
pure function inv66(A) result(B)

  real(pREAL), dimension(6,6), intent(in) :: A
  real(pREAL), dimension(6,6) :: B

  real(pREAL), dimension(6,12) :: W
  real(pREAL), dimension(12) :: r
  real(pREAL) :: q
  integer :: c, j, p


  W = 0.0_pREAL
  W(1:6,1:6) = A
  do c = 1, 6
    W(c,6+c) = 1.0_pREAL
  end do
  do c = 1, 6
    p = c - 1 + maxloc(abs(W(c:6,c)),1)
    if (p /= c) then
      r = W(c,:); W(c,:) = W(p,:); W(p,:) = r
    end if
    q = 1.0_pREAL/W(c,c)
    W(c,:) = W(c,:)*q
    do j = 1, 6
      if (j /= c) W(j,:) = W(j,:) - W(j,c)*W(c,:)
    end do
  end do
  B = W(1:6,7:12)

end function inv66


#if defined(METAL) && defined(GRID)
!--------------------------------------------------------------------------------------------------
!> @brief Upload the model of a phase to the GPU constitutive kernel (cp_model_tdt, cp_core.h).
!> @details blob layout: see C_Metal_cp_add_phase_tdt (DAMASK_Metal.h). Returns the GPU handle or a
!> negative number if the phase stays on the CPU.
!--------------------------------------------------------------------------------------------------
module function plastic_tdt_GPUadd(ph,C66,numerics) result(handle)

  integer, intent(in) :: ph
  real(pREAL), dimension(6,6), intent(in) :: C66
  real(pREAL), dimension(7), intent(in) :: numerics
  integer :: handle

  real(pREAL), dimension(:), allocatable :: blob
  real(pREAL), dimension(6,6,param(ph)%sum_N_tw) :: C_tw
  integer :: k, o
  character(len=:), allocatable :: reason


  handle = -1
  associate(prm => param(ph))
    if (prm%sum_N_sl > 36 .or. prm%sum_N_tw > 12 .or. prm%N_fam > 6) then
      reason = 'more than 36 slip systems, 12 twin systems, or 6 slip families'
    else
      C_tw = crystal_C66_twin(prm%N_tw_fam,C66,phase_lattice(ph),phase_cOverA(ph))
      blob = [reshape(prm%P_sl,[9*prm%sum_N_sl]), reshape(prm%P_sl_tw,[9*prm%sum_N_sl*prm%sum_N_tw]), &
              reshape(prm%P_tw,[9*prm%sum_N_tw]), &
              reshape(merge(inv66(C66),C66,prm%Reuss),[36])]
      do k = 1, prm%sum_N_tw
        blob = [blob, reshape(merge(inv66(C_tw(:,:,k)),C_tw(:,:,k),prm%Reuss),[36])]
      end do
      blob = [blob, prm%dot_gamma_0_sl, prm%n_sl, prm%a_sl, prm%h_0_sl_sl, prm%xi_inf_sl, real(prm%fam,pREAL), &
              prm%xi_inf_sl_tw, prm%h_0_sl_tw_f, prm%a_sl_tw, &
              prm%dot_gamma_0_tw, prm%n_tw, prm%gamma_char, prm%dxi_nuc, prm%dxi_retw, prm%H_chi, prm%h_0_dtw, &
              reshape(prm%h_sl_sl,[prm%sum_N_sl**2]), &
              reshape(spread(prm%h_0_sl_tw,2,prm%sum_N_tw)*prm%h_sl_tw*spread(prm%gamma_char,1,prm%sum_N_sl), &
                      [prm%sum_N_sl*prm%sum_N_tw]), &
              reshape(spread(prm%h_0_tw_sl,2,prm%sum_N_sl)*prm%h_tw_sl,[prm%sum_N_tw*prm%sum_N_sl]), &
              reshape(spread(prm%h_0_tw_tw,2,prm%sum_N_tw)*prm%h_tw_tw*spread(prm%gamma_char,1,prm%sum_N_tw), &
                      [prm%sum_N_tw**2]), &
              prm%f_n, prm%f_sat, F_MIN]
      o = 2*prm%sum_N_sl
      handle = Metal_cpAddPhaseTdt(Metal_ctx, &
                                   int([prm%sum_N_sl,prm%sum_N_tw,prm%N_fam,merge(1,0,prm%Reuss)],C_INT), &
                                   C66,blob, &
                                   [plasticState(ph)%atol(1),plasticState(ph)%atol(prm%sum_N_sl+1), &
                                    plasticState(ph)%atol(o+1)],numerics)
      reason = 'GPU initialization failed'
    end if
  end associate

  if (handle >= 0) then
    print'(/,1x,a,i0,a)', 'phase ',ph,': tdt constitutive update on GPU'
  else
    print'(/,1x,a,i0,a)', 'phase ',ph,': tdt constitutive update on CPU ('//reason//')'
  end if
  flush(IO_STDOUT)

end function plastic_tdt_GPUadd


#endif

end submodule tdt
