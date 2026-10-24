! SPDX-License-Identifier: AGPL-3.0-or-later
!--------------------------------------------------------------------------------------------------
!> @brief  TWIP/TRIP model 'twintrip': dislocation-density slip in fcc austenite, twinning, gamma->epsilon
!> and epsilon->alpha' martensite, with crystal plasticity inside twins and both martensites
!> @details Every material point is a composite (iso-stress, Kalidindi 1998) of the austenite parent
!> (volume fraction w_g = 1 - sum(f_tw) - sum(f_tr)) and, on each of the four {111} planes p, a twin
!> region (w_T,p = sum of f_tw on p, rotated by the twin rotation), an epsilon region (w_E,p = sum of
!> f_tr on p - f_a,p, Shoji-Nishiyama) and an alpha' region (w_A,p = f_a,p, Kurdjumov-Sachs):
!>   Lp = w_g sum_i gd_i P_i + sum_p sum_R w_R,p sum_j gd_j^(R,p) Q P_j^R Q^T
!>      + gamma_tw sum_k fd_tw,k P_tw,k + gamma_eps sum_k fd_tr,k P_tw,k
!> Parent: slip, dipoles and twin/epsilon nucleation of dislotwin (Ma & Roters 2004, Wong et al. 2016)
!> with these corrections: |dot_gamma_sl| in the nucleation rate also in Lp, transformation shear
!> 1/sqrt(8) applied once on the unit {111}<112> Schmid tensor, thermally activated glide without
!> NaN above tau_0 (softplus) and without overflow at low temperature, bounded mean free paths for
!> sum(f) -> 1. Regions: phenomenological power law with Voce hardening per slip family.
!> alpha' forms where epsilon lamellae on plane p are sheared on the other planes (Olson-Cohen type
!> intersection kinetics) with a driving force factor 1/(1+exp(Delta_G_alpha(T)/G_s)); its volume
!> change gives the isotropic eigenstrain Fi = (1 + Delta_V sum(f_a))^(1/3) I.
!> Elasticity: Reuss (default, consistent with iso-stress) or Voigt average of the region stiffnesses.
!--------------------------------------------------------------------------------------------------
submodule(phase:plastic) twintrip

  real(pREAL), parameter :: &
    GAMMA_EPS = sqrt(0.125_pREAL), &                                                                !< shear of epsilon formation (a/sqrt(6) partial on every 2nd {111})
    F_MIN     = 1.0e-12_pREAL, &                                                                    !< regions with w <= F_MIN carry no slip
    W_FLOOR   = 1.0e-3_pREAL                                                                        !< lower bound of 1 - sum(f) in mean free paths

  type :: tRegions                                                                                  !< slip in one kind of product region (one region per {111} plane)
    logical :: plastic = .false.
    integer :: N_sys = 0, N_fam = 0
    integer,     allocatable, dimension(:) :: fam                                                   !< family (1..N_fam) of each system
    real(pREAL), allocatable, dimension(:) :: dot_gamma_0, n, xi_0, xi_inf, h_0, a                  !< per family
    real(pREAL), allocatable, dimension(:,:,:,:) :: P                                               !< (3,3,N,4) Schmid tensors in the parent frame
    real(pREAL), dimension(3,3,4) :: Q                                                              !< lattice rotation per plane
    integer :: o_xi = 0, o_gamma = 0                                                                !< state offsets
  end type tRegions

  type :: tParameters
    real(pREAL) :: &
      Q_cl, omega, D, i_tw, i_tr, L_tw, L_tr, x_c, V_cs, h, V_mol, rho, delta_sl, &
      k_alpha, G_s, Delta_V
    type(tPolynomial) :: Gamma_sf, Delta_G, Delta_G_alpha
    real(pREAL), allocatable, dimension(:) :: &
      b_sl, Q_sl, v_0, i_sl, p, q, tau_0, B, d_caron, &                                            !< per slip system
      b_tw, t_tw, r, gamma_char_tw, &                                                               !< per twin system
      b_tr, t_tr, s                                                                                 !< per transformation system
    real(pREAL), allocatable, dimension(:,:) :: &
      h_sl_sl, h_sl_tw, h_sl_tr, h_tw_tw, h_tr_tr, n0_sl, forestProjection
    real(pREAL), allocatable, dimension(:,:,:) :: &
      P_sl, P_tw                                                                                    !< P_tw: also the (unit) transformation shear
    real(pREAL), dimension(6,6,3,4) :: C_T                                                          !< twin stiffness per plane: rotated cubic basis (C_11, C_12, C_44)
    real(pREAL), dimension(6,6,5,4) :: C_E                                                          !< epsilon stiffness per plane: rotated hP basis (C_11, C_12, C_44, C_13, C_33)
    real(pREAL), dimension(6,6,4)   :: C_A                                                          !< alpha' stiffness per plane
    real(pREAL), dimension(5)       :: C_hP = 0.0_pREAL                                             !< user epsilon constants (else from the austenite, Martin 1972)
    logical :: user_C_hP = .false., Reuss = .true., alpha = .false., &
               extendedDislocations = .false., omitDipoles = .false.
    type(tRegions) :: Tw, Ep, Al
    integer :: sum_N_sl, sum_N_tw, sum_N_tr, o_fa = 0
    integer, allocatable, dimension(:,:) :: pair                                                    !< slip systems of the nucleation pair of each twin/trans system
    character(len=:), allocatable :: isotropic_bound
    character(len=pSTRLEN), allocatable, dimension(:) :: output
    character(len=:), allocatable, dimension(:) :: systems_sl, systems_tw
  end type tParameters

  type :: tTwintripState
    real(pREAL), pointer, dimension(:,:) :: rho_mob, rho_dip, gamma_sl, f_tw, f_tr, f_a
  end type tTwintripState

  type :: tTwintripDependentState
    real(pREAL), allocatable, dimension(:,:) :: Lambda_sl, Lambda_tw, Lambda_tr, tau_pass
    real(pREAL), allocatable, dimension(:,:)   :: C_key                                             !< volume fractions and constants of C_bar
    real(pREAL), allocatable, dimension(:,:,:) :: C_bar                                             !< last homogenized stiffness
  end type tTwintripDependentState

  type(tParameters),             allocatable, dimension(:) :: param
  type(tTwintripState),          allocatable, dimension(:) :: state
  type(tTwintripDependentState), allocatable, dimension(:) :: dependentState

contains


!--------------------------------------------------------------------------------------------------
!> @brief Perform module initialization.
!--------------------------------------------------------------------------------------------------
module function plastic_twintrip_init() result(myPlasticity)

  logical, dimension(:), allocatable :: myPlasticity
  integer :: ph, i, p, Nmembers, sizeState, o
  integer, dimension(:), allocatable :: N_sl
  real(pREAL) :: a_cF
  real(pREAL), dimension(:), allocatable :: f_edge, rho_mob_0, rho_dip_0
  real(pREAL), dimension(6,6) :: C
  real(pREAL), dimension(3,3,12) :: Q_tw
  real(pREAL), dimension(3,3,4) :: Q_SN, Q_KS
  character(len=:), allocatable :: refs, extmsg, mixing
  type(tDict), pointer :: phases, phase, mech, pl


  myPlasticity = plastic_active('twintrip')
  if (count(myPlasticity) == 0) return

  print'(/,1x,a)', '<<<+-  phase:mechanical:plastic:twintrip init  -+>>>'
  print'(/,1x,a)', 'A. Ma and F. Roters, Acta Materialia 52(12):3603–3612, 2004'
  print'(  1x,a)', 'https://doi.org/10.1016/j.actamat.2004.04.012'
  print'(/,1x,a)', 'S.L. Wong et al., Acta Materialia 118:140–151, 2016'
  print'(  1x,a)', 'https://doi.org/10.1016/j.actamat.2016.07.032'
  print'(/,1x,a)', 'S.R. Kalidindi, Journal of the Mechanics and Physics of Solids 46:267-290, 1998'
  print'(  1x,a)', 'https://doi.org/10.1016/S0022-5096(97)00051-3'
  print'(/,1x,a)', 'G.B. Olson and M. Cohen, Metallurgical Transactions A 6:791-795, 1975'
  print'(  1x,a)', 'https://doi.org/10.1007/BF02672301'
  print'(/,1x,a,1x,i0)', '# phases:',count(myPlasticity); flush(IO_STDOUT)

  phases => config_material%get_dict('phase')
  allocate(param(size(phases)))
  allocate(state(size(phases)))
  allocate(dependentState(size(phases)))
  extmsg = ''

  Q_tw = crystal_rotation_twin([12],'cF',0.0_pREAL)
  Q_SN = crystal_rotation_SN()
  Q_KS = crystal_rotation_KS()

  do ph = 1, size(phases)
    if (.not. myPlasticity(ph)) cycle

    associate(prm => param(ph), stt => state(ph), dst => dependentState(ph))

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
    if (phase_lattice(ph) /= 'cF') extmsg = trim(extmsg)//' lattice (cF only)'
    prm%isotropic_bound = pl%get_asStr('isotropic_bound',defaultVal='isostrain')
    mixing    = pl%get_asStr('elastic_mixing',defaultVal='Reuss')
    prm%Reuss = mixing == 'Reuss'
    if (all(mixing /= ['Reuss','Voigt'])) extmsg = trim(extmsg)//' elastic_mixing'

!--------------------------------------------------------------------------------------------------
! austenite slip (dislotwin)
    N_sl = pl%get_as1dInt('N_sl')
    prm%sum_N_sl = sum(abs(N_sl))
    if (any(N_sl /= [12])) extmsg = trim(extmsg)//' N_sl ([12] only)'
    prm%P_sl = crystal_SchmidMatrix_slip(N_sl,phase_lattice(ph),phase_cOverA(ph))
    prm%systems_sl = crystal_labels_slip(N_sl,phase_lattice(ph))
    prm%n0_sl = crystal_slip_normal(N_sl,phase_lattice(ph),phase_cOverA(ph))
    prm%h_sl_sl = crystal_interaction_SlipBySlip(N_sl,pl%get_as1dReal('h_sl-sl'),phase_lattice(ph))
    prm%extendedDislocations = pl%get_asBool('extend_dislocations',defaultVal=.false.)
    prm%omitDipoles          = pl%get_asBool('omit_dipoles',       defaultVal=.false.)
    prm%Q_cl = pl%get_asReal('Q_cl')
    f_edge     = math_expand(pl%get_as1dReal('f_edge',requiredSize=size(N_sl), &
                                             defaultVal=[(0.5_pREAL,i=1,size(N_sl))]),N_sl)
    rho_mob_0  = math_expand(pl%get_as1dReal('rho_mob_0',requiredSize=size(N_sl)),N_sl)
    rho_dip_0  = math_expand(pl%get_as1dReal('rho_dip_0',requiredSize=size(N_sl)),N_sl)
    prm%v_0    = math_expand(pl%get_as1dReal('v_0',      requiredSize=size(N_sl)),N_sl)
    prm%b_sl   = math_expand(pl%get_as1dReal('b_sl',     requiredSize=size(N_sl)),N_sl)
    prm%Q_sl   = math_expand(pl%get_as1dReal('Q_sl',     requiredSize=size(N_sl)),N_sl)
    prm%i_sl   = math_expand(pl%get_as1dReal('i_sl',     requiredSize=size(N_sl)),N_sl)
    prm%p      = math_expand(pl%get_as1dReal('p_sl',     requiredSize=size(N_sl)),N_sl)
    prm%q      = math_expand(pl%get_as1dReal('q_sl',     requiredSize=size(N_sl)),N_sl)
    prm%tau_0  = math_expand(pl%get_as1dReal('tau_0',    requiredSize=size(N_sl)),N_sl)
    prm%B      = math_expand(pl%get_as1dReal('B',        requiredSize=size(N_sl), &
                                             defaultVal=[(0.0_pREAL,i=1,size(N_sl))]),N_sl)
    prm%d_caron = prm%b_sl * pl%get_asReal('D_a')
    prm%delta_sl = pl%get_asReal('delta_sl',defaultVal=1.0e-3_pREAL)
    prm%forestProjection = spread(          f_edge,1,prm%sum_N_sl) &
                         * crystal_forestProjection_edge (N_sl,phase_lattice(ph),phase_cOverA(ph)) &
                         + spread(1.0_pREAL-f_edge,1,prm%sum_N_sl) &
                         * crystal_forestProjection_screw(N_sl,phase_lattice(ph),phase_cOverA(ph))
    prm%omega = pl%get_asReal('omega', defaultVal=1000.0_pREAL) * 12.0_pREAL
    prm%D = pl%get_asReal('D')
    prm%pair = crystal_CF_TWINNUCLEATIONSLIPPAIR

    if (    prm%Q_cl     <= 0.0_pREAL)          extmsg = trim(extmsg)//' Q_cl'
    if (any(rho_mob_0    <  0.0_pREAL))         extmsg = trim(extmsg)//' rho_mob_0'
    if (any(rho_dip_0    <  0.0_pREAL))         extmsg = trim(extmsg)//' rho_dip_0'
    if (any(prm%v_0      <= 0.0_pREAL))         extmsg = trim(extmsg)//' v_0'
    if (any(prm%b_sl     <= 0.0_pREAL))         extmsg = trim(extmsg)//' b_sl'
    if (any(prm%Q_sl     <= 0.0_pREAL))         extmsg = trim(extmsg)//' Q_sl'
    if (any(prm%i_sl     <= 0.0_pREAL))         extmsg = trim(extmsg)//' i_sl'
    if (any(prm%tau_0    <= 0.0_pREAL))         extmsg = trim(extmsg)//' tau_0'
    if (any(prm%B        <  0.0_pREAL))         extmsg = trim(extmsg)//' B'
    if (any(prm%d_caron  <  0.0_pREAL))         extmsg = trim(extmsg)//' D_a'
    if (any(prm%p<=0.0_pREAL .or. prm%p>1.0_pREAL)) extmsg = trim(extmsg)//' p_sl'
    if (any(prm%q< 1.0_pREAL .or. prm%q>2.0_pREAL)) extmsg = trim(extmsg)//' q_sl'
    if (prm%delta_sl <= 0.0_pREAL)              extmsg = trim(extmsg)//' delta_sl'
    if (prm%D        <= 0.0_pREAL)              extmsg = trim(extmsg)//' D'

!--------------------------------------------------------------------------------------------------
! twinning and gamma->epsilon (both nucleate on the 12 {111}<112> systems, Wong et al. 2016)
    prm%sum_N_tw = sum(abs(pl%get_as1dInt('N_tw',defaultVal=emptyIntArray)))
    prm%sum_N_tr = sum(abs(pl%get_as1dInt('N_tr',defaultVal=emptyIntArray)))
    if (all(prm%sum_N_tw /= [0,12])) extmsg = trim(extmsg)//' N_tw ([12] or none)'
    if (all(prm%sum_N_tr /= [0,12])) extmsg = trim(extmsg)//' N_tr ([12] or none)'
    prm%P_tw = crystal_SchmidMatrix_twin([12],'cF',0.0_pREAL)
    prm%systems_tw = crystal_labels_twin([12],'cF')

    if (prm%sum_N_tw + prm%sum_N_tr > 0) then
      prm%Gamma_sf = polynomial(pl,'Gamma_sf','T')
      prm%x_c  = pl%get_asReal('x_c')
      prm%V_cs = pl%get_asReal('V_cs')
      if (prm%x_c  < 0.0_pREAL) extmsg = trim(extmsg)//' x_c'
      if (prm%V_cs < 0.0_pREAL) extmsg = trim(extmsg)//' V_cs'
    elseif (prm%extendedDislocations) then
      prm%Gamma_sf = polynomial(pl,'Gamma_sf','T')
    end if

    if (prm%sum_N_tw > 0) then
      prm%gamma_char_tw = abs(crystal_characteristicShear_twin([12],'cF',0.0_pREAL))
      prm%h_tw_tw = crystal_interaction_TwinByTwin([12],pl%get_as1dReal('h_tw-tw'),'cF')
      prm%h_sl_tw = crystal_interaction_SlipByTwin(N_sl,[12],pl%get_as1dReal('h_sl-tw'),'cF')
      prm%L_tw = pl%get_asReal('L_tw')
      prm%i_tw = pl%get_asReal('i_tw')
      prm%b_tw = math_expand(pl%get_as1dReal('b_tw',requiredSize=1),[12])
      prm%t_tw = math_expand(pl%get_as1dReal('t_tw',requiredSize=1),[12])
      prm%r    = math_expand(pl%get_as1dReal('p_tw',requiredSize=1),[12])
      if (prm%L_tw <= 0.0_pREAL)       extmsg = trim(extmsg)//' L_tw'
      if (prm%i_tw <= 0.0_pREAL)       extmsg = trim(extmsg)//' i_tw'
      if (any(prm%b_tw <= 0.0_pREAL))  extmsg = trim(extmsg)//' b_tw'
      if (any(prm%t_tw <= 0.0_pREAL))  extmsg = trim(extmsg)//' t_tw'
      if (any(prm%r    <= 0.0_pREAL))  extmsg = trim(extmsg)//' p_tw'
    else
      allocate(prm%gamma_char_tw,prm%b_tw,prm%t_tw,prm%r,source=emptyRealArray)
      allocate(prm%h_tw_tw(0,0),prm%h_sl_tw(prm%sum_N_sl,0))
    end if

    if (prm%sum_N_tr > 0) then
      prm%h_tr_tr = crystal_interaction_TransByTrans([12],pl%get_as1dReal('h_tr-tr'),'cF')
      prm%h_sl_tr = crystal_interaction_SlipByTrans(N_sl,[12],pl%get_as1dReal('h_sl-tr'),'cF')
      prm%Delta_G = polynomial(pl,'Delta_G','T')
      prm%i_tr  = pl%get_asReal('i_tr')
      prm%L_tr  = pl%get_asReal('L_tr')
      prm%V_mol = pl%get_asReal('V_mol')
      prm%b_tr = math_expand(pl%get_as1dReal('b_tr',requiredSize=1),[12])
      prm%t_tr = math_expand(pl%get_as1dReal('t_tr',requiredSize=1),[12])
      prm%s    = math_expand(pl%get_as1dReal('p_tr',requiredSize=1),[12])
      a_cF     = prm%b_tr(1)*sqrt(6.0_pREAL)                                                        ! b_tr is a Shockley partial
      prm%h    = 5.0_pREAL*a_cF/sqrt(3.0_pREAL)
      prm%rho  = 4.0_pREAL/(sqrt(3.0_pREAL)*a_cF**2)/N_A
      if (prm%L_tr  <= 0.0_pREAL)      extmsg = trim(extmsg)//' L_tr'
      if (prm%i_tr  <= 0.0_pREAL)      extmsg = trim(extmsg)//' i_tr'
      if (prm%V_mol <= 0.0_pREAL)      extmsg = trim(extmsg)//' V_mol'
      if (any(prm%b_tr <= 0.0_pREAL))  extmsg = trim(extmsg)//' b_tr'
      if (any(prm%t_tr <= 0.0_pREAL))  extmsg = trim(extmsg)//' t_tr'
      if (any(prm%s    <= 0.0_pREAL))  extmsg = trim(extmsg)//' p_tr'
      if (prm%sum_N_tw > 0) then
        if (dNeq(prm%b_tw(1),prm%b_tr(1))) extmsg = trim(extmsg)//' b_tw != b_tr'
      end if
    else
      allocate(prm%b_tr,prm%t_tr,prm%s,source=emptyRealArray)
      allocate(prm%h_tr_tr(0,0),prm%h_sl_tr(prm%sum_N_sl,0))
    end if

!--------------------------------------------------------------------------------------------------
! product regions: twins (austenite slip systems), epsilon (hP), alpha' (cI)
    do p = 1, 4
      prm%Tw%Q(1:3,1:3,p) = Q_tw(1:3,1:3,3*p-2)
    end do
    prm%Ep%Q = Q_SN
    prm%Al%Q = Q_KS
    if (prm%sum_N_tw > 0) call regions_init(prm%Tw,'tw',N_sl,'cF',0.0_pREAL)
    if (prm%sum_N_tr > 0) then
      call regions_init(prm%Ep,'eps',pl%get_as1dInt('N_sl_eps',defaultVal=emptyIntArray),'hP', &
                        pl%get_asReal('c/a_eps',defaultVal=sqrt(8.0_pREAL/3.0_pREAL)))
      prm%k_alpha = pl%get_asReal('k_alpha',defaultVal=0.0_pREAL)
      prm%alpha = prm%k_alpha > 0.0_pREAL
      if (prm%k_alpha < 0.0_pREAL) extmsg = trim(extmsg)//' k_alpha'
      if (prm%alpha) then
        prm%Delta_G_alpha = polynomial(pl,'Delta_G_alpha','T')
        prm%G_s     = pl%get_asReal('G_s')
        prm%Delta_V = pl%get_asReal('Delta_V_alpha',defaultVal=0.03_pREAL)
        if (prm%G_s <= 0.0_pREAL)     extmsg = trim(extmsg)//' G_s'
        if (prm%Delta_V < 0.0_pREAL)  extmsg = trim(extmsg)//' Delta_V_alpha'
        call regions_init(prm%Al,'alpha',pl%get_as1dInt('N_sl_alpha',defaultVal=emptyIntArray),'cI',0.0_pREAL)
        C = crystal_assembleStiffness(pl%get_asReal('C_11_alpha'),pl%get_asReal('C_12_alpha'), &
                                      pl%get_asReal('C_44_alpha'),lattice='cI')
        do p = 1, 4
          prm%C_A(1:6,1:6,p) = crystal_C66_rotate(C,Q_KS(1:3,1:3,p))
        end do
      end if
      prm%user_C_hP = pl%contains('C_11_eps')
      if (prm%user_C_hP) &
        prm%C_hP = [pl%get_asReal('C_11_eps'),pl%get_asReal('C_12_eps'),pl%get_asReal('C_44_eps'), &
                    pl%get_asReal('C_13_eps'),pl%get_asReal('C_33_eps')]
    end if

    ! rotated basis stiffnesses: region stiffness is linear in the (possibly temperature dependent) constants
    do p = 1, 4
      do i = 1, 3
        C = crystal_assembleStiffness(merge(1.0_pREAL,0.0_pREAL,i==1),merge(1.0_pREAL,0.0_pREAL,i==2), &
                                      merge(1.0_pREAL,0.0_pREAL,i==3),lattice='cF')
        prm%C_T(1:6,1:6,i,p) = crystal_C66_rotate(C,prm%Tw%Q(1:3,1:3,p))
      end do
      do i = 1, 5
        C = crystal_assembleStiffness(merge(1.0_pREAL,0.0_pREAL,i==1),merge(1.0_pREAL,0.0_pREAL,i==2), &
                                      merge(1.0_pREAL,0.0_pREAL,i==3),merge(1.0_pREAL,0.0_pREAL,i==4), &
                                      merge(1.0_pREAL,0.0_pREAL,i==5),lattice='hP')
        prm%C_E(1:6,1:6,i,p) = crystal_C66_rotate(C,Q_SN(1:3,1:3,p))
      end do
    end do

!--------------------------------------------------------------------------------------------------
! state: rho_mob, rho_dip, gamma_sl, f_tw, f_tr, f_a, [xi, gamma] per plastic region kind
    Nmembers = count(material_ID_phase == ph)
    o = 3*prm%sum_N_sl + prm%sum_N_tw + prm%sum_N_tr
    if (prm%alpha) then
      prm%o_fa = o; o = o + 4
    end if
    call regions_offsets(prm%Tw,o)
    call regions_offsets(prm%Ep,o)
    call regions_offsets(prm%Al,o)
    sizeState = o
    call phase_allocateState(plasticState(ph),Nmembers,sizeState,sizeState,0)
    deallocate(plasticState(ph)%dotState)

    o = 0
    stt%rho_mob  => plasticState(ph)%state(o+1:o+prm%sum_N_sl,:); o = o + prm%sum_N_sl
    stt%rho_mob  = spread(rho_mob_0,2,Nmembers)
    stt%rho_dip  => plasticState(ph)%state(o+1:o+prm%sum_N_sl,:); o = o + prm%sum_N_sl
    stt%rho_dip  = spread(rho_dip_0,2,Nmembers)
    plasticState(ph)%atol(1:o) = pl%get_asReal('atol_rho',defaultVal=1.0_pREAL)
    stt%gamma_sl => plasticState(ph)%state(o+1:o+prm%sum_N_sl,:)
    plasticState(ph)%atol(o+1:o+prm%sum_N_sl) = pl%get_asReal('atol_gamma',defaultVal=1.0e-6_pREAL)
    o = o + prm%sum_N_sl
    stt%f_tw     => plasticState(ph)%state(o+1:o+prm%sum_N_tw,:)
    plasticState(ph)%atol(o+1:o+prm%sum_N_tw) = pl%get_asReal('atol_f_tw',defaultVal=1.0e-9_pREAL)
    o = o + prm%sum_N_tw
    stt%f_tr     => plasticState(ph)%state(o+1:o+prm%sum_N_tr,:)
    plasticState(ph)%atol(o+1:o+prm%sum_N_tr) = pl%get_asReal('atol_f_tr',defaultVal=1.0e-9_pREAL)
    o = o + prm%sum_N_tr
    if (prm%alpha) then
      stt%f_a    => plasticState(ph)%state(o+1:o+4,:)
      plasticState(ph)%atol(o+1:o+4) = pl%get_asReal('atol_f_tr',defaultVal=1.0e-9_pREAL)
    else
      stt%f_a    => plasticState(ph)%state(o+1:o,:)
    end if
    call regions_state(prm%Tw)
    call regions_state(prm%Ep)
    call regions_state(prm%Al)
    if (any(plasticState(ph)%atol < 0.0_pREAL)) extmsg = trim(extmsg)//' atol'

    allocate(dst%tau_pass (prm%sum_N_sl,Nmembers),source=0.0_pREAL)
    allocate(dst%Lambda_sl(prm%sum_N_sl,Nmembers),source=0.0_pREAL)
    allocate(dst%Lambda_tw(prm%sum_N_tw,Nmembers),source=0.0_pREAL)
    allocate(dst%Lambda_tr(prm%sum_N_tr,Nmembers),source=0.0_pREAL)
    allocate(dst%C_key(15,Nmembers),source=-1.0_pREAL)
    allocate(dst%C_bar(6,6,Nmembers),source=0.0_pREAL)

    end associate

    if (extmsg /= '') call IO_error(211,ext_msg=trim(extmsg))

  end do

  contains

  !> slip systems and parameters of one kind of product region (YAML keys with suffix _sl_<sfx>)
  subroutine regions_init(R,sfx,N,lattice,cOverA)
    type(tRegions),        intent(inout) :: R
    character(len=*),      intent(in)    :: sfx, lattice
    integer, dimension(:), intent(in)    :: N
    real(pREAL),           intent(in)    :: cOverA
    real(pREAL), dimension(:,:,:), allocatable :: P0
    integer :: f, k, j, pp
    R%plastic = sum(abs(N)) > 0 .and. pl%contains('xi_0_sl_'//sfx)
    if (.not. R%plastic) return
    R%N_sys = sum(abs(N))
    R%N_fam = count(N > 0)
    allocate(R%fam(R%N_sys))
    k = 0
    do f = 1, size(N)
      if (N(f) <= 0) cycle
      k = k + 1
      R%fam(sum(N(1:f-1))+1:sum(N(1:f))) = k
    end do
    R%dot_gamma_0 = pack(pl%get_as1dReal('dot_gamma_0_sl_'//sfx,requiredSize=size(N)),N > 0)
    R%n           = pack(pl%get_as1dReal('n_sl_'//sfx,          requiredSize=size(N)),N > 0)
    R%xi_0        = pack(pl%get_as1dReal('xi_0_sl_'//sfx,       requiredSize=size(N)),N > 0)
    R%xi_inf      = pack(pl%get_as1dReal('xi_inf_sl_'//sfx,     requiredSize=size(N)),N > 0)
    R%h_0         = pack(pl%get_as1dReal('h_0_sl_'//sfx,        requiredSize=size(N)),N > 0)
    R%a           = pack(pl%get_as1dReal('a_sl_'//sfx,          requiredSize=size(N)),N > 0)
    if (any(R%dot_gamma_0 <= 0.0_pREAL)) extmsg = trim(extmsg)//' dot_gamma_0_sl_'//sfx
    if (any(R%n           <= 0.0_pREAL)) extmsg = trim(extmsg)//' n_sl_'//sfx
    if (any(R%xi_0        <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_0_sl_'//sfx
    if (any(R%xi_inf      <= 0.0_pREAL)) extmsg = trim(extmsg)//' xi_inf_sl_'//sfx
    if (any(R%a           <= 0.0_pREAL)) extmsg = trim(extmsg)//' a_sl_'//sfx
    P0 = crystal_SchmidMatrix_slip(N,lattice,cOverA)
    allocate(R%P(3,3,R%N_sys,4))
    do pp = 1, 4; do j = 1, R%N_sys
      R%P(1:3,1:3,j,pp) = matmul(matmul(R%Q(1:3,1:3,pp),P0(1:3,1:3,j)),transpose(R%Q(1:3,1:3,pp)))
    end do; end do
  end subroutine regions_init

  subroutine regions_offsets(R,o)
    type(tRegions), intent(inout) :: R
    integer,        intent(inout) :: o
    if (.not. R%plastic) return
    R%o_xi = o;    o = o + 4*R%N_fam
    R%o_gamma = o; o = o + 4
  end subroutine regions_offsets

  subroutine regions_state(R)
    type(tRegions), intent(in) :: R
    integer :: pp
    if (.not. R%plastic) return
    do pp = 1, 4
      plasticState(ph)%state(R%o_xi+R%N_fam*(pp-1)+1:R%o_xi+R%N_fam*pp,:) = spread(R%xi_0,2,Nmembers)
    end do
    plasticState(ph)%atol(R%o_xi+1:R%o_xi+4*R%N_fam)  = pl%get_asReal('atol_xi',   defaultVal=1.0_pREAL)
    plasticState(ph)%atol(R%o_gamma+1:R%o_gamma+4)    = pl%get_asReal('atol_gamma',defaultVal=1.0e-6_pREAL)
  end subroutine regions_state

end function plastic_twintrip_init


!--------------------------------------------------------------------------------------------------
!> @brief Plastic velocity gradient and its tangent.
!--------------------------------------------------------------------------------------------------
module subroutine twintrip_LpAndItsTangent(Lp,dLp_dMp,Mp,ph,en)

  real(pREAL), dimension(3,3),     intent(out) :: Lp
  real(pREAL), dimension(3,3,3,3), intent(out) :: dLp_dMp
  real(pREAL), dimension(3,3),     intent(in)  :: Mp
  integer,                         intent(in)  :: ph, en

  real(pREAL), dimension(param(ph)%sum_N_sl) :: gd
  real(pREAL), dimension(param(ph)%sum_N_tw) :: fd_tw
  real(pREAL), dimension(param(ph)%sum_N_tr) :: fd_tr
  real(pREAL), dimension(4) :: gT, gE, gA


  call kinetics(Mp,thermal_T(ph,en),ph,en,gd,fd_tw,fd_tr,gT,gE,gA,Lp,dLp_dMp)

end subroutine twintrip_LpAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief Isotropic eigenstrain of the alpha' volume change.
!> @details Li such that Fi = (I - dt Li)^-1 Fi0 (after the push forward of phase_LiAndItsTangents)
!> is exactly (1 + Delta_V sum(f_a))^(1/3) I for the current state, given Fi0 = j0 I.
!--------------------------------------------------------------------------------------------------
module subroutine twintrip_LiAndItsTangent(Li,dLi_dMi,detFi,detFi0,Delta_t,ph,en)

  real(pREAL), dimension(3,3),     intent(out) :: Li
  real(pREAL), dimension(3,3,3,3), intent(out) :: dLi_dMi
  real(pREAL),                     intent(in)  :: detFi, detFi0, Delta_t
  integer,                         intent(in)  :: ph, en

  real(pREAL) :: j, j0


  Li = 0.0_pREAL
  dLi_dMi = 0.0_pREAL
  associate(prm => param(ph), stt => state(ph))
    if (.not. prm%alpha .or. .not. Delta_t > 0.0_pREAL) return
    j  = (1.0_pREAL + prm%Delta_V*sum(stt%f_a(:,en)))**(1.0_pREAL/3.0_pREAL)
    j0 = detFi0**(1.0_pREAL/3.0_pREAL)
    Li = math_I3 * (1.0_pREAL - j0/j)/(Delta_t*detFi)
  end associate

end subroutine twintrip_LiAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief Rate of change of the state.
!--------------------------------------------------------------------------------------------------
module function twintrip_dotState(Mp,ph,en) result(dotState)

  real(pREAL), dimension(3,3), intent(in) :: Mp
  integer,                     intent(in) :: ph, en
  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: dotState

  integer :: i, ns, nt, nr
  real(pREAL) :: d_hat, v_cl, tau, sigma_cl, b_d, mu, nu, T
  real(pREAL), dimension(param(ph)%sum_N_sl) :: dot_rho_dip_formation, dot_rho_dip_climb, gd, agd
  real(pREAL), dimension(param(ph)%sum_N_tw) :: fd_tw
  real(pREAL), dimension(param(ph)%sum_N_tr) :: fd_tr
  real(pREAL), dimension(4) :: gT, gE, gA, fd_a


  associate(prm => param(ph), stt => state(ph), dst => dependentState(ph))

    ns = prm%sum_N_sl; nt = prm%sum_N_tw; nr = prm%sum_N_tr
    mu = elastic_mu(ph,en,prm%isotropic_bound)
    nu = elastic_nu(ph,en,prm%isotropic_bound)
    T  = thermal_T(ph,en)

    call kinetics(Mp,T,ph,en,gd,fd_tw,fd_tr,gT,gE,gA,fd_a=fd_a)
    agd = abs(gd)

    do i = 1, ns
      tau = math_tensordot(Mp,prm%P_sl(1:3,1:3,i))
      if (dEq0(tau) .or. prm%omitDipoles) then
        d_hat = dst%Lambda_sl(i,en)
        dot_rho_dip_formation(i) = 0.0_pREAL
      else
        d_hat = 3.0_pREAL*mu*prm%b_sl(i)/(16.0_pREAL*PI*abs(tau))
        d_hat = math_clip(d_hat, right = dst%Lambda_sl(i,en))
        d_hat = math_clip(d_hat, left  = prm%d_caron(i))
        dot_rho_dip_formation(i) = 2.0_pREAL*(d_hat-prm%d_caron(i))/prm%b_sl(i) * stt%rho_mob(i,en)*agd(i)
      end if
      if (dEq(d_hat,prm%d_caron(i))) then
        dot_rho_dip_climb(i) = 0.0_pREAL
      else
        sigma_cl = dot_product(prm%n0_sl(1:3,i),matmul(Mp,prm%n0_sl(1:3,i)))
        if (prm%extendedDislocations) then
          b_d = 24.0_pREAL*PI*(1.0_pREAL - nu)/(2.0_pREAL + nu) * prm%Gamma_sf%at(T) / (mu*prm%b_sl(i))
        else
          b_d = 1.0_pREAL
        end if
        v_cl = 2.0_pREAL*prm%omega*b_d**2*exp(-prm%Q_cl/(K_B*T)) &
             * (exp(abs(sigma_cl)*prm%b_sl(i)**3/(K_B*T)) - 1.0_pREAL)
        dot_rho_dip_climb(i) = 4.0_pREAL*v_cl*stt%rho_dip(i,en)/(d_hat-prm%d_caron(i))
      end if
    end do

    dotState(1:ns)        = agd/(prm%b_sl*dst%Lambda_sl(:,en)) - dot_rho_dip_formation &
                          - 2.0_pREAL*prm%d_caron/prm%b_sl * stt%rho_mob(:,en)*agd
    dotState(ns+1:2*ns)   = dot_rho_dip_formation - 2.0_pREAL*prm%d_caron/prm%b_sl * stt%rho_dip(:,en)*agd &
                          - dot_rho_dip_climb
    dotState(2*ns+1:3*ns) = agd
    dotState(3*ns+1:3*ns+nt)       = fd_tw
    dotState(3*ns+nt+1:3*ns+nt+nr) = fd_tr
    if (prm%alpha) dotState(prm%o_fa+1:prm%o_fa+4) = fd_a
    call regions_dot(prm%Tw,gT)
    call regions_dot(prm%Ep,gE)
    call regions_dot(prm%Al,gA)

  end associate

  contains

  !> Voce hardening per family, driven by the total slip rate in the region
  subroutine regions_dot(R,g)
    type(tRegions),            intent(in) :: R
    real(pREAL), dimension(4), intent(in) :: g
    integer :: pp, f, o
    real(pREAL) :: x
    if (.not. R%plastic) return
    do pp = 1, 4
      do f = 1, R%N_fam
        o = R%o_xi + R%N_fam*(pp-1) + f
        x = 1.0_pREAL - plasticState(ph)%state(o,en)/R%xi_inf(f)
        dotState(o) = R%h_0(f)*sign(abs(x)**R%a(f),x)*g(pp)
      end do
    end do
    dotState(R%o_gamma+1:R%o_gamma+4) = g
  end subroutine regions_dot

end function twintrip_dotState


!--------------------------------------------------------------------------------------------------
!> @brief Diagonal of d(dotState)/d(state) (<= 0) for the preconditioned state update (slip
!> resistances inside product regions; tiny regions have stiff, unconstrained local slip rates).
!--------------------------------------------------------------------------------------------------
module function twintrip_dotStateDiag(Mp,ph,en) result(Jd)

  real(pREAL), dimension(3,3), intent(in) :: Mp
  integer,                     intent(in) :: ph, en
  real(pREAL), dimension(plasticState(ph)%sizeDotState) :: Jd

  real(pREAL), dimension(param(ph)%sum_N_sl) :: gd
  real(pREAL), dimension(param(ph)%sum_N_tw) :: fd_tw
  real(pREAL), dimension(param(ph)%sum_N_tr) :: fd_tr
  real(pREAL), dimension(4) :: gT, gE, gA
  real(pREAL), dimension(max(param(ph)%Tw%N_fam,1),4) :: nT                                         ! automatic: no heap allocation per call
  real(pREAL), dimension(max(param(ph)%Ep%N_fam,1),4) :: nE
  real(pREAL), dimension(max(param(ph)%Al%N_fam,1),4) :: nA


  associate(prm => param(ph))
    call kinetics(Mp,thermal_T(ph,en),ph,en,gd,fd_tw,fd_tr,gT,gE,gA,nT=nT,nE=nE,nA=nA)
    Jd = 0.0_pREAL
    call regions_diag(prm%Tw,gT,nT)
    call regions_diag(prm%Ep,gE,nE)
    call regions_diag(prm%Al,gA,nA)
  end associate

  contains

  !> d/dxi of h0 sign(|x|^a,x) G(xi), x = 1 - xi/xi_inf, dG/dxi = -nG/xi (own family), clamped <= 0
  subroutine regions_diag(R,G,nG)
    type(tRegions),              intent(in) :: R
    real(pREAL), dimension(4),   intent(in) :: G
    real(pREAL), dimension(:,:), intent(in) :: nG
    integer :: pp, f, o
    real(pREAL) :: x, xi, l, dl
    if (.not. R%plastic) return
    do pp = 1, 4
      do f = 1, R%N_fam
        o = R%o_xi + R%N_fam*(pp-1) + f
        xi = plasticState(ph)%state(o,en)
        x = 1.0_pREAL - xi/R%xi_inf(f)
        l  = merge(abs(x)**R%a(f),0.0_pREAL,abs(x) > 0.0_pREAL)
        dl = merge(R%a(f)*l/abs(x),0.0_pREAL,abs(x) > 0.0_pREAL)
        Jd(o) = min(R%h_0(f)*(-(dl/R%xi_inf(f))*G(pp) - sign(l,x)*nG(f,pp)/xi),0.0_pREAL)
      end do
    end do
  end subroutine regions_diag

end function twintrip_dotStateDiag


!--------------------------------------------------------------------------------------------------
!> @brief Mean free paths and passing stress (dislotwin, with bounded 1/(1 - sum(f))).
!--------------------------------------------------------------------------------------------------
module subroutine twintrip_dependentState(ph,en)

  integer, intent(in) :: ph, en

  real(pREAL) :: sumf_tw, sumf_tr, mu
  real(pREAL), dimension(param(ph)%sum_N_sl) :: inv_lambda_sl
  real(pREAL), dimension(param(ph)%sum_N_tw) :: f_over_t_tw
  real(pREAL), dimension(param(ph)%sum_N_tr) :: f_over_t_tr


  associate(prm => param(ph), stt => state(ph), dst => dependentState(ph))

    mu = elastic_mu(ph,en,prm%isotropic_bound)
    sumf_tw = sum(stt%f_tw(:,en))
    sumf_tr = sum(stt%f_tr(:,en))
    f_over_t_tw = stt%f_tw(:,en)/prm%t_tw
    f_over_t_tr = sumf_tr/prm%t_tr

    inv_lambda_sl = sqrt(matmul(prm%forestProjection,stt%rho_mob(:,en)+stt%rho_dip(:,en)))/prm%i_sl
    if (prm%sum_N_tw > 0) &
      inv_lambda_sl = inv_lambda_sl + matmul(prm%h_sl_tw,f_over_t_tw)/max(1.0_pREAL-sumf_tw,W_FLOOR)
    if (prm%sum_N_tr > 0) &
      inv_lambda_sl = inv_lambda_sl + matmul(prm%h_sl_tr,f_over_t_tr)/max(1.0_pREAL-sumf_tr,W_FLOOR)
    dst%Lambda_sl(:,en) = prm%D/(1.0_pREAL+prm%D*inv_lambda_sl)

    if (prm%sum_N_tw > 0) &
      dst%Lambda_tw(:,en) = prm%i_tw*prm%D &
                          / (1.0_pREAL+prm%D*matmul(prm%h_tw_tw,f_over_t_tw)/max(1.0_pREAL-sumf_tw,W_FLOOR))
    if (prm%sum_N_tr > 0) &
      dst%Lambda_tr(:,en) = prm%i_tr*prm%D &
                          / (1.0_pREAL+prm%D*matmul(prm%h_tr_tr,f_over_t_tr)/max(1.0_pREAL-sumf_tr,W_FLOOR))

    dst%tau_pass(:,en) = mu*prm%b_sl*sqrt(matmul(prm%h_sl_sl,stt%rho_mob(:,en)+stt%rho_dip(:,en)))

  end associate

end subroutine twintrip_dependentState


!--------------------------------------------------------------------------------------------------
!> @brief Homogenized stiffness of all regions (Reuss or Voigt).
!--------------------------------------------------------------------------------------------------
module function plastic_twintrip_homogenizedC(ph,en) result(homogenizedC)

  integer, intent(in) :: ph, en
  real(pREAL), dimension(6,6) :: homogenizedC

  real(pREAL), dimension(4) :: wT, wE, wA
  real(pREAL), dimension(6,6) :: C
  real(pREAL), dimension(15) :: key


  call fractions(ph,en,wT=wT,wE=wE,wA=wA)
  C = elastic_C66(ph,en)
  key = [wT,wE,wA,C(1,1),C(1,2),C(4,4)]
  associate(dst => dependentState(ph))                                                              ! called many times per state: reuse the last result
    if (any(key /= dst%C_key(:,en))) then
      dst%C_key(:,en) = key
      dst%C_bar(:,:,en) = twintrip_C66(C,wT,wE,wA,ph)
    end if
    homogenizedC = dst%C_bar(:,:,en)
  end associate

end function plastic_twintrip_homogenizedC


!--------------------------------------------------------------------------------------------------
!> @brief Homogenized stiffness for given austenite stiffness and region volume fractions.
!--------------------------------------------------------------------------------------------------
function twintrip_C66(C,wT,wE,wA,ph) result(Cbar)

  real(pREAL), dimension(6,6), intent(in) :: C
  real(pREAL), dimension(4),   intent(in) :: wT, wE, wA
  integer,                     intent(in) :: ph
  real(pREAL), dimension(6,6) :: Cbar

  real(pREAL), dimension(5) :: c_hP
  real(pREAL), dimension(6,6) :: C_r
  real(pREAL) :: w_g, C11, C12, C13, C33, C44, C14
  integer :: p, i


  associate(prm => param(ph))

    if (sum(wT) + sum(wE) + sum(wA) <= 0.0_pREAL) then                                              ! austenite only: exactly C
      Cbar = C
      return
    end if
    w_g = max(1.0_pREAL - sum(wT) - sum(wE) - sum(wA),0.0_pREAL)
    if (prm%user_C_hP) then
      c_hP = prm%C_hP
    else                                                                                            ! Martin 1972, Wong et al. 2016 eqs. (47), (48)
      C11 = (C(1,1) + C(1,2) + 2.0_pREAL*C(4,4))/2.0_pREAL
      C12 = (C(1,1) + 5.0_pREAL*C(1,2) - 2.0_pREAL*C(4,4))/6.0_pREAL
      C33 = (C(1,1) + 2.0_pREAL*C(1,2) + 4.0_pREAL*C(4,4))/3.0_pREAL
      C13 = (C(1,1) + 2.0_pREAL*C(1,2) - 2.0_pREAL*C(4,4))/3.0_pREAL
      C44 = (C(1,1) - C(1,2) + C(4,4))/3.0_pREAL
      C14 = (C(1,1) - C(1,2) - 2.0_pREAL*C(4,4))/(3.0_pREAL*sqrt(2.0_pREAL))
      c_hP = [C11 - C14**2/C44, C12 + C14**2/C44, C44 - C14**2/(0.5_pREAL*(C11 - C12)), C13, C33]
    end if

    Cbar = w_g*inv66_if(C,prm%Reuss)
    do p = 1, 4
      if (wT(p) > 0.0_pREAL) then
        C_r = C(1,1)*prm%C_T(:,:,1,p) + C(1,2)*prm%C_T(:,:,2,p) + C(4,4)*prm%C_T(:,:,3,p)
        Cbar = Cbar + wT(p)*inv66_if(C_r,prm%Reuss)
      end if
      if (wE(p) > 0.0_pREAL) then
        C_r = 0.0_pREAL
        do i = 1, 5
          C_r = C_r + c_hP(i)*prm%C_E(:,:,i,p)
        end do
        Cbar = Cbar + wE(p)*inv66_if(C_r,prm%Reuss)
      end if
      if (wA(p) > 0.0_pREAL) Cbar = Cbar + wA(p)*inv66_if(prm%C_A(:,:,p),prm%Reuss)
    end do
    if (prm%Reuss) Cbar = inv66(Cbar)

  end associate

end function twintrip_C66


!--------------------------------------------------------------------------------------------------
!> @brief Volume fractions of the austenite and of the product regions on each plane.
!--------------------------------------------------------------------------------------------------
subroutine fractions(ph,en,w_g,wT,wE,wA)

  integer, intent(in) :: ph, en
  real(pREAL),               intent(out), optional :: w_g
  real(pREAL), dimension(4), intent(out), optional :: wT, wE, wA

  real(pREAL), dimension(4) :: T_, E_, A_
  integer :: p


  associate(prm => param(ph), stt => state(ph))
    T_ = 0.0_pREAL; E_ = 0.0_pREAL; A_ = 0.0_pREAL
    do p = 1, 4
      if (prm%sum_N_tw > 0) T_(p) = max(sum(stt%f_tw(3*p-2:3*p,en)),0.0_pREAL)
      if (prm%sum_N_tr > 0) E_(p) = sum(stt%f_tr(3*p-2:3*p,en))
      if (prm%alpha)        A_(p) = max(stt%f_a(p,en),0.0_pREAL)
    end do
    E_ = max(E_ - A_,0.0_pREAL)
    if (present(w_g)) w_g = 1.0_pREAL - sum(stt%f_tw(:,en)) - sum(stt%f_tr(:,en))
    if (present(wT)) wT = T_
    if (present(wE)) wE = E_
    if (present(wA)) wA = A_
  end associate

end subroutine fractions


!--------------------------------------------------------------------------------------------------
!> @brief Write results to HDF5 output file.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_twintrip_result(ph,group)

  integer,          intent(in) :: ph
  character(len=*), intent(in) :: group

  integer :: ou, en
  real(pREAL), dimension(:,:), allocatable :: w


  associate(prm => param(ph), stt => state(ph), dst => dependentState(ph))
    do ou = 1, size(prm%output)
      select case(trim(prm%output(ou)))
        case('rho_mob')
          call result_writeDataset(stt%rho_mob,group,'rho_mob','mobile dislocation density','1/m²',prm%systems_sl)
        case('rho_dip')
          call result_writeDataset(stt%rho_dip,group,'rho_dip','dislocation dipole density','1/m²',prm%systems_sl)
        case('gamma_sl')
          call result_writeDataset(stt%gamma_sl,group,'gamma_sl','plastic shear (austenite)','1',prm%systems_sl)
        case('Lambda_sl')
          call result_writeDataset(dst%Lambda_sl,group,'Lambda_sl','mean free path for slip','m',prm%systems_sl)
        case('tau_pass')
          call result_writeDataset(dst%tau_pass,group,'tau_pass','passing stress for slip','Pa',prm%systems_sl)
        case('f_tw')
          if (prm%sum_N_tw > 0) call result_writeDataset(stt%f_tw,group,'f_tw','twinned volume fraction', &
                                                         'm³/m³',prm%systems_tw)
        case('gamma_tw')
          if (prm%sum_N_tw > 0) call result_writeDataset(stt%f_tw*spread(prm%gamma_char_tw,2,size(stt%f_tw,2)), &
                                                         group,'gamma_tw','twinning shear','1',prm%systems_tw)
        case('Lambda_tw')
          if (prm%sum_N_tw > 0) call result_writeDataset(dst%Lambda_tw,group,'Lambda_tw', &
                                                         'mean free path for twinning','m',prm%systems_tw)
        case('f_tr')
          if (prm%sum_N_tr > 0) call result_writeDataset(stt%f_tr,group,'f_tr', &
                                                         'transformed volume fraction (epsilon and alpha'')', &
                                                         'm³/m³',prm%systems_tw)
        case('f_eps','f_alpha')
          allocate(w(4,size(stt%rho_mob,2)))
          do en = 1, size(w,2)
            if (trim(prm%output(ou)) == 'f_eps') then
              call fractions(ph,en,wE=w(:,en))
            else
              call fractions(ph,en,wA=w(:,en))
            end if
          end do
          call result_writeDataset(w,group,trim(prm%output(ou)),'volume fraction per {111} plane','m³/m³')
          deallocate(w)
        case('xi_sl_tw','xi_sl_eps','xi_sl_alpha','gamma_sl_tw','gamma_sl_eps','gamma_sl_alpha')
          call regions_result(trim(prm%output(ou)))
      end select
    end do
  end associate

  contains

  subroutine regions_result(label)
    character(len=*), intent(in) :: label
    type(tRegions) :: R
    select case(label(index(label,'_',back=.true.)+1:))
      case('tw');    R = param(ph)%Tw
      case('eps');   R = param(ph)%Ep
      case('alpha'); R = param(ph)%Al
    end select
    if (.not. R%plastic) return
    if (label(1:2) == 'xi') then
      call result_writeDataset(plasticState(ph)%state(R%o_xi+1:R%o_xi+4*R%N_fam,:),group,label, &
                               'slip resistance per {111} plane (slip family fastest)','Pa')
    else
      call result_writeDataset(plasticState(ph)%state(R%o_gamma+1:R%o_gamma+4,:),group,label, &
                               'accumulated slip per {111} plane','1')
    end if
  end subroutine regions_result

end subroutine plastic_twintrip_result


!--------------------------------------------------------------------------------------------------
!> @brief Slip, twinning and transformation rates and (optionally) Lp and dLp/dMp.
!> @details gd: austenite slip rates (signed), fd_tw, fd_tr: df/dt, gT/gE/gA: sum of |slip rates| in
!> the twin/epsilon/alpha' region of each plane, fd_a: alpha' rates, nT/nE/nA: sum n |gd| per family.
!--------------------------------------------------------------------------------------------------
subroutine kinetics(Mp,T,ph,en,gd,fd_tw,fd_tr,gT,gE,gA,Lp,dLp_dMp,fd_a,nT,nE,nA)

  real(pREAL), dimension(3,3), intent(in) :: Mp
  real(pREAL),                 intent(in) :: T
  integer,                     intent(in) :: ph, en
  real(pREAL), dimension(param(ph)%sum_N_sl), intent(out) :: gd
  real(pREAL), dimension(param(ph)%sum_N_tw), intent(out) :: fd_tw
  real(pREAL), dimension(param(ph)%sum_N_tr), intent(out) :: fd_tr
  real(pREAL), dimension(4),                  intent(out) :: gT, gE, gA
  real(pREAL), dimension(3,3),     intent(out), optional :: Lp
  real(pREAL), dimension(3,3,3,3), intent(out), optional :: dLp_dMp
  real(pREAL), dimension(4),       intent(out), optional :: fd_a
  real(pREAL), dimension(:,:),     intent(out), optional :: nT, nE, nA

  real(pREAL), dimension(param(ph)%sum_N_sl) :: dgd
  real(pREAL), dimension(4) :: wT, wE, wA, S_p
  real(pREAL) :: w_g, mu, nu, Gamma_sf, x0, tau_r, tau_hat
  integer :: k, l, m, n, i, p
  logical :: deriv


  deriv = present(Lp)
  if (deriv) then
    Lp = 0.0_pREAL
    dLp_dMp = 0.0_pREAL
  end if

  associate(prm => param(ph), stt => state(ph), dst => dependentState(ph))

    call fractions(ph,en,w_g,wT,wE,wA)
    w_g = max(w_g,0.0_pREAL)

    ! austenite slip
    call kinetics_sl(Mp,T,ph,en,gd,dgd)
    if (deriv) then
      do i = 1, prm%sum_N_sl
        if (dEq0(gd(i))) cycle
        Lp = Lp + w_g*gd(i)*prm%P_sl(1:3,1:3,i)
        forall (k=1:3,l=1:3,m=1:3,n=1:3) &
          dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) + w_g*dgd(i)*prm%P_sl(k,l,i)*prm%P_sl(m,n,i)
      end do
    end if

    ! twinning and gamma->epsilon: nucleation on the twin systems
    if (prm%sum_N_tw + prm%sum_N_tr > 0) then
      mu = elastic_mu(ph,en,prm%isotropic_bound)
      nu = elastic_nu(ph,en,prm%isotropic_bound)
      Gamma_sf = prm%Gamma_sf%at(T)
      x0 = mu*prm%b_sl(1)**2*(2.0_pREAL+nu)/(Gamma_sf*8.0_pREAL*PI*(1.0_pREAL-nu))
      tau_r = mu*prm%b_sl(1)/(2.0_pREAL*PI)*(1.0_pREAL/(x0+prm%x_c)+cos(PI/3.0_pREAL)/x0)
    end if
    if (prm%sum_N_tw > 0) then
      tau_hat = 3.0_pREAL*prm%b_tw(1)*mu/prm%L_tw + Gamma_sf/(3.0_pREAL*prm%b_tw(1))
      call nucleation(fd_tw,prm%r,prm%L_tw,prm%t_tw,dst%Lambda_tw(:,en),prm%gamma_char_tw)
    end if
    if (prm%sum_N_tr > 0) then
      tau_hat = 3.0_pREAL*prm%b_tr(1)*mu/prm%L_tr &
              + (Gamma_sf + (prm%h/prm%V_mol - 2.0_pREAL*prm%rho)*prm%Delta_G%at(T))/(3.0_pREAL*prm%b_tr(1))
      call nucleation(fd_tr,prm%s,prm%L_tr,prm%t_tr,dst%Lambda_tr(:,en),spread(GAMMA_EPS,1,12))
    end if

    ! slip in the product regions
    call regions_kinetics(prm%Tw,wT,gT,nT)
    call regions_kinetics(prm%Ep,wE,gE,nE)
    call regions_kinetics(prm%Al,wA,gA,nA)

    ! alpha' at epsilon lamellae sheared on the other planes
    if (present(fd_a)) then
      fd_a = 0.0_pREAL
      if (prm%alpha) then
        do p = 1, 4
          S_p(p) = wE(p)*gE(p) + GAMMA_EPS*sum(fd_tr(3*p-2:3*p))
        end do
        do p = 1, 4
          fd_a(p) = prm%k_alpha/(1.0_pREAL + exp(prm%Delta_G_alpha%at(T)/prm%G_s)) &
                  * wE(p)*(sum(S_p) - S_p(p))
        end do
      end if
    end if

  end associate

  contains

  !> volume fraction rates of twins or epsilon (Wong et al. 2016) with |dot_gamma_sl| in the nucleation
  !> rate; adds gamma_char fd P_tw to Lp and its tangent, including the nucleation-rate dependence
  subroutine nucleation(fd,r,L_n,t_n,Lambda,gamma_char)
    real(pREAL), dimension(12), intent(out) :: fd
    real(pREAL), dimension(12), intent(in)  :: r, t_n, Lambda, gamma_char
    real(pREAL),                intent(in)  :: L_n
    real(pREAL) :: tau, P, dP_dtau, P_ncs, dP_ncs_dtau, N0, V, c, dfd_dtau
    real(pREAL), dimension(3,3) :: dN0_dMp
    integer :: kk, s1, s2
    associate(prm => param(ph), stt => state(ph))
      do kk = 1, 12
        fd(kk) = 0.0_pREAL
        tau = math_tensordot(Mp,prm%P_tw(1:3,1:3,kk))
        if (.not. (tau > tol_math_check .and. tau < tau_r)) cycle
        P = exp(-(tau_hat/tau)**r(kk))
        dP_dtau = r(kk)*(tau_hat/tau)**r(kk)/tau*P
        s1 = prm%pair(1,kk); s2 = prm%pair(2,kk)
        N0 = (abs(gd(s2))*(stt%rho_mob(s1,en)+stt%rho_dip(s1,en)) &
            + abs(gd(s1))*(stt%rho_mob(s2,en)+stt%rho_dip(s2,en)))/(3.0_pREAL*L_n)
        P_ncs = 1.0_pREAL - exp(-prm%V_cs/(K_B*T)*(tau_r-tau))
        dP_ncs_dtau = prm%V_cs/(K_B*T)*(P_ncs - 1.0_pREAL)
        V = PI/4.0_pREAL*Lambda(kk)**2*t_n(kk)
        c = w_g*V*P_ncs*P
        fd(kk) = c*N0
        if (.not. deriv) cycle
        dfd_dtau = w_g*V*N0*(P*dP_ncs_dtau + P_ncs*dP_dtau)
        dN0_dMp = (sign(1.0_pREAL,gd(s2))*dgd(s2)*(stt%rho_mob(s1,en)+stt%rho_dip(s1,en))*prm%P_sl(1:3,1:3,s2) &
                 + sign(1.0_pREAL,gd(s1))*dgd(s1)*(stt%rho_mob(s2,en)+stt%rho_dip(s2,en))*prm%P_sl(1:3,1:3,s1)) &
                / (3.0_pREAL*L_n)
        Lp = Lp + gamma_char(kk)*fd(kk)*prm%P_tw(1:3,1:3,kk)
        forall (k=1:3,l=1:3,m=1:3,n=1:3) &
          dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) + gamma_char(kk)*prm%P_tw(k,l,kk) &
                                               * (dfd_dtau*prm%P_tw(m,n,kk) + c*dN0_dMp(m,n))
      end do
    end associate
  end subroutine nucleation

  !> power-law slip in the product regions of one kind
  subroutine regions_kinetics(R,w,g,nG)
    type(tRegions),              intent(in) :: R
    real(pREAL), dimension(4),   intent(in) :: w
    real(pREAL), dimension(4),   intent(out) :: g
    real(pREAL), dimension(:,:), intent(out), optional :: nG
    real(pREAL) :: tau, xi, gg, dg
    integer :: pp, j, f
    g = 0.0_pREAL
    if (present(nG)) nG = 0.0_pREAL
    if (.not. R%plastic) return
    do pp = 1, 4
      if (w(pp) <= F_MIN) cycle
      do j = 1, R%N_sys
        f = R%fam(j)
        tau = math_tensordot(Mp,R%P(1:3,1:3,j,pp))
        xi  = plasticState(ph)%state(R%o_xi+R%N_fam*(pp-1)+f,en)
        gg  = sign(R%dot_gamma_0(f)*(abs(tau)/xi)**R%n(f),tau)
        g(pp) = g(pp) + abs(gg)
        if (present(nG)) nG(f,pp) = nG(f,pp) + R%n(f)*abs(gg)
        if (deriv .and. dNeq0(gg)) then
          Lp = Lp + w(pp)*gg*R%P(1:3,1:3,j,pp)
          dg = w(pp)*gg*R%n(f)/tau
          forall (k=1:3,l=1:3,m=1:3,n=1:3) &
            dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) + dg*R%P(k,l,j,pp)*R%P(m,n,j,pp)
        end if
      end do
    end do
  end subroutine regions_kinetics

end subroutine kinetics


!--------------------------------------------------------------------------------------------------
!> @brief Thermally activated glide in the austenite (dislotwin), written without overflow
!> (rate = rho b v_0 e^-A / (1 + v_0 e^-A B/(tau_eff b)), A = Q/kT x^q) and with the barrier term
!> x = softplus_delta(1 - (tau_eff/tau_0)^p) > 0, i.e. drag-controlled glide above tau_0 instead of NaN.
!--------------------------------------------------------------------------------------------------
pure subroutine kinetics_sl(Mp,T,ph,en,dot_gamma_sl,ddot_gamma_dtau_sl)

  real(pREAL), dimension(3,3), intent(in) :: Mp
  real(pREAL),                 intent(in) :: T
  integer,                     intent(in) :: ph, en
  real(pREAL), dimension(param(ph)%sum_N_sl), intent(out) :: dot_gamma_sl, ddot_gamma_dtau_sl

  real(pREAL) :: tau, tau_eff, SR, y, x, sg, A, u, du, den, rate, dA, Q_kB_T
  integer :: i


  associate(prm => param(ph), stt => state(ph), dst => dependentState(ph))
    do i = 1, prm%sum_N_sl
      tau = math_tensordot(Mp,prm%P_sl(1:3,1:3,i))
      tau_eff = abs(tau) - dst%tau_pass(i,en)
      if (.not. tau_eff > tol_math_check) then
        dot_gamma_sl(i) = 0.0_pREAL
        ddot_gamma_dtau_sl(i) = 0.0_pREAL
        cycle
      end if
      Q_kB_T = prm%Q_sl(i)/(K_B*T)
      SR = tau_eff/prm%tau_0(i)
      y  = (1.0_pREAL - SR**prm%p(i))/prm%delta_sl
      if (y > 30.0_pREAL) then                                                                      ! softplus(y) = y to double precision
        x  = y*prm%delta_sl
        sg = 1.0_pREAL
      else
        x  = prm%delta_sl*log(1.0_pREAL + exp(y))
        sg = 1.0_pREAL/(1.0_pREAL + exp(-y))
      end if
      A   = Q_kB_T*x**prm%q(i)
      u   = prm%v_0(i)*exp(-A)
      dA  = Q_kB_T*prm%q(i)*x**(prm%q(i)-1.0_pREAL)*sg*(-prm%p(i)*SR**(prm%p(i)-1.0_pREAL)/prm%tau_0(i))
      du  = -u*dA
      den = tau_eff*prm%b_sl(i) + u*prm%B(i)
      rate = stt%rho_mob(i,en)*prm%b_sl(i)*u*tau_eff*prm%b_sl(i)/den
      dot_gamma_sl(i) = sign(rate,tau)
      ddot_gamma_dtau_sl(i) = stt%rho_mob(i,en)*prm%b_sl(i) &
                            * (du*(tau_eff*prm%b_sl(i))**2 + u**2*prm%b_sl(i)*prm%B(i))/den**2
    end do
  end associate

end subroutine kinetics_sl


#if defined(METAL) && defined(GRID)
!--------------------------------------------------------------------------------------------------
!> @brief Upload the model of a phase to the GPU constitutive kernel (cp_model_twintrip,
!> cp_models_composite.h). Returns the GPU handle or a negative number if the phase stays on the CPU.
!> @details dims = [N_tw, N_tr, alpha', N_sys/N_fam of the twin, epsilon, alpha' regions, Reuss, sizeState],
!> blob = [13 region compliances (Reuss) or stiffnesses (Voigt): austenite, twin p, epsilon p, alpha' p;
!> P_sl, n0_sl, P_tw; forestProjection, h_sl_sl, (h_sl_tw), (h_sl_tr), (h_tw_tw), (h_tr_tr);
!> b_sl, Q_sl, v_0, i_sl, p, q, tau_0, B, d_caron; (t_tw, r, gamma_char_tw); (t_tr, s); pair;
!> per plastic region kind: P, fam, dot_gamma_0, n, xi_inf, h_0, a; Gamma_sf, Delta_G, Delta_G_alpha as
!> [degree, x_ref, coef(0:4)]; 20 scalars, extend_dislocations, omit_dipoles] (see cp_twintrip_pack).
!--------------------------------------------------------------------------------------------------
module function plastic_twintrip_GPUadd(ph,C66,numerics,ndep) result(handle)

  integer, intent(in) :: ph
  real(pREAL), dimension(6,6), intent(in) :: C66
  real(pREAL), dimension(7), intent(in) :: numerics
  integer, intent(out) :: ndep                                                                     !< size of the dependent state passed along with the state
  integer :: handle

  real(pREAL), dimension(:), allocatable :: blob
  real(pREAL), dimension(13,6,6) :: M
  real(pREAL), dimension(6,6) :: C_r
  real(pREAL), dimension(5) :: c_hP
  real(pREAL), dimension(20) :: sc
  real(pREAL) :: C11, C12, C13, C33, C44, C14
  integer :: p, i, k_rnd
  real(pREAL), dimension(:), allocatable :: x_g                                                     ! GPU_golden
  character(len=:), allocatable :: reason


  handle = -1
  associate(prm => param(ph))

    ndep = 36 + 2*prm%sum_N_sl + prm%sum_N_tw + prm%sum_N_tr + merge(1,0,prm%alpha)
    if (prm%user_C_hP) then
      c_hP = prm%C_hP
    else                                                                                            ! as twintrip_C66
      C11 = (C66(1,1) + C66(1,2) + 2.0_pREAL*C66(4,4))/2.0_pREAL
      C12 = (C66(1,1) + 5.0_pREAL*C66(1,2) - 2.0_pREAL*C66(4,4))/6.0_pREAL
      C33 = (C66(1,1) + 2.0_pREAL*C66(1,2) + 4.0_pREAL*C66(4,4))/3.0_pREAL
      C13 = (C66(1,1) + 2.0_pREAL*C66(1,2) - 2.0_pREAL*C66(4,4))/3.0_pREAL
      C44 = (C66(1,1) - C66(1,2) + C66(4,4))/3.0_pREAL
      C14 = (C66(1,1) - C66(1,2) - 2.0_pREAL*C66(4,4))/(3.0_pREAL*sqrt(2.0_pREAL))
      c_hP = [C11 - C14**2/C44, C12 + C14**2/C44, C44 - C14**2/(0.5_pREAL*(C11 - C12)), C13, C33]
    end if
    M = 0.0_pREAL
    M(1,:,:) = inv66_if(C66,prm%Reuss)
    do p = 1, 4
      C_r = C66(1,1)*prm%C_T(:,:,1,p) + C66(1,2)*prm%C_T(:,:,2,p) + C66(4,4)*prm%C_T(:,:,3,p)
      M(1+p,:,:) = inv66_if(C_r,prm%Reuss)
      C_r = 0.0_pREAL
      do i = 1, 5
        C_r = C_r + c_hP(i)*prm%C_E(:,:,i,p)
      end do
      if (prm%sum_N_tr > 0) M(5+p,:,:) = inv66_if(C_r,prm%Reuss)
      if (prm%alpha)        M(9+p,:,:) = inv66_if(prm%C_A(:,:,p),prm%Reuss)
    end do

    blob = [(reshape(M(i,:,:),[36]),i=1,13), &
            reshape(prm%P_sl,[108]), reshape(prm%n0_sl,[36]), reshape(prm%P_tw,[108]), &
            reshape(prm%forestProjection,[144]), reshape(prm%h_sl_sl,[144])]
    if (prm%sum_N_tw > 0) blob = [blob, reshape(prm%h_sl_tw,[144])]
    if (prm%sum_N_tr > 0) blob = [blob, reshape(prm%h_sl_tr,[144])]
    if (prm%sum_N_tw > 0) blob = [blob, reshape(prm%h_tw_tw,[144])]
    if (prm%sum_N_tr > 0) blob = [blob, reshape(prm%h_tr_tr,[144])]
    blob = [blob, prm%b_sl, prm%Q_sl, prm%v_0, prm%i_sl, prm%p, prm%q, prm%tau_0, prm%B, prm%d_caron]
    if (prm%sum_N_tw > 0) blob = [blob, prm%t_tw, prm%r, prm%gamma_char_tw]
    if (prm%sum_N_tr > 0) blob = [blob, prm%t_tr, prm%s]
    blob = [blob, real(reshape(prm%pair,[24]),pREAL)]
    call region_blob(prm%Tw)
    call region_blob(prm%Ep)
    call region_blob(prm%Al)
    reason = ''
    call polynomial_blob(prm%Gamma_sf, prm%sum_N_tw + prm%sum_N_tr > 0 .or. prm%extendedDislocations)
    call polynomial_blob(prm%Delta_G,  prm%sum_N_tr > 0)
    call polynomial_blob(prm%Delta_G_alpha, prm%alpha)

    sc = 0.0_pREAL
    sc([3,4,5,15]) = [prm%Q_cl, prm%omega, prm%D, prm%delta_sl]
    if (prm%sum_N_tw > 0) sc([1,6,8]) = [prm%b_tw(1), prm%i_tw, prm%L_tw]
    if (prm%sum_N_tr > 0) sc([2,7,9,12,13,14]) = [prm%b_tr(1), prm%i_tr, prm%L_tr, prm%h, prm%V_mol, prm%rho]
    if (prm%sum_N_tw + prm%sum_N_tr > 0) sc([10,11]) = [prm%x_c, prm%V_cs]
    if (prm%alpha) sc([16,17,18]) = [prm%k_alpha, prm%G_s, prm%Delta_V]
    sc(19) = elastic_mu(ph,1,prm%isotropic_bound)
    sc(20) = elastic_nu(ph,1,prm%isotropic_bound)
    blob = [blob, sc, merge(1.0_pREAL,0.0_pREAL,prm%extendedDislocations), merge(1.0_pREAL,0.0_pREAL,prm%omitDipoles)]

    block
      integer :: l
      character(len=pSTRLEN) :: golden
      call get_environment_variable('DAMASK_TWINTRIP_GOLDEN',golden,length=l)
      if (l > 0) then                                                                               ! diagnostics: reference values for the GPU policy, then stop
        call GPU_golden(trim(golden))
        call quit(0)
      end if
    end block
    if (reason == '') then
      handle = Metal_cpAddPhaseTwintrip(Metal_ctx, &
                                        int([prm%sum_N_tw,prm%sum_N_tr,merge(1,0,prm%alpha), &
                                             prm%Tw%N_sys,prm%Tw%N_fam,prm%Ep%N_sys,prm%Ep%N_fam,prm%Al%N_sys,prm%Al%N_fam, &
                                             merge(1,0,prm%Reuss),plasticState(ph)%sizeState],C_INT), &
                                        int(size(blob),C_INT),C66,blob,plasticState(ph)%atol,numerics)
      reason = 'region slip systems not supported or GPU initialization failed'
    end if
  end associate

  if (handle >= 0) then
    print'(/,1x,a,i0,a)', 'phase ',ph,': twintrip constitutive update on GPU'
  else
    print'(/,1x,a,i0,a)', 'phase ',ph,': twintrip constitutive update on CPU ('//reason//')'
  end if
  flush(IO_STDOUT)

  contains

  !> Fortran results for synthetic states and stresses of entry 1 (test_twintrip_policy.cpp)
  subroutine GPU_golden(fname)
    character(len=*), intent(in) :: fname
    integer, parameter :: N_SAMPLES = 200
    real(pREAL), dimension(plasticState(ph)%sizeState) :: x_saved, dot, Jd
    real(pREAL), dimension(3,3) :: Mp, Lp
    real(pREAL), dimension(3,3,3,3) :: dLp_dMp
    real(pREAL), dimension(6,6) :: C
    real(pREAL) :: sc_M
    integer :: u, t, pp, f, o, NS
    NS = plasticState(ph)%sizeState
    associate(prm => param(ph), dst => dependentState(ph))
    x_saved = plasticState(ph)%state(:,1)
    open(newunit=u,file=fname,access='stream',form='unformatted',status='replace')
    write(u) int([prm%sum_N_tw,prm%sum_N_tr,merge(1,0,prm%alpha),prm%Tw%N_sys,prm%Tw%N_fam,prm%Ep%N_sys,prm%Ep%N_fam, &
                  prm%Al%N_sys,prm%Al%N_fam,merge(1,0,prm%Reuss),NS],C_INT), int(size(blob),C_INT)
    write(u) C66, blob, plasticState(ph)%atol, int(N_SAMPLES,C_INT)
    k_rnd = 0
    do t = 1, N_SAMPLES
      x_g = x_saved
      x_g(1:2*prm%sum_N_sl) = x_g(1:2*prm%sum_N_sl)*(0.5_pREAL + 10.0_pREAL*[(rnd(),o=1,2*prm%sum_N_sl)])
      x_g(2*prm%sum_N_sl+1:3*prm%sum_N_sl) = 0.1_pREAL*[(rnd(),o=1,prm%sum_N_sl)]
      o = 3*prm%sum_N_sl
      if (t > 50) then                                                                              ! products present
        x_g(o+1:o+prm%sum_N_tw+prm%sum_N_tr) = 0.03_pREAL*[(rnd(),f=1,prm%sum_N_tw+prm%sum_N_tr)]
        if (prm%alpha) then
          do pp = 1, 4
            x_g(prm%o_fa+pp) = 0.5_pREAL*rnd()*sum(x_g(o+prm%sum_N_tw+3*pp-2:o+prm%sum_N_tw+3*pp))
          end do
        end if
      end if
      call region_state(prm%Tw); call region_state(prm%Ep); call region_state(prm%Al)
      sc_M = 5.0e7_pREAL + 5.5e8_pREAL*rnd()
      Mp = reshape([(rnd()-0.5_pREAL,f=1,9)],[3,3])
      Mp = sc_M*(Mp + transpose(Mp))
      plasticState(ph)%state(:,1) = x_g
      dst%C_key(:,1) = -1.0_pREAL
      call twintrip_dependentState(ph,1)
      C = plastic_twintrip_homogenizedC(ph,1)
      call twintrip_LpAndItsTangent(Lp,dLp_dMp,Mp,ph,1)
      dot = twintrip_dotState(Mp,ph,1)
      Jd  = twintrip_dotStateDiag(Mp,ph,1)
      write(u) x_g, Mp, thermal_T(ph,1), dst%Lambda_sl(:,1), dst%Lambda_tw(:,1), dst%Lambda_tr(:,1), dst%tau_pass(:,1), &
               C, plastic_twintrip_GPUj(ph,1), Lp, dLp_dMp, dot, Jd
    end do
    close(u)
    plasticState(ph)%state(:,1) = x_saved
    dst%C_key(:,1) = -1.0_pREAL
    call twintrip_dependentState(ph,1)
    end associate
    print'(/,1x,a,i0,a)', 'twintrip: ',N_SAMPLES,' reference samples written to '//fname

  end subroutine GPU_golden

  real(pREAL) function rnd()                                                                       ! deterministic pseudo random numbers in [0,1)
    k_rnd = k_rnd + 1
    rnd = modulo(sin(real(k_rnd,pREAL)*12.9898_pREAL)*43758.5453_pREAL,1.0_pREAL)
  end function rnd

  subroutine region_state(R)
    type(tRegions), intent(in) :: R
    integer :: q, ff
    if (.not. R%plastic) return
    do q = 1, 4; do ff = 1, R%N_fam
      x_g(R%o_xi+R%N_fam*(q-1)+ff) = R%xi_0(ff) + rnd()*(R%xi_inf(ff) - R%xi_0(ff))
    end do; end do
    x_g(R%o_gamma+1:R%o_gamma+4) = [(rnd(),q=1,4)]
  end subroutine region_state

  subroutine region_blob(R)
    type(tRegions), intent(in) :: R
    if (.not. R%plastic) return
    blob = [blob, reshape(R%P,[36*R%N_sys]), real(R%fam,pREAL), R%dot_gamma_0, R%n, R%xi_inf, R%h_0, R%a]
  end subroutine region_blob

  subroutine polynomial_blob(P,used)
    type(tPolynomial), intent(in) :: P
    logical,           intent(in) :: used
    real(pREAL), dimension(7) :: v
    integer :: n
    v = 0.0_pREAL
    if (used) then
      n = ubound(P%coef,1)
      if (n > 4) reason = 'polynomial of degree > 4'
      v(1) = real(min(n,4),pREAL)
      if (n > 0) v(2) = P%x_ref
      v(3:3+min(n,4)) = P%coef(0:min(n,4))
    end if
    blob = [blob, v]
  end subroutine polynomial_blob

end function plastic_twintrip_GPUadd


!--------------------------------------------------------------------------------------------------
!> @brief Get (set = .false.) or set the dependent state of an entry for the GPU kernel:
!> D = [C66, Lambda_sl, Lambda_tw, Lambda_tr, tau_pass, (j^2)]; computed on the GPU from the state.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_twintrip_GPUdependent(ph,en,D,set)

  integer, intent(in) :: ph, en
  real(pREAL), dimension(:), intent(inout) :: D
  logical, intent(in) :: set

  integer :: o


  if (.not. set) then
    D = 0.0_pREAL
    return
  end if
  associate(prm => param(ph), dst => dependentState(ph))
    o = 36
    dst%Lambda_sl(:,en) = D(o+1:o+prm%sum_N_sl); o = o + prm%sum_N_sl
    dst%Lambda_tw(:,en) = D(o+1:o+prm%sum_N_tw); o = o + prm%sum_N_tw
    dst%Lambda_tr(:,en) = D(o+1:o+prm%sum_N_tr); o = o + prm%sum_N_tr
    dst%tau_pass(:,en)  = D(o+1:o+prm%sum_N_sl)
  end associate

end subroutine plastic_twintrip_GPUdependent


!--------------------------------------------------------------------------------------------------
!> @brief Eigenstrain Fi = j I of the current state (for the GPU results).
!--------------------------------------------------------------------------------------------------
module function plastic_twintrip_GPUj(ph,en) result(j)

  integer, intent(in) :: ph, en
  real(pREAL) :: j


  associate(prm => param(ph), stt => state(ph))
    j = 1.0_pREAL
    if (prm%alpha) j = (1.0_pREAL + prm%Delta_V*sum(stt%f_a(:,en)))**(1.0_pREAL/3.0_pREAL)
  end associate

end function plastic_twintrip_GPUj
#endif

end submodule twintrip
