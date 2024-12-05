#arXiv:2410.21590

from numpy import array, arcsinh, arctanh, abs, sign
from math import pi, sqrt, log10, exp
from scipy.optimize import fsolve

#all quantities are in units of fm.  Use 1=hbar*c=197.3 MeV*fm to convert as necessary when finished

#NEXT SECTION IS FUNCTIONS AND CONSTANTS NEEDED FOR ALL MODELS

NEUTRON_MASS = 939.57/197.3
PROTON_MASS = 938.27/197.3
ELECTRON_MASS = 0.511 / 197.3
NUCLEON_MASS = (NEUTRON_MASS + PROTON_MASS) / 2
MUON_MASS = 105.7/197.3

def _density(kf): return kf**3/(3*pi**2)

def _lepton_density(mue):
    if mue > MUON_MASS:
        return _density(mue) + _density(sqrt(mue**2 - MUON_MASS**2))
    elif mue > 50 * ELECTRON_MASS:
        return _density(mue)
    elif mue < ELECTRON_MASS:
        return 0
    return _density(sqrt(mue**2 - ELECTRON_MASS**2))

def _fermion_energy_density(kf, mass): #includes spin 1/2 degeneracy but not color/flavor
    if kf / mass > 0.1:
        return 1/(8*pi**2) * (kf * sqrt(kf**2 + mass**2) * (2*kf**2 + mass**2) - mass**4 * arcsinh(kf/mass))
    return mass * _density(kf) + 1 / (10 * pi**2) * kf**5 / mass
def _lepton_energy_density(mue):
    #include muons if mue is larger than the muon mass
    if mue > MUON_MASS:
        return mue**4/(4*pi**2) + _fermion_energy_density(sqrt(mue**2-MUON_MASS**2), MUON_MASS)
    elif mue > 50 * ELECTRON_MASS:
        return mue**4/(4*pi**2)
    elif mue < ELECTRON_MASS:
        return 0
    return _fermion_energy_density(sqrt(mue**2 - ELECTRON_MASS**2), ELECTRON_MASS)

def _fermion_pressure(kf, mass): #includes spin 1/2 degeneracy, but not color/flavor degeneracy
    if kf / mass > 0.1:
        return 1/(24*pi**2) * (kf*(2*kf**2-3*mass**2) * sqrt(kf**2+mass**2) +
            3*mass**4 * arctanh(kf/sqrt(kf**2+mass**2)))
    return 1 / (15 * pi**2) * kf**5 / mass
def _lepton_pressure(mue):
    #include muons if mue is larger than the muon mass
    if mue > MUON_MASS:
        return mue**4/(12*pi**2) + _fermion_pressure(sqrt(mue**2-MUON_MASS**2), MUON_MASS)
    elif mue > 50 * ELECTRON_MASS:
        return mue**4/(12*pi**2)
    elif mue < ELECTRON_MASS:
        return 0
    return _fermion_pressure(sqrt(mue**2 - ELECTRON_MASS**2), ELECTRON_MASS)
    
#scalar density needed for RMF equations of motion
def _scalar_density(kf, mass):
    if kf / mass > 0.1:
        return mass / (2*pi**2) * (kf * sqrt(kf**2 + mass**2) - mass**2 * arctanh(kf / sqrt(kf**2 + mass**2)))
    return _density(kf) - kf**5 / (10 * pi**2 * mass**2)

def _outer_crust_solver(inputs, mub, baryon_per_nuc, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
    binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants
    mue, xp = inputs

    val = [-mue + NEUTRON_MASS - PROTON_MASS + 4 * (1 - 2 * xp) * symm_energy - 2 * xp * coul_energy * baryon_per_nuc**(2/3)]
    val.append(-mub + NEUTRON_MASS - binding_energy + symm_energy * (1 - 4 * xp**2) 
               - 1 / 3 * coul_energy * xp**2 * baryon_per_nuc**(2/3) + 2 / 3 * surf_energy / baryon_per_nuc**(1/3))
    return val

#NEXT SECTION IS RMF FUNCTIONS AND CLASSES

# U = - kappa3/6M gsig msig^2 sig^3 - kappa4/24M^2 gsig^2 msig^2 sig^4 +lambdasigdelta sig^2 delta^2 +zeta0/24 gw^2 w^4
# + eta1/2m gsig mw^2 sig omega^2 + eta2/4m^2 gsig^2 mw^2 sig^2 w^2 + etarho/2m gsig mrho^2 sig rho^2 
# + eta1rho/4m^2 gsig^2 mrho^2 sig^2 rho^2 + eta2rho/4m^2 gw^2 mrho^2 w^2 rho^2

# Following are equations of motion based on the above meson potential. All are mean field, beta equilibrated, some are charge neutral
# All have option to include deltas or not (if delta isn't needed, omitting it improves code stability)

def _rmf_eom_neutral(fields, baryon_density, constants): #rmf eoms for charge neutral nuclear matter
    #rmf model constants
    [msig,mw,mrho,mdelta,gsig,gw,grho,gdelta,kappa3,kappa4,lambdasigdelta,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #fields to solve for
    [sigma, omega, rho, delta, kfp, kfn, mue, mub] = fields

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma * gsig - delta * gdelta
    mn_star = NEUTRON_MASS - sigma * gsig + delta * gdelta

    #meson equations of motion
    val = [gsig * (_scalar_density(kfp, mp_star) + _scalar_density(kfn, mn_star)) - msig**2*sigma - kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        - kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3 + 2*lambdasigdelta*sigma*delta**2 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2 + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2 + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(gdelta * (_scalar_density(kfp,mp_star) - _scalar_density(kfn,mn_star)) - mdelta**2*delta + 2*lambdasigdelta*sigma**2*delta)
    val.append(-gw * (_density(kfp) + _density(kfn))+mw**2*omega + eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2 + 1/6*zeta0*gw**2*omega**3)
    val.append(-1/2 * grho * (_density(kfp) - _density(kfn)) + mrho**2*rho + etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #apply beta equilibrium
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1/2 * grho * rho - mub)
    val.append(sqrt(kfp**2 + mp_star**2) + gw * omega + 1/2 * grho * rho - mub + mue)
    #apply charge neutrality
    if mue > MUON_MASS:
        val.append(mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3)
    else:
        val.append(mue - kfp)
    #fix baryon density
    val.append(baryon_density - _density(kfn) - _density(kfp))
    return val

def _rmf_eom_neutral_no_delta(fields, baryon_density, constants): #rmf eoms for charge neutral nuclear matter
    #rmf model constants
    [msig,mw,mrho,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho]=constants
    #fields to solve for
    [sigma, omega, rho, kfp, kfn, mue, mub] = fields

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma*gsig
    mn_star = NEUTRON_MASS - sigma*gsig

    #meson equations of motion
    val = [gsig * (_scalar_density(kfp, mp_star) + _scalar_density(kfn, mn_star)) - msig**2*sigma - kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        - kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2 + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2 + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw * (_density(kfp) + _density(kfn)) + mw**2*omega + eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2 + 1/6*zeta0*gw**2*omega**3)
    val.append(-1/2 * grho * (_density(kfp) - _density(kfn)) + mrho**2*rho + etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #apply beta equilibrium
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1/2 * grho * rho - mub)
    val.append(sqrt(kfp**2 + mp_star**2) + gw * omega + 1/2 * grho * rho - mub + mue)
    #apply charge neutrality
    if mue > MUON_MASS:
        val.append(mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3)
    else:
        val.append(mue - kfp)
    #fix baryon density
    val.append(baryon_density - _density(kfn) - _density(kfp))
    return val

def _rmf_eom_neutral_mub(fields, mub, constants): #rmf eoms for charge neutral nuclear matter
    #rmf model constants
    [msig,mw,mrho,mdelta,gsig,gw,grho,gdelta,kappa3,kappa4,lambdasigdelta,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #fields to solve for
    [sigma, omega, rho, delta, kfp, kfn, mue] = fields

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma * gsig - delta * gdelta
    mn_star = NEUTRON_MASS - sigma * gsig + delta * gdelta

    #meson equations of motion
    val = [gsig * (_scalar_density(kfp, mp_star) + _scalar_density(kfn,mn_star)) - msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        - kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3 + 2*lambdasigdelta*sigma*delta**2 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2 + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2 + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(gdelta * (_scalar_density(kfp,mp_star) - _scalar_density(kfn,mn_star)) - mdelta**2*delta + 2*lambdasigdelta*sigma**2*delta)
    val.append(-gw*(_density(kfp)+_density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2 + 1/6*zeta0*gw**2*omega**3)
    val.append(-1/2 * grho * (_density(kfp) - _density(kfn)) + mrho**2*rho + etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #apply beta equilibrium
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1/2 * grho * rho - mub)
    val.append(sqrt(kfp**2 + mp_star**2) + gw * omega + 1/2 * grho * rho - mub + mue)
    #apply charge neutrality
    if mue > MUON_MASS:
        val.append(mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3)
    else:
        val.append(mue - kfp)
    return val

def _rmf_eom_neutral_mub_no_delta(fields, mub, constants): #rmf eoms for charge neutral nuclear matter
    #rmf model constants
    [msig,mw,mrho,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #fields to solve for
    [sigma,omega,rho,kfp,kfn,mue] = fields

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma*gsig
    mn_star = NEUTRON_MASS - sigma*gsig

    if mp_star < 0 or mn_star < 0 or kfp < 0 or kfn < 0 or mue < 0:
        return [10, 10, 10, 10, 10, 10]

    #meson equations of motion
    val = [gsig * (_scalar_density(kfp, mp_star) + _scalar_density(kfn, mn_star)) - msig**2*sigma - kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        - kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2 + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2 + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw * (_density(kfp) + _density(kfn)) + mw**2*omega + eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2 + 1/6*zeta0*gw**2*omega**3)
    val.append(-1/2 * grho * (_density(kfp) - _density(kfn)) + mrho**2*rho + etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #apply beta equilibrium
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1/2 * grho * rho - mub)
    val.append(sqrt(kfp**2 + mp_star**2) + gw * omega + 1/2 * grho * rho - mub + mue)
    #apply charge neutrality
    if mue > MUON_MASS:
        val.append(mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3)
    else:
        val.append(mue - kfp)
    return val

def _rmf_eom_charged(fields, mub, mue, constants): #rmf eoms for charge neutral nuclear matter
    #rmf model constants
    [msig,mw,mrho,mdelta,gsig,gw,grho,gdelta,kappa3,kappa4,lambdasigdelta,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #fields to solve for
    [sigma, omega, rho, delta, kfp, kfn] = fields

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma * gsig - delta * gdelta
    mn_star = NEUTRON_MASS - sigma * gsig + delta * gdelta

    #meson equations of motion
    val = [gsig * (_scalar_density(kfp, mp_star) + _scalar_density(kfn,mn_star)) - msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        - kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3 + 2*lambdasigdelta*sigma*delta**2 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2 + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2 + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(gdelta * (_scalar_density(kfp,mp_star) - _scalar_density(kfn,mn_star)) - mdelta**2*delta + 2*lambdasigdelta*sigma**2*delta)
    val.append(-gw*(_density(kfp)+_density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2 + 1/6*zeta0*gw**2*omega**3)
    val.append(-1/2 * grho * (_density(kfp) - _density(kfn)) + mrho**2*rho + etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #apply beta equilibrium
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1/2 * grho * rho - mub)
    val.append(sqrt(kfp**2 + mp_star**2) + gw * omega + 1/2 * grho * rho - mub + mue)

    return val

def _rmf_eom_charged_no_delta(fields, mub, mue, constants): #rmf eoms for charge neutral nuclear matter
    #rmf model constants
    [msig,mw,mrho,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #fields to solve for
    [sigma,omega,rho,kfp,kfn] = fields

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma*gsig
    mn_star = NEUTRON_MASS - sigma*gsig

    if mp_star < 0 or mn_star < 0 or kfp < 0 or kfn < 0:
        return [10, 10, 10, 10, 10, 10]

    #meson equations of motion
    val = [gsig * (_scalar_density(kfp, mp_star) + _scalar_density(kfn, mn_star)) - msig**2*sigma - kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        - kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2 + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2 + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw * (_density(kfp) + _density(kfn)) + mw**2*omega + eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2 + 1/6*zeta0*gw**2*omega**3)
    val.append(-1/2 * grho * (_density(kfp) - _density(kfn)) + mrho**2*rho + etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #apply beta equilibrium
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1/2 * grho * rho - mub)
    val.append(sqrt(kfp**2 + mp_star**2) + gw * omega + 1/2 * grho * rho - mub + mue)

    return val

def _rmf_eom_pnm(fields, mub, constants): #rmf eoms for charge neutral nuclear matter
    #rmf model constants
    [msig,mw,mrho,mdelta,gsig,gw,grho,gdelta,kappa3,kappa4,lambdasigdelta,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #fields to solve for
    [sigma, omega, rho, delta, kfn] = fields

    #effective nucleon masses
    mn_star = NEUTRON_MASS - sigma * gsig + delta * gdelta

    #meson equations of motion
    val = [gsig * (_scalar_density(kfn,mn_star)) - msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        - kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3 + 2*lambdasigdelta*sigma*delta**2 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2 + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2 + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(gdelta * (- _scalar_density(kfn,mn_star)) - mdelta**2*delta + 2*lambdasigdelta*sigma**2*delta)
    val.append(-gw*(_density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2 + 1/6*zeta0*gw**2*omega**3)
    val.append(-1/2 * grho * (- _density(kfn)) + mrho**2*rho + etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #apply beta equilibrium
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1/2 * grho * rho - mub)

    return val

def _rmf_eom_pnm_no_delta(fields, mub, constants): #rmf eoms for charge neutral nuclear matter
    #rmf model constants
    [msig,mw,mrho,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #fields to solve for
    [sigma,omega,rho,kfn] = fields

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma*gsig
    mn_star = NEUTRON_MASS - sigma*gsig

    if mp_star < 0 or mn_star < 0 or kfn < 0:
        return [10, 10, 10, 10]

    #meson equations of motion
    val = [gsig * (_scalar_density(kfn, mn_star)) - msig**2*sigma - kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        - kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2 + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2 + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw * ( _density(kfn)) + mw**2*omega + eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega + eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2 + 1/6*zeta0*gw**2*omega**3)
    val.append(-1/2 * grho * (- _density(kfn)) + mrho**2*rho + etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho + eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        + eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #apply beta equilibrium
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1/2 * grho * rho - mub)

    return val

def _rmf_energy_density(fields, constants, no_leptons = False): #energy_density in RMF nuclear phase, includes leptons
    [msig,mw,mrho,mdelta,gsig,gw,gdelta,kappa3,kappa4,lambdasigdelta,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = list(constants[:6]) + list(constants[7:])
    [sigma,omega,rho,delta,kfp,kfn,mue] = fields[0:7]

    #effective nucleon masses
    mp_star = PROTON_MASS - gsig * sigma - gdelta * delta
    mn_star = NEUTRON_MASS - gsig * sigma + gdelta * delta

    #energy density from nucleons and from mesons
    nucleon_energy_density = _fermion_energy_density(kfn, mn_star) + _fermion_energy_density(kfp, mp_star)
    meson_energy_density = (1/2 * (msig**2*sigma**2 + mw**2*omega**2 + mrho**2*rho**2 + mdelta**2*delta**2) + kappa3/(6*NUCLEON_MASS)*gsig*msig**2*sigma**3 + kappa4/(24*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**4
        - lambdasigdelta * sigma**2*delta**2 + zeta0/8*gw**2*omega**4 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*sigma*omega**2 + eta2/(4*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*sigma*rho**2 + eta1rho/(4*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho**2 + 3*eta2rho/(4*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho**2)
    
    #add energy density of leptons
    if no_leptons:
        return nucleon_energy_density + meson_energy_density
    return nucleon_energy_density + meson_energy_density + _lepton_energy_density(mue)
    
def _rmf_pressure(fields, constants, no_leptons = False): #pressure in the RMF nuclear phase, includes leptons
    [msig,mw,mrho,mdelta,gsig,gw,gdelta,kappa3,kappa4,lambdasigdelta,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = list(constants[:6]) + list(constants[7:])
    [sigma, omega, rho, delta, kfp, kfn, mue] = fields[0:7]

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma * gsig - gdelta * delta
    mn_star = NEUTRON_MASS - gsig * sigma + gdelta * delta

    #pressure from nucleons and mesons
    nucleon_pressure=_fermion_pressure(kfp, mp_star)+_fermion_pressure(kfn, mn_star)
    meson_pressure=(1/2 * (-msig**2*sigma**2 + mw**2*omega**2 + mrho**2*rho**2 - mdelta**2*delta**2) - kappa3/(6*NUCLEON_MASS)*gsig*msig**2*sigma**3 - kappa4/(24*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**4
        + lambdasigdelta*sigma**2*delta**2 + zeta0/24*gw**2*omega**4 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*sigma*omega**2 + eta2/(4*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*sigma*rho**2 + eta1rho/(4*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho**2 + eta2rho/(4*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho**2)
    
    #add pressure from leptons
    if no_leptons:
        return nucleon_pressure + meson_pressure
    return nucleon_pressure + meson_pressure + _lepton_pressure(mue)

def _rmf_energy_density_no_delta(fields, constants, no_leptons = False): #energy_density in RMF nuclear phase, includes leptons
    [msig,mw,mrho,gsig,gw,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = list(constants[:5]) + list(constants[6:])
    [sigma, omega, rho, kfp, kfn, mue] = fields[0:6]

    #effective nucleon masses
    mp_star = PROTON_MASS - gsig * sigma
    mn_star = NEUTRON_MASS - gsig * sigma

    #energy density of nucleons and mesons
    nucleon_energy_density = _fermion_energy_density(kfn, mn_star) + _fermion_energy_density(kfp, mp_star)
    meson_energy_density = (1/2 * (msig**2*sigma**2 + mw**2*omega**2 + mrho**2*rho**2) + kappa3/(6*NUCLEON_MASS)*gsig*msig**2*sigma**3 + kappa4/(24*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**4
        + zeta0/8*gw**2*omega**4 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*sigma*omega**2 + eta2/(4*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*sigma*rho**2 + eta1rho/(4*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho**2 + 3*eta2rho/(4*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho**2)
    
    #add energy density of leptons
    if no_leptons:
        return nucleon_energy_density + meson_energy_density
    return nucleon_energy_density + meson_energy_density + _lepton_energy_density(mue)
    
def _rmf_pressure_no_delta(fields, constants, no_leptons = False): #pressure in the RMF nuclear phase, includes leptons
    [msig,mw,mrho,gsig,gw,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = list(constants[:5]) + list(constants[6:])
    [sigma,omega,rho,kfp,kfn,mue] = fields[0:6]

    #effective nucleon masses
    mp_star = PROTON_MASS - sigma * gsig
    mn_star = NEUTRON_MASS - gsig * sigma

    #pressure from nucleons and mesons
    nucleon_pressure = _fermion_pressure(kfp, mp_star) + _fermion_pressure(kfn, mn_star)
    meson_pressure = (1/2 * (-msig**2*sigma**2 + mw**2*omega**2 + mrho**2*rho**2) - kappa3/(6*NUCLEON_MASS)*gsig*msig**2*sigma**3 - kappa4/(24*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**4
        + zeta0/24*gw**2*omega**4 + eta1/(2*NUCLEON_MASS)*gsig*mw**2*sigma*omega**2 + eta2/(4*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega**2
        + etarho/(2*NUCLEON_MASS)*gsig*mrho**2*sigma*rho**2 + eta1rho/(4*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho**2 + eta2rho/(4*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho**2)
    
    #add pressure from leptons
    if no_leptons:
        return nucleon_pressure + meson_pressure
    return nucleon_pressure + meson_pressure + _lepton_pressure(mue)
    
def _rmf_pressure_equil(mue, mub, neut_pressure, constants, fields_guess = (0.1, 0.1, -0.01, -0.01, 0.8, 0.8), no_delta = True):
    if mue[0] > fields_guess[3]:
        fields_guess = list(fields_guess[0:3]) + [mue[0],] + [fields_guess[4],]
    if no_delta:
        fields_temp = fsolve(_rmf_eom_charged_no_delta, fields_guess[0:3] + fields_guess[4:], args = (mub, mue[0], constants), xtol = 1e-12)
        pressure_temp = _rmf_pressure_no_delta(fields_temp, constants, no_leptons = True)
    else:
        fields_temp = fsolve(_rmf_eom_charged, fields_guess, args = (mub, mue[0], constants), xtol = 1e-12)
        pressure_temp = _rmf_pressure(fields_temp, constants, no_leptons = True)
    return pressure_temp - neut_pressure

class RMFModel:
    #initialize value of constants when the class is created
    def __init__(self,msig,mw,mrho,mdelta,gsig,gw,grho,gdelta,kappa3,kappa4,lambdasigdelta,zeta0,eta1,eta2,etarho,eta1rho,eta2rho):
        self.msig = msig
        self.mw = mw
        self.mrho = mrho
        self.mdelta = mdelta
        self.gsig = gsig
        self.gw = gw
        self.grho = grho
        self.gdelta = gdelta
        self.kappa3 = kappa3
        self.kappa4 = kappa4
        self.lambdasigdelta = lambdasigdelta
        self.zeta0 = zeta0
        self.eta1 = eta1
        self.eta2 = eta2
        self.etarho = etarho
        self.eta1rho = eta1rho
        self.eta2rho = eta2rho
    
    #calculates eos data given baryon density
    def eos_data_density(self, baryon_density, nuc_guess = (0.2,0.3,-0.1,-0.1,1,2,6,1)):
        #nuclear eom has been rewritten to use density as free variable to match skyrme
        #nuc_guess is in the form [sigma,omega,rho,delta,kfp,kfn,mub,mue]
        #use no_delta functions if this rmf model has no deltas
        if not self.gdelta == 0:
            #vector of constants to pass to RMF functions
            constants = [self.msig,self.mw,self.mrho,self.mdelta,self.gsig,self.gw,self.grho,self.gdelta,self.kappa3,self.kappa4,self.lambdasigdelta,self.zeta0,
                self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
            #solve equations of motion and store data
            fields_temp = fsolve(_rmf_eom_neutral, nuc_guess, args=(baryon_density, constants))
            kfp = fields_temp[4]; kfn = fields_temp[5]; mub = fields_temp[7]; mue = fields_temp[6]
            #calculate eos data from solution to equations of motion
            energy_density = _rmf_energy_density(fields_temp, constants)
            pressure = _rmf_pressure(fields_temp, constants)
            mp_star = PROTON_MASS - self.gsig * fields_temp[0] - self.gdelta * fields_temp[3]
            mn_star = NEUTRON_MASS - self.gsig * fields_temp[0] + self.gdelta * fields_temp[3]
        else:
            #vector of constants to pass to RMF functions
            constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
                self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
            #solve equations of motion and store data
            fields_temp = fsolve(_rmf_eom_neutral_no_delta, nuc_guess[0:3] + nuc_guess[4:], args = (baryon_density, constants))
            kfp = fields_temp[3]; kfn = fields_temp[4]; mub = fields_temp[6]; mue = fields_temp[5]
            #calculate eos data from solutions of equations of motion
            energy_density = _rmf_energy_density_no_delta(fields_temp, constants)
            pressure = _rmf_pressure_no_delta(fields_temp, constants)
            mp_star = PROTON_MASS - self.gsig * fields_temp[0]
            mn_star = NEUTRON_MASS - self.gsig * fields_temp[0]
        return [kfp, kfn, mub, mue, energy_density, pressure, mp_star, mn_star]
    
    #calculates eos data given mub
    def eos_data(self, mub, nuc_guess = (0.2, 0.1, -0.05, -0.05, 0.8, 2, 0.8)):
        #nuc_guess is in the form [sigma,omega,rho,delta,kfp,kfn,mub,mue]
        #use no_delta functions if there is no delta meson in this RMF model
        if not self.gdelta == 0:
            #vector of constants to pass to RMF functions
            constants = [self.msig,self.mw,self.mrho,self.mdelta,self.gsig,self.gw,self.grho,self.gdelta,self.kappa3,self.kappa4,self.lambdasigdelta,self.zeta0,
                self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
            #solve equations of motion and store data
            fields_temp = fsolve(_rmf_eom_neutral_mub, nuc_guess, args = (mub,constants))
            kfp = fields_temp[4]; kfn = fields_temp[5]; mue = fields_temp[6]
            #calculate eos data from solutions of equations of motion
            energy_density = _rmf_energy_density(fields_temp, constants)
            pressure = _rmf_pressure(fields_temp, constants)
            mp_star = PROTON_MASS - self.gsig * fields_temp[0] - self.gdelta * fields_temp[3]
            mn_star = NEUTRON_MASS - self.gsig * fields_temp[0] + self.gdelta * fields_temp[3]
        else:
            #vector of constants to pass to RMF functions
            constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
                self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
            #solve equations of motion and store data
            fields_temp = fsolve(_rmf_eom_neutral_mub_no_delta, nuc_guess[0:3] + nuc_guess[4:], args = (mub, constants))
            kfp = fields_temp[3]; kfn = fields_temp[4]; mue = fields_temp[5]
            #calculate eos data from solutions of equations of motion
            energy_density = _rmf_energy_density_no_delta(fields_temp, constants)
            pressure = _rmf_pressure_no_delta(fields_temp, constants)
            mp_star = PROTON_MASS - self.gsig * fields_temp[0]
            mn_star = NEUTRON_MASS - self.gsig * fields_temp[0]
        return [kfp, kfn, mue, energy_density, pressure, mp_star, mn_star]
    def eos_data_inner_crust(self, mub, nuc_guess = [0.01,0.01,-0.01,-0.01,0.3,0.5], nuc_guess_pnm = [0.2,0.2,-0.01,-0.01,0.5], mue_guess = 0.2):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]

        if self.gdelta == 0:
            pnm_fields = fsolve(_rmf_eom_pnm_no_delta, nuc_guess_pnm[:3] + [nuc_guess[4]], args = (mub, constants))
            pnm_pressure = _rmf_pressure_no_delta(list(pnm_fields[0:3]) + [0, pnm_fields[3]], constants, no_leptons = True)
            mue = fsolve(_rmf_pressure_equil, mue_guess, args = (mub, pnm_pressure, constants, nuc_guess[:3] + nuc_guess[4:], True), xtol = 1e-10)[0]
            pressure = pnm_pressure + _lepton_pressure(mue)
            fields_nucleus = fsolve(_rmf_eom_charged_no_delta, nuc_guess[:3] + nuc_guess[4:], args = (mub, mue, constants))

            kfn_pnm = pnm_fields[3]
            kfn_nucleus = fields_nucleus[4]
            kfp = fields_nucleus[3]

            no_prot_charge = -_lepton_density(mue)
            prot_charge = no_prot_charge + _density(kfp)
            nucleus_vol_frac = 1 / (1 - prot_charge / no_prot_charge)

            energy_density = _lepton_energy_density(mue) + (1 - nucleus_vol_frac) * _rmf_energy_density_no_delta(
                list(pnm_fields[0:3]) + [0, pnm_fields[3]], constants, no_leptons=True) + nucleus_vol_frac * _rmf_energy_density_no_delta(fields_nucleus, constants, no_leptons = True)
        else:
            pnm_fields = fsolve(_rmf_eom_pnm, nuc_guess_pnm, args = (mub, constants))
            pnm_pressure = _rmf_pressure(list(pnm_fields[0:4]) + [0, pnm_fields[4]], constants, no_leptons = True)
            mue = fsolve(_rmf_pressure_equil, mue_guess, args = (mub, pnm_pressure, constants, nuc_guess, False), xtol = 1e-10)[0]
            pressure = pnm_pressure + _lepton_pressure(mue)
            fields_nucleus = fsolve(_rmf_eom_charged, nuc_guess, args = (mub, mue, constants))

            kfn_pnm = pnm_fields[4]
            kfn_nucleus = fields_nucleus[5]
            kfp = fields_nucleus[4]

            no_prot_charge = -_lepton_density(mue)
            prot_charge = no_prot_charge + _density(kfp)
            nucleus_vol_frac = 1 / (1 - prot_charge / no_prot_charge)

            energy_density = _lepton_energy_density(mue) + (1 - nucleus_vol_frac) * _rmf_energy_density(
                list(pnm_fields[0:4]) + [0, pnm_fields[4]], constants, no_leptons=True) + nucleus_vol_frac * _rmf_energy_density(fields_nucleus, constants, no_leptons = True)

        return [kfp, kfn_nucleus, kfn_pnm, mue, energy_density, pressure, nucleus_vol_frac]
    def eos_data_outer_crust(self, mub, baryon_per_nuc, mue_xp_guess = (1e-3, 1e-2), rho_sat = 0.153, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
        binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants

        mue, xp = fsolve(_outer_crust_solver, mue_xp_guess, args = (mub, baryon_per_nuc, list(nuc_constants)))
        kfp = (3 * pi**2 * rho_sat * xp)**(1/3)
        kfn = (3 * pi**2 * rho_sat * (1 - xp))**(1/3)

        no_nuc_charge = -_lepton_density(mue)
        nuc_charge = _density(kfp) + no_nuc_charge
        nucleus_vol_frac = 1 / (1 - nuc_charge / no_nuc_charge)
        interaction_energy = -binding_energy + symm_energy * (1 - 2 * xp)**2 + \
            surf_energy / baryon_per_nuc**(1/3) + coul_energy * xp**2 * baryon_per_nuc**(2/3)
        energy_density = _lepton_energy_density(mue) + nucleus_vol_frac * (rho_sat * xp * (PROTON_MASS + 3 / (10 * PROTON_MASS) * kfp**2) 
            + rho_sat * (1 - xp) * (NEUTRON_MASS + 3 / (10 * NEUTRON_MASS) * kfn**2) + interaction_energy * rho_sat)
        pressure = _lepton_pressure(mue)

        return [kfp, kfn, mue, energy_density, pressure, nucleus_vol_frac]
    
#NEXT SECTION IS SKYRME MODEL FUNCTIONS AND CLASSES

def _skyrme_mub(proton_frac, baryon_density, constants):
    #constants is the skyrme model constants
    [t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3] = constants

    #these coefficients are used for calculation of asymmetric matter
    h2 = 2**(2-1) * (proton_frac**2 + (1-proton_frac)**2)
    h53 = 2**(5/3-1) * (proton_frac**(5/3) + (1-proton_frac)**(5/3))
    
    #rest mass and kinetic energy
    mass_term = NEUTRON_MASS
    first_term = 1 / (2 * NEUTRON_MASS) * (3 * pi**2 * baryon_density * (1 - proton_frac))**(2/3)
    #mean field potential
    second_term = t0 / 8 * (2 * (x0 + 2) - (2 * x0 + 1) * 2 * (1 - proton_frac)) * 2 * baryon_density
    #this term is density dependent many body effects
    third_term = 1/48 * (t31 * baryon_density**(1 + sigma1) * ((2 + sigma1) * 2*(x31+2) - (sigma1 * h2 + 2 * 2 * (1 - proton_frac)) * (2*x31 + 1))
                       + t32 * baryon_density**(1 + sigma2) * ((2 + sigma2) * 2*(x32+2) - (sigma2 * h2 + 2 * 2 * (1 - proton_frac)) * (2*x32 + 1))
                       + t33 * baryon_density**(1 + sigma3) * ((2 + sigma3) * 2*(x33+2) - (sigma3 * h2 + 2 * 2 * (1 - proton_frac)) * (2*x33 + 1)))
    #effective mass modification
    a = t1*(x1+2)+t2*(x2+2)
    b = 1/2*(t2*(2*x2+1)-t1*(2*x1+1))
    fourth_term = 3 / 40 * (3 * pi**2 / 2)**(2/3) * baryon_density**(5/3) * (
        a * (h53 + 5/3 * (2 * (1 - proton_frac))**(2/3)) + b * 8 / 3 * (2 * (1 - proton_frac))**(5/3))

    return mass_term + first_term + second_term + third_term + fourth_term 
    
def _skyrme_mue(proton_frac, baryon_density, constants):
    #constants is the skyrme model constants
    [t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3] = constants

    #rest mass and kinetic energy
    mass_term = NEUTRON_MASS - PROTON_MASS
    first_term = 1 / 2 * (3 * pi**2 * baryon_density)**(2/3) * ((1 - proton_frac)**(2/3) / NEUTRON_MASS - proton_frac**(2/3) / PROTON_MASS)
    #mean field potential
    second_term = t0 / 8 * ( - (2 * x0 + 1) * 2 * (1 - 2 * proton_frac)) * 2 * baryon_density
    #this term is density dependent many body effects
    third_term = 1/48 * (t31 * baryon_density**(1 + sigma1) * (- (2 * 2 * (1 - 2 * proton_frac)) * (2*x31 + 1))
                       + t32 * baryon_density**(1 + sigma2) * (- (2 * 2 * (1 - 2 * proton_frac)) * (2*x32 + 1))
                       + t33 * baryon_density**(1 + sigma3) * (- (2 * 2 * (1 - 2 * proton_frac)) * (2*x33 + 1)))
    #effective mass modification
    a = t1*(x1+2)+t2*(x2+2)
    b = 1/2*(t2*(2*x2+1)-t1*(2*x1+1))
    fourth_term = 3 / 40 * (3 * pi**2 / 2)**(2/3) * baryon_density**(5/3) * (
        a * 5/3 * ((2 * (1 - proton_frac))**(2/3) - (2 * proton_frac)**(2/3)) + b * 8 / 3 * ((2 * (1 - proton_frac))**(5/3) - (2 * proton_frac)**(5/3)))

    return mass_term + first_term + second_term + third_term + fourth_term

def _skyrme_energy_density(proton_frac, baryon_density, constants):
    #constants is the skyrme model constants
    [t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3] = constants

    #these coefficients are used for calculation of asymmetric matter
    h2 = 2**(2-1) * (proton_frac**2 + (1-proton_frac)**2)
    h53 = 2**(5/3-1) * (proton_frac**(5/3) + (1-proton_frac)**(5/3))
    h83 = 2**(8/3-1) * (proton_frac**(8/3) + (1-proton_frac)**(8/3))
    
    #rest mass and kinetic energy
    rest_mass_ed = baryon_density * (proton_frac * PROTON_MASS + (1 - proton_frac) * NEUTRON_MASS)
    first_term = 3 / 10 * (3 * pi**2)**(2/3) * baryon_density**(5/3) * (proton_frac**(5/3) / PROTON_MASS + (1 - proton_frac)**(5/3) / NEUTRON_MASS)
    #mean field potential
    second_term = t0 / 8 * (2 * (x0 + 2) - (2 * x0 + 1) * h2) * baryon_density**2
    #this term is density dependent many body effects
    third_term = 1/48 * (t31 * baryon_density**(2+sigma1) * (2*(x31+2) - (2*x31 + 1) * h2) 
                        + t32 * baryon_density**(2+sigma2) * (2*(x32+2) - (2*x32 + 1) * h2)
                        + t33 * baryon_density**(2+sigma3) * (2*(x33+2) - (2*x33+1) * h2))
    #effective mass modification
    a = t1*(x1+2)+t2*(x2+2)
    b = 1/2*(t2*(2*x2+1)-t1*(2*x1+1))
    fourth_term = 3 / 40 * (3 * pi**2 / 2)**(2/3) * baryon_density**(8/3) * (a * h53 + b * h83)

    return rest_mass_ed + first_term + second_term + third_term + fourth_term 

def _skyrme_pressure(proton_frac, baryon_density, constants):
    #constants is the Skyrme model constants
    [t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3] = constants

    #these coefficients are used for asymmetric matter
    h53 = 2**(5/3-1) * (proton_frac**(5/3) + (1-proton_frac)**(5/3))
    h2 = 2 * (proton_frac**2 + (1-proton_frac)**2)
    h83 = 2**(8/3-1) * (proton_frac**(8/3) + (1-proton_frac)**(8/3))

    #kinetic energy
    first_term = 1 / 5 * (3 * pi**2)**(2/3) * baryon_density**(5/3) * (proton_frac**(5/3) / PROTON_MASS + (1 - proton_frac)**(5/3) / NEUTRON_MASS)
    #mean field potential
    second_term = t0 / 8 * baryon_density**2 * (2*(x0+2) - (2*x0+1)*h2)
    #this terms includes density dependence in the potential due to many body effects
    third_term = 1/48 * (t31 * (sigma1+1) * baryon_density**(sigma1+2) * (2*(x31+2) - (2*x31+1) * h2) 
                        + t32 * (sigma2+1) * baryon_density**(sigma2+2) * (2*(x32+2) - (2*x32+1) * h2)
                        + t33 * (sigma3+1) * baryon_density**(sigma3+2) * (2*(x33+2) - (2*x33+1) * h2))
    #effective mass modification
    a = t1*(x1+2)+t2*(x2+2)
    b = 1/2*(t2*(2*x2+1)-t1*(2*x1+1))
    fourth_term = 1 / 8 * (3 * pi**2 / 2)**(2/3) * baryon_density**(8/3) * (a * h53 + b * h83)

    return first_term + second_term + third_term + fourth_term
    
def _mue_solver(mue, kfp):
    if mue > MUON_MASS:
        return mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3
    return mue - kfp

def _skyrme_proton_frac_density_solver(inputs, mub, constants):
    [proton_frac, baryon_density]=inputs
    if proton_frac<0 or proton_frac>1 or baryon_density<0:
        return [10,10]
    
    kfp = (3*pi**2 * baryon_density * proton_frac)**(1/3)
    mue = fsolve(_mue_solver, kfp, args = (kfp))[0]

    val = [mue - _skyrme_mue(proton_frac, baryon_density, constants)]
    val.append(mub - _skyrme_mub(proton_frac, baryon_density, constants))
    return val  

def _skyrme_density_solver(baryon_density, mub, proton_frac, constants):
    if baryon_density < 0:
        return exp(-baryon_density)
    return mub - _skyrme_mub(proton_frac, baryon_density, constants)

def _skyrme_pressure_equil(proton_frac, mub, neut_pressure, constants, yp_min = 0):
    yp = proton_frac
    if yp < yp_min:
        return exp(yp_min - yp)
    nb = fsolve(_skyrme_density_solver, 0.15, args = (mub, yp, constants))[0]
    pressure = _skyrme_pressure(yp, nb, constants)
    return 1e5 * (pressure - neut_pressure)

class SkyrmeModel:
    #initialize values of constants of Skyrme model
    def __init__(self,t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3):
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t31 = t31
        self.t32 = t32
        self.t33 = t33
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x31 = x31
        self.x32 = x32
        self.x33 = x33
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
    #calculates equation of state data given mub
    def eos_data(self, mub, proton_frac_guess = (0.2, 0.3)):
        constants = [self.t0,self.t1,self.t2,self.t31,self.t32,self.t33,self.x0,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]

        [proton_frac, baryon_density] = fsolve(_skyrme_proton_frac_density_solver, 
            proton_frac_guess, args = (mub, constants))
        kfp = (3*pi**2 * baryon_density * proton_frac)**(1/3)
        kfn = (3*pi**2 * baryon_density * (1 - proton_frac))**(1/3)
        mue = fsolve(_mue_solver, kfp, args = (kfp))[0]

        pressure = _skyrme_pressure(proton_frac, baryon_density, constants) + _lepton_pressure(mue)
        energy_density = _skyrme_energy_density(proton_frac, baryon_density, constants) + _lepton_energy_density(mue)

        return [kfp, kfn, mue, energy_density, pressure]
        
    def eos_data_inner_crust(self, mub, nb_guess = 0.02, yp_guess = 0.2, yp_min = 0):
        #ftheta_val must be 0 or ftheta_min if specified
        constants = [self.t0,self.t1,self.t2,self.t31,self.t32,self.t33,self.x0,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]

        pnm_nb = fsolve(_skyrme_density_solver, nb_guess, args = (mub, 0, constants))[0]
        pnm_pressure = _skyrme_pressure(0, pnm_nb, constants)
        yp = fsolve(_skyrme_pressure_equil, yp_guess, args = (mub, pnm_pressure, constants, yp_min))
        nucleus_nb = fsolve(_skyrme_density_solver, 0.2, args = (mub, yp, constants))[0]

        mue = _skyrme_mue(yp, nucleus_nb, 1, constants)
        kfp = (3 * pi**2 * nucleus_nb * yp)**(1/3)
        kfn_nucleus = (3 * pi**2 * nucleus_nb * (1 - yp))**(1/3)
        kfn_pnm = (3 * pi**2 * pnm_nb)**(1/3)
        
        charge_pnm = -_lepton_density(mue)
        charge_nucleus = _density(kfp) - _lepton_density(mue)
        nucleus_vol_frac = 1 / (1 - charge_nucleus / charge_pnm)

        energy_density = nucleus_vol_frac * _skyrme_energy_density(yp, nucleus_nb, constants) + (
            1 - nucleus_vol_frac) * _skyrme_energy_density(0, pnm_nb, constants) + _lepton_energy_density(mue)
        pressure = pnm_pressure + _lepton_pressure(mue)

        return [kfp, kfn_nucleus, kfn_pnm, energy_density, pressure]
    
    def eos_data_outer_crust(self, mub, baryon_per_nuc, mue_xp_guess = (1e-3, 1e-2), rho_sat = 0.153, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
        binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants

        mue, xp = fsolve(_outer_crust_solver, mue_xp_guess, args = (mub, baryon_per_nuc, list(nuc_constants)))
        kfp = (3 * pi**2 * rho_sat * xp)**(1/3)
        kfn = (3 * pi**2 * rho_sat * (1 - xp))**(1/3)

        no_nuc_charge = -_lepton_density(mue)
        nuc_charge = _density(kfp) + no_nuc_charge
        nucleus_vol_frac = 1 / (1 - nuc_charge / no_nuc_charge)
        interaction_energy = -binding_energy + symm_energy * (1 - 2 * xp)**2 + \
            surf_energy / baryon_per_nuc**(1/3) + coul_energy * xp**2 * baryon_per_nuc**(2/3)
        energy_density = _lepton_energy_density(mue) + nucleus_vol_frac * (rho_sat * xp * (PROTON_MASS + 3 / (10 * PROTON_MASS) * kfp**2) 
            + rho_sat * (1 - xp) * (NEUTRON_MASS + 3 / (10 * NEUTRON_MASS) * kfn**2) + interaction_energy * rho_sat)
        pressure = _lepton_pressure(mue)

        return [kfp, kfn, mue, energy_density, pressure, nucleus_vol_frac]

#NEXT SECTION IS A WHOLE BUNCH OF RMF AND SKYRME MODELS 

#prints list of constants names
def rmf_constants_names():
    print('bsp\niufsu_star\niufsu\ng1\ng2\ntm1\ntm1_star\nln1\nnl3\nglendenning1\nglendenning5\nglendenning9\nnbl\nomeg1\nomeg2\nomeg3\nConstants names are \'[name]_constants\'')
    return

#list of rmf constants
bsp_constants = RMFModel(msig=NUCLEON_MASS*0.5383,mw=NUCLEON_MASS*0.8333,mrho=NUCLEON_MASS*0.82,mdelta=NUCLEON_MASS,gsig=4*pi*0.8764,gw=4*pi*1.1481,grho=4*pi*1.0508,gdelta=0,kappa3=1.0681,kappa4=14.9857,
    lambdasigdelta=0,zeta0=0,eta1=0.0872,eta2=3.1265,etarho=0,eta1rho=0,eta2rho=53.7642)

iufsu_star_constants = RMFModel(msig=NUCLEON_MASS*0.543,mw=NUCLEON_MASS*0.8331,mrho=NUCLEON_MASS*0.8198,mdelta=NUCLEON_MASS,gsig=4*pi*0.8379,gw=4*pi*1.0666,grho=4*pi*0.9889,gdelta=0,kappa3=1.1418,
    kappa4=1.0328,lambdasigdelta=0,zeta0=5.3895,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=41.3066)

iufsu_constants = RMFModel(msig=NUCLEON_MASS*0.5234,mw=NUCLEON_MASS*0.8333,mrho=NUCLEON_MASS*0.8216,mdelta=NUCLEON_MASS,gsig=4*pi*0.7935,gw=4*pi*1.0371,grho=4*pi*1.0815,gdelta=0,kappa3=1.3066,
    kappa4=0.1074,lambdasigdelta=0,zeta0=5.0951,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=51.4681)

g1_constants = RMFModel(msig=NUCLEON_MASS*0.5396,mw=NUCLEON_MASS*0.8328,mrho=NUCLEON_MASS*0.82,mdelta=NUCLEON_MASS,gsig=4*pi*0.7853,gw=4*pi*0.9651,grho=4*pi*0.6984,gdelta=0,kappa3=2.2067,
    kappa4=-10.09,lambdasigdelta=0,zeta0=3.5249,eta1=0.0706,eta2=-0.9616,etarho=-0.2722,eta1rho=0,eta2rho=0)

g2_constants = RMFModel(msig=NUCLEON_MASS*0.5541,mw=NUCLEON_MASS*0.8328,mrho=NUCLEON_MASS*0.82,mdelta=NUCLEON_MASS,gsig=4*pi*0.8352,gw=4*pi*1.0156,grho=4*pi*0.7547,gdelta=0,kappa3=3.2467,
    kappa4=0.6315,lambdasigdelta=0,zeta0=2.6416,eta1=0.6499,eta2=0.1098,etarho=0.3901,eta1rho=0,eta2rho=0)

tm1_constants = RMFModel(msig=511.2/197.3,mw=783/197.3,mrho=770/197.3,mdelta=NUCLEON_MASS,gsig=10.0289,gw=12.6139,grho=4.6322,gdelta=0,kappa3=2*NUCLEON_MASS/(9.6959*(510/197.3)**2)*7.2325,
    kappa4=6*NUCLEON_MASS**2/(9.6959**2*(510/197.3)**2)*0.6183,lambdasigdelta=0,zeta0=6/(12.6139**2)*731.3075,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

tm1_star_constants = RMFModel(msig=NUCLEON_MASS*0.545,mw=NUCLEON_MASS*0.8348,mrho=NUCLEON_MASS*0.8209,mdelta=NUCLEON_MASS,gsig=4*pi*0.893,gw=4*pi*1.192,grho=4*pi*0.796,gdelta=0,kappa3=2.513,
    kappa4=8.97,lambdasigdelta=0,zeta0=3.6,eta1=1.1,eta2=0.1,etarho=0.45, eta1rho=0,eta2rho=0)

ln1_constants = RMFModel(msig=492/197.3,mw=795.359/197.3,mrho=763/197.3,mdelta=NUCLEON_MASS,gsig=10.138,gw=13.285,grho=4.6322,gdelta=0,kappa3=2*NUCLEON_MASS/(9.6959*(510/197.3)**2)*12.172,
    kappa4=6*NUCLEON_MASS**2/(9.6959**2*(510/197.3)**2)*(-36.259),lambdasigdelta=0,zeta0=0,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

nl3_constants = RMFModel(msig=NUCLEON_MASS*0.5412,mw=NUCLEON_MASS*0.8333,mrho=NUCLEON_MASS*0.8126,mdelta=NUCLEON_MASS,gsig=4*pi*0.8131,gw=4*pi*1.024,grho=4*pi*0.7121,gdelta=0,kappa3=1.4661,
    kappa4=-5.6718,lambdasigdelta=0,zeta0=0,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

glendenning1_constants = RMFModel(msig=NUCLEON_MASS*0.54,mw=NUCLEON_MASS*0.833,mrho=NUCLEON_MASS*8.2,mdelta=NUCLEON_MASS,gsig=NUCLEON_MASS*0.54*sqrt(12.684),gw=NUCLEON_MASS*0.833*sqrt(7.148),grho=NUCLEON_MASS*0.82*sqrt(4.41),gdelta=0,
    kappa3=2*NUCLEON_MASS**2*12.684*0.00561,kappa4=6*NUCLEON_MASS**2*12.684*(-0.006986),lambdasigdelta=0,zeta0=0,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

glendenning5_constants = RMFModel(msig=NUCLEON_MASS*0.54,mw=NUCLEON_MASS*0.833,mrho=NUCLEON_MASS*8.2,mdelta=NUCLEON_MASS,gsig=NUCLEON_MASS*0.54*sqrt(10.727),gw=NUCLEON_MASS*0.833*sqrt(5.696),grho=NUCLEON_MASS*0.82*sqrt(4.656),gdelta=0,
    kappa3=2*NUCLEON_MASS**2*12.684*0.006275,kappa4=6*NUCLEON_MASS**2*12.684*(-0.003409),lambdasigdelta=0,zeta0=0,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

glendenning9_constants = RMFModel(msig=NUCLEON_MASS*0.54,mw=NUCLEON_MASS*0.833,mrho=NUCLEON_MASS*0.82,mdelta=NUCLEON_MASS,gsig=NUCLEON_MASS*0.54*sqrt(8.403),gw=NUCLEON_MASS*0.833*sqrt(4.233),grho=NUCLEON_MASS*0.82*sqrt(4.876),gdelta=0,
    kappa3=2*NUCLEON_MASS**2*12.684*0.00248,kappa4=6*NUCLEON_MASS**2*12.684*(0.027997),lambdasigdelta=0,zeta0=0,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

nbl_constants = RMFModel(msig=510/197.3,mw=786/197.3,mrho=770/197.3,mdelta=NUCLEON_MASS,gsig=9.6959,gw=12.5889,grho=8.544,gdelta=0,kappa3=2*NUCLEON_MASS/(9.6959*(510/197.3)**2)*2.03,
    kappa4=6*NUCLEON_MASS**2/(9.6959**2*(510/197.3)**2)*1.666,lambdasigdelta=0,zeta0=0,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

omeg1_constants = RMFModel(msig=497.825/197.3,mw=782.66/197.3,mrho=775.26/197.3,mdelta=980/197.3,gsig=sqrt(99.645),gw=sqrt(166.268),grho=sqrt(44.591),gdelta=sqrt(30),
    kappa3=2*NUCLEON_MASS*7.824/(sqrt(99.645)*(497.825/197.3)**2),kappa4=6*NUCLEON_MASS**2*(-1.115)/(99.645*(497.825/197.3)**2),lambdasigdelta=95,zeta0=6*100/166.268,eta1=0,eta2=0,
    etarho=0,eta1rho=0,eta2rho=4*NUCLEON_MASS**2*75.677/(166.268*(775.26/197.3)**2))

omeg2_constants = RMFModel(msig=497.82/197.3,mw=782.66/197.3,mrho=775.26/197.3,mdelta=980/197.3,gsig=sqrt(99.641),gw=sqrt(166.269),grho=sqrt(44.364),gdelta=sqrt(20),
    kappa3=2*NUCLEON_MASS*7.823/(sqrt(99.641)*(497.82/197.3)**2),kappa4=6*NUCLEON_MASS**2*(-1.113)/(99.641*(497.82/197.3)**2),lambdasigdelta=85,zeta0=6*100/166.269,eta1=0,eta2=0,
    etarho=0,eta1rho=0,eta2rho=4*NUCLEON_MASS**2*288.859/(166.269*(775.26/197.3)**2))

omeg3_constants = RMFModel(msig=498.015/197.3,mw=782.66/197.3,mrho=775.26/197.3,mdelta=980/197.3,gsig=sqrt(99.713),gw=sqrt(166.272),grho=sqrt(57.55),gdelta=sqrt(15),
    kappa3=2*NUCLEON_MASS*7.827/(sqrt(99.713)*(498.015/197.3)**2),kappa4=6*NUCLEON_MASS**2*(-1.105)/(99.713*(498.015/197.3)**2),lambdasigdelta=70,zeta0=6*100/166.272,eta1=0,eta2=0,
    etarho=0,eta1rho=0,eta2rho=4*NUCLEON_MASS**2*909.825/(166.272*(775.26/197.3)**2))

qmc_rmf2_constants = RMFModel(msig = 491.5 / 197.3, mw = 782.5 / 197.3, mrho = 763 / 197.3, mdelta = NUCLEON_MASS, gsig = 7.54,
    gw = 8.43, grho = 11.24, gdelta = 0, kappa3 = 2 * (938 / 491.5)**2 * 7.54**2 * 0.0073, kappa4 = 6 * 938**2 / 491.5**2 * 7.54**2 * 0.0035,
    lambdasigdelta = 0, zeta0 = 0, eta1 = 0, eta2 = 0, etarho = 0, eta1rho = 0, eta2rho = 4 * 938**2 / 763**2 * 11.24**2 / 8.43**2 * 7.89)

#prints list of skyrme constant names
def skyrme_constants_names():
    print('gsk1\nskt3\nlns\nkde0v1\nskxs20\nnrapr\nsqmc650\ntov_min\nsly4\nsv\nskl4\nqmc_rmf2\nConstants names are \'[name]_constants\'')
    return

#list of skyrme constants
gsk1_constants = SkyrmeModel(t0=-1855.5/197.3,t1=397.2/197.3,t2=264/197.3,t31=13858/197.3,t32=-2694.1/197.3,t33=-319.9/197.3,x0=0.12,x1=-1.76,
    x2=-1.81,x31=0.13,x32=-1.19,x33=-0.46,sigma1=0.33,sigma2=0.67,sigma3=1)

skt3_constants = SkyrmeModel(t0=-1791.8/197.3,t1=298.5/197.3,t2=-99.5/197.3,t31=12794/197.3,t32=0,t33=0,x0=0.14,x1=-1,x2=1,x31=0.08,
    x32=0,x33=0,sigma1=0.33,sigma2=0,sigma3=0)

lns_constants = SkyrmeModel(t0=-2485/197.3,t1=266.7/197.3,t2=-337.1/197.3,t31=14588.2/197.3,t32=0,t33=0,x0=0.06,x1=0.66,x2=-0.95,x31=-0.03,x32=0,x33=0,sigma1=0.17,sigma2=0,sigma3=0)

kde0v1_constants = SkyrmeModel(t0=-2553.1/197.3,t1=411.7/197.3,t2=-419.9/197.3,t31=14603.6/197.3,t32=0,t33=0,x0=0.65,x1=-0.35,x2=-0.93,x31=0.95,x32=0,x33=0,sigma1=0.17,sigma2=0,sigma3=0)

skxs20_constants = SkyrmeModel(t0=-2885.2/197.3,t1=302.7/197.3,t2=-323.4/197.3,t31=18237.5/197.3,t32=0,t33=0,x0=0.14,x1=-0.26,x2=-0.61,x31=0.05,x32=0,x33=0,sigma1=0.17,sigma2=0,sigma3=0)

gsk2_constants = SkyrmeModel(t0=-1856/197.3,t1=393.1/197.3,t2=266.1/197.3,t31=13842.9/197.3,t32=-2689.7/197.3,t33=0,x0=0.09,x1=-0.72,x2=-1.84,x31=-0.1,x32=-0.35,x33=0,sigma1=0.33,sigma2=0.67,sigma3=0)

nrapr_constants = SkyrmeModel(t0=-2719.7/197.3,t1=417.6/197.3,t2=-66.7/197.3,t31=15042/197.3,t32=0,t33=0,x0=0.16,x1=-0.05,x2=0.03,x31=0.14,x32=0,x33=0,sigma1=0.14,sigma2=0,sigma3=0)

sqmc650_constants = SkyrmeModel(t0=-2462.7/197.3,t1=436.1/197.3,t2=-151.9/197.3,t31=14154.5/197.3,t32=0,t33=0,x0=0.13,x1=0,x2=0,x31=0,x32=0,x33=0,sigma1=0.17,sigma2=0,sigma3=0)

tov_min_constants = SkyrmeModel(t0=-2129.735/197.3,t1=305.398/197.3,t2=362.532/197.3,t31=13957.064/197.3,t32=0,t33=0,x0=0.169949,
    x1=-3.39948,x2=-1.782177,x31=0.402634,x32=0,x33=0,sigma1=0.2504,sigma2=0,sigma3=0)

sly4_constants = SkyrmeModel(t0=-2488.91/197.3,t1=486.82/197.3,t2=-546.39/197.3,t31=13777/197.3,t32=0,t33=0,x0=0.834,x1=-0.344,x2=-1,x31=1.354,x32=0,x33=0,sigma1=1/6,sigma2=0,sigma3=0)

sv_constants = SkyrmeModel(t0=-1248.29/197.3,t1=970.56/197.3,t2=107.22/197.3,t31=0,t32=0,t33=0,x0=-0.17,x1=0,x2=0,x31=0,x32=0,x33=0,sigma1=0,sigma2=0,sigma3=0)

skl4_constants = SkyrmeModel(t0=-1855.8/197.3,t1=473.8/197.3,t2=1006.9/197.3,t31=9703/197.3,t32=0,t33=0,x0=0.41,x1=-2.89,x2=-1.33,x31=1.15,x32=0,x33=0,sigma1=1/4,sigma2=0,sigma3=0)

#NEXT SECTION IS BPS EQUATION OF STATE TABULATED DATA

#data for BPS equation of state
_bps_raw = []

_bps_raw.append([6.793866120000000E-005, 3.631720090000000E-007, 8.000000000000000E-002])
_bps_raw.append([6.367101020000000E-005, 3.253464300000000E-007, 7.500000000000000E-002])
_bps_raw.append([5.940584100000000E-005, 2.892898000000000E-007, 7.000000000000001E-002])
_bps_raw.append([5.514316700000000E-005, 2.554833620000000E-007, 6.500000000000000E-002])
_bps_raw.append([5.088301060000000E-005, 2.238381650000000E-007, 6.000000000000000E-002])
_bps_raw.append([4.662540540000000E-005, 1.942702640000000E-007, 5.500000000000000E-002])
_bps_raw.append([4.237039810000000E-005, 1.667020910000000E-007, 5.000000000000000E-002])
_bps_raw.append([3.811805260000000E-005, 1.410644700000000E-007, 4.500000000000000E-002])
_bps_raw.append([3.386845390000000E-005, 1.172996210000000E-007, 4.000000000000000E-002])
_bps_raw.append([2.962171490000000E-005, 9.536578390000000E-008, 3.500000000000000E-002])
_bps_raw.append([2.537798550000000E-005, 7.524456380000000E-008, 3.000000000000000E-002])
_bps_raw.append([2.113746650000000E-005, 5.695326890000000E-008, 2.500000000000000E-002])
_bps_raw.append([1.690043200000000E-005, 4.056747240000000E-008, 2.000000000000000E-002])
_bps_raw.append([1.266726710000000E-005, 2.626978270000000E-008, 1.500000000000000E-002])
_bps_raw.append([8.438540529999999E-006, 1.451397180000000E-008, 1.000000000000000E-002])
_bps_raw.append([4.215165390000000E-006, 7.047787820000000E-009, 5.000000010000000E-003])
_bps_raw.append([7.355000090000000E-007, 7.824300100000000E-010, 8.785999960000000E-004])
_bps_raw.append([5.979999860000000E-007, 6.811999850000000E-010, 7.143000260000000E-004])
_bps_raw.append([4.863999830000000E-007, 6.056999900000000E-010, 5.812000020000000E-004])
_bps_raw.append([3.982000010000000E-007, 5.498300150000000E-010, 4.758999860000000E-004])
_bps_raw.append([3.305000010000000E-007, 5.088299780000000E-010, 3.951000110000000E-004])
_bps_raw.append([2.614000040000000E-007, 4.671100170000000E-010, 3.125999940000000E-004])
_bps_raw.append([2.230000010000000E-007, 4.412699930000000E-010, 2.669999960000000E-004])
_bps_raw.append([2.149500060000000E-007, 4.365199870000000E-010, 2.572000080000000E-004])
_bps_raw.append([2.094000000000000E-007, 4.383599870000000E-010, 2.505999870000000E-004])
_bps_raw.append([1.662500040000000E-007, 3.252799970000000E-010, 1.990000020000000E-004])
_bps_raw.append([1.319999970000000E-007, 2.501699960000000E-010, 1.580999960000000E-004])
_bps_raw.append([1.048000020000000E-007, 1.839999960000000E-010, 1.255999960000000E-004])
_bps_raw.append([9.219999699999999E-008, 1.617399960000000E-010, 1.104999970000000E-004])
_bps_raw.append([8.319999980000000E-008, 1.452500060000000E-010, 9.975999999999999E-005])
_bps_raw.append([6.610000010000000E-008, 1.137000010000000E-010, 7.924000240000001E-005])
_bps_raw.append([5.245000170000000E-008, 8.361199930000000E-011, 6.294000200000000E-005])
_bps_raw.append([4.166000170000000E-008, 6.152099660000000E-011, 4.999999870000000E-005])
_bps_raw.append([3.308500140000000E-008, 4.523999920000000E-011, 3.971999830000000E-005])
_bps_raw.append([2.626999950000000E-008, 3.327199970000000E-011, 3.155000010000000E-005])
_bps_raw.append([2.085999990000000E-008, 2.588300060000000E-011, 2.506000060000000E-005])
_bps_raw.append([1.656500000000000E-008, 1.903800030000000E-011, 1.990000060000000E-005])
_bps_raw.append([1.315499980000000E-008, 1.399899990000000E-011, 1.580999920000000E-005])
_bps_raw.append([1.045000000000000E-008, 1.083899980000000E-011, 1.256000000000000E-005])
_bps_raw.append([8.295000240000000E-009, 7.969700420000000E-012, 9.976000000000001E-006])
_bps_raw.append([6.590000100000000E-009, 5.861300160000000E-012, 7.923999870000000E-006])
_bps_raw.append([5.229999990000000E-009, 4.307599850000000E-012, 6.293999830000000E-006])
_bps_raw.append([4.155999990000000E-009, 3.166600010000000E-012, 4.999999870000000E-006])
_bps_raw.append([3.300500100000000E-009, 2.439599950000000E-012, 3.971999830000000E-006])
_bps_raw.append([2.081999910000000E-009, 1.317699960000000E-012, 2.506000100000000E-006])
_bps_raw.append([1.313000040000000E-009, 7.114000140000000E-013, 1.580999990000000E-006])
_bps_raw.append([8.284999910000000E-010, 3.836700110000000E-013, 9.975999549999999E-007])
_bps_raw.append([6.579999880000000E-010, 2.816500020000000E-013, 7.923999870000000E-007])
_bps_raw.append([5.224999880000000E-010, 2.309299940000000E-013, 6.294000060000000E-007])
_bps_raw.append([4.150499940000000E-010, 1.694100050000000E-013, 4.999999990000000E-007])
_bps_raw.append([2.618499860000000E-010, 9.110699930000000E-014, 3.154999890000000E-007])
_bps_raw.append([1.652000060000000E-010, 4.886999960000000E-014, 1.989999990000000E-007])
_bps_raw.append([1.311999950000000E-010, 3.733799840000000E-014, 1.580999940000000E-007])
_bps_raw.append([6.575000270000000E-011, 1.456399970000000E-014, 7.924000300000000E-008])
_bps_raw.append([3.294500090000000E-011, 5.626399830000000E-015, 3.971999970000000E-008])
_bps_raw.append([1.651000000000000E-011, 2.143699990000000E-015, 1.989999990000000E-008])
_bps_raw.append([8.274999610000000E-012, 8.025700170000000E-016, 9.975999580000000E-009])
_bps_raw.append([4.146499980000000E-012, 2.941299940000000E-016, 4.999999970000000E-009])
_bps_raw.append([3.294000060000000E-012, 2.187300060000000E-016, 3.972000060000000E-009])
_bps_raw.append([1.311000030000000E-012, 5.455800090000000E-017, 1.581000000000000E-009])
_bps_raw.append([5.219999780000000E-013, 1.296399970000000E-017, 6.293999770000000E-010])
_bps_raw.append([2.078000040000000E-013, 2.945199940000000E-018, 2.506000130000000E-010])
_bps_raw.append([8.270000080000000E-014, 6.437299960000000E-019, 9.975999890000000E-011])
_bps_raw.append([3.293499910000000E-014, 1.359599940000000E-019, 3.972000160000000E-011])
_bps_raw.append([1.310999990000000E-014, 2.778500000000000E-020, 1.580999910000000E-011])
_bps_raw.append([5.220000200000000E-015, 5.449599970000000E-021, 6.295000110000000E-012])
_bps_raw.append([5.750000050000000E-016, 1.062600060000000E-022, 6.930000190000000E-013])
_bps_raw.append([1.059999990000000E-016, 3.255000040000000E-024, 1.270000000000000E-013])
_bps_raw.append([2.255000060000000E-017, 9.507800290000000E-026, 2.719999990000000E-014])
_bps_raw.append([8.200000230000000E-018, 7.829899770000000E-027, 9.899999960000001E-015])
_bps_raw.append([5.800000200000000E-018, 6.767299900000000E-028, 6.990000060000000E-015])
_bps_raw.append([4.074999820000000E-018, 5.648700170000000E-029, 4.910000120000000E-015])
_bps_raw.append([3.950000010000000E-018, 5.648699860000000E-030, 4.759999900000000E-015])
_bps_raw.append([3.930000020000000E-018, 5.648699860000000E-031, 4.730000110000000E-015])

#change units and reorganize bps raw data
bps_pressure = [elt[1] / (8.96057e-7*197.3) for elt in _bps_raw]
bps_energy_density = [elt[0] / (8.96057e-7*197.3) for elt in _bps_raw]
bps_number_density = [elt[2] for elt in _bps_raw]
#reverse order so that bps eos goes from low to high density
bps_pressure.reverse()
bps_energy_density.reverse()
bps_number_density.reverse()
bps_mub = list((array(bps_pressure) + array(bps_energy_density)) / array(bps_number_density))

bps_fit_consts = [1.54223623e-05, 7.77095294e-04, 1.47191494e-02, 1.25172664e-01, 5.09709458e+00]
bps_mub_fit = [elt**4 * bps_fit_consts[0] + elt**3 * bps_fit_consts[1] + elt**2 * bps_fit_consts[2] 
               + elt * bps_fit_consts[3] + bps_fit_consts[4] for elt in [log10(elt) for elt in bps_pressure[11:58]]] + bps_mub[58:]
bps_pressure_fit = bps_pressure[11:]
bps_energy_density_fit = bps_energy_density[11:]
bps_number_density_fit = bps_number_density[11:]