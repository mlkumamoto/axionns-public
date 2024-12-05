#arXiv:2410.21590

from numpy import arcsinh, arctanh, floor
from math import pi, sqrt, exp, log
from scipy.optimize import fsolve, minimize_scalar

#all quantities are in units of fm

NEUTRON_MASS = 939.57/197.3
PROTON_MASS = 938.27/197.3
NUCLEON_MASS = (NEUTRON_MASS + PROTON_MASS) / 2
MUON_MASS = 105.7/197.3
ELECTRON_MASS = 0.511 / 197.3
F_PION = 94/197.3 
PION_MASS = 134/197.3
UP_MASS = 2.2/197.3
DOWN_MASS = 4.7/197.3
FTHETA_MIN = (DOWN_MASS - UP_MASS) / (DOWN_MASS + UP_MASS)
FINE_STRUCTURE = 1 / 137

def density(x): return x**3/(3*pi**2)

def lepton_density(mue):
    if mue > MUON_MASS:
        return density(mue) + density(sqrt(mue**2 - MUON_MASS**2))
    elif mue > 50 * ELECTRON_MASS:
        return density(mue)
    elif mue < ELECTRON_MASS:
        return 0
    return density(sqrt(mue**2 - ELECTRON_MASS**2))

def fermion_energy_density(kf, mass): #includes spin 1/2 degeneracy but not color/flavor degeneracy
    if kf / mass > 0.01:
        return 1/(8*pi**2) * (kf * sqrt(kf**2+mass**2) * (2*kf**2+mass**2) - mass**4 * arcsinh(kf/mass))
    return mass * density(kf) + 1 / (10 * pi**2) * kf**5 / mass
def lepton_energy_density(mue):
    if mue > MUON_MASS:
        return mue**4/(4*pi**2) + fermion_energy_density(sqrt(mue**2 - MUON_MASS**2), MUON_MASS)
    elif mue > 50 * ELECTRON_MASS:
        return mue**4/(4*pi**2)
    elif mue < ELECTRON_MASS:
        return 0
    return fermion_energy_density(sqrt(mue**2 - ELECTRON_MASS**2), ELECTRON_MASS)

def fermion_pressure(kf, mass): #includes spin 1/2 degeneracy, but not color/flavor degeneracy
    if kf / mass > 0.01:
        return 1/(24*pi**2)*(kf*(2*kf**2-3*mass**2)*sqrt(kf**2+mass**2)+3*mass**4*arctanh(kf/sqrt(kf**2+mass**2)))
    return 1 / (15 * pi**2) * kf**5 / mass
def lepton_pressure(mue):
    if mue > MUON_MASS:
        return mue**4/(12*pi**2) + fermion_pressure(sqrt(mue**2 - MUON_MASS**2), MUON_MASS)
    elif mue > 50 * ELECTRON_MASS:
        return mue**4/(12*pi**2)
    elif mue < ELECTRON_MASS:
        return 0
    return fermion_pressure(sqrt(mue**2 - ELECTRON_MASS**2), ELECTRON_MASS)
    
def scalar_density(kf, mass):
    if kf / mass > 0.01:
        return mass/(2*pi**2) * (kf * sqrt(kf**2+mass**2) - mass**2 * arctanh(kf / sqrt(kf**2 + mass**2)))
    return density(kf) - kf**5 / (10 * pi**2 * mass**2)

#beta equilibrated charge neutral rmf equation of motion with axion, specifying mub
def rmf_axion_eom_mub(fields, mub, ftheta, constants, axion_constants):
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants
    #fields is nucleon and meson fields
    [sigma, omega, rho, kfp, kfn, mue] = fields

    #don't allow any Fermi momenta to be negative
    if kfp < 0 or kfn < 0 or mue < 0:
        return [10,10,10,10,10,10]
    
    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig * ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    
    #proton and neutron mass are modified by axions and by mean field, 
    mp_star = PROTON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #equations of motion for meson fields
    val=[gsig*(scalar_density(kfp,mp_star)+scalar_density(kfn,mn_star))-msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        -kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3+eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw*(density(kfp)+density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2+1/6*zeta0*gw**2*omega**3)
    val.append(-1/2*grho*(density(kfp)-density(kfn))+mrho**2*rho+etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #mean field modified dispersions for proton and neutron
    val.append(sqrt(kfn**2+mn_star**2)+gw*omega-1/2*grho*rho-mub)
    val.append(sqrt(kfp**2+mp_star**2)+gw*omega+1/2*grho*rho-mub+mue)
    if mue > MUON_MASS:
        val.append(mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3)
    else:
        val.append(mue - kfp)
    return val

def rmf_axion_eom_pnm(fields, mub, ftheta, constants, axion_constants):
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants
    #fields is nucleon and meson fields
    [sigma, omega, rho, kfn] = fields

    #don't allow any Fermi momenta to be negative
    if kfn < 0:
        return [10, 10, 10, 10]
    
    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig*ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    
    #proton and neutron mass are modified by axions and by mean field, 
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #equations of motion for meson fields
    val=[gsig*(scalar_density(kfn,mn_star))-msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        -kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3+eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw*(density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2+1/6*zeta0*gw**2*omega**3)
    val.append(-1/2*grho*(-density(kfn))+mrho**2*rho+etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #mean field modified dispersions for proton and neutron
    val.append(sqrt(kfn**2+mn_star**2)+gw*omega-1/2*grho*rho-mub)

    return val

def rmf_axion_eom_density(fields, baryon_density, ftheta, constants, axion_constants):
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants
    #fields is nucleon and meson fields
    [sigma, omega, rho, kfp, kfn, mue] = fields

    #don't allow any Fermi momenta to be negative
    if kfp < 0 or kfn < 0 or mue < 0:
        return [10,10,10,10,10,10]
    
    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig*ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    
    #proton and neutron mass are modified by axions and by mean field, 
    mp_star = PROTON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #equations of motion for meson fields
    val=[gsig*(scalar_density(kfp,mp_star)+scalar_density(kfn,mn_star))-msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        -kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3+eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw*(density(kfp)+density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2+1/6*zeta0*gw**2*omega**3)
    val.append(-1/2*grho*(density(kfp)-density(kfn))+mrho**2*rho+etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    val.append(baryon_density - density(kfn) - density(kfp))
    val.append(sqrt(kfn**2 + mn_star**2) - sqrt(kfp**2 + mp_star**2) - grho * rho - mue)
    if mue > MUON_MASS:
        val.append(mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3)
    else:
        val.append(mue - kfp)
    return val

def rmf_axion_snm(fields, baryon_density, ftheta, constants, axion_constants):
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants
    #fields is nucleon and meson fields
    #fields is nucleon and meson fields
    [sigma, omega] = fields
    kfn = (3 * pi**2 * baryon_density / 2)**(1/3)
    
    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig*ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    
    #proton and neutron mass are modified by axions and by mean field, 
    mp_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #equations of motion for meson fields
    val=[gsig*(scalar_density(kfn,mn_star) + scalar_density(kfn, mp_star)) -msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        -kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3+eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2]
    val.append(-gw*baryon_density+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
               +1/6*zeta0*gw**2*omega**3)
    return val

def rmf_axion_xp(fields, baryon_density, proton_fraction, ftheta, constants, axion_constants):
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants
    #fields is nucleon and meson fields
    #fields is nucleon and meson fields
    [sigma, omega, rho] = fields
    baryon_density = float(baryon_density)
    kfn = (3 * pi**2 * baryon_density * (1 - proton_fraction))**(1/3)
    kfp = (3 * pi**2 * baryon_density * proton_fraction)**(1/3)
    
    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig*ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    
    #proton and neutron mass are modified by axions and by mean field, 
    mp_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #equations of motion for meson fields
    val=[gsig*(scalar_density(kfp,mp_star)+scalar_density(kfn,mn_star))-msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        -kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3+eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw*(density(kfp)+density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2+1/6*zeta0*gw**2*omega**3)
    val.append(-1/2*grho*(density(kfp)-density(kfn))+mrho**2*rho+etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    return val

def rmf_axion_epb_xp(ftheta, baryon_density, proton_fraction, constants, axion_constants, nuc_guess):
    if ftheta > 1 or ftheta < FTHETA_MIN:
        return 10
    kfn = (3 * pi**2 * baryon_density * (1 - proton_fraction))**(1/3)
    kfp = (3 * pi**2 * baryon_density * proton_fraction)**(1/3)
    if kfp < MUON_MASS:
        mue = kfp
    else:
        mue = fsolve(lambda mue: mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3, kfp)[0]

    fields_temp = fsolve(rmf_axion_xp, nuc_guess, args = (baryon_density, proton_fraction, ftheta, constants, axion_constants))
    energy_density = rmf_axion_energy_density(ftheta, list(fields_temp) + [kfp, kfn, mue], constants, axion_constants)
    return energy_density / baryon_density

def rmf_axion_epb_beta(ftheta, baryon_density, constants, axion_constants, nuc_guess):
    if ftheta > 1 or ftheta < FTHETA_MIN:
        return 10
    fields_temp = fsolve(rmf_axion_eom_density, nuc_guess, args = (baryon_density, ftheta, constants, axion_constants))
    energy_density = rmf_axion_energy_density(ftheta, fields_temp, constants, axion_constants)
    return energy_density / baryon_density

#beta equilibrated rmf equation of motion with axion, specifying mub and mue
#for use with solving with mixed phase
def rmf_axion_eom_mixed(fields, mub, mue, ftheta, constants, axion_constants, has_protons):
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants

    #use has_protons to solve for region with nuclei, without protons is dripped neutron region
    if has_protons:
        [sigma, omega, rho, kfp, kfn] = fields
        if kfp < 0 or kfn < 0 or sigma < 0 or omega < 0:
            return [10,10,10,10,10]
    else:
        [sigma, omega, rho, kfn] = fields
        if kfn < 0 or sigma < 0 or omega < 0:
            return [10,10,10,10]

    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig*ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    
    #proton and neutron mass are modified by axions and mean field
    #MUST USE DIFFERENT PROTON AND NEUTRON MASSES IN CRUST SINCE E_F SMALL
    mp_star = PROTON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #equations of motion for meson fields
    if has_protons:
        val=[gsig*(scalar_density(kfp,mp_star)+scalar_density(kfn,mn_star))-msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
            -kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3+eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
            +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
        val.append(-gw*(density(kfp)+density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
            +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2+1/6*zeta0*gw**2*omega**3)
        val.append(-1/2*grho*(density(kfp)-density(kfn))+mrho**2*rho+etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
            +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    else:
        val=[gsig*(scalar_density(kfn,mn_star))-msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
            -kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3+eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
            +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
        val.append(-gw*(density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
            +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2+1/6*zeta0*gw**2*omega**3)
        val.append(-1/2*grho*(-density(kfn))+mrho**2*rho+etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
            +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #neutron dispersion
    val.append(sqrt(kfn**2 + mn_star**2) + gw * omega - 1 / 2 * grho * rho - mub)

    #include proton dispersion only in region with protons, discard kfp<mue solution since in that case both regions are negatively charged
    #if has_protons:
    #    if density(kfp) < lepton_density(mue):
    #        return [10,10,10,10,10]
    #    else:
    val.append(sqrt(kfp**2 + mp_star**2) + gw * omega + 1 / 2 * grho * rho - mub + mue)

    return val

#energy_density in RMF nuclear phase with axions
def rmf_axion_energy_density(ftheta, fields, constants, axion_constants, no_leptons = False): 
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants
    #fields is nucleon and meson fields, only take first 6 elements since in some codes (old?) mub is in the fields vector
    if no_leptons:
        [sigma, omega, rho, kfp, kfn] = fields[0:5]
        lepton_ed = 0
    else:
        [sigma, omega, rho, kfp, kfn, mue] = fields[0:6]
        lepton_ed = lepton_energy_density(mue)

    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig*ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    #nucleon masses are modified by axions and mean field
    mp_star = PROTON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    nucleon_energy_density = fermion_energy_density(kfp, mp_star) + fermion_energy_density(kfn, mn_star)
    axion_energy_density = eps * F_PION**2 * PION_MASS**2 * (1-ftheta)

    meson_energy_density = (1/2*(msig**2*sigma**2+mw**2*omega**2+mrho**2*rho**2)+kappa3/(6*NUCLEON_MASS)*gsig*msig**2*sigma**3+kappa4/(24*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**4
        +zeta0/8*gw**2*omega**4+eta1/(2*NUCLEON_MASS)*gsig*mw**2*sigma*omega**2+eta2/(4*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega**2
        +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*sigma*rho**2+eta1rho/(4*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho**2+3*eta2rho/(4*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho**2)  
    return nucleon_energy_density + axion_energy_density + meson_energy_density + lepton_ed
    
#pressure in the RMF nuclear phase with axions
def rmf_axion_pressure(ftheta, fields, constants, axion_constants, no_leptons = False): 
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants
    #fields is nucleon and meson fields, only take first 6 elements since some codes (old?) mub is in the fields vector
    if no_leptons:
        [sigma, omega, rho, kfp, kfn] = fields[0:5]
        lepton_p = 0
    else:
        [sigma, omega, rho, kfp, kfn, mue] = fields[0:6]
        lepton_p = lepton_pressure(mue)

    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig*ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    #nucleon masses are modified by axions and mean field
    mp_star = PROTON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    nucleon_pressure = fermion_pressure(kfp, mp_star) + fermion_pressure(kfn, mn_star)
    axion_pressure = -eps * F_PION**2 * PION_MASS**2 * (1-ftheta)

    meson_pressure = (1/2*(-msig**2*sigma**2+mw**2*omega**2+mrho**2*rho**2)-kappa3/(6*NUCLEON_MASS)*gsig*msig**2*sigma**3-kappa4/(24*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**4
        +zeta0/24*gw**2*omega**4+eta1/(2*NUCLEON_MASS)*gsig*mw**2*sigma*omega**2+eta2/(4*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega**2
        +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*sigma*rho**2+eta1rho/(4*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho**2+eta2rho/(4*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho**2)
    #print(nucleon_pressure, axion_pressure, meson_pressure)
    return nucleon_pressure + meson_pressure + lepton_p + axion_pressure

def rmf_pressure_equil(mue, mub, ftheta, neut_pressure, constants, axion_constants, fields_guess = (0.1, 0.1, -0.01, 0.8, 0.8)):
    #if mue[0] > fields_guess[3]:
    #    fields_guess = list(fields_guess[0:3]) + [mue[0],] + [fields_guess[4],]
    fields_temp = fsolve(rmf_axion_eom_mixed, fields_guess, args = (mub, mue[0], ftheta, constants, axion_constants, True), xtol = 1e-12)
    if max(rmf_axion_eom_mixed(fields_temp, mub, mue, ftheta, constants, axion_constants, True)) > 1e-5:
        print('possible error')
        return 10
    
    pressure_temp = rmf_axion_pressure(ftheta, fields_temp, constants, axion_constants, no_leptons = True)
    #print(pressure_temp - neut_pressure)
    return pressure_temp - neut_pressure

def rmf_outer_crust_solver(fields, mue, ftheta, constants, axion_constants, shift = 0):
    #constants is all of the RMF model constants
    [msig_old,mw_old,mrho_old,gsig,gw,grho,kappa3,kappa4,zeta0,eta1,eta2,etarho,eta1rho,eta2rho] = constants
    #axion_constants are constants for axion model
    [sigma_pi_n, delta_sigma, eps, dsig, dw, drho] = axion_constants
    #fields is nucleon and meson fields
    [sigma, omega, rho, kfp, kfn] = fields

    #don't allow any Fermi momenta to be negative
    if kfp < 0 or kfn < 0 or mue < 0:
        return [10,10,10,10,10,10]
    
    #ansatz for axions only modifies sigma mass
    msig = msig_old * sqrt((1 + dsig*ftheta) / (1+dsig))
    mw = mw_old * sqrt((1 + dw * ftheta) / (1 + dw))
    mrho = mrho_old * sqrt((1 + drho * ftheta) / (1 + drho))
    
    #proton and neutron mass are modified by axions and by mean field, 
    mp_star = PROTON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn_star = NEUTRON_MASS - sigma * gsig + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #equations of motion for meson fields
    val=[gsig*(scalar_density(kfp,mp_star)+scalar_density(kfn,mn_star))-msig**2*sigma-kappa3/(2*NUCLEON_MASS)*gsig*msig**2*sigma**2
        -kappa4/(6*NUCLEON_MASS**2)*gsig**2*msig**2*sigma**3+eta1/(2*NUCLEON_MASS)*gsig*mw**2*omega**2+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma*omega**2
        +etarho/(2*NUCLEON_MASS)*gsig*mrho**2*rho**2+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma*rho**2]
    val.append(-gw*(density(kfp)+density(kfn))+mw**2*omega+eta1/NUCLEON_MASS*gsig*mw**2*sigma*omega+eta2/(2*NUCLEON_MASS**2)*gsig**2*mw**2*sigma**2*omega
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega*rho**2+1/6*zeta0*gw**2*omega**3)
    val.append(-1/2*grho*(density(kfp)-density(kfn))+mrho**2*rho+etarho/NUCLEON_MASS*gsig*mrho**2*sigma*rho+eta1rho/(2*NUCLEON_MASS**2)*gsig**2*mrho**2*sigma**2*rho
        +eta2rho/(2*NUCLEON_MASS**2)*gw**2*mrho**2*omega**2*rho)
    #mean field modified dispersions for proton and neutron
    val.append(sqrt(kfn**2+mn_star**2) - sqrt(kfp**2 + mp_star**2) - grho*rho-mue)
    val.append(rmf_axion_pressure(ftheta, fields, constants, axion_constants, no_leptons = True) + eps * F_PION**2 * PION_MASS**2 * (1 - ftheta) - shift)
    return val

def outer_crust_solver(inputs, mub, baryon_per_nuc, mp, mn, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
    binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants
    mue, xp = inputs

    val = [-mue + mn - mp + 4 * (1 - 2 * xp) * symm_energy - 2 * xp * coul_energy * baryon_per_nuc**(2/3)]
    val.append(-mub + mn - binding_energy + symm_energy * (1 - 4 * xp**2) 
               - 1 / 3 * coul_energy * xp**2 * baryon_per_nuc**(2/3) + 2 / 3 * surf_energy / baryon_per_nuc**(1/3))
    return val

def outer_crust_solver_mue(xp, mue, baryon_per_nuc, mp, mn, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
    binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants
    return -mue + mn - mp + 4 * (1 - 2 * xp) * symm_energy - 2 * xp * coul_energy * baryon_per_nuc**(2/3)

def outer_crust_energy_min(baryon_per_nuc, mue, mp, mn, xp_guess, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
    binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants
    xp = fsolve(outer_crust_solver_mue, xp_guess, args = (mue, baryon_per_nuc, mp, mn, nuc_constants))[0]
    return symm_energy * (1 - 2 * xp)**2 + surf_energy / baryon_per_nuc**(1/3) + coul_energy * xp**2 * baryon_per_nuc**(2/3)

def surf_energy(rare_vol_frac, surf_tension, d, r): return rare_vol_frac * surf_tension * d / r

def coul_energy(rare_vol_frac, charge_diff, d, r):
    if d == 3:
        fd = 1 / 5 * (2 - 3 * rare_vol_frac**(1/3) + rare_vol_frac)
    elif d == 2:
        fd = 1 / 4 * (rare_vol_frac - 1 - log(rare_vol_frac))
    elif d == 1:
        fd = (rare_vol_frac - 1)**2 / (3 * rare_vol_frac)
    else:
        print('invalid d')
        return
    return 2 * pi * charge_diff**2 * FINE_STRUCTURE * r**2 * rare_vol_frac * fd

def rmf_pressure_equil_finite_size(mue, mub, ftheta_val, constants, axion_constants, d, has_dripped_neutrons, surf_tension,
                                   nuc_guess_prot, nuc_guess_no_prot = (0.2, 0.2, -0.1, 0.5), nuc_is_rare = True):
    mue = mue[0]
    if not has_dripped_neutrons:
        pressure_no_nuc = lepton_pressure(mue) - (1 - ftheta_val) * F_PION**2 * PION_MASS**2 * axion_constants[2]
    else:
        pnm_fields = fsolve(rmf_axion_eom_pnm, nuc_guess_no_prot, args = (mub, ftheta_val, constants, axion_constants))
        pressure_no_nuc = rmf_axion_pressure(ftheta_val, list(pnm_fields[0:3]) + [0, pnm_fields[3], mue], constants, axion_constants)

    nuc_fields = fsolve(rmf_axion_eom_mixed, nuc_guess_prot, args = (mub, mue, ftheta_val, constants, axion_constants, True))
    pressure_nuc = rmf_axion_pressure(ftheta_val, list(nuc_fields) + [mue], constants, axion_constants)

    no_prot_charge = -lepton_density(mue)
    prot_charge = no_prot_charge + density(nuc_fields[3])
    nucleus_vol_frac = 1 / (1 - prot_charge / no_prot_charge)

    if nucleus_vol_frac > 1 or nucleus_vol_frac <= 0:
        return 10
    elif nuc_is_rare:
        r = fsolve(lambda r: surf_energy(nucleus_vol_frac, surf_tension, d, r) 
                   - 2 * coul_energy(nucleus_vol_frac, density(nuc_fields[3]), d, r), 20)[0]
        if d == 3:
            coul_pressure = 2 * pi * density(nuc_fields[3])**2 * FINE_STRUCTURE * r**2 * 2 / 15 * (5 - 9 * nucleus_vol_frac**(1/3) + 4 * nucleus_vol_frac)#* (nucleus_vol_frac - 1)
        elif d == 2:
            coul_pressure = 2 * pi * density(nuc_fields[3])**2 * FINE_STRUCTURE * r**2 * 1 / 4 * (-3 + 3 * nucleus_vol_frac - 2 * log(nucleus_vol_frac))#* (nucleus_vol_frac - 1)
        elif d == 1:
            coul_pressure = 2 * pi * density(nuc_fields[3])**2 * FINE_STRUCTURE * r**2 * 2 / 3 * (2 * nucleus_vol_frac - 3 + 1 / nucleus_vol_frac)#* (nucleus_vol_frac - 1)
        return pressure_nuc - pressure_no_nuc - (d - 1) * surf_tension / r - coul_pressure
    else:
        r = fsolve(lambda r: surf_energy(1 - nucleus_vol_frac, surf_tension, d, r) 
                   - 2 * coul_energy(1 - nucleus_vol_frac, density(nuc_fields[3]), d, r), 20)[0]
        if d == 3:
            coul_pressure = 2 * pi * density(nuc_fields[3])**2 * FINE_STRUCTURE * r**2 * 2 / 15 * (5 - 9 * (1 - nucleus_vol_frac)**(1/3) + 4 * (1 - nucleus_vol_frac))
        elif d == 2:
            coul_pressure = 2 * pi * density(nuc_fields[3])**2 * FINE_STRUCTURE * r**2 * 1 / 4 * (-3 * nucleus_vol_frac - 2 * log(1 - nucleus_vol_frac))
        elif d == 1:
            coul_pressure = 2 * pi * density(nuc_fields[3])**2 * FINE_STRUCTURE * r**2 * 2 / 3 * (-1 - 2 * nucleus_vol_frac + 1 / (1 - nucleus_vol_frac))
        return pressure_no_nuc - pressure_nuc - (d - 1) * surf_tension / r - coul_pressure

#relativistic mean field model that holds all RMF and axion constants and has functions to solve for EoS data
class rmfAxionModel:
    def __init__(self, sigma_pi_n, delta_sigma, eps, dsig, dw, drho, msig, mw, mrho, gsig, gw, grho, kappa3, 
                 kappa4, zeta0, eta1, eta2, etarho, eta1rho, eta2rho):
        self.msig = msig
        self.mw = mw
        self.mrho = mrho
        self.gsig = gsig
        self.gw = gw
        self.grho = grho
        self.kappa3 = kappa3
        self.kappa4 = kappa4
        self.zeta0 = zeta0
        self.eta1 = eta1
        self.eta2 = eta2
        self.etarho = etarho
        self.eta1rho = eta1rho
        self.eta2rho = eta2rho
        self.sigma_pi_n = sigma_pi_n
        self.delta_sigma = delta_sigma
        self.eps = eps 
        self.dsig = dsig
        self.dw = dw
        self.drho = drho
    def eos_data(self, mub, nuc_guess = (0.2,0.3,-0.01,1,3,1), ftheta_val=False):
        #specifying ftheta_val will use that value of ftheta, otherwise the code will solve to minimize free energy
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps, self.dsig, self.dw, self.drho]

        if not ftheta_val:
            #solve for theta = pi
            fields_temp_axion=fsolve(rmf_axion_eom_mub,nuc_guess,args=(mub, FTHETA_MIN, constants, axion_constants), xtol=10**-10)
            pressure_axion=rmf_axion_pressure(FTHETA_MIN, fields_temp_axion, constants, axion_constants)

            #solve for theta = 0
            fields_temp_no_axion = fsolve(rmf_axion_eom_mub, nuc_guess, args=(mub, 1, constants, axion_constants))
            pressure_no_axion = rmf_axion_pressure(1, fields_temp_no_axion, constants, axion_constants)

            #phase with maximum pressure (min free energy) will be favored
            if pressure_axion>pressure_no_axion:
                fthetaTemp = FTHETA_MIN
                fields_temp = fields_temp_axion
                pressure = pressure_axion
            else:
                fthetaTemp = 1
                fields_temp = fields_temp_no_axion
                pressure = pressure_no_axion
        else:
            fthetaTemp = ftheta_val
            fields_temp = fsolve(rmf_axion_eom_mub, nuc_guess, args=(mub, fthetaTemp, constants, axion_constants))
            pressure = rmf_axion_pressure(fthetaTemp, fields_temp, constants, axion_constants)
        
        kfp = fields_temp[3]; kfn = fields_temp[4]; mue = fields_temp[5]

        energy_density = rmf_axion_energy_density(fthetaTemp, fields_temp, constants, axion_constants)
        mp_star = (PROTON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (fthetaTemp-1)
            - self.delta_sigma * (1/fthetaTemp-1))
        mn_star = (NEUTRON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (fthetaTemp-1)
            + self.delta_sigma * (1/fthetaTemp-1))
        return [kfp, kfn, mue, energy_density, pressure, mp_star, mn_star, fthetaTemp]
    def eos_data_snm(self, baryon_density, ftheta, snm_guess = (0.2, 0.2)):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n,self.delta_sigma,self.eps,self.dsig, self.dw, self.drho]
        fields_temp = fsolve(rmf_axion_snm, snm_guess, args = (baryon_density, ftheta, constants, axion_constants))
        kfn = (3 * pi**2 * baryon_density / 2)**(1/3)
        kfp = kfn
        mp_star = (PROTON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (ftheta-1)
            - self.delta_sigma * (1/ftheta-1))
        mn_star = (NEUTRON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (ftheta-1)
            + self.delta_sigma * (1/ftheta-1))
        energy_density = rmf_axion_energy_density(ftheta, list(fields_temp) + [0, kfp, kfn, 0], constants, axion_constants)
        pressure = rmf_axion_pressure(ftheta, list(fields_temp) + [0, kfp, kfn, 0], constants, axion_constants)
        return [kfp, kfn, 0, energy_density, pressure, mp_star, mn_star, ftheta]
    def eos_data_beta(self, baryon_density, nuc_guess = (0.2, 0.3, -0.01, 1, 3, 1), no_axion = False, ftheta_val = False, no_leptons = False):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n,self.delta_sigma,self.eps,self.dsig, self.dw, self.drho]

        if no_axion:
            ftheta = 1
        else:
            if not ftheta_val:
                ftheta = minimize_scalar(rmf_axion_epb_beta, bounds = [FTHETA_MIN, 1], 
                    args = (baryon_density, constants, axion_constants, nuc_guess)).x
            else:
                ftheta = ftheta_val

        fields_temp = fsolve(rmf_axion_eom_density, nuc_guess, args = (baryon_density, ftheta, constants, axion_constants))

        mp_star = (PROTON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (ftheta-1)
            - self.delta_sigma * (1/ftheta-1))
        mn_star = (NEUTRON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (ftheta-1)
            + self.delta_sigma * (1/ftheta-1))
        energy_density = rmf_axion_energy_density(ftheta, fields_temp, constants, axion_constants, no_leptons)
        pressure = rmf_axion_pressure(ftheta, fields_temp, constants, axion_constants, no_leptons)
        mub = sqrt(fields_temp[4]**2 + mn_star**2) + self.gw * fields_temp[1] - 1 / 2 * self.grho * fields_temp[2]
        return [fields_temp[3], fields_temp[4], fields_temp[5], mub, energy_density, pressure, mp_star, mn_star, ftheta]
    def eos_data_xp(self, baryon_density, proton_fraction, nuc_guess = (0.2, 0.2, -0.01), no_axion = False, ftheta_val = False, no_leptons = False):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n,self.delta_sigma,self.eps,self.dsig, self.dw, self.drho]
        
        if no_axion:
            ftheta = 1
        else:
            if not ftheta_val:
                ftheta = minimize_scalar(rmf_axion_epb_xp, bounds = [FTHETA_MIN, 1], 
                    args = (baryon_density, proton_fraction, constants, axion_constants, nuc_guess)).x
            else:
                ftheta = ftheta_val

        kfp = (3 * pi**2 * baryon_density * proton_fraction)**(1/3)
        kfn = (3 * pi**2 * baryon_density * (1 - proton_fraction))**(1/3)

        fields_temp = fsolve(rmf_axion_xp, nuc_guess, args = (baryon_density, proton_fraction, ftheta, constants, axion_constants))

        mp_star = (PROTON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (ftheta-1)
            - self.delta_sigma * (1/ftheta-1))
        mn_star = (NEUTRON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (ftheta-1)
            + self.delta_sigma * (1/ftheta-1))
        mue = sqrt(kfn**2 + mn_star**2) - sqrt(kfp**2 + mp_star**2) - self.grho * fields_temp[2]
        
        mub = sqrt(kfn**2 + mn_star**2) + self.gw * fields_temp[1] - 1 / 2 * self.grho * fields_temp[2]
        total_fields = list(fields_temp) + [kfp, kfn, mue]
        energy_density = rmf_axion_energy_density(ftheta, total_fields, constants, axion_constants, no_leptons)
        pressure = rmf_axion_pressure(ftheta, total_fields, constants, axion_constants, no_leptons)

        return [kfp, kfn, mue, mub, energy_density, pressure, mp_star, mn_star, ftheta]
    def free_energy(self, mub, baryon_density, ftheta, nuc_guess = (0.2,0.3,-0.01,1,3,1)):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n,self.delta_sigma,self.eps,self.dsig, self.dw, self.drho]

        fields_temp = fsolve(rmf_axion_eom_density, nuc_guess, args = (baryon_density, ftheta, constants, axion_constants))
        energy_density = rmf_axion_energy_density(ftheta, fields_temp, constants, axion_constants)
        return energy_density - mub * baryon_density
    def eos_data_inner_crust(self, mub, nuc_guess_prot = (0.3,0.3,-0.01,1,1.4), nuc_guess_no_prot = (0.2,0.2,-0.01,0.5),
                     nuc_guess_prot_axion = (0.3,0.3,-0.005,1.3,1.6), nuc_guess_no_prot_axion = (0.2,0.2,-0.1,1), mue_guess = 0.2,
                     ftheta_val = False):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n,self.delta_sigma,self.eps,self.dsig, self.dw, self.drho]

        pressure_no_axion = -10
        pressure_axion = -10

        if not ftheta_val or ftheta_val == 1:
            pnm_fields_norm = fsolve(rmf_axion_eom_pnm, nuc_guess_no_prot, args = (mub, 1, constants, axion_constants))
            pnm_pressure_norm = rmf_axion_pressure(1, list(pnm_fields_norm[0:3]) + [0, pnm_fields_norm[3]], constants, axion_constants, no_leptons = True)

            mue_norm = fsolve(rmf_pressure_equil, mue_guess, args = (mub, 1, pnm_pressure_norm, constants, axion_constants, nuc_guess_prot), xtol = 1e-12)[0]
            print(rmf_pressure_equil([mue_norm], mub, 1, pnm_pressure_norm, constants, axion_constants, nuc_guess_prot))
            
            fields_prot_test = fsolve(rmf_axion_eom_mixed, nuc_guess_prot, args = (mub, mue_norm, 1, constants, axion_constants, True))
            #print(rmf_axion_pressure(1, fields_prot_test, constants, axion_constants, no_leptons = True))

            pressure_no_axion = pnm_pressure_norm + lepton_pressure(mue_norm)
        elif not ftheta_val or ftheta_val == FTHETA_MIN:
            pnm_fields_ax = fsolve(rmf_axion_eom_pnm, nuc_guess_no_prot_axion, args = (mub, FTHETA_MIN, constants, axion_constants))
            pnm_pressure_ax = rmf_axion_pressure(FTHETA_MIN, list(pnm_fields_ax[0:3]) + [0, pnm_fields_ax[3]], constants, axion_constants, no_leptons = True)
            #print(pnm_pressure_ax)
            mue_ax = fsolve(rmf_pressure_equil, (mue_guess,), args = (mub, FTHETA_MIN, pnm_pressure_ax, constants, axion_constants, nuc_guess_prot_axion), xtol = 10**-12)[0]
            print(rmf_pressure_equil([mue_ax], mub, FTHETA_MIN, pnm_pressure_ax, constants, axion_constants, nuc_guess_prot_axion))
            pressure_axion = pnm_pressure_ax + lepton_pressure(mue_ax)

            fields_prot_test = fsolve(rmf_axion_eom_mixed, nuc_guess_prot_axion, args = (mub, mue_ax, FTHETA_MIN, constants, axion_constants, True))
            #print(rmf_axion_pressure(FTHETA_MIN, fields_prot_test, constants, axion_constants, no_leptons = True))
        else:
            pnm_fields_ax = fsolve(rmf_axion_eom_pnm, nuc_guess_no_prot_axion, args = (mub, ftheta_val, constants, axion_constants))
            pnm_pressure_ax = rmf_axion_pressure(ftheta_val, list(pnm_fields_ax[0:3]) + [0, pnm_fields_ax[3]], constants, axion_constants, no_leptons = True)
            #print(pnm_pressure_ax)
            mue_ax = fsolve(rmf_pressure_equil, (mue_guess,), args = (mub, ftheta_val, pnm_pressure_ax, constants, axion_constants, nuc_guess_prot_axion), xtol = 10**-12)[0]
            print(rmf_pressure_equil([mue_ax], mub, FTHETA_MIN, pnm_pressure_ax, constants, axion_constants, nuc_guess_prot_axion))
            pressure_axion = pnm_pressure_ax + lepton_pressure(mue_ax)

        if pressure_no_axion > pressure_axion:
            pressure = pressure_no_axion
            fields_prot = fsolve(rmf_axion_eom_mixed, nuc_guess_prot, args = (mub, mue_norm, 1, constants, axion_constants, True))
            pnm_fields = pnm_fields_norm
            mue = mue_norm
            ftheta = 1
        else:
            pressure = pressure_axion
            if not ftheta_val:
                fields_prot = fsolve(rmf_axion_eom_mixed, nuc_guess_prot_axion, args = (mub, mue_ax, FTHETA_MIN, constants, axion_constants, True))
                ftheta = FTHETA_MIN
            else:
                fields_prot = fsolve(rmf_axion_eom_mixed, nuc_guess_prot_axion, args = (mub, mue_ax, ftheta_val, constants, axion_constants, True))
                ftheta = ftheta_val
            pnm_fields = pnm_fields_ax
            mue = mue_ax

        #print(fields_prot)
        #print(pnm_fields)

        kfn_no_prot = pnm_fields[3]
        kfn_prot = fields_prot[4]
        kfp = fields_prot[3]

        #print(NEUTRON_MASS - self.gsig * fields_prot[0])
        #print(PROTON_MASS - self.gsig * fields_prot[0])
        #print(fields_prot[2])
        #print(fields_prot)
        #print(pnm_fields)

        no_prot_charge = -lepton_density(mue)
        prot_charge = no_prot_charge + density(kfp)
        nucleus_vol_frac = 1 / (1 - prot_charge / no_prot_charge)

        energy_density = lepton_energy_density(mue) + (1 - nucleus_vol_frac) * rmf_axion_energy_density(ftheta, list(pnm_fields[0:3]) + [0, pnm_fields[3]], constants, 
            axion_constants, no_leptons=True) + nucleus_vol_frac * rmf_axion_energy_density(ftheta, fields_prot, constants, axion_constants, no_leptons = True)        
        return [kfp, kfn_prot, kfn_no_prot, mue, energy_density, pressure, nucleus_vol_frac, ftheta]
    def eos_data_inner_crust_finite_size(self, mub, ftheta_val, surf_tension, d, nuc_guess_prot = (0.3,0.3,-0.01,1,1.4), nuc_guess_no_prot = (0.2,0.2,-0.01,0.5),
                     nuc_guess_prot_axion = (0.3, 0.2, -0.01, 1.2, 1.4), nuc_guess_no_prot_axion = (0.2,0.2,-0.1,1), mue_guess = 0.2, nuc_is_rare = True, surf_off = False):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps, self.dsig, self.dw, self.drho]
        
        if ftheta_val == 1:
            mue = fsolve(rmf_pressure_equil_finite_size, mue_guess, args = (mub, ftheta_val, constants, axion_constants, d, True, surf_tension,
                nuc_guess_prot, nuc_guess_no_prot, nuc_is_rare))[0]
            print(rmf_pressure_equil_finite_size([mue], mub, ftheta_val, constants, axion_constants, d, True, surf_tension,
                nuc_guess_prot, nuc_guess_no_prot, nuc_is_rare))
            pnm_fields = fsolve(rmf_axion_eom_pnm, nuc_guess_no_prot, args = (mub, ftheta_val, constants, axion_constants))
            pressure_no_nuc = rmf_axion_pressure(ftheta_val, list(pnm_fields[0:3]) + [0, pnm_fields[3], mue], constants, axion_constants)
            nuc_fields = fsolve(rmf_axion_eom_mixed, nuc_guess_prot, args = (mub, mue, ftheta_val, constants, axion_constants, True))
            pressure_nuc = rmf_axion_pressure(ftheta_val, list(nuc_fields) + [mue], constants, axion_constants)
        else:
            mue = fsolve(rmf_pressure_equil_finite_size, mue_guess, args = (mub, ftheta_val, constants, axion_constants, d, True, surf_tension,
                nuc_guess_prot_axion, nuc_guess_no_prot_axion, nuc_is_rare))[0]
            print(rmf_pressure_equil_finite_size([mue], mub, ftheta_val, constants, axion_constants, d, True, surf_tension,
                nuc_guess_prot_axion, nuc_guess_no_prot_axion, nuc_is_rare))
            pnm_fields = fsolve(rmf_axion_eom_pnm, nuc_guess_no_prot_axion, args = (mub, ftheta_val, constants, axion_constants))
            pressure_no_nuc = rmf_axion_pressure(ftheta_val, list(pnm_fields[0:3]) + [0, pnm_fields[3], mue], constants, axion_constants)
            nuc_fields = fsolve(rmf_axion_eom_mixed, nuc_guess_prot_axion, args = (mub, mue, ftheta_val, constants, axion_constants, True))
            pressure_nuc = rmf_axion_pressure(ftheta_val, list(nuc_fields) + [mue], constants, axion_constants)

        no_prot_charge = -lepton_density(mue)
        prot_charge = no_prot_charge + density(nuc_fields[3])
        nucleus_vol_frac = 1 / (1 - prot_charge / no_prot_charge)

        if nuc_is_rare:
            r = fsolve(lambda r: surf_energy(nucleus_vol_frac, surf_tension, d, r) 
                    - 2 * coul_energy(nucleus_vol_frac, density(nuc_fields[3]), d, r), 20)[0]
            if surf_off:
                finite_ed = 0
            else:
                finite_ed = surf_energy(nucleus_vol_frac, surf_tension, d, r) + coul_energy(nucleus_vol_frac, density(nuc_fields[3]), d, r)
            energy_density = lepton_energy_density(mue) + (1 - nucleus_vol_frac) * rmf_axion_energy_density(ftheta_val, list(pnm_fields[0:3]) + [0, pnm_fields[3]], constants, 
                axion_constants, no_leptons=True) + nucleus_vol_frac * rmf_axion_energy_density(ftheta_val, nuc_fields, constants, axion_constants, no_leptons = True) + (
                finite_ed)
                
            #print(surf_energy(nucleus_vol_frac, surf_tension, d, r), coul_energy(nucleus_vol_frac, density(nuc_fields[3]), d, r))
            rws = r / nucleus_vol_frac**(1 / d)
            pressure = pressure_no_nuc
        else:
            r = fsolve(lambda r: surf_energy(1 - nucleus_vol_frac, surf_tension, d, r) 
                    - 2 * coul_energy(1 - nucleus_vol_frac, density(nuc_fields[3]), d, r), 20)[0]
            energy_density = lepton_energy_density(mue) + (1 - nucleus_vol_frac) * rmf_axion_energy_density(ftheta_val, list(pnm_fields[0:3]) + [0, pnm_fields[3]], constants, 
                axion_constants, no_leptons=True) + nucleus_vol_frac * rmf_axion_energy_density(ftheta_val, nuc_fields, constants, axion_constants, no_leptons = True) + (
                surf_energy(1 - nucleus_vol_frac, surf_tension, d, r) + coul_energy(1 - nucleus_vol_frac, density(nuc_fields[3]), d, r)) 
            #print(surf_energy(1 - nucleus_vol_frac, surf_tension, d, r), coul_energy(1 - nucleus_vol_frac, density(nuc_fields[3]), d, r))
            rws = r / (1 - nucleus_vol_frac)**(1 / d)
            pressure = pressure_nuc
        #pressure = pressure_nuc * nucleus_vol_frac + pressure_no_nuc * (1 - nucleus_vol_frac)
        kfp = nuc_fields[3]
        kfn_prot = nuc_fields[4]
        kfn_no_prot = pnm_fields[3]
        mp_star = (PROTON_MASS - self.gsig * nuc_fields[0] + self.sigma_pi_n * (ftheta_val - 1)
            - self.delta_sigma * (1 / ftheta_val-1))
        mn_star = (NEUTRON_MASS - self.gsig * nuc_fields[0] + self.sigma_pi_n * (ftheta_val - 1)
            + self.delta_sigma * (1 / ftheta_val - 1))
        mn_star_drip = (NEUTRON_MASS - self.gsig * pnm_fields[0] + self.sigma_pi_n * (ftheta_val - 1)
            + self.delta_sigma * (1 / ftheta_val - 1))
        
        return [kfp, kfn_prot, kfn_no_prot, mue, energy_density, pressure, nucleus_vol_frac, ftheta_val, r, rws, pressure_nuc, pressure_no_nuc, mp_star, mn_star, mn_star_drip]
    def eos_data_outer_crust_finite_size(self, mub, ftheta_val, surf_tension, nuc_guess_norm = (0.2, 0.1, -0.001, 1.4, 1.5), 
                                         nuc_guess_ax = (0.2, 0.1, -0.001, 1.4, 1.5), mue_guess = 0.1, d = 3, surf_off = False):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps, self.dsig, self.dw, self.drho]
        if ftheta_val == 1:
            mue = fsolve(rmf_pressure_equil_finite_size, mue_guess, args = (mub, ftheta_val, constants, axion_constants, d, False, surf_tension,
                nuc_guess_norm))[0]
            print(rmf_pressure_equil_finite_size([mue], mub, ftheta_val, constants, axion_constants, d, False, surf_tension,
                nuc_guess_norm))
            nuc_fields = fsolve(rmf_axion_eom_mixed, nuc_guess_norm, args = (mub, mue, ftheta_val, constants, axion_constants, True))
        else:
            mue = fsolve(rmf_pressure_equil_finite_size, mue_guess, args = (mub, ftheta_val, constants, axion_constants, d, False, surf_tension,
                nuc_guess_ax))[0]
            print(rmf_pressure_equil_finite_size([mue], mub, ftheta_val, constants, axion_constants, d, False, surf_tension,
                nuc_guess_ax))
            nuc_fields = fsolve(rmf_axion_eom_mixed, nuc_guess_ax, args = (mub, mue, ftheta_val, constants, axion_constants, True))
        
        pressure_nuc = rmf_axion_pressure(ftheta_val, list(nuc_fields) + [mue], constants, axion_constants)
        pressure_elec = lepton_pressure(mue) - self.eps * F_PION**2 * PION_MASS**2 * (1 - ftheta_val)
        no_prot_charge = -lepton_density(mue)
        prot_charge = no_prot_charge + density(nuc_fields[3])
        nucleus_vol_frac = 1 / (1 - prot_charge / no_prot_charge)
        r = fsolve(lambda r: surf_energy(nucleus_vol_frac, surf_tension, d, r) 
            - 2 * coul_energy(nucleus_vol_frac, density(nuc_fields[3]), d, r), 20)[0]
        if surf_off:
            finite_ed = 0
        else:
            finite_ed = surf_energy(nucleus_vol_frac, surf_tension, d, r) + coul_energy(nucleus_vol_frac, density(nuc_fields[3]), d, r)
        energy_density = lepton_energy_density(mue) + (1 - nucleus_vol_frac) * (self.eps * F_PION**2 * PION_MASS**2 * (1 - ftheta_val)) + (
            nucleus_vol_frac * rmf_axion_energy_density(ftheta_val, nuc_fields, constants, axion_constants, no_leptons = True) + 
            finite_ed) 
        pressure = pressure_elec #* (1 - nucleus_vol_frac) + pressure_nuc * nucleus_vol_frac
        rws = r / nucleus_vol_frac**(1 / d)
        mp_star = (PROTON_MASS - self.gsig * nuc_fields[0] + self.sigma_pi_n * (ftheta_val - 1)
            - self.delta_sigma * (1 / ftheta_val-1))
        mn_star = (NEUTRON_MASS - self.gsig * nuc_fields[0] + self.sigma_pi_n * (ftheta_val - 1)
            + self.delta_sigma * (1 / ftheta_val - 1))
        return [nuc_fields[3], nuc_fields[4], mue, energy_density, pressure, nucleus_vol_frac, ftheta_val, r, rws, pressure_nuc, pressure_elec, mp_star, mn_star]

    def eos_data_outer_crust(self, mue, ftheta, nuc_guess = (0.2, 0.1, -0.001, 1.4, 1.5)):
        constants = [self.msig,self.mw,self.mrho,self.gsig,self.gw,self.grho,self.kappa3,self.kappa4,self.zeta0,
            self.eta1,self.eta2,self.etarho,self.eta1rho,self.eta2rho]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps, self.dsig, self.dw, self.drho]
        fields_temp = fsolve(rmf_outer_crust_solver, nuc_guess, args = (mue, ftheta, constants, axion_constants))
        kfp = fields_temp[3]
        kfn = fields_temp[4]
        
        mp_star = (PROTON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (ftheta-1)
            - self.delta_sigma * (1/ftheta-1))
        mn_star = (NEUTRON_MASS - self.gsig * fields_temp[0] + self.sigma_pi_n * (ftheta-1)
            + self.delta_sigma * (1/ftheta-1))
        mub = sqrt(kfn**2 + mn_star**2) + self.gw * fields_temp[1] - 1 / 2 * self.grho * fields_temp[2]

        if mue > 50 * ELECTRON_MASS:
            no_prot_charge = -lepton_density(mue)
        else:
            no_prot_charge = -(mue**2 - ELECTRON_MASS**2)**(3/2) / (3 * pi**2)
        prot_charge = no_prot_charge + density(kfp)
        nucleus_vol_frac = 1 / (1 - prot_charge / no_prot_charge)

        pressure = lepton_pressure(mue) - self.eps * F_PION**2 * PION_MASS**2 * (1-ftheta)
        lepton_ed = lepton_energy_density(mue) + self.eps * F_PION**2 * PION_MASS**2 * (1 - ftheta)
        
        energy_density = lepton_ed * (1 - nucleus_vol_frac) + nucleus_vol_frac * rmf_axion_energy_density(ftheta, list(fields_temp) + [mue], constants, axion_constants)
        return [kfp, kfn, mue, mub, energy_density, pressure, nucleus_vol_frac, mp_star, mn_star, ftheta]
    def eos_data_liquid_drop_mub(self, mub, ftheta, baryon_per_nuc, mue_xp_guess = (1e-3, 1e-2), rho_sat = 0.153, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
        mp = PROTON_MASS + self.sigma_pi_n * (ftheta-1) - self.delta_sigma * (1/ftheta-1)
        mn = NEUTRON_MASS + self.sigma_pi_n * (ftheta-1) + self.delta_sigma * (1/ftheta-1)
        binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants

        mue, xp = fsolve(outer_crust_solver, mue_xp_guess, args = (mub, baryon_per_nuc, mp, mn, list(nuc_constants)))
        kfp = (3 * pi**2 * rho_sat * xp)**(1/3)
        kfn = (3 * pi**2 * rho_sat * (1 - xp))**(1/3)

        if mue > 50 * ELECTRON_MASS:
            no_nuc_charge = -lepton_density(mue)
        else:
            no_nuc_charge = -(mue**2 - ELECTRON_MASS**2)**(3/2) / (3 * pi**2)
        nuc_charge = density(kfp) + no_nuc_charge
        nucleus_vol_frac = 1 / (1 - nuc_charge / no_nuc_charge)
        interaction_energy = -binding_energy + symm_energy * (1 - 2 * xp)**2 + \
            surf_energy / baryon_per_nuc**(1/3) + coul_energy * xp**2 * baryon_per_nuc**(2/3)
        if mue > 50 * ELECTRON_MASS:
            pressure = lepton_pressure(mue) - self.eps * F_PION**2 * PION_MASS**2 * (1-ftheta)
            lepton_ed = lepton_energy_density(mue)
        else:
            pressure = fermion_pressure(sqrt(mue**2 - ELECTRON_MASS**2), ELECTRON_MASS) - self.eps * F_PION**2 * PION_MASS**2 * (1-ftheta)
            lepton_ed = fermion_energy_density(sqrt(mue**2 - ELECTRON_MASS**2), ELECTRON_MASS)
        
        energy_density = lepton_ed + nucleus_vol_frac * (rho_sat * xp * (mp + 3 / (10 * mp) * kfp**2) 
            + rho_sat * (1 - xp) * (mn + 3 / (10 * mn) * kfn**2) + interaction_energy * rho_sat) + self.eps * F_PION**2 * PION_MASS**2 * (1 - ftheta)

        return [kfp, kfn, mue, energy_density, pressure, nucleus_vol_frac, ftheta]
    def eos_data_liquid_drop(self, mue, ftheta, xp_guess = 0.4, rho_sat = 0.153, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
        mp = PROTON_MASS + self.sigma_pi_n * (ftheta-1) - self.delta_sigma * (1/ftheta-1)
        mn = NEUTRON_MASS + self.sigma_pi_n * (ftheta-1) + self.delta_sigma * (1/ftheta-1)
        binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants
        baryon_per_nuc = int(floor(minimize_scalar(outer_crust_energy_min, bounds = (1, 300), args = (mue, mp, mn, xp_guess, nuc_constants)).x + 0.5))
        xp = fsolve(outer_crust_solver_mue, xp_guess, args = (mue, baryon_per_nuc, mp, mn, nuc_constants))[0]
        kfp = (3 * pi**2 * rho_sat * xp)**(1/3)
        kfn = (3 * pi**2 * rho_sat * (1 - xp))**(1/3)

        if mue > 50 * ELECTRON_MASS:
            no_nuc_charge = -lepton_density(mue)
        else:
            no_nuc_charge = -(mue**2 - ELECTRON_MASS**2)**(3/2) / (3 * pi**2)
        nuc_charge = density(kfp) + no_nuc_charge
        nucleus_vol_frac = 1 / (1 - nuc_charge / no_nuc_charge)
        interaction_energy = -binding_energy + symm_energy * (1 - 2 * xp)**2 + \
            surf_energy / baryon_per_nuc**(1/3) + coul_energy * xp**2 * baryon_per_nuc**(2/3)
        if mue > 50 * ELECTRON_MASS:
            pressure = lepton_pressure(mue) - self.eps * F_PION**2 * PION_MASS**2 * (1-ftheta)
            lepton_ed = lepton_energy_density(mue)
        else:
            pressure = fermion_pressure(sqrt(mue**2 - ELECTRON_MASS**2), ELECTRON_MASS) - self.eps * F_PION**2 * PION_MASS**2 * (1-ftheta)
            lepton_ed = fermion_energy_density(sqrt(mue**2 - ELECTRON_MASS**2), ELECTRON_MASS)
        energy_density = lepton_ed + nucleus_vol_frac * (rho_sat * xp * (mp + 3 / (10 * mp) * kfp**2) 
            + rho_sat * (1 - xp) * (mn + 3 / (10 * mn) * kfn**2) + interaction_energy * rho_sat) + self.eps * F_PION**2 * PION_MASS**2 * (1 - ftheta)

        return [kfp, kfn, mue, energy_density, pressure, nucleus_vol_frac, ftheta]
#these are a selection of standard rmf models with axions included
def iufsu_star(sigpin, delta_sigma, eps, dsig, dw = 0, drho = 0):
    return rmfAxionModel(sigpin, delta_sigma, eps, dsig, dw, drho, msig=NEUTRON_MASS*0.543,mw=NEUTRON_MASS*0.8331,mrho=NEUTRON_MASS*0.8198,gsig=4*pi*0.8379,gw=4*pi*1.0666,grho=4*pi*0.9889,kappa3=1.1418,
    kappa4=1.0328,zeta0=5.3895,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=41.3066)

def nl3(sigpin, delta_sigma, eps, dsig, dw = 0, drho = 0):
    return rmfAxionModel(sigpin, delta_sigma, eps, dsig, dw, drho, msig=NEUTRON_MASS*0.5412,mw=NEUTRON_MASS*0.8333,mrho=NEUTRON_MASS*0.8126,gsig=4*pi*0.8131,gw=4*pi*1.024,grho=4*pi*0.7121,kappa3=1.4661,
    kappa4=-5.6718,zeta0=0,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

def free(sig_pin, delta_sigma, eps):
    return rmfAxionModel(sig_pin, delta_sigma, eps, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

def bsp(sigpin, delta_sigma, eps, dsig, dw = 0, drho = 0):
    return rmfAxionModel(sigpin, delta_sigma, eps, dsig, dw, drho, msig=NEUTRON_MASS*0.5383,mw=NEUTRON_MASS*0.8333,mrho=NEUTRON_MASS*0.82,gsig=4*pi*0.8764,gw=4*pi*1.1481,grho=4*pi*1.0508,kappa3=1.0681,kappa4=14.9857,
    zeta0=0,eta1=0.0872,eta2=3.1265,etarho=0,eta1rho=0,eta2rho=53.7642)

def glendenning9(sigpin, delta_sigma, eps, dsig, dw = 0, drho = 0):
    return rmfAxionModel(sigpin, delta_sigma, eps, dsig, dw, drho, msig=NUCLEON_MASS*0.54,mw=NUCLEON_MASS*0.833,mrho=NUCLEON_MASS*0.82,gsig=NUCLEON_MASS*0.54*sqrt(8.403),gw=NUCLEON_MASS*0.833*sqrt(4.233),grho=NUCLEON_MASS*0.82*sqrt(4.876),
    kappa3=2*NUCLEON_MASS**2*12.684*0.00248,kappa4=6*NUCLEON_MASS**2*12.684*(0.027997),zeta0=0,eta1=0,eta2=0,etarho=0,eta1rho=0,eta2rho=0)

def g1(sigpin, delta_sigma, eps, dsig, dw = 0, drho = 0):
    return rmfAxionModel(sigpin, delta_sigma, eps, dsig, dw, drho, msig=NUCLEON_MASS*0.5396,mw=NUCLEON_MASS*0.8328,mrho=NUCLEON_MASS*0.82,gsig=4*pi*0.7853,gw=4*pi*0.9651,grho=4*pi*0.6984,kappa3=2.2067,
    kappa4=-10.09, zeta0=3.5249, eta1=0.0706, eta2=-0.9616, etarho=-0.2722, eta1rho=0, eta2rho=0)

def qmc_rmf2(sigpin, delta_sigma, eps, dsig, dw = 0, drho = 0):
    return rmfAxionModel(sigpin, delta_sigma, eps, dsig, dw, drho, msig = 491.5 / 197.3, mw = 782.5 / 197.3, mrho = 763 / 197.3, gsig = 7.82,
    gw = 8.99, grho = 11.24, kappa3 = 2 * (938 / 491.5)**2 * 7.54**2 * 0.0063, kappa4 = 6 * (938 / 491.5)**2 * 7.54**2 * (-0.0009),
    zeta0 = 0, eta1 = 0, eta2 = 0, etarho = 0, eta1rho = 0, eta2rho = 4 * 938**2 / 763**2 * 11.24**2 / 8.43**2 * 8.02)

#this is a model for a free fermi gas
def free_model(axion_constants):
    return rmfAxionModel(axion_constants[0],axion_constants[1],axion_constants[2],axion_constants[3],NUCLEON_MASS,NUCLEON_MASS,NUCLEON_MASS,0,0,0,0,0,0,0,0,0,0,0)
    
def skyrme_energy_density_axion(proton_frac, baryon_density, ftheta, constants, axion_constants):
    #constants is the skyrme model constants
    [t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3] = constants
    #axion_constants is the axion model constants, note that t0 and x0 are modified by the axion field
    [sigma_pi_n, delta_sigma, eps] = axion_constants

    mp = PROTON_MASS + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn = NEUTRON_MASS + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #these coefficients are used for calculation of asymmetric matter
    h2 = 2**(2-1) * (proton_frac**2 + (1-proton_frac)**2)
    h53 = 2**(5/3-1) * (proton_frac**(5/3) + (1-proton_frac)**(5/3))
    h83 = 2**(8/3-1) * (proton_frac**(8/3) + (1-proton_frac)**(8/3))
    
    #rest mass and kinetic energy
    rest_mass_ed = baryon_density * (proton_frac * mp + (1 - proton_frac) * mn)
    first_term = 3 / 10 * (3 * pi**2)**(2/3) * baryon_density**(5/3) * (proton_frac**(5/3) / mp + (1 - proton_frac)**(5/3) / mn)
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

    axion_energy_density = eps * F_PION**2 * PION_MASS**2 * (1-ftheta)

    return rest_mass_ed + first_term + second_term + third_term + fourth_term + axion_energy_density

def skyrme_mub_axion(proton_frac, baryon_density, ftheta, constants, axion_constants):
    #constants is the skyrme model constants
    [t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3] = constants
    #axion_constants is the axion model constants, note that t0 and x0 are modified by the axion field
    [sigma_pi_n, delta_sigma, eps] = axion_constants

    #these coefficients are used for calculation of asymmetric matter
    h2 = 2**(2-1) * (proton_frac**2 + (1-proton_frac)**2)
    h53 = 2**(5/3-1) * (proton_frac**(5/3) + (1-proton_frac)**(5/3))

    mn = NEUTRON_MASS + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)
    
    #rest mass and kinetic energy
    mass_term = mn
    first_term = 1 / (2 * mn) * (3 * pi**2 * baryon_density * (1 - proton_frac))**(2/3)
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
    
def skyrme_mue_axion(proton_frac, baryon_density, ftheta, constants, axion_constants):
    #constants is the skyrme model constants
    [t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3] = constants
    #axion_constants is the axion model constants, note that t0 and x0 are modified by the axion field
    [sigma_pi_n, delta_sigma, eps] = axion_constants

    mp = PROTON_MASS + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn = NEUTRON_MASS + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #rest mass and kinetic energy
    mass_term = mn - mp
    first_term = 1 / 2 * (3 * pi**2 * baryon_density)**(2/3) * ((1 - proton_frac)**(2/3) / mn - proton_frac**(2/3) / mp)
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

def skyrme_pressure_axion(proton_frac, baryon_density, ftheta, constants, axion_constants):
    #constants is the Skyrme model constants
    [t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3] = constants
    #axion_constants is the axion model constants, note that t0 and x0 are modified by the axion field
    [sigma_pi_n, delta_sigma, eps] = axion_constants

    #these coefficients are used for asymmetric matter
    h53 = 2**(5/3-1) * (proton_frac**(5/3) + (1-proton_frac)**(5/3))
    h2 = 2 * (proton_frac**2 + (1-proton_frac)**2)
    h83 = 2**(8/3-1) * (proton_frac**(8/3) + (1-proton_frac)**(8/3))

    mp = PROTON_MASS + sigma_pi_n * (ftheta-1) - delta_sigma * (1/ftheta-1)
    mn = NEUTRON_MASS + sigma_pi_n * (ftheta-1) + delta_sigma * (1/ftheta-1)

    #kinetic energy
    first_term = 1 / 5 * (3 * pi**2)**(2/3) * baryon_density**(5/3) * (proton_frac**(5/3) / mp + (1 - proton_frac)**(5/3) / mn)
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

    axion_pressure = -eps * F_PION**2 * PION_MASS**2 * (1 - ftheta)

    return first_term + second_term + third_term + fourth_term + axion_pressure
    
def mue_solver(mue, kfp):
    if mue > MUON_MASS:
        return mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3
    else:
        return mue - kfp

def skyrme_proton_frac_density_solver(inputs, mub, ftheta, constants, axion_constants):
    [proton_frac, baryon_density]=inputs
    if proton_frac<0 or proton_frac>1 or baryon_density<0:
        return [10,10]
    
    kfp = (3*pi**2 * baryon_density * proton_frac)**(1/3)
    mue = fsolve(mue_solver, kfp, args = (kfp))[0]

    val = [mue - skyrme_mue_axion(proton_frac, baryon_density, ftheta, constants, axion_constants)]
    val.append(mub - skyrme_mub_axion(proton_frac, baryon_density, ftheta, constants, axion_constants))
    return val  

def skyrme_proton_frac_solver(proton_frac, baryon_density, ftheta, constants, axion_constants):
    if proton_frac < 0 or proton_frac > 1:
        return 10
    kfp = (3*pi**2 * baryon_density * proton_frac)**(1/3)
    mue = fsolve(mue_solver, kfp, args = (kfp))[0]
    
    return mue - skyrme_mue_axion(proton_frac, baryon_density, ftheta, constants, axion_constants)

def skyrme_density_solver(baryon_density, mub, proton_frac, ftheta, constants, axion_constants):
    if baryon_density < 0:
        return exp(-baryon_density)
    return mub - skyrme_mub_axion(proton_frac, baryon_density, ftheta, constants, axion_constants)

def skyrme_pressure_equil(proton_frac, mub, ftheta, neut_pressure, constants, axion_constants, yp_min = 0):
    yp = proton_frac
    if yp < yp_min:
        return exp(yp_min - yp)
    nb = fsolve(skyrme_density_solver, 0.15, args = (mub, yp, ftheta, constants, axion_constants))[0]
    pressure = skyrme_pressure_axion(yp, nb, ftheta, constants, axion_constants)
    return 1e5 * (pressure - neut_pressure)

class skyrmeAxionModel:
    def __init__(self,sigma_pi_n,delta_sigma,eps,d1s0,d3s1,t0,t1,t2,t31,t32,t33,x0,x1,x2,x31,x32,x33,sigma1,sigma2,sigma3):
        self.sigma_pi_n = sigma_pi_n
        self.delta_sigma = delta_sigma
        self.eps = eps
        self.d1s0 = d1s0
        self.d3s1 = d3s1
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

    def eos_data(self, mub, proton_frac_guess_no_axion = (0.2, 0.3), proton_frac_guess_axion = (0.1, 0.5), ftheta_val = False):
        pressure_axion = -5
        pressure_no_axion = -5
        
        if not ftheta_val:
            t0_new = self.t0 + (self.d1s0 + self.d3s1) / 2 * PION_MASS**2 * (FTHETA_MIN - 1)
            x0_new = (self.x0 + (self.d3s1 - self.d1s0) / (2 * self.t0) * PION_MASS**2 * (FTHETA_MIN - 1) ) / ( 
                1 + (self.d1s0 + self.d3s1) / (2 * self.t0) * PION_MASS**2 * (FTHETA_MIN - 1))

            constants_axion = [t0_new,self.t1,self.t2,self.t31,self.t32,self.t33,x0_new,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
            constants_norm = [self.t0,self.t1,self.t2,self.t31,self.t32,self.t33,self.x0,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
            axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps]

            [proton_frac_no_axion, baryon_density_no_axion] = fsolve(skyrme_proton_frac_density_solver, 
                proton_frac_guess_no_axion, args = (mub, 1, constants_norm, axion_constants))
            kfp_no_axion = (3*pi**2 * baryon_density_no_axion * proton_frac_no_axion)**(1/3)
            kfn_no_axion = (3*pi**2 * baryon_density_no_axion * (1 - proton_frac_no_axion))**(1/3)
            mue_no_axion = fsolve(mue_solver, kfp_no_axion, args = (kfp_no_axion))[0]

            pressure_no_axion = skyrme_pressure_axion(proton_frac_no_axion, baryon_density_no_axion, 1, constants_norm, axion_constants) + lepton_pressure(mue_no_axion)
            energy_density_no_axion = skyrme_energy_density_axion(proton_frac_no_axion, baryon_density_no_axion, 1, constants_norm, axion_constants) + lepton_energy_density(mue_no_axion)
        
            [proton_frac_axion, baryon_density_axion] = fsolve(skyrme_proton_frac_density_solver, proton_frac_guess_axion, 
                args = (mub, FTHETA_MIN, constants_axion, axion_constants))

            kfp_axion = (3*pi**2 * baryon_density_axion * proton_frac_axion)**(1/3)
            kfn_axion = (3*pi**2 * baryon_density_axion * (1 - proton_frac_axion))**(1/3)
            mue_axion = fsolve(mue_solver, kfp_axion, args = (kfp_axion))[0]

            pressure_axion = skyrme_pressure_axion(proton_frac_axion, baryon_density_axion, FTHETA_MIN, constants_axion, axion_constants) + lepton_pressure(mue_axion)
            energy_density_axion = skyrme_energy_density_axion(proton_frac_axion, baryon_density_axion, FTHETA_MIN, constants_axion, axion_constants) + lepton_energy_density(mue_axion)

            if pressure_axion > pressure_no_axion:
                return [kfp_axion, kfn_axion, mue_axion, energy_density_axion, pressure_axion, FTHETA_MIN]
            return [kfp_no_axion, kfn_no_axion, mue_no_axion, energy_density_no_axion, pressure_no_axion, 1]
        else:
            #use saturation of LECs to modify x0 and t0
            t0_new = self.t0 + (self.d1s0 + self.d3s1) / 2 * PION_MASS**2 * (ftheta_val - 1)
            x0_new = (self.x0 + (self.d3s1 - self.d1s0) / (2 * self.t0) * PION_MASS**2 * (ftheta_val - 1) ) / ( 
                1 + (self.d1s0 + self.d3s1) / (2 * self.t0) * PION_MASS**2 * (ftheta_val - 1))

            constants = [t0_new,self.t1,self.t2,self.t31,self.t32,self.t33,x0_new,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
            axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps]

            if ftheta_val==1:
                [proton_frac, baryon_density] = fsolve(skyrme_proton_frac_density_solver, proton_frac_guess_no_axion, args = (mub, ftheta_val, constants, axion_constants))
            else:
                [proton_frac, baryon_density] = fsolve(skyrme_proton_frac_density_solver, proton_frac_guess_axion, args = (mub, ftheta_val, constants, axion_constants))

            kfp = (3*pi**2 * baryon_density * proton_frac)**(1/3)
            kfn = (3*pi**2 * baryon_density * (1 - proton_frac))**(1/3)
            mue = fsolve(mue_solver, kfp, args = (kfp))[0]

            pressure = skyrme_pressure_axion(proton_frac, baryon_density, ftheta_val, constants, axion_constants) + lepton_pressure(mue)
            energy_density = skyrme_energy_density_axion(proton_frac, baryon_density, ftheta_val, constants, axion_constants) + lepton_energy_density(mue)

            return [kfp, kfn, mue, energy_density, pressure, ftheta_val] 
    def eos_data_density(self, baryon_density, ftheta_val, xp_guess = 0.1, no_leptons = False):
        #use saturation of LECs to modify x0 and t0
        t0_new = self.t0 + (self.d1s0 + self.d3s1) / 2 * PION_MASS**2 * (ftheta_val - 1)
        x0_new = (self.x0 + (self.d3s1 - self.d1s0) / (2 * self.t0) * PION_MASS**2 * (ftheta_val - 1) ) / ( 
            1 + (self.d1s0 + self.d3s1) / (2 * self.t0) * PION_MASS**2 * (ftheta_val - 1))

        constants = [t0_new,self.t1,self.t2,self.t31,self.t32,self.t33,x0_new,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps]

        proton_frac = fsolve(skyrme_proton_frac_solver, xp_guess, args = (baryon_density, ftheta_val, constants, axion_constants))[0]

        kfp = (3*pi**2 * baryon_density * proton_frac)**(1/3)
        kfn = (3*pi**2 * baryon_density * (1 - proton_frac))**(1/3)
        mue = fsolve(mue_solver, kfp, args = (kfp))[0]

        if no_leptons:
            pressure = skyrme_pressure_axion(proton_frac, baryon_density, ftheta_val, constants, axion_constants)
            energy_density = skyrme_energy_density_axion(proton_frac, baryon_density, ftheta_val, constants, axion_constants)
        else:
            pressure = skyrme_pressure_axion(proton_frac, baryon_density, ftheta_val, constants, axion_constants) + lepton_pressure(mue)
            energy_density = skyrme_energy_density_axion(proton_frac, baryon_density, ftheta_val, constants, axion_constants) + lepton_energy_density(mue)

        return [kfp, kfn, mue, energy_density, pressure, ftheta_val] 
    def eos_data_inner_crust(self, mub, ftheta_val = False, nb_guess_norm = 0.02, 
                                 yp_guess_norm = 0.45, nb_guess_ax = 0.1, yp_guess_ax = 0.45, yp_min = 0):
        #ftheta_val must be 0 or ftheta_min if specified
        t0_new = self.t0 + (self.d1s0 + self.d3s1) / 2 * PION_MASS**2 * (FTHETA_MIN - 1)
        x0_new = (self.x0 + (self.d3s1 - self.d1s0) / (2 * self.t0) * PION_MASS**2 * (FTHETA_MIN - 1) ) / ( 
            1 + (self.d1s0 + self.d3s1) / (2 * self.t0) * PION_MASS**2 * (FTHETA_MIN - 1))

        constants_axion = [t0_new,self.t1,self.t2,self.t31,self.t32,self.t33,x0_new,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
        constants_norm = [self.t0,self.t1,self.t2,self.t31,self.t32,self.t33,self.x0,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps]

        #this section finds coexistence of two phases as fxn of mue with and without axions and compares thems
        pnm_pressure_norm = -10
        pnm_pressure_ax = -10

        if not ftheta_val or ftheta_val == 1:
            pnm_nb_norm = fsolve(skyrme_density_solver, nb_guess_norm, args = (mub, 0, 1, constants_norm, axion_constants))[0]
            pnm_pressure_norm = skyrme_pressure_axion(0, pnm_nb_norm, 1, constants_norm, axion_constants)
            yp_norm = fsolve(skyrme_pressure_equil, yp_guess_norm, args = (mub, 1, pnm_pressure_norm, constants_norm, axion_constants, yp_min))
            nucleus_nb = fsolve(skyrme_density_solver, 0.2, args = (mub, yp_norm, 1, constants_norm, axion_constants))[0]
        elif not ftheta_val or ftheta_val == FTHETA_MIN:
            pnm_nb_ax = fsolve(skyrme_density_solver, nb_guess_ax, args = (mub, 0, FTHETA_MIN, constants_axion, axion_constants))[0]
            pnm_pressure_ax = skyrme_pressure_axion(0, pnm_nb_ax, FTHETA_MIN, constants_axion, axion_constants)
            yp_ax = fsolve(skyrme_pressure_equil, yp_guess_ax, args = (mub, FTHETA_MIN, pnm_pressure_ax, constants_axion, axion_constants))[0]
            nucleus_nb = fsolve(skyrme_density_solver, 0.2, args = (mub, yp_ax, FTHETA_MIN, constants_axion, axion_constants))[0]
            
        if pnm_pressure_norm > pnm_pressure_ax:
            ftheta = 1
            nucleus_nb = fsolve(skyrme_density_solver, 0.2, args = (mub, yp_norm, 1, constants_norm, axion_constants))[0]
            mue = skyrme_mue_axion(yp_norm, nucleus_nb, 1, constants_norm, axion_constants)
            kfp = (3 * pi**2 * nucleus_nb * yp_norm)**(1/3)
            kfn_nucleus = (3 * pi**2 * nucleus_nb * (1 - yp_norm))**(1/3)
            kfn_pnm = (3 * pi**2 * pnm_nb_norm)**(1/3)
            
            charge_pnm = -lepton_density(mue)
            charge_nucleus = density(kfp) - lepton_density(mue)
            nucleus_vol_frac = 1 / (1 - charge_nucleus / charge_pnm)

            energy_density = nucleus_vol_frac * skyrme_energy_density_axion(yp_norm, nucleus_nb, 1, constants_norm, axion_constants) + (
                1 - nucleus_vol_frac) * skyrme_energy_density_axion(0, pnm_nb_norm, 1, constants_norm, axion_constants) + lepton_energy_density(mue)
            pressure = pnm_pressure_norm + lepton_pressure(mue)
        else:
            ftheta = FTHETA_MIN
            nucleus_nb = fsolve(skyrme_density_solver, 0.2, args = (mub, yp_ax, FTHETA_MIN, constants_axion, axion_constants))[0]
            mue = skyrme_mue_axion(yp_ax, nucleus_nb, FTHETA_MIN, constants_axion, axion_constants)
            kfp = (3 * pi**2 * nucleus_nb * yp_ax)**(1/3)
            kfn_nucleus = (3 * pi**2 * nucleus_nb * (1 - yp_ax))**(1/3)
            kfn_pnm = (3 * pi**2 * pnm_nb_ax)**(1/3)

            charge_pnm = -lepton_density(mue)
            charge_nucleus = density(kfp) - lepton_density(mue)
            nucleus_vol_frac = 1 / (1 - charge_nucleus / charge_pnm)
            energy_density = nucleus_vol_frac * skyrme_energy_density_axion(yp_ax, nucleus_nb, FTHETA_MIN, constants_axion, axion_constants) + (
                1 - nucleus_vol_frac) * skyrme_energy_density_axion(0, pnm_nb_ax, FTHETA_MIN, constants_axion, axion_constants) + lepton_energy_density(mue)
            pressure = pnm_pressure_ax + lepton_pressure(mue)
          
        return [kfp, kfn_nucleus, kfn_pnm, mue, energy_density, pressure, nucleus_vol_frac, ftheta]   
    def free_energy(self, mub, baryon_density, ftheta, proton_frac_guess = 0.1):
        t0_new = self.t0 + (self.d1s0 + self.d3s1) / 2 * PION_MASS**2 * (ftheta - 1)
        x0_new = (self.x0 + (self.d3s1 - self.d1s0) / (2 * self.t0) * PION_MASS**2 * (ftheta - 1) ) / ( 
            1 + (self.d1s0 + self.d3s1) / (2 * self.t0) * PION_MASS**2 * (ftheta - 1))

        constants = [t0_new,self.t1,self.t2,self.t31,self.t32,self.t33,x0_new,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps]

        proton_frac = fsolve(skyrme_proton_frac_solver, proton_frac_guess, args = (baryon_density, ftheta, constants, axion_constants))[0]
        energy_density = skyrme_energy_density_axion(proton_frac, baryon_density, ftheta, constants, axion_constants)

        kfp = (3*pi**2 * baryon_density * proton_frac)**(1/3)
        mue = fsolve(mue_solver, kfp, args = (kfp))[0]
        if mue > MUON_MASS:
            lepton_density = density(mue) + density(sqrt(mue**2 - MUON_MASS**2))
        else:
            lepton_density = density(mue)

        return energy_density - mub * baryon_density + lepton_energy_density(mue) - mue * lepton_density
    def _epb_for_min(self, ftheta, baryon_density, proton_frac):
        t0_new = self.t0 + (self.d1s0 + self.d3s1) / 2 * PION_MASS**2 * (ftheta - 1)
        x0_new = (self.x0 + (self.d3s1 - self.d1s0) / (2 * self.t0) * PION_MASS**2 * (ftheta - 1) ) / ( 
            1 + (self.d1s0 + self.d3s1) / (2 * self.t0) * PION_MASS**2 * (ftheta - 1))

        constants = [t0_new,self.t1,self.t2,self.t31,self.t32,self.t33,x0_new,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps]

        return skyrme_energy_density_axion(proton_frac, baryon_density, ftheta, constants, axion_constants) / baryon_density
    def _ebp_for_beta_min(self, ftheta, baryon_density, no_leptons = False):
        fields_temp = self.eos_data_density(baryon_density, ftheta, no_leptons = no_leptons)
        return fields_temp[3] / baryon_density
    def eos_data_xp(self, baryon_density, proton_frac, no_leptons = False, ftheta_val = False):
        if not ftheta_val:
            ftheta = minimize_scalar(self._epb_for_min, bounds = [FTHETA_MIN, 1], args = (baryon_density, proton_frac)).x
        else:
            ftheta = ftheta_val
        t0_new = self.t0 + (self.d1s0 + self.d3s1) / 2 * PION_MASS**2 * (ftheta - 1)
        x0_new = (self.x0 + (self.d3s1 - self.d1s0) / (2 * self.t0) * PION_MASS**2 * (ftheta - 1) ) / ( 
            1 + (self.d1s0 + self.d3s1) / (2 * self.t0) * PION_MASS**2 * (ftheta - 1))

        constants = [t0_new,self.t1,self.t2,self.t31,self.t32,self.t33,x0_new,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps]

        kfp = (3 * pi**2 * baryon_density * proton_frac)**(1/3)
        kfn = (3 * pi**2 * baryon_density * (1 - proton_frac))**(1/3)
        if kfp < MUON_MASS:
            mue = kfp
        else:
            mue = fsolve(lambda mue: mue**3 + (mue**2 - MUON_MASS**2)**(3/2) - kfp**3, kfp)[0]
            
        if no_leptons:
            energy_density = skyrme_energy_density_axion(proton_frac, baryon_density, ftheta, constants, axion_constants)
            pressure = skyrme_pressure_axion(proton_frac, baryon_density, ftheta, constants, axion_constants)
        else:
            energy_density = skyrme_energy_density_axion(proton_frac, baryon_density, ftheta, constants, axion_constants) + lepton_energy_density(mue)
            pressure = skyrme_pressure_axion(proton_frac, baryon_density, ftheta, constants, axion_constants) + lepton_pressure(mue)

        mub = skyrme_mub_axion(proton_frac, baryon_density, ftheta, constants, axion_constants)
        return [kfp, kfn, mue, mub, energy_density, pressure, ftheta]
    def eos_data_beta_opt(self, baryon_density, no_leptons = False, no_axion = False):
        if no_axion:
            ftheta = 1
        else:
            ftheta = minimize_scalar(self._ebp_for_beta_min, bounds = [FTHETA_MIN, 1], args = (baryon_density, no_leptons)).x
        return self.eos_data_density(baryon_density, ftheta, no_leptons = no_leptons)
    def eos_data_snm(self, baryon_density, ftheta):
        t0_new = self.t0 + (self.d1s0 + self.d3s1) / 2 * PION_MASS**2 * (ftheta - 1)
        x0_new = (self.x0 + (self.d3s1 - self.d1s0) / (2 * self.t0) * PION_MASS**2 * (ftheta - 1) ) / ( 
            1 + (self.d1s0 + self.d3s1) / (2 * self.t0) * PION_MASS**2 * (ftheta - 1))

        constants = [t0_new,self.t1,self.t2,self.t31,self.t32,self.t33,x0_new,self.x1,self.x2,self.x31,self.x32,self.x33,self.sigma1,self.sigma2,self.sigma3]
        axion_constants = [self.sigma_pi_n, self.delta_sigma, self.eps]
        
        proton_frac = 0.5
        kfp = (3 * pi**2 * baryon_density * proton_frac)**(1/3)
        kfn = (3 * pi**2 * baryon_density * (1 - proton_frac))**(1/3)
        mue = skyrme_mue_axion(proton_frac, baryon_density, ftheta, constants, axion_constants)

        pressure = skyrme_pressure_axion(proton_frac, baryon_density, ftheta, constants, axion_constants) + lepton_pressure(mue)
        energy_density = skyrme_energy_density_axion(proton_frac, baryon_density, ftheta, constants, axion_constants) + lepton_energy_density(mue)

        return [kfp, kfn, mue, energy_density, pressure, ftheta]
    def eos_data_outer_crust(self, mub, ftheta, baryon_per_nuc, mue_xp_guess = (1e-3, 1e-2), rho_sat = 0.153, nuc_constants = (16 / 197.3, 25 / 197.3, 17 / 197.3, 0.7 / 197.3)):
        mp = PROTON_MASS + self.sigma_pi_n * (ftheta-1) - self.delta_sigma * (1/ftheta-1)
        mn = NEUTRON_MASS + self.sigma_pi_n * (ftheta-1) + self.delta_sigma * (1/ftheta-1)
        binding_energy, symm_energy, surf_energy, coul_energy = nuc_constants

        mue, xp = fsolve(outer_crust_solver, mue_xp_guess, args = (mub, baryon_per_nuc, mp, mn, list(nuc_constants)))
        kfp = (3 * pi**2 * rho_sat * xp)**(1/3)
        kfn = (3 * pi**2 * rho_sat * (1 - xp))**(1/3)

        no_nuc_charge = -lepton_density(mue)
        nuc_charge = density(kfp) + no_nuc_charge
        nucleus_vol_frac = 1 / (1 - nuc_charge / no_nuc_charge)
        interaction_energy = -binding_energy + symm_energy * (1 - 2 * xp)**2 + \
            surf_energy / baryon_per_nuc**(1/3) + coul_energy * xp**2 * baryon_per_nuc**(2/3)
        energy_density = lepton_energy_density(mue) + nucleus_vol_frac * (rho_sat * xp * (mp + 3 / (10 * mp) * kfp**2) 
            + rho_sat * (1 - xp) * (mn + 3 / (10 * mn) * kfn**2) + interaction_energy * rho_sat)
        pressure = lepton_pressure(mue)

        return [kfp, kfn, mue, energy_density, pressure, nucleus_vol_frac, ftheta]
    
def sly4(axion_constants):
    return skyrmeAxionModel(axion_constants[0],axion_constants[1],axion_constants[2],axion_constants[3],axion_constants[4],t0=-2488.91/197.3,t1=486.82/197.3,t2=-546.39/197.3,t31=13777/197.3,t32=0,t33=0,x0=0.834,x1=-0.344,x2=-1,x31=1.354,x32=0,x33=0,sigma1=1/6,sigma2=0,sigma3=0)

def lns(axion_constants):
    return skyrmeAxionModel(axion_constants[0],axion_constants[1],axion_constants[2],axion_constants[3],axion_constants[4],t0=-2485/197.3,t1=266.7/197.3,t2=-337.1/197.3,t31=14588.2/197.3,t32=0,t33=0,x0=0.06,x1=0.66,x2=-0.95,x31=-0.03,x32=0,x33=0,sigma1=0.17,sigma2=0,sigma3=0)

def gsk1(axion_constants):
    return skyrmeAxionModel(axion_constants[0],axion_constants[1],axion_constants[2],axion_constants[3],axion_constants[4],t0=-1855.5/197.3,t1=397.2/197.3,t2=264/197.3,t31=13858/197.3,t32=-2694.1/197.3,t33=-319.9/197.3,x0=0.12,x1=-1.76,x2=-1.81,x31=0.13,x32=-1.19,x33=-0.46,sigma1=0.33,sigma2=0.67,sigma3=1)
    
