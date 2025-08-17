#arXiv:2410.21590

from math import pi, log10
from numpy import linspace, logspace
from tovmlk import tov_solver, tov_solver_domain_wall
from eosmlk import bps_pressure_fit, bps_energy_density_fit, bps_number_density_fit, bps_mub_fit
from scipy.interpolate import interp1d

#all quantities are in units of fm

NEUTRON_MASS = 939.57/197.3
PROTON_MASS = 938.27/197.3
NUCLEON_MASS = (NEUTRON_MASS + PROTON_MASS) / 2
MUON_MASS = 105.7/197.3
F_PION = 94/197.3 
PION_MASS = 134/197.3
UP_MASS = 2.2/197.3
DOWN_MASS = 4.7/197.3
FTHETA_MIN = (DOWN_MASS - UP_MASS) / (DOWN_MASS + UP_MASS)
MEVFM3_TO_GCM3 = 1.78e12

def _density(kf): return kf**3 / (3 * pi**2)
def _eps0(p): return p / (PION_MASS**2 * F_PION**2 * (1 - FTHETA_MIN))
def _closest_ix(val, list_of_vals): return list(list_of_vals).index(min(list_of_vals, key = lambda x: abs(val - x)))

def axion_crust_surface_rmf(constants, mub_range, mub_outer, mub_inner, mass_number = 56, ftheta_val = FTHETA_MIN, outer_guess = (0.2, 0.2, -0.02, 0.8, 0.8)):
    eps = []
    n0 = []
    pressure = []
    ed = []

    for mub in mub_range:
        if mub > mub_inner:
            results_temp = constants.eos_data(mub, ftheta_val = ftheta_val)
            eps.append(_eps0(results_temp[4]))
            n0.append(_density(results_temp[0])+ _density(results_temp[1]))
            pressure.append(results_temp[4])
            ed.append(results_temp[3])
        elif mub > mub_outer:
            results_temp = constants.eos_data_inner_crust(mub, ftheta_val = ftheta_val)
            eps.append(_eps0(results_temp[5]))
            n0.append((_density(results_temp[0]) + _density(results_temp[1])) * results_temp[6] + _density(results_temp[2]) * (1 - results_temp[6]))
            pressure.append(results_temp[5])
            ed.append(results_temp[4])
        else:
            results_temp = constants.eos_data_outer_crust(mub, ftheta_val, mass_number)
            eps.append(_eps0(results_temp[4]))
            n0.append((_density(results_temp[0]) + _density(results_temp[1])) * results_temp[5])
            pressure.append(results_temp[4])
            ed.append(results_temp[3])

    return mub_range, eps, n0, pressure, ed

def axion_crust_surface_skyrme(constants, mub_range, mub_outer, mub_inner, mass_number = 56, ftheta_val = FTHETA_MIN):
    eps = []
    n0 = []
    pressure = []
    ed = []

    for mub in mub_range:
        if mub > mub_inner:
            results_temp = constants.eos_data(mub, ftheta_val = ftheta_val)
            eps.append(_eps0(results_temp[4]))
            n0.append(_density(results_temp[0])+ _density(results_temp[1]))
            pressure.append(results_temp[4])
            ed.append(results_temp[3])
        elif mub > mub_outer:
            results_temp = constants.eos_data_inner_crust(mub, ftheta_val = ftheta_val)
            eps.append(_eps0(results_temp[5]))
            n0.append((_density(results_temp[0]) + _density(results_temp[1])) * results_temp[6] + _density(results_temp[2]) * (1 - results_temp[6]))
            pressure.append(results_temp[5])
            ed.append(results_temp[4])
        else:
            results_temp = constants.eos_data_outer_crust(mub, ftheta_val, mass_number)
            eps.append(_eps0(results_temp[4]))
            n0.append((_density(results_temp[0]) + _density(results_temp[1])) * results_temp[5])
            pressure.append(results_temp[4])
            ed.append(results_temp[3])

    return mub_range, eps, n0, pressure, ed

def tail_added(pressure, energy_density, sound_speed_sq, num_points = 100):
    pressure_tail = linspace(pressure[-1] + 0.001, pressure[-1] + 5 * sound_speed_sq, num_points)
    ed_tail = linspace(energy_density[-1] + 0.001, energy_density[-1] + 5, num_points)

    pressure_full = list(pressure) + list(pressure_tail)
    ed_full = list(energy_density) + list (ed_tail)

    return pressure_full, ed_full

p_of_mub = interp1d(bps_mub_fit, bps_pressure_fit)
ed_of_mub = interp1d(bps_mub_fit, bps_energy_density_fit)
nb_of_mub = interp1d(bps_mub_fit, bps_number_density_fit)

def rmf_eos_generator(constants, mub_range, nuc_end_mub = bps_mub_fit[0], no_axion = False):
    energy_density = []
    pressure = []
    number_density = []
    bps_used = []

    for mub in mub_range:
        if no_axion:
            results_temp = constants.eos_data(mub, ftheta_val = 1)
        else:
            results_temp = constants.eos_data(mub)

        if bps_mub_fit[-1] > mub > bps_mub_fit[0]:
            bps_pressure_interp = p_of_mub(mub)
            bps_ed_interp = ed_of_mub(mub)
            bps_nb_interp = nb_of_mub(mub)
            if bps_pressure_interp > results_temp[4] or (mub < nuc_end_mub and results_temp[7] == 1):
                p_temp = bps_pressure_interp
                ed_temp = bps_ed_interp
                nb_temp = bps_nb_interp
                bps_used.append(True)
            else:
                p_temp = results_temp[4]
                ed_temp = results_temp[3]
                nb_temp = _density(results_temp[0]) + _density(results_temp[1])
                bps_used.append(False)
        else:
            p_temp = results_temp[4]
            ed_temp = results_temp[3]
            nb_temp = _density(results_temp[0]) + _density(results_temp[1])
            bps_used.append(False)

        energy_density.append(ed_temp)
        pressure.append(p_temp)
        number_density.append(nb_temp)
    return mub_range, pressure, energy_density, number_density, bps_used

def rmf_inner_crust_eos_generator(constants, mub_range):
    energy_density = []
    pressure = []
    number_density = []
    bps_used = []

    for mub in mub_range:
        results_temp = constants.eos_data_inner_crust(mub, ftheta_val = FTHETA_MIN)

        if bps_mub_fit[-1] > mub > bps_mub_fit[0]:
            bps_pressure_interp = p_of_mub(mub)
            bps_ed_interp = ed_of_mub(mub)
            bps_nb_interp = nb_of_mub(mub)
            if bps_pressure_interp > results_temp[5]:
                pressure.append(bps_pressure_interp)
                energy_density.append(bps_ed_interp)
                number_density.append(bps_nb_interp)
                bps_used.append(True)
            else:
                pressure.append(results_temp[5])
                energy_density.append(results_temp[4])
                number_density.append((_density(results_temp[0]) + _density(results_temp[1])) * results_temp[6]
                                      + _density(results_temp[2]) * (1 - results_temp[6]))
                bps_used.append(False)
        else:
            pressure.append(results_temp[5])
            energy_density.append(results_temp[4])
            number_density.append((_density(results_temp[0]) + _density(results_temp[1])) * results_temp[6]
                                    + _density(results_temp[2]) * (1 - results_temp[6]))
            bps_used.append(False)

    return mub_range, pressure, energy_density, number_density, bps_used

def rmf_outer_crust_eos_generator(constants, mub_range, no_bps = True, nuc_end_mub = bps_mub_fit[0], no_axion = False):
    energy_density = []
    pressure = []
    number_density = []
    bps_used = []
    no_bps_from_now_on = False
    for mub in mub_range:
        if no_axion:
            results_temp = constants.eos_data_outer_crust(mub, ftheta_val = 1)
        else:
            results_temp = constants.eos_data_outer_crust(mub)

        if mub < bps_mub_fit[-1]:
            bps_pressure_interp = p_of_mub(mub)
            bps_ed_interp = ed_of_mub(mub)
            bps_nb_interp = nb_of_mub(mub)
        else:
            no_bps_from_now_on = True

        if (bps_pressure_interp > results_temp[4] or (mub < nuc_end_mub and results_temp[7] == 1)) and not no_bps and not no_bps_from_now_on:
            p_temp = bps_pressure_interp
            ed_temp = bps_ed_interp
            nb_temp = bps_nb_interp
            bps_used.append(True)
        else:
            p_temp = results_temp[4]
            ed_temp = results_temp[3]
            nb_temp = _density(results_temp[0]) + _density(results_temp[1])
            bps_used.append(False)

        energy_density.append(ed_temp)
        pressure.append(p_temp)
        number_density.append(nb_temp)
    return mub_range, pressure, energy_density, number_density, bps_used

def rmf_eos_normal_crust(constants, mub_range, mub_inner, mub_outer):
    energy_density = []
    pressure = []
    number_density = []
    is_crust = []
    for mub in mub_range:
        if mub > mub_inner:
            results_temp = constants.eos_data(mub, ftheta_val = 1)
            p_temp = results_temp[4]
            ed_temp = results_temp[3]
            n_temp = _density(results_temp[0]) + _density(results_temp[1])
            crust_temp = False
        elif mub > mub_outer:
            results_temp = constants.eos_data_inner_crust(mub, ftheta_val = 1)
            p_temp = results_temp[5]
            ed_temp = results_temp[4]
            n_temp = (_density(results_temp[0]) + _density(results_temp[1])) * (1 - results_temp[6]) + (
                _density(results_temp[2]) * results_temp[6])
            crust_temp = True
        else:
            results_temp = constants.eos_data_outer_crust(mub, ftheta_val = 1)
            p_temp = results_temp[4]
            ed_temp = results_temp[3]
            n_temp = (_density(results_temp[0]) + _density(results_temp[1])) * results_temp[5]
            crust_temp = True

        results_temp_axion = constants.eos_data(mub, ftheta_val = FTHETA_MIN)
        if results_temp_axion[4] > p_temp:
            energy_density.append(results_temp_axion[3])
            pressure.append(results_temp_axion[3])
            number_density.append(_density(results_temp_axion[0]) + _density(results_temp_axion[1]))
            is_crust.append(False)
        else:
            energy_density.append(ed_temp)
            pressure.append(p_temp)
            number_density.append(n_temp)
            is_crust.append(crust_temp)

    return mub_range, pressure, energy_density, number_density, is_crust

def skyrme_eos_generator(constants, mub_range, no_axion = False):
    energy_density = []
    pressure = []
    bps_used = []

    for mub in mub_range:
        if mub > 70/197.3 + NUCLEON_MASS:
            guess = (0.08, 0.25)
        else:
            guess = (0.05, 0.12)

        if no_axion:
            results_temp = constants.eos_data(mub, guess, ftheta_val = 1)
        else:
            results_temp = constants.eos_data(mub, guess)

        if bps_mub_fit[0] < mub < bps_mub_fit[-1]:
            p_bps = p_of_mub(mub)
            if p_bps > results_temp[4] or no_axion:
                pressure.append(p_bps)
                energy_density.append(ed_of_mub(mub))
                bps_used.append(True)
            else:
                energy_density.append(results_temp[3])
                pressure.append(results_temp[4])
                bps_used.append(False)
        else:
            energy_density.append(results_temp[3])
            pressure.append(results_temp[4])
            bps_used.append(False)

    return energy_density, pressure, bps_used

def skyrme_inner_crust_eos_generator(constants, mub_range):
    energy_density = []
    pressure = []
    number_density = []
    bps_used = []

    for mub in mub_range:
        if mub < NUCLEON_MASS - 18 / 197.3:
            results_temp = constants.eos_data_inner_crust(mub, ftheta_val = FTHETA_MIN, yp_guess_norm = 0.4, yp_guess_ax = 0.4)
        else:
            results_temp = constants.eos_data_inner_crust(mub, ftheta_val = FTHETA_MIN)
        if bps_mub_fit[-1] > mub > bps_mub_fit[0]:
            bps_pressure_interp = p_of_mub(mub)
            bps_ed_interp = ed_of_mub(mub)
            bps_nb_interp = nb_of_mub(mub)
            if bps_pressure_interp > results_temp[5]:
                pressure.append(bps_pressure_interp)
                energy_density.append(bps_ed_interp)
                number_density.append(bps_nb_interp)
                bps_used.append(True)
            else:
                pressure.append(results_temp[5])
                energy_density.append(results_temp[4])
                number_density.append((_density(results_temp[0]) + _density(results_temp[1])) * results_temp[6]
                                      + _density(results_temp[2]) * (1 - results_temp[6]))
                bps_used.append(False)
        else:
            pressure.append(results_temp[5])
            energy_density.append(results_temp[4])
            number_density.append((_density(results_temp[0]) + _density(results_temp[1])) * results_temp[6]
                                    + _density(results_temp[2]) * (1 - results_temp[6]))
            bps_used.append(False)

    return mub_range, pressure, energy_density, number_density, bps_used

def crust_data_constructor(inputs):
    cp, pressure, energy_density, crust_pressure, drip_pressure, axion_ed_surf = inputs
    
    results_temp = tov_solver(cp, pressure, energy_density, num_steps = 10**4)
    radius = results_temp.radius_range[-1]
    mass = results_temp.contained_mass[-1]

    r_of_p = interp1d(results_temp.pressure, results_temp.radius_range, fill_value = 'extrapolate')
    m_of_p = interp1d(results_temp.pressure, results_temp.contained_mass, fill_value = 'extrapolate')
    p_of_ed = interp1d(energy_density, pressure, fill_value = 'extrapolate')
    eos = interp1d(pressure, energy_density, fill_value = 'extrapolate')
    if eos(results_temp.pressure[-1]) - axion_ed_surf > 1e10 / MEVFM3_TO_GCM3 / 1000:
        r_env = r_of_p(p_of_ed(1e10 / MEVFM3_TO_GCM3 / 1000 + axion_ed_surf))
        env_thickness = radius - r_env
    else:
        print('no envelope, surf p = ' + str(results_temp.pressure[-1]))
        env_thickness = 0

    if crust_pressure > 0:
        delta_rad = radius - r_of_p(crust_pressure)
        delta_mass = mass - m_of_p(crust_pressure)
    else:
        delta_rad = 0
        delta_mass = 0

    if drip_pressure > 0:
        outer_rad = radius - r_of_p(drip_pressure)
        outer_mass = mass - m_of_p(drip_pressure)
    else:
        outer_rad = 0
        outer_mass = 0

    return [radius, mass, delta_rad, delta_mass, cp, outer_rad, outer_mass, env_thickness]

def crust_data_constructor_dw(inputs):
    cp, pressure, energy_density, crust_pressure, drip_pressure, transition_pressure, dw_tension = inputs
    
    results_temp, p_out, p_in = tov_solver_domain_wall(cp, transition_pressure, dw_tension, pressure, energy_density, num_steps = 10**4)
    radius = results_temp.radius_range[-1]
    mass = results_temp.contained_mass[-1]
    
    r_of_p = interp1d(results_temp.pressure, results_temp.radius_range, fill_value = 'extrapolate')
    m_of_p = interp1d(results_temp.pressure, results_temp.contained_mass, fill_value = 'extrapolate')

    if crust_pressure > 0:
        delta_rad = radius - r_of_p(crust_pressure)
        delta_mass = mass - m_of_p(crust_pressure)
    else:
        delta_rad = 0
        delta_mass = 0

    if drip_pressure > 0:
        outer_rad = radius - r_of_p(drip_pressure)
        outer_mass = mass - m_of_p(drip_pressure)
    else:
        outer_rad = 0
        outer_mass = 0

    return [radius, mass, delta_rad, delta_mass, cp, outer_rad, outer_mass]

def phase_diagram_boundaries(constants, ax_crust_core_mub, norm_crust_core_mub, upper_mub, eps_min):
    bps_mub_min = bps_mub_fit[0]    
    eps_range_crust_crust = []
    eps_range_crust_core = []
    eps_range_core_core = []
    p_coex_crust_crust = []
    p_coex_crust_core = []
    p_coex_core_core = []

    n_edge_norm = []
    n_edge_ax = []

    p_of_mu_bps = interp1d(bps_mub_fit, bps_pressure_fit, fill_value = 'extrapolate')
    n_of_mu_bps = interp1d(bps_mub_fit, bps_number_density_fit, fill_value='extrapolate')

    #make boundary line between axion crust and normal crust
    if not ax_crust_core_mub or ax_crust_core_mub < bps_mub_min:
        p_coex_crust_crust = []
        eps_range_crust_crust = []
    else:
        mub_range_crust_crust = linspace(bps_mub_min, ax_crust_core_mub, 10**3)
        for mub in mub_range_crust_crust:
            p_bps = p_of_mu_bps(mub)
            p_coex_crust_crust.append(197.3 * p_bps)
            n_edge_norm.append(n_of_mu_bps(mub))

            results_temp = constants.eos_data_inner_crust(mub, ftheta_val = FTHETA_MIN)
            p_ax = results_temp[5]
            n_edge_ax.append(_density(results_temp[0]) + _density(results_temp[1]))

            eps_range_crust_crust.append(_eps0(p_ax - p_bps))

    #make boundary line between axion crust and axion core
    if not ax_crust_core_mub:
        eps_range_ax_crust = []
        p_coex_ax_crust = []
    else:
        crust_boundary_data = constants.eos_data(ax_crust_core_mub, ftheta_val = FTHETA_MIN)
        p_crust_boundary = crust_boundary_data[4]
        if len(eps_range_crust_crust) > 0:
            eps_range_ax_crust = logspace(log10(eps_min), log10(eps_range_crust_crust[-1]), 10**3)
        else:
            eps_range_ax_crust = logspace(log10(eps_min), log10(0.3), 10**3)
        p_coex_ax_crust = [197.3 * (p_crust_boundary - eps * F_PION**2 * PION_MASS**2 * (1 - FTHETA_MIN)) for eps in eps_range_ax_crust]

    #make boundary line between axion core and normal crust
    if not ax_crust_core_mub:
        mub_range_crust_core = linspace(NUCLEON_MASS - 27.7 / 197.3, norm_crust_core_mub, 10**3)
    else:
        mub_range_crust_core = linspace(ax_crust_core_mub, norm_crust_core_mub, 10**3)
    
    for mub in mub_range_crust_core:
        p_bps = p_of_mu_bps(mub)
        p_coex_crust_core.append(197.3 * p_bps)
        n_edge_norm.append(n_of_mu_bps(mub))

        results_temp = constants.eos_data(mub, ftheta_val = FTHETA_MIN)
        p_ax = results_temp[4]
        n_edge_ax.append(_density(results_temp[0]) + _density(results_temp[1]))

        eps_range_crust_core.append(_eps0(p_ax - p_bps))

    #make boundary line between axion core and normal core
    mub_range_core_core = linspace(norm_crust_core_mub, upper_mub, 10**3)
    for mub in mub_range_core_core:
        results_norm = constants.eos_data(mub, ftheta_val=1)
        p_norm = results_norm[4]
        p_coex_core_core.append(197.3 * p_norm)
        n_edge_norm.append(_density(results_norm[0]) + _density(results_norm[1]))

        results_ax = constants.eos_data(mub, ftheta_val=FTHETA_MIN)
        p_ax = results_ax[4]
        n_edge_ax.append(_density(results_ax[0]) + _density(results_ax[1]))

        eps_range_core_core.append(_eps0(p_ax - p_norm))

    p_nd_norm = [bps_pressure_fit[38] * 197.3, bps_pressure_fit[38] * 197.3]

    if p_coex_crust_core[0] < bps_pressure_fit[38]:
        eps_nd_norm = [eps_range_crust_core[_closest_ix(bps_pressure_fit[38] * 197.3, p_coex_crust_core)], 1]
    else:
        eps_nd_norm = [eps_range_crust_crust[_closest_ix(bps_pressure_fit[38] * 197.3, p_coex_crust_crust)], 1]

    return ([eps_range_ax_crust, p_coex_ax_crust], [eps_range_crust_crust, p_coex_crust_crust], 
            [eps_range_crust_core, p_coex_crust_core], [eps_range_core_core, p_coex_core_core],
            [eps_nd_norm, p_nd_norm], n_edge_ax, n_edge_norm)

def phase_diagram_lines(constants, mub_range_ax_crust, ax_crust_core_mub, ax_inner_outer_mub, 
                        p_coex_full, eps_range_full, n_edge_ax, n_edge_norm, density_range_crust, density_range_norm):
    dump1, eps_ax_crust, n0_ax_crust, pressure_crust, dump2 = axion_crust_surface_rmf(constants, 
        mub_range_ax_crust, ax_inner_outer_mub, ax_crust_core_mub)
    
    axion_lines = []
    normal_lines = []

    for density_temp in density_range_crust:
        axion_ix = _closest_ix(density_temp, n0_ax_crust)
        eps_range_temp = logspace(-4, log10(eps_ax_crust[axion_ix]), 100)
        p_range_temp = [- 197.3 * (elt - eps_ax_crust[axion_ix]) * F_PION**2 * PION_MASS**2 * (1 - FTHETA_MIN) for elt in eps_range_temp]
        axion_lines.append([eps_range_temp, p_range_temp])

        no_axion_ix = _closest_ix(density_temp, n_edge_norm)
        normal_lines.append([[eps_range_full[no_axion_ix], 1],[p_coex_full[no_axion_ix], p_coex_full[no_axion_ix]]])

    for density_temp in density_range_norm:
        axion_ix = _closest_ix(density_temp, n_edge_ax)
        eps_range_temp = logspace(-4, log10(eps_range_full[axion_ix]), 100)
        p_range_temp = [p_coex_full[axion_ix] - 197.3 * (elt - eps_range_full[axion_ix]) * F_PION**2 * PION_MASS**2 * (1 - FTHETA_MIN) for elt in eps_range_temp]

        axion_lines.append([eps_range_temp, p_range_temp])

        no_axion_ix = _closest_ix(density_temp, n_edge_norm)
        normal_lines.append([[eps_range_full[no_axion_ix], 1],[p_coex_full[no_axion_ix], p_coex_full[no_axion_ix]]])

    return normal_lines, axion_lines

def pressure_lines(p_coex_full, eps_range_full):
    eps_range_surf_tension = list(logspace(-8, 0, 10**3))
    factor = F_PION**2 * PION_MASS**2 * UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * 197.3**2 / 12e3
    p_ma8 = [elt * factor / 10 for elt in eps_range_surf_tension]
    p_ma6 = [elt * factor / 10**3 for elt in eps_range_surf_tension]

    p8_found = False
    p6_found = False

    for ix, eps in enumerate(eps_range_surf_tension):
        if eps > eps_range_full[0]:
            p_coex_temp = p_coex_full[_closest_ix(eps, eps_range_full)]
            eps_coex_temp = eps_range_full[_closest_ix(eps, eps_range_full)]
            if p_ma6[ix] < p_coex_temp and not p6_found:
                p6_end_ix = ix
                p6_found = True
                p6_add = p_coex_temp
                eps6_add = eps_coex_temp
            if p_ma8[ix] < p_coex_temp and not p8_found:
                p8_end_ix = ix
                p8_found = True
                p8_add = p_coex_temp
                eps8_add = eps_coex_temp

    eps_surf_6 = eps_range_surf_tension[:p6_end_ix] + [eps6_add,]
    p_ma6_short = p_ma6[:p6_end_ix] + [p6_add,]
    eps_surf_8 = eps_range_surf_tension[:p8_end_ix] + [eps8_add,]
    p_ma8_short = p_ma8[:p8_end_ix] + [p8_add,]

    return ([eps_surf_8, p_ma8_short], [eps_surf_6, p_ma6_short])