#arXiv:2410.21590

#solves TOV equations and calculates tidal deformability for a given equation of state
#includes two fluid solver

from numpy import where, linspace, logical_and, append, array as np_array, sqrt as np_sqrt, exp as np_exp, diag, gradient, newaxis, zeros, diff, concatenate, ones, vstack, log as np_log
from math import pi, log, floor, sin, sqrt, exp
from scipy.integrate import solve_ivp, simpson, solve_bvp
from scipy.linalg import eigvals
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

F_PION = 94/197.3 
PION_MASS = 134/197.3
UP_MASS = 2.2/197.3
DOWN_MASS = 4.7/197.3
FTHETA_MIN = (DOWN_MASS - UP_MASS) / (DOWN_MASS + UP_MASS)
FINE_STRUCTURE = 1 / 137
#used in tov solver, comes up because of m / r factor out front
MSOL_TO_SCHWARZ = 0.5
#convertz r**3 * pressure (in schwarz rad and Gev/fm^3) to solar masses
PR_CUBE_TO_MSOL = 1/41.3
#prefactor on phi in metric
PHI_PREFACTOR = 0.5
#Schwarzchild radius of sun, used in tov solvers
SUN_SCHWARZ_RAD = 2.95
#unit conversions
FA2_OVER_R2_TO_GEVFM3 = 1e-6 / SUN_SCHWARZ_RAD**2 / 0.1973
NUCLEON_MASS = 938 / 197.3

def closest_ix(val, list_of_vals): return list(list_of_vals).index(min(list_of_vals, key = lambda x: abs(val - x)))

def ftheta_func(theta): return sqrt(1 - 4 * UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * sin(theta / 2)**2)

#class tovData is output of tov solver, has vectors of radius, pressure, contained mass within radius, and phi value found in metric
#phi is normalized to zero at r=inf, radius in km, pressure in Gev/fm^3, contained mass in solar masses
class tovDataAxion:
    def __init__(self, radius_range, pressure, contained_mass, tov_phi, theta):
        self.radius_range = radius_range
        self.pressure = pressure
        self.contained_mass = contained_mass
        self.tov_phi = tov_phi
        self.theta = theta

# mass in solar masses, radius in units of scharzchild radius of sun
# pressure and energy density in GeV/fm^3, fa in units 10^15 GeV, sig_pin and delta_sigma in fm^-1, scalar density in fm^-3
def _axion_tov_diff_eq(radius, inputs, pressure_full, energy_density_full, theta_full, proton_scalar_density, neutron_scalar_density, fa15, eps, sig_pin, delta_sigma, p_steps): 
    pressure, contained_mass, phi, theta, thetapr = inputs
    t_ix = closest_ix(theta, theta_full[::p_steps])
    shift = eps * PION_MASS**2 * F_PION**2 * (1 - ftheta_func(theta))
    p_temp = [elt - shift for elt in pressure_full[t_ix * p_steps: (t_ix + 1) * p_steps]]
    ed_temp = [elt + shift for elt in  energy_density_full[t_ix * p_steps: (t_ix + 1) * p_steps]]
    eos = interp1d(p_temp, ed_temp, fill_value = 'extrapolate')
    psd = interp1d(p_temp, proton_scalar_density[t_ix * p_steps: (t_ix + 1) * p_steps], fill_value = 'extrapolate')
    nsd = interp1d(p_temp, neutron_scalar_density[t_ix * p_steps: (t_ix + 1) * p_steps], fill_value = 'extrapolate')

    ftheta = sqrt(1 - 4 * UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * sin(theta / 2)**2)
    ftheta_pr = - UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * sin(theta) / ftheta
    uf = ftheta_pr * 0.1973 * (-eps * F_PION**2 * PION_MASS**2 + sig_pin * (psd(pressure) + nsd(pressure)) 
                      + delta_sigma / ftheta**2 * (psd(pressure) - nsd(pressure)))
    uf_no_matter = ftheta_pr * 0.1973 * (-eps * F_PION**2 * PION_MASS**2)
    print(uf, uf_no_matter, thetapr, 'uutpr')

    #solve up until pressure is negative, after that everything is constant (and will be cut off)
    if pressure > 0:
        #derivatives given by tov equations
        spatial_correction = 1 - contained_mass / radius
        contained_mass_correction = (1 + 4 * pi * PR_CUBE_TO_MSOL * radius**3 / contained_mass * (pressure + fa15**2 * FA2_OVER_R2_TO_GEVFM3 * thetapr**2 / 2 * spatial_correction))
        phi_prime = PHI_PREFACTOR  * contained_mass * contained_mass_correction / (radius**2 * spatial_correction)
        pressure_prime = - (pressure + eos(pressure)) * phi_prime - thetapr * uf
        mass_prime = 4 * pi * PR_CUBE_TO_MSOL * radius**2 * (eos(pressure) + thetapr**2 * fa15**2 * FA2_OVER_R2_TO_GEVFM3 / 2 * spatial_correction)
        theta_prpr = (- 2 * thetapr / radius * (1 - MSOL_TO_SCHWARZ * contained_mass / radius - pi * PR_CUBE_TO_MSOL * radius**2 * (eos(pressure) - pressure)) 
                      + uf / (fa15**2 * FA2_OVER_R2_TO_GEVFM3)) / spatial_correction
        print(pressure, radius * 2.95, theta, thetapr, theta_prpr, 'prttprtprpr')
        return [pressure_prime, mass_prime, phi_prime, thetapr, theta_prpr]
    else:
        spatial_correction = 1 - contained_mass / radius
        contained_mass_correction = (1 + 4 * pi * PR_CUBE_TO_MSOL * radius**3 / contained_mass * (fa15**2 * FA2_OVER_R2_TO_GEVFM3 * thetapr**2 / 2 * spatial_correction))
        phi_prime = PHI_PREFACTOR  * contained_mass * contained_mass_correction / (radius**2 * spatial_correction)
        axion_ed = eps * F_PION**2 * PION_MASS**2 * (1 - ftheta)
        theta_prpr = (- 2 * thetapr / radius * (1 - MSOL_TO_SCHWARZ * contained_mass / radius - pi * PR_CUBE_TO_MSOL * radius**2 * 2 * axion_ed) 
                      + uf_no_matter / (fa15**2 * FA2_OVER_R2_TO_GEVFM3)) / spatial_correction
        mass_prime = 4 * pi * PR_CUBE_TO_MSOL * radius**2 * (thetapr**2 * fa15**2 * FA2_OVER_R2_TO_GEVFM3 / 2 * spatial_correction)
        print(pressure, radius * 2.95, theta, thetapr, theta_prpr, 'prttprtprpr')
        return [0, mass_prime, phi_prime, thetapr, theta_prpr]
    
def _axion_bounded_de(radius, inputs, theta_range, energy_density, pressure, psd, nsd, chi_loc, ns_mass, fa15, eps, sig_pin, delta_sigma): 
    y, ypr = inputs
    if -100 < y < 100:
        theta = pi / (1 + exp(y))
    elif y > 100:
        theta = 1e-5
    else:
        theta = pi

    if -100 < y < 100:
        thetapr = -pi * exp(y) / (1 + exp(y))**2 * ypr
    else:
        thetapr = -100

    nsd_func = interp1d(theta_range, nsd)
    psd_func = interp1d(theta_range, psd)
    ed_func = interp1d(theta_range, energy_density)
    p_func = interp1d(theta_range, pressure)
    chi_loc_func = interp1d(theta_range, chi_loc)

    ftheta = sqrt(1 - 4 * UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * sin(theta / 2)**2)
    ftheta_pr = - UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * sin(theta) / ftheta
    uf = ftheta_pr * 0.1973 * (-eps * F_PION**2 * PION_MASS**2 + sig_pin * (1 + chi_loc_func(theta)) * (psd_func(theta) + nsd_func(theta)) 
                      + delta_sigma / ftheta**2 * (psd_func(theta) - nsd_func(theta)))

    #derivatives given by tov equations
    spatial_correction = 1 - ns_mass / radius
    theta_prpr = (- 2 * thetapr / radius * (1 - MSOL_TO_SCHWARZ * ns_mass / radius - pi * PR_CUBE_TO_MSOL * radius**2 * (ed_func(theta) - p_func(theta))) 
                    + uf / (fa15**2 * FA2_OVER_R2_TO_GEVFM3)) / spatial_correction
    y_prpr = theta_prpr * ypr / thetapr - 2 * thetapr * ypr / theta - ypr**2
    return [ypr, y_prpr]

def _ax_tov_bounded_wrapper(r_list, inputs_list, theta_range, energy_density, pressure, psd, nsd, ns_mass, fa15, eps, sig_pin, delta_sigma):
    output = []
    y_list, ypr_list = inputs_list
    for ix, r_temp in enumerate(r_list):
        output.append(_axion_bounded_de(r_temp, [y_list[ix], ypr_list[ix]], theta_range, energy_density, pressure, psd, nsd, ns_mass, fa15, eps, sig_pin, delta_sigma))
    return np_array(output).T

def bc_bounded(xa, xb, y0, y_end):
    return [xa[0] - y0, xb[0] - y_end]

#solves the tov equations for a given central pressure and equation of state, pressure and energy densities in GeV/fm^3
def axion_tov_solver(central_pressure, central_theta, eos_data, axion_data, num_steps = 500, len_r_eval = 10, eval_steps = 10**6, p_steps = 13000, rtol = 1e-10, atol = 1e-9):
    pressure_full, energy_density_full, theta_full, psd, nsd = eos_data
    fa15, eps, sig_pin, delta_sigma = axion_data
    print('eps', eps)
    #interpolating function for equation of state
    t_ix = closest_ix(central_theta, theta_full[::p_steps])
    shift = eps * PION_MASS**2 * F_PION**2 * (1 - ftheta_func(central_theta))
    p_temp = [elt - shift for elt in pressure_full[t_ix * p_steps: (t_ix + 1) * p_steps]]
    ed_temp = [elt + shift for elt in  energy_density_full[t_ix * p_steps: (t_ix + 1) * p_steps]]
    eos_temp = interp1d(p_temp, ed_temp)
    #initial radius and contained mass
    r0 = 1e-8
    m0 = PR_CUBE_TO_MSOL * eos_temp(central_pressure) * 4 / 3 * pi * r0**3
    #solve TOV equations out to 10 * Schwarz radius of the sun (~30km), use many r steps so that they can be sampled over when done
    solution = solve_ivp(_axion_tov_diff_eq, [r0, len_r_eval], [central_pressure, m0, 0, central_theta, 0],
        t_eval = linspace(r0, len_r_eval, eval_steps), 
        args = (pressure_full, energy_density_full, theta_full, psd, nsd, fa15, eps, sig_pin, delta_sigma, p_steps),
        rtol = rtol, atol = atol)
    
    #find the edge of the star where pressure goes below zero
    if solution.y[3][-1] <= 0.0001:
        edge_index = where(np_array([elt <= 0 and solution.y[3][ix] <= 0.0001 for ix, elt in enumerate(solution.y[0])]))[0][0]
    else:
        edge_index = -1
    #sample tov solution data to have num_steps points
    step_size = max([1, int(floor(edge_index / num_steps))])
    #convert radius to km
    radius_range = solution.t[0:edge_index:step_size] * SUN_SCHWARZ_RAD
    pressure = solution.y[0][0:edge_index:step_size]
    contained_mass = solution.y[1][0:edge_index:step_size]
    theta = solution.y[3][0:edge_index:step_size]
    #if pi < theta[-1] < 3 * pi:
    #    theta = [2 * pi - elt for elt in theta]
    #elif theta[-1] > 3 * pi or theta[-1] < -pi:
    #    print('theta in unexpected domain')

    #set e^phi to sqrt(1 - 2GM/r) at edge of star
    phi_offset = 1/2 * log(1 - solution.y[1][edge_index] / solution.t[edge_index]) - solution.y[2][edge_index]
    tov_phi = solution.y[2][0:edge_index:step_size] + phi_offset

    return tovDataAxion(radius_range, pressure, contained_mass, tov_phi, theta)

def axion_tov_solver_bounded(r0, m0, y0, y_max, eos_data, axion_data, len_r_eval = 10, eval_steps = 10**6, tol = 1e-10):
    theta_range, energy_density, pressure, psd, nsd = eos_data
    fa15, eps, sig_pin, delta_sigma = axion_data

    init_guess = vstack((linspace(y0, y_max, eval_steps), (y_max - y0) / len_r_eval * ones(eval_steps)))

    solution = solve_bvp(lambda x, y: _ax_tov_bounded_wrapper(x, y, theta_range, energy_density, pressure, psd, nsd, m0, fa15, eps, sig_pin, delta_sigma),
        lambda xa, xb: bc_bounded(xa, xb, y0, y_max), linspace(r0 / SUN_SCHWARZ_RAD, r0 / SUN_SCHWARZ_RAD + len_r_eval, eval_steps), init_guess, tol = tol)
    return solution

def axion_shooter(r0, m0, y0, ypr0, eos_data, axion_data, len_r_eval = 10, eval_steps = 10**4):
    theta_range, energy_density, pressure, psd, nsd, chi_loc = eos_data
    fa15, eps, sig_pin, delta_sigma = axion_data
    solution = solve_ivp(lambda r, x: _axion_bounded_de(r, x, theta_range, energy_density, pressure, psd, nsd, chi_loc,
        m0, fa15, eps, sig_pin, delta_sigma), [r0 / SUN_SCHWARZ_RAD, r0 / SUN_SCHWARZ_RAD + len_r_eval], [y0, ypr0],
        t_eval = linspace(r0 / SUN_SCHWARZ_RAD, r0 / SUN_SCHWARZ_RAD + len_r_eval, eval_steps))
    return solution

def axion_shooter_wrapper(inputs):
    r0, m0, y0, ypr0, eos_data, axion_data, len_r_eval = inputs
    return axion_shooter(r0, m0, y0, ypr0, eos_data, axion_data, len_r_eval)

def _omega_de(r, omompr, bulk_star):
    omega, omegapr = omompr
    j = np_exp(-bulk_star.tov_phi) * np_sqrt(1 - bulk_star.contained_mass / (bulk_star.radius_range / SUN_SCHWARZ_RAD))
    j_of_r = interp1d(bulk_star.radius_range, j)
    jpr_of_r = interp1d(bulk_star.radius_range, gradient(j, bulk_star.radius_range))
    omegaprpr = - (4 * r**3 * (j_of_r(r) * omegapr + jpr_of_r(r) * omega) + r**4 * jpr_of_r(r) * omegapr) / (r**4 * j_of_r(r))
    return [omegapr, omegaprpr]

def crust_moment_of_inertia(bulk_star, ypr, crust_core_in_dw, crust_core_boundary_param, eos_data, axion_data, p_full, ed_full, len_r_eval = 1):
    theta_range, ed_range, p_range, dump1, dump2, dump3 = eos_data
    shooter_sol = axion_shooter(bulk_star.radius_range[-1], bulk_star.contained_mass[-1], -12, ypr, eos_data, axion_data, len_r_eval)
    omega_bulk_sol = solve_ivp(lambda r, x: _omega_de(r, x, bulk_star), [bulk_star.radius_range[0], bulk_star.radius_range[-1]], [1, 0], t_eval = bulk_star.radius_range)
    factor = 1 / (omega_bulk_sol.y[0][-1] + bulk_star.radius_range[-1] * omega_bulk_sol.y[1][-1] / 3)
    omega_bulk = omega_bulk_sol.y[0] * factor
    c1 = bulk_star.radius_range[-1]**3 * (omega_bulk[-1] - 1)
    #print(c1)
    omega_dw = c1 / (SUN_SCHWARZ_RAD * shooter_sol.t)**3 + 1
    theta_of_r = pi / (1 + np_exp(shooter_sol.y[0]))
    ed_of_theta = interp1d(theta_range, ed_range)
    p_of_theta = interp1d(theta_range, p_range)
    ed_dw = ed_of_theta(theta_of_r)
    p_dw = p_of_theta(theta_of_r)

    eos = interp1d(p_full, ed_full)
    integrand_star = 8 * pi / 3 * bulk_star.radius_range**4 * np_exp(-bulk_star.tov_phi) / np_sqrt(
        1 - bulk_star.contained_mass * SUN_SCHWARZ_RAD / bulk_star.radius_range) * (
        bulk_star.pressure + eos(bulk_star.pressure)) * omega_bulk
    integrand_dw_full = 8 * pi / 3 * (SUN_SCHWARZ_RAD * shooter_sol.t)**4 / (1 - bulk_star.contained_mass[-1] * SUN_SCHWARZ_RAD / bulk_star.radius_range[-1]) * (
            ed_dw + p_dw) * omega_dw
    total_moment_of_inertia = simpson(integrand_star, x = bulk_star.radius_range) + simpson(integrand_dw_full, x = SUN_SCHWARZ_RAD * shooter_sol.t)

    if crust_core_in_dw:
        theta_cc = crust_core_boundary_param
        edge_ix = list(theta_of_r).index([elt for elt in theta_of_r if elt < theta_cc][0])
        integrand_dw = 8 * pi / 3 * (SUN_SCHWARZ_RAD * shooter_sol.t[edge_ix:])**4 / (1 - bulk_star.contained_mass[-1] * SUN_SCHWARZ_RAD / bulk_star.radius_range[-1]) * (
            ed_dw[edge_ix:] + p_dw[edge_ix:]) * omega_dw[edge_ix:]
        return simpson(integrand_dw, x = shooter_sol.t[edge_ix:] * SUN_SCHWARZ_RAD), total_moment_of_inertia
    else:
        p_cc = crust_core_boundary_param
        integrand_dw = 8 * pi / 3 * (SUN_SCHWARZ_RAD * shooter_sol.t)**4 / (1 - bulk_star.contained_mass[-1] * SUN_SCHWARZ_RAD / bulk_star.radius_range[-1]) * (
            ed_dw + p_dw) * omega_dw
        if p_cc > bulk_star.pressure[-1]:
            edge_ix = list(bulk_star.pressure).index([elt for elt in bulk_star.pressure if elt < p_cc][0])
            integrand_bulk = 8 * pi / 3 * bulk_star.radius_range[edge_ix:]**4 * np_exp(-bulk_star.tov_phi[edge_ix:]) / np_sqrt(
                1 - bulk_star.contained_mass[edge_ix:] * SUN_SCHWARZ_RAD / bulk_star.radius_range[edge_ix:]) * (
                bulk_star.pressure[edge_ix:] + eos(bulk_star.pressure[edge_ix:])) * omega_bulk[edge_ix:]
            return simpson(integrand_bulk, x = bulk_star.radius_range[edge_ix:]) + simpson(integrand_dw, x = shooter_sol.t * SUN_SCHWARZ_RAD), total_moment_of_inertia
        else:
            return simpson(integrand_dw, x = shooter_sol.t * SUN_SCHWARZ_RAD), total_moment_of_inertia
        
def crust_moment_of_inertia_inside(bulk_star, ypr, rc, theta_cc, eos_data, axion_data, p_full, ed_full, len_r_eval = 1):
    theta_range, ed_range, p_range, dump1, dump2, dump3 = eos_data
    ed_of_theta_dw = interp1d(theta_range, ed_range)
    p_of_theta_dw = interp1d(theta_range, p_range)
    radius_range_inner = np_array([elt for elt in bulk_star.radius_range if elt < rc])
    contained_mass_inner = np_array([elt for ix, elt in enumerate(bulk_star.contained_mass) if bulk_star.radius_range[ix] < rc])
    pressure_inner = np_array([elt for ix, elt in enumerate(bulk_star.pressure) if bulk_star.radius_range[ix] < rc])
    tov_phi_inner = np_array([elt for ix, elt in enumerate(bulk_star.tov_phi) if bulk_star.radius_range[ix] < rc])

    shooter_sol = axion_shooter(radius_range_inner[-1], contained_mass_inner[-1], -12, ypr, eos_data, axion_data, len_r_eval)
    
    theta_of_r = pi / (1 + np_exp(shooter_sol.y[0][1:]))
    radius_range_dw = shooter_sol.t[1:] * SUN_SCHWARZ_RAD
    pressure_dw = p_of_theta_dw(theta_of_r)
    tov_phi_dw = ones(len(shooter_sol.t) - 1) * tov_phi_inner[-1]
    contained_mass_integrand = 4 * pi * shooter_sol.t[1:]**2 * ed_of_theta_dw(theta_of_r) * PR_CUBE_TO_MSOL
    contained_mass_dw = np_array([contained_mass_inner[-1] + simpson(contained_mass_integrand[:ix+1], x = shooter_sol.t[:ix+1])
                                  for ix, dump in enumerate(shooter_sol.t[1:])])

    radius_range_outer = np_array([elt + (shooter_sol.t[-1] - shooter_sol.t[0]) * SUN_SCHWARZ_RAD for elt in bulk_star.radius_range if elt > rc])
    contained_mass_outer = np_array([elt + simpson(contained_mass_integrand, x = shooter_sol.t[1:])
                                     for ix, elt in enumerate(bulk_star.contained_mass) if bulk_star.radius_range[ix] > rc])
    pressure_outer = np_array([elt for ix, elt in enumerate(bulk_star.pressure) if bulk_star.radius_range[ix] > rc])
    tov_phi_outer = np_array([elt for ix, elt in enumerate(bulk_star.tov_phi) if bulk_star.radius_range[ix] > rc])

    full_star = tovDataAxion(concatenate((radius_range_inner, radius_range_dw, radius_range_outer)), 
                             concatenate((pressure_inner, pressure_dw, pressure_outer)),
                             concatenate((contained_mass_inner, contained_mass_dw, contained_mass_outer)),
                             concatenate((tov_phi_inner, tov_phi_dw, tov_phi_outer)), 
                             concatenate((ones(len(radius_range_inner)) * pi, theta_of_r, zeros(len(radius_range_outer)))))

    eos = interp1d(p_full, ed_full)
    ed = concatenate((eos(pressure_inner), ed_of_theta_dw(theta_of_r), eos(pressure_outer)))

    omega_bulk_sol = solve_ivp(lambda r, x: _omega_de(r, x, full_star), [full_star.radius_range[0], full_star.radius_range[-1]], [1, 0], t_eval = full_star.radius_range)
    factor = 1 / (omega_bulk_sol.y[0][-1] + bulk_star.radius_range[-1] * omega_bulk_sol.y[1][-1] / 3)
    omega_bulk = omega_bulk_sol.y[0] * factor

    edge_ix = list(full_star.theta).index([elt for elt in full_star.theta if elt < theta_cc][0])
    integrand_crust = 8 * pi / 3 * full_star.radius_range[edge_ix:]**4 * np_exp(-full_star.tov_phi[edge_ix:]) / np_sqrt(1 - full_star.contained_mass[edge_ix:] * SUN_SCHWARZ_RAD / full_star.radius_range[edge_ix:]) * (
        ed[edge_ix:] + full_star.pressure[edge_ix:]) * omega_bulk[edge_ix:]
    integrand_full = 8 * pi / 3 * full_star.radius_range**4 * np_exp(-full_star.tov_phi) / np_sqrt(1 - full_star.contained_mass * SUN_SCHWARZ_RAD / full_star.radius_range) * (
        ed + full_star.pressure) * omega_bulk
    
    return simpson(integrand_crust, x = full_star.radius_range[edge_ix:]), simpson(integrand_full, x = full_star.radius_range)


def star_baryon_number(tov_results, nb_full, pressure_full):
    nb_of_p =interp1d(pressure_full, nb_full, fill_value='extrapolate')

    radius_np = np_array(tov_results.radius_range)
    pressure_np = np_array(tov_results.pressure)
    nb_np = nb_of_p(pressure_np)
    mass_np = np_array(tov_results.contained_mass)

    nb_vec = nb_np * 4 * pi * radius_np**2 / np_sqrt(1 - mass_np / (radius_np / SUN_SCHWARZ_RAD)) 

    return simpson(nb_vec, radius_np)

#solves for the inverse sound speed squared with a 3 step centered derivative
def _de_dp(pressure, pressure_full, ed_full, dp_frac = 1e-5, singularities = []):
    eos = interp1d(pressure_full, ed_full, fill_value = 'extrapolate')
    dp = pressure * dp_frac

    #if near a singularity, move away from it
    if not all([abs(pressure - elt) > 3 * dp for elt in singularities]):
        print('test')
        closest_singularity = min(singularities, key = lambda x: abs(x - pressure))
        if pressure > closest_singularity:
            if not closest_singularity == max(singularities):
                next_singularity = min([elt for elt in singularities if elt - closest_singularity > 0])
                if next_singularity - closest_singularity < 7 * dp:
                    dp = (next_singularity - closest_singularity) / 7
            pressure = pressure + 3.01 * dp
        else:
            if not closest_singularity == min(singularities):
                prev_singularity = max([elt for elt in singularities if closest_singularity - elt > 0])
                if closest_singularity - prev_singularity < 7 * dp:
                    dp = (closest_singularity - prev_singularity) / 7
            pressure = pressure - 3.01 * dp

    ed_min3 = eos(pressure - 3 * dp)
    ed_min2 = eos(pressure - 2 * dp)
    ed_min1 = eos(pressure - dp)
    ed_pl1 = eos(pressure + dp)
    ed_pl2 = eos(pressure + 2 * dp)
    ed_pl3 = eos(pressure + 3 * dp)
    return (1 / 60 * (ed_pl3 - ed_min3) - 3 / 20 * (ed_pl2 - ed_min2) + 3 / 4 * (ed_pl1 - ed_min1)) / dp  

#differential equation for the tidal deformability
#remember that r is in units of schwarz radius so GM(kg)/c^2 r(kg) = 1/2 M(msol)/r(schwarz)
def _h_diff_eq(radius, h_h_prime, tov_results, pressure_full, ed_full, singularities = []):
    #radius_range in km, contained mass in solar masses, pressure and ed in Gev/fm^3
    #pressure range is as a fxn of r, pressure_full is as a function of energy density
    [h, h_prime] = h_h_prime

    #equation of state, contained_mass and pressure as fxn of radius
    eos = interp1d(pressure_full, ed_full, fill_value = 'extrapolate')
    m_of_r = interp1d(tov_results.radius_range, tov_results.contained_mass)
    p_of_r = interp1d(tov_results.radius_range, tov_results.pressure)

    contained_mass = m_of_r(radius)
    pressure = p_of_r(radius)
    energy_density = eos(pressure)
    radius = radius / SUN_SCHWARZ_RAD

    #radial element of metric
    exp_lambda = 1 / (1 - contained_mass / radius)
    cs_sq = 1 / _de_dp(pressure, pressure_full, ed_full, singularities = singularities)

    phi_prime = PHI_PREFACTOR * (contained_mass + 4 * pi * PR_CUBE_TO_MSOL * radius**3 * pressure) / (
        radius**2 - contained_mass * radius)

    h_prime_factor = 2 / radius + 1 / radius**2 * exp_lambda * (contained_mass
        + 2 * pi * PR_CUBE_TO_MSOL * radius**3 * (pressure - energy_density))
    hFactor = exp_lambda * (-6 / radius**2 + 4 * pi * PR_CUBE_TO_MSOL * MSOL_TO_SCHWARZ * (5 * energy_density + 9 * pressure
        + (energy_density + pressure) / cs_sq)) - (phi_prime / PHI_PREFACTOR)**2

    return [h_prime, - h_prime * h_prime_factor / SUN_SCHWARZ_RAD - h * hFactor / SUN_SCHWARZ_RAD**2]

#calculates the tidal deformability using tov results object and eos
def tidal_deformability(tov_results, pressure_full, ed_full, rtol_deq = 1e-5, atol_deq = 1e-8, singularities = []):
    #this is in the equation for h(0), but it doesn't look like it matters since the final answer depends on hprime/h
    a0 = 1
    #eos
    eos = interp1d(pressure_full, ed_full, fill_value = 'extrapolate')
    #boundary values
    r0 = tov_results.radius_range[0]
    r_max = tov_results.radius_range[-2]
    p0 = tov_results.pressure[0]
    ed0 = eos(tov_results.pressure[0])
    cs0 = 1 / _de_dp(p0, pressure_full, ed_full)

    h0 = a0 * r0**2 * (1 - MSOL_TO_SCHWARZ * 2 * pi / 7 * PR_CUBE_TO_MSOL * (5 * ed0 + 9 * p0 + (ed0 + p0) / cs0) * r0**2)

    h_results = solve_ivp(_h_diff_eq, (r0, r_max), [h0, 0], args = (tov_results, pressure_full, ed_full, singularities), rtol = rtol_deq, atol = atol_deq)

    #y = r * hprime / h at edge of star, c is compactness, need to divide by 2 since RS = 2 G M_sol / c^2 and we want G M_sol / c^2
    y = h_results.t[-1] * h_results.y[1][-1] / h_results.y[0][-1]
    c = tov_results.contained_mass[-1] / (2 * h_results.t[-1] / SUN_SCHWARZ_RAD)
    k2 = 8 * c**5 / 5 * (1 - 2 * c)**2 * (2 + 2 * c * (y - 1) - y) / (2 * c * (6 - 3 * y + 3 * c * (5 * y - 8)) 
        + 4 * c**3 * (13 - 11 * y + c * (3 * y - 2) + 2 * c**2 * (1 + y)) 
        + 3 * (1 - 2 * c)**2 * (2 - y + 2 * c * (y - 1)) * log(1 - 2 * c))
    return 2 / 3 * k2 / c**5

#convenient wrapper to do tov calculation and tidal deformability at the same time
def tov_lambda_solver(central_pressure, pressure_full, energy_density_full, num_steps = 500, rtol_input = 1e-5, atol_input = 1e-8, singularities = []):
    tov_results = axion_tov_solver(central_pressure, pressure_full, energy_density_full, num_steps)
    td = tidal_deformability(tov_results, pressure_full, energy_density_full, rtol_deq = rtol_input, atol_deq = atol_input, singularities = singularities)
    return (tov_results, td)

#solving S-L eq. of form grad(p(r) u'(r)) + (q(r) + lambda (w(r)) u(r) = 0
def _sl_funcs(tov_results, pressure_full, energy_density_full):
    cs_sq = np_array([1 / _de_dp(pressure, pressure_full, energy_density_full) for pressure in tov_results.pressure])
    exp_lambda = 1 / np_sqrt(1 - SUN_SCHWARZ_RAD * tov_results.contained_mass / tov_results.radius_range)
    exp_3phi = np_exp(3 * tov_results.tov_phi)
    eos = interp1d(pressure_full, energy_density_full, fill_value= 'extrapolate')
    ed_array = eos(tov_results.pressure)
    p_plus_e = tov_results.pressure + ed_array
    p_prime = gradient(tov_results.pressure, tov_results.radius_range)

    p = exp_lambda * exp_3phi / tov_results.radius_range**2 * p_plus_e * cs_sq

    q =  - 4 * exp_lambda * exp_3phi / tov_results.radius_range**3 * p_prime \
        - 8 * pi * PR_CUBE_TO_MSOL * MSOL_TO_SCHWARZ / SUN_SCHWARZ_RAD**2 * exp_lambda**3 * exp_3phi / tov_results.radius_range**2 * tov_results.pressure * p_plus_e \
        + exp_lambda * exp_3phi / tov_results.radius_range**2 * p_prime**2 / p_plus_e
    
    w = exp_lambda**3 * np_exp(tov_results.tov_phi) / tov_results.radius_range**2 * p_plus_e
    return p, q, w

def _d2_mat(dx_list):
    size = len(dx_list) + 1
    result = zeros([size, size])
    
    for ix in range(1, size - 1):
        dx_plus = dx_list[ix]
        dx_minus = dx_list[ix - 1]
        
        result[ix, ix] = - 2 / (dx_plus + dx_minus) * (1 / dx_plus + 1 / dx_minus)
        result[ix, ix - 1] = 2 / (dx_minus * (dx_plus + dx_minus)) 
        result[ix, ix + 1] = 2 / (dx_plus * (dx_plus + dx_minus))

    result[0, :3] = result[1, :3]
    result[-1, -3:] = result[-2, -3:]
    return result

def _d1_mat(dx_list):
    size = len(dx_list) + 1
    result = zeros([size, size])

    for ix in range(1, size - 1):
        dx = dx_list[ix] + dx_list[ix - 1]
        result[ix, ix - 1] = - 1 / dx
        result[ix, ix + 1] = 1 / dx
    return result

def normal_modes(tov_results, pressure_full, energy_density_full, num_steps = 500):
    step_size = max([1, int(floor(len(tov_results.radius_range) / num_steps))])
    p, q, w = _sl_funcs(tov_results, pressure_full, energy_density_full)
    p = p[::step_size]
    q = q[::step_size]
    w = w[::step_size]
    #right_mat = - diag(w)
    dr = diff(tov_results.radius_range[::step_size])
    d2 = _d2_mat(dr)
    d1 = _d1_mat(dr)
    left_mat = - d2 * p[:,newaxis] / w[:, newaxis] - d1 * gradient(p, tov_results.radius_range[::step_size])[:,newaxis] / w[:newaxis] - diag(q) / w[:newaxis]
    return eigvals(left_mat)

#stores data for two fluid solution to tov equations, stores total contained mass and contained mass of a second fluid (usually the exotic one)
#phi is normalized to zero at r=inf, radius in km, pressure in Gev/fm^3, contained mass in solar masses
class tovDataTwoFluid:
    def __init__(self, radius_range, pressure1, pressure2, contained_mass, contained_mass2, tov_phi):
        self.radius_range = radius_range
        self.pressure1 = pressure1
        self.pressure2 = pressure2
        self.contained_mass = contained_mass
        self.contained_mass2 = contained_mass2
        self.tov_phi = tov_phi

#mass in solar masses, radius in units of scharzchild radius of sun, pressure and energy density in GeV/fm^3
#WARNING THIS ASSUMES THERE IS NO REGION WHERE ONLY FLUID 2 IS PRESENT
def _tov_diff_eq_two_fluid(radius,pmphi,pressure_full1,energy_density_full1,pressure_full2,energy_density_full2): 
    [pressure1, pressure2, contained_mass] = pmphi[0:3]
    
    #equations of state for two fluids
    eos1=interp1d(pressure_full1, energy_density_full1, fill_value='extrapolate')
    eos2=interp1d(pressure_full2, energy_density_full2, fill_value='extrapolate')

    if pressure1>0:
        if pressure2>0:
            #case one, both fluids are present
            phi_prime = PHI_PREFACTOR * (contained_mass + 4 * pi * PR_CUBE_TO_MSOL * radius**3 * (pressure1 + pressure2)) / (radius**2 - contained_mass * radius)

            pressure1_prime = - (pressure1 + eos1(pressure1)) * phi_prime

            pressure2_prime = - (pressure2 + eos2(pressure2)) * phi_prime

            mass_prime = 4 * pi * PR_CUBE_TO_MSOL * radius**2 * (eos1(pressure1) + eos2(pressure2))
            mass2_prime = 4 * pi * PR_CUBE_TO_MSOL * radius**2 * eos2(pressure2)

            return [pressure1_prime, pressure2_prime, mass_prime, mass2_prime, phi_prime]
        else:
            #case two, only fluid 1 is present
            phi_prime = PHI_PREFACTOR * (contained_mass + 4 * pi * PR_CUBE_TO_MSOL * radius**3 * pressure1) / (radius**2 - contained_mass * radius)

            pressure1_prime = - (pressure1 + eos1(pressure1)) * phi_prime

            mass_prime = 4 * pi * PR_CUBE_TO_MSOL * radius**2 * eos1(pressure1)

            return[pressure1_prime, 0, mass_prime, 0, phi_prime]
    else:
        #end process if both pressures are negative
        return [0,0,0,0,0]

#solves tov equations with two fluids, central pressure and energy density in GeV/fm^3
def tov_solver_two_fluid(central_pressure1, central_pressure2, pressure_full1, energy_density_full1, pressure_full2, energy_density_full2, num_steps = 500, len_r_eval = 10):
    #radii of two fluids
    eos1 = interp1d(pressure_full1, energy_density_full1, fill_value='extrapolate')
    eos2 = interp1d(pressure_full2, energy_density_full2, fill_value='extrapolate')
    
    #initial radius and contained mass
    r0 = 1e-8
    if central_pressure2 == 0:
        central_energy_density = eos1(central_pressure1)
        central_energy_density2 = 0
    else:
        central_energy_density = eos1(central_pressure1) + eos2(central_pressure2)
        central_energy_density2 = eos2(central_pressure2)
    m0 = PR_CUBE_TO_MSOL * central_energy_density * 4 / 3 * pi * r0**3
    m0_2 = PR_CUBE_TO_MSOL * central_energy_density2 * 4 / 3 * pi * r0**3
    
    #integrate out to 10 schwarzchild radii ~ 30km
    solution=solve_ivp(_tov_diff_eq_two_fluid, [r0, len_r_eval], [central_pressure1, central_pressure2, m0,
        m0_2, 0], t_eval=linspace(r0, len_r_eval, 10**6),
        args = (pressure_full1, energy_density_full1, pressure_full2, energy_density_full2),
        rtol=10**-10, atol=10**-9)
    
    #find index where fluid 1 pressure is zero to find edge of star (assumes there is no region where fluid 2 is present and not fluid 1)
    edge_index = where(solution.y[0] <= 0)[0][0]
    #sample parameters num_steps times
    step_size = int(floor(edge_index / num_steps))
    #convert radius to km
    radius_range = [elt * SUN_SCHWARZ_RAD for elt in solution.t[0:edge_index:step_size]]
    pressure_range1 = solution.y[0][0:edge_index:step_size]
    pressure_range2 = solution.y[1][0:edge_index:step_size]
    contained_mass = solution.y[2][0:edge_index:step_size]
    contained_mass2 = solution.y[3][0:edge_index:step_size]

    #sets e^phi to sqrt(1- 2Gm/r) at edge of star
    phi_offset = 1/2 * log(1 - (solution.y[2][edge_index] + solution.y[3][edge_index]) / solution.t[edge_index]) - solution.y[4][edge_index]
    tov_phi = [elt + phi_offset for elt in solution.y[4][0:edge_index:step_size]]


    return tovDataTwoFluid(radius_range, pressure_range1, pressure_range2, contained_mass, contained_mass2, tov_phi)
