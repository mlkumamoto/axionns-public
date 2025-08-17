#arXiv:2410.21590

#solves TOV equations and calculates tidal deformability for a given equation of state
#includes two fluid solver

from numpy import where, linspace, logical_and, append, array as np_array, sqrt as np_sqrt, exp as np_exp, diag, gradient, newaxis, zeros, diff, concatenate
from math import pi, log, floor
from scipy.integrate import solve_ivp, simpson
from scipy.linalg import eigvals
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from mpmath import fabs, mpmathify, odefun, findroot, linspace as mp_linspace
from functools import partial

#used in tov solver, comes up because of m / r factor out front
MSOL_TO_SCHWARZ = 0.5
#convertz r**3 * pressure (in schwarz rad and Gev/fm^3) to solar masses
PR_CUBE_TO_MSOL = 1/41.3
#prefactor on phi in metric
PHI_PREFACTOR = 0.5
#Schwarzchild radius of sun, used in tov solvers
SUN_SCHWARZ_RAD = 2.95

def closest_ix(val, list_of_vals): return list(list_of_vals).index(min(list_of_vals, key = lambda x: abs(val - x)))

#class tovData is output of tov solver, has vectors of radius, pressure, contained mass within radius, and phi value found in metric
#phi is normalized to zero at r=inf, radius in km, pressure in Gev/fm^3, contained mass in solar masses
class tovData:
    def __init__(self, radius_range, pressure, contained_mass, tov_phi):
        self.radius_range = radius_range
        self.pressure = pressure
        self.contained_mass = contained_mass
        self.tov_phi = tov_phi

# mass in solar masses, radius in units of scharzchild radius of sun
# pressure and energy density in GeV/fm^3
def _tov_diff_eq(radius, pmphi, pressure_full, energy_density_full): 
    [pressure, contained_mass] = pmphi[:2]
    eos=interp1d(pressure_full,energy_density_full,fill_value='extrapolate')

    #solve up until pressure is negative, after that everything is constant (and will be cut off)
    if pressure > 0:
        #derivatives given by tov equations
        phi_prime = PHI_PREFACTOR  * (contained_mass + 4 * pi * PR_CUBE_TO_MSOL * radius**3 * pressure) / (radius**2 - contained_mass * radius)
        pressure_prime = - (pressure + eos(pressure)) * phi_prime
        mass_prime = 4 * pi * PR_CUBE_TO_MSOL * radius**2 * eos(pressure)
        return [pressure_prime, mass_prime, phi_prime]
    else:
        return [0,0,0]

#solves the tov equations for a given central pressure and equation of state, pressure and energy densities in GeV/fm^3
def tov_solver(central_pressure, pressure_full, energy_density_full, num_steps = 500, len_r_eval = 10, eval_steps = 10**6):
    #interpolating function for equation of state
    eos = interp1d(pressure_full, energy_density_full)

    #initial radius and contained mass
    r0 = 1e-8
    m0 = PR_CUBE_TO_MSOL * eos(central_pressure) * 4 / 3 * pi * r0**3
    
    #solve TOV equations out to 10 * Schwarz radius of the sun (~30km), use many r steps so that they can be sampled over when done
    solution = solve_ivp(_tov_diff_eq, [r0, len_r_eval], [central_pressure, m0, 0],
        t_eval = linspace(r0, len_r_eval, eval_steps), args = (pressure_full, energy_density_full), rtol=10**-10, atol=10**-9)
    
    #find the edge of the star where pressure goes below zero
    edge_index = where(solution.y[0] <= 0)[0][0]
    #sample tov solution data to have num_steps points
    step_size = max([1, int(floor(edge_index / num_steps))])
    #convert radius to km
    radius_range = solution.t[0:edge_index:step_size] * SUN_SCHWARZ_RAD
    pressure = solution.y[0][0:edge_index:step_size]
    contained_mass = solution.y[1][0:edge_index:step_size]

    #set e^phi to sqrt(1 - 2GM/r) at edge of star
    phi_offset = 1/2 * log(1 - solution.y[1][edge_index] / solution.t[edge_index]) - solution.y[2][edge_index]
    tov_phi = solution.y[2][0:edge_index:step_size] + phi_offset

    return tovData(radius_range, pressure, contained_mass, tov_phi)

def star_baryon_number(tov_results, nb_full, pressure_full):
    nb_of_p =interp1d(pressure_full, nb_full, fill_value='extrapolate')

    radius_np = np_array(tov_results.radius_range)
    pressure_np = np_array(tov_results.pressure)
    nb_np = nb_of_p(pressure_np)
    mass_np = np_array(tov_results.contained_mass)

    nb_vec = nb_np * 4 * pi * radius_np**2 / np_sqrt(1 - mass_np / (radius_np / SUN_SCHWARZ_RAD)) 

    return simpson(nb_vec, radius_np)

#solves the tov equations for a given central pressure and equation of state, pressure and energy densities in GeV/fm^3
def tov_solver_domain_wall(central_pressure, transition_pressure, domain_wall_tension, pressure_full, energy_density_full, num_steps = 500, eval_steps = 10**6, len_r_eval = 10):
    initial_data = tov_solver(central_pressure, pressure_full, energy_density_full, num_steps, len_r_eval, eval_steps)

    if not transition_pressure:
        p_of_r = interp1d(list(initial_data.radius_range), list(initial_data.pressure), fill_value = 'extrapolate')
        transition_r = fsolve(lambda r: p_of_r(r) - domain_wall_tension / (r * SUN_SCHWARZ_RAD), 4)[0]
        transition_ix = closest_ix(transition_r, initial_data.radius_range)
        return tovData(initial_data.radius_range[:transition_ix], initial_data.pressure[:transition_ix], 
            initial_data.contained_mass[:transition_ix], initial_data.tov_phi[:transition_ix]), 0, initial_data.pressure[transition_ix]
    #interpolating function for equation of state
    transition_ix = closest_ix(transition_pressure, initial_data.pressure)
    r0 = initial_data.radius_range[transition_ix - 1] / SUN_SCHWARZ_RAD
    m0 = initial_data.contained_mass[transition_ix - 1]
    #domain wall tension must be in GeV*km/fm^3
    p0 = initial_data.pressure[transition_ix - 1] - domain_wall_tension / (r0 * SUN_SCHWARZ_RAD)
    phi0 = initial_data.tov_phi[transition_ix - 1]

    p_in = initial_data.pressure[transition_ix - 1]
    p_out = p0
    
    #solve TOV equations out to 10 * Schwarz radius of the sun (~30km), use many r steps so that they can be sampled over when done
    solution = solve_ivp(_tov_diff_eq, [r0, len_r_eval], [p0, m0, phi0],
        t_eval = linspace(r0, len_r_eval, eval_steps), args = (pressure_full, energy_density_full), rtol=10**-10, atol=10**-9)
    
    #find the edge of the star where pressure goes below zero
    edge_index = where(solution.y[0] <= 0)[0][0]
    #sample tov solution data to have num_steps points
    step_size = max([1, int(floor(edge_index / num_steps))])
    #convert radius to km
    radius_range = concatenate((initial_data.radius_range[:transition_ix - 1], solution.t[0:edge_index:step_size] * SUN_SCHWARZ_RAD))
    pressure = concatenate((initial_data.pressure[:transition_ix - 1], solution.y[0][0:edge_index:step_size]))
    contained_mass = concatenate((initial_data.contained_mass[:transition_ix - 1], solution.y[1][0:edge_index:step_size]))

    #set e^phi to sqrt(1 - 2GM/r) at edge of star
    phi_offset = 1/2 * log(1 - solution.y[1][edge_index] / solution.t[edge_index]) - solution.y[2][edge_index]
    tov_phi = concatenate((initial_data.tov_phi[:transition_ix - 1], solution.y[2][0:edge_index:step_size])) + phi_offset

    return tovData(radius_range, pressure, contained_mass, tov_phi), p_out, p_in

def nb_matcher(central_pressure_shifted, cp_shift, nb_desired, transition_pressure, domain_wall_tension, pressure_full, energy_density_full, number_density_full,
               num_steps = 10**5, eval_steps = 10**7):
    central_pressure = central_pressure_shifted[0] + cp_shift
    results, p_out, p_in = tov_solver_domain_wall(central_pressure, transition_pressure, domain_wall_tension, pressure_full,
        energy_density_full, num_steps, eval_steps)
    print(central_pressure)
    if p_out < 0:
        print('transition pressure too low resulting in negative pressure')
        return 10
    return star_baryon_number(results, number_density_full, pressure_full) - nb_desired

def _interp1d_mp(x_val, x_data, y_data):
    if not len(x_data) == len(y_data):
        print('length of vectors not the same')
        return
    x_ix = list(x_data).index(min(x_data, key = lambda x: fabs(x_val - x)))

    if x_val < x_data[0]:
        x_plus = x_data[1]
        x_minus = x_data[0]
        y_plus = y_data[1]
        y_minus = y_data[0]
    elif x_val > x_data[-1]:
        x_plus = x_data[-1]
        x_minus = x_data[-2]
        y_plus = y_data[-1]
        y_minus = y_data[-2]
    elif x_val >= x_data[x_ix]:
        x_plus = x_data[x_ix + 1]
        x_minus = x_data[x_ix]
        y_plus = y_data[x_ix + 1]
        y_minus = y_data[x_ix]
    else:
        x_plus = x_data[x_ix]
        x_minus = x_data[x_ix - 1]
        y_plus = y_data[x_ix]
        y_minus = y_data[x_ix - 1]

    return y_minus + (y_plus - y_minus) / (x_plus - x_minus) * (x_val - x_minus)

# mass in solar masses, radius in units of scharzchild radius of sun
# pressure and energy density in GeV/fm^3
def _tov_diff_eq_mp(radius, pmphi, pressure_full, energy_density_full): 
    [pressure, contained_mass] = pmphi[:2]
    eos = interp1d(pressure_full,energy_density_full,fill_value='extrapolate')

    #solve up until pressure is negative, after that everything is constant (and will be cut off)
    if pressure > 0:
        #derivatives given by tov equations
        phi_prime = mpmathify(PHI_PREFACTOR)  * (contained_mass + 4 * mpmathify(pi * PR_CUBE_TO_MSOL) * radius**3 * pressure) / (radius**2 - contained_mass * radius)
        pressure_prime = - (pressure + _interp1d_mp(pressure, pressure_full, energy_density_full)) * phi_prime
        mass_prime = 4 * mpmathify(pi * PR_CUBE_TO_MSOL) * radius**2 * _interp1d_mp(pressure, pressure_full, energy_density_full)
        return [pressure_prime, mass_prime, phi_prime]
    else:
        return [0,0,0]

#calculates TOV solution out to point where pressure - domain wall pressure = max bps pressure
def tov_domain_wall_mp(central_pressure, max_bps_pressure, domain_wall_tension, pressure_full, energy_density_full, num_radii = 10):
    #interpolating function for equation of state
    pressure_mp = [mpmathify(elt) for elt in pressure_full]
    ed_mp = [mpmathify(elt) for elt in energy_density_full]

    ed_c = _interp1d_mp(mpmathify(central_pressure), pressure_mp, ed_mp)

    #initial radius and contained mass
    r0 = 1e-8
    m0 = ed_c * 4 / 3 * mpmathify(pi * PR_CUBE_TO_MSOL) * r0**3
    
    #solve TOV equations out to 10 * Schwarz radius of the sun (~30km), use many r steps so that they can be sampled over when done
    inner_solution = odefun(partial(_tov_diff_eq_mp, pressure_full = pressure_mp, energy_density_full = ed_mp), r0, [mpmathify(central_pressure), m0, 0], tol = 0.01)
    min_dw_radius = findroot(lambda x: inner_solution(x)[0] - mpmathify(domain_wall_tension) / (x * mpmathify(SUN_SCHWARZ_RAD)) - mpmathify(max_bps_pressure), 10)
    max_dw_radius = findroot(lambda x: inner_solution(x)[0] - mpmathify(domain_wall_tension) / (x * mpmathify(SUN_SCHWARZ_RAD)), 10)
    dw_radii = mp_linspace(min_dw_radius, max_dw_radius, num_radii)
    print('inner done')

    outer_data = []

    for boundary_radius in dw_radii:
        p_inner, m0, phi0 = inner_solution(boundary_radius)
        p_outer = p_inner - mpmathify(domain_wall_tension) / (boundary_radius * mpmathify(SUN_SCHWARZ_RAD))
        outer_solution = odefun(partial(_tov_diff_eq_mp, pressure_full = pressure_mp, energy_density_full = ed_mp), boundary_radius, [p_outer, m0, phi0])
        edge_radius = findroot(lambda x: inner_solution(x)[0])
        outer_data.append([outer_solution, boundary_radius, edge_radius])
        print('one outer done')
    return inner_solution, outer_data

"""
    #find the edge of the star where pressure goes below zero
    edge_index = where(logical_and(solution.y[0] - domain_wall_tension / (solution.t * SUN_SCHWARZ_RAD) <= max_bps_pressure, solution.t > 1))[0][0]
    #sample tov solution data to have num_steps points
    step_size = max([1, int(floor(edge_index / num_steps))])
    #convert radius to km
    if step_size == 1:
        radius_range = solution.t[0:edge_index:step_size] * SUN_SCHWARZ_RAD
        pressure = solution.y[0][0:edge_index:step_size]
        contained_mass = solution.y[1][0:edge_index:step_size]

        #set e^phi to sqrt(1 - 2GM/r) at edge of star
        phi_offset = 1/2 * log(1 - solution.y[1][edge_index] / solution.t[edge_index]) - solution.y[2][edge_index]
        tov_phi = solution.y[2][0:edge_index:step_size] + phi_offset
    else:
        radius_range = append(solution.t[0:edge_index:step_size], solution.t[edge_index]) * SUN_SCHWARZ_RAD
        pressure = append(solution.y[0][0:edge_index:step_size], solution.y[0][edge_index])
        contained_mass = append(solution.y[1][0:edge_index:step_size], solution.y[1][edge_index])

        #set e^phi to sqrt(1 - 2GM/r) at edge of star
        phi_offset = 1/2 * log(1 - solution.y[1][edge_index] / solution.t[edge_index]) - solution.y[2][edge_index]
        tov_phi = append(solution.y[2][0:edge_index:step_size], solution.y[2][edge_index]) + phi_offset

    return tovData(radius_range, pressure, contained_mass, tov_phi)


def tov_solver_outer(tov_results_inner, transition_pressure_list, domain_wall_tension, pressure_full, energy_density_full, num_steps = 500, len_r_eval = 10, eval_steps = 10**6):
    #interpolating function for equation of state
    r0 = tov_results_inner.radius_range[-1] / SUN_SCHWARZ_RAD
    m0 = tov_results_inner.contained_mass[-1]
    #domain wall tension must be in GeV*km/fm^3
    p0 = tov_results_inner.pressure[-1]
    phi0 = tov_results_inner.tov_phi[-1]
    
    #solve TOV equations out to 10 * Schwarz radius of the sun (~30km), use many r steps so that they can be sampled over when done
    solution_before = solve_ivp(_tov_diff_eq, [r0, len_r_eval], [p0, m0, phi0],
        t_eval = linspace(r0, len_r_eval, eval_steps), args = (pressure_full, energy_density_full), rtol=10**-10, atol=10**-9)

    tov_data_list = []

    for transition_pressure in transition_pressure_list:
        #interpolating function for equation of state
        transition_ix = closest_ix(transition_pressure, solution_before.y[0])
        r0 = solution_before.t[transition_ix]
        m0 = solution_before.y[1][transition_ix]
        #domain wall tension must be in GeV*km/fm^3
        p0 = solution_before.y[0][transition_ix] - domain_wall_tension / (r0 * SUN_SCHWARZ_RAD)
        phi0 = solution_before.y[2][transition_ix]

        p_in = solution_before.y[0][transition_ix]
        p_out = p0
        
        #solve TOV equations out to 10 * Schwarz radius of the sun (~30km), use many r steps so that they can be sampled over when done
        solution_after = solve_ivp(_tov_diff_eq, [r0, len_r_eval], [p0, m0, phi0],
            t_eval = linspace(r0, len_r_eval, eval_steps), args = (pressure_full, energy_density_full), rtol=10**-10, atol=10**-9)

        #find the edge of the star where pressure goes below zero
        edge_index = where(solution_after.y[0] <= 0)[0][0]
        #sample tov solution data to have num_steps points
        step_size = max([1, int(floor(edge_index / num_steps))])
        #convert radius to km
        radius_range = concatenate((solution_before.t[:transition_ix], solution_after.t[0:edge_index:step_size])) * SUN_SCHWARZ_RAD
        pressure = concatenate((solution_before.y[0][:transition_ix], solution_after.y[0][0:edge_index:step_size]))
        contained_mass = concatenate((solution_before.y[1][:transition_ix], solution_after.y[1][0:edge_index:step_size]))

        crust_mass = contained_mass[-1] - tov_results_inner.contained_mass[-1]
        #set e^phi to sqrt(1 - 2GM/r) at edge of star
        phi_offset = 1/2 * log(1 - solution_after.y[1][edge_index] / solution_after.t[edge_index]) - solution_after.y[2][edge_index]
        tov_phi = concatenate((solution_before.y[2][:transition_ix], solution_after.y[2][0:edge_index:step_size])) + phi_offset
        
        tov_data_list.append([tovData(radius_range, pressure, contained_mass, tov_phi), p_out, p_in, crust_mass])
    return tov_data_list
"""
    
#solves for the inverse sound speed squared with a 3 step centered derivative
def _de_dp(pressure, pressure_full, ed_full, dp_frac = 1e-5, singularities = [], counter = 0):
    eos = interp1d(pressure_full, ed_full, fill_value = 'extrapolate')
    dp = pressure * dp_frac

    #if near a singularity, move away from it
    if not all([abs(pressure - elt) > 3 * dp for elt in singularities]):
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
    temp = (1 / 60 * (ed_pl3 - ed_min3) - 3 / 20 * (ed_pl2 - ed_min2) + 3 / 4 * (ed_pl1 - ed_min1)) / dp
    if counter > 5:
        print('five iterations reached, terminating')
        return temp
    elif temp < 1:
        return _de_dp(pressure, pressure_full, ed_full, dp_frac = dp_frac * 1e-3, singularities = singularities, counter = counter + 1)
    return temp

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
    h_factor = exp_lambda * (-6 / radius**2 + 4 * pi * PR_CUBE_TO_MSOL * MSOL_TO_SCHWARZ * (5 * energy_density + 9 * pressure
        + (energy_density + pressure) / cs_sq)) - (phi_prime / PHI_PREFACTOR)**2

    return [h_prime, - h_prime * h_prime_factor / SUN_SCHWARZ_RAD - h * h_factor / SUN_SCHWARZ_RAD**2]

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
    return 2 / 3 * k2 / c**5, h_results

#convenient wrapper to do tov calculation and tidal deformability at the same time
def tov_lambda_solver(central_pressure, pressure_full, energy_density_full, num_steps = 500, rtol_input = 1e-5, atol_input = 1e-8, singularities = []):
    tov_results = tov_solver(central_pressure, pressure_full, energy_density_full, num_steps)
    td, h_results = tidal_deformability(tov_results, pressure_full, energy_density_full, rtol_deq = rtol_input, atol_deq = atol_input, singularities = singularities)
    return (tov_results, td, h_results)

def tov_lambda_multi(inputs):
    if len(inputs) == 4:
        central_pressure, pressure_full, energy_density_full, singularities = inputs
        rtol = 1e-8
        atol = 1e-8
        num_steps = 500
    else:
        central_pressure, pressure_full, energy_density_full, singularities, rtol, atol, num_steps = inputs
    tov_results = tov_solver(central_pressure, pressure_full, energy_density_full, num_steps = num_steps)
    td, h_results = tidal_deformability(tov_results, pressure_full, energy_density_full, rtol, atol, singularities)
    return td

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
