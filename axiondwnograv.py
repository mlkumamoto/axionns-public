#arXiv:2410.21590

from numpy import linspace, concatenate, array as np_array, zeros, ones, vstack, log as np_log, exp as np_exp, gradient, sqrt as np_sqrt, sin as np_sin, abs as np_abs
from math import pi, log, floor, sin, sqrt, exp
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d

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

def closest_ix(val, list_of_vals): return list(list_of_vals).index(min(list_of_vals, key = lambda x: abs(val - x)))

def ftheta_func(theta): return sqrt(1 - 4 * UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * sin(theta / 2)**2)

def _axion_log_de_no_grav(radius, inputs, p_full, ed_full, theta_full, psd_full, nsd_full, fa15, eps, sig_pin, delta_sigma, p_steps):
    log_theta, log_thetapr, pressure = inputs
    if log_theta >= log(pi + 0.01):
        theta = pi
        thetapr = 10
    elif log_theta < -16:
        theta = 1e-16
        thetapr = 0
    else:
        theta = exp(log_theta)
        thetapr = log_thetapr * exp(log_theta)
    t_ix = closest_ix(theta, theta_full[::p_steps])
    shift = eps * PION_MASS**2 * F_PION**2 * (1 - ftheta_func(theta))
    p_temp = [elt - shift for elt in p_full[t_ix * p_steps: (t_ix + 1) * p_steps]]
    psd = interp1d(p_temp, psd_full[t_ix * p_steps: (t_ix + 1) * p_steps], fill_value = 'extrapolate')
    nsd = interp1d(p_temp, nsd_full[t_ix * p_steps: (t_ix + 1) * p_steps], fill_value = 'extrapolate')

    ftheta = sqrt(1 - 4 * UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * sin(theta / 2)**2)
    ftheta_pr = - UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * sin(theta) / ftheta
    if pressure > p_temp[0]:
        uf = ftheta_pr * 0.1973 * (-eps * F_PION**2 * PION_MASS**2 + sig_pin * (psd(pressure) + nsd(pressure)) 
                      + delta_sigma / ftheta**2 * (psd(pressure) - nsd(pressure)))
    else:
        uf = ftheta_pr * 0.1973 * (-eps * F_PION**2 * PION_MASS**2)
    pressure_prime = - thetapr * uf

    if log_theta >= log(pi + 0.01):
        log_theta_prpr = 10
    else:
        log_theta_prpr = -(thetapr / theta)**2 + 1 / theta * uf / (fa15**2 * FA2_OVER_R2_TO_GEVFM3)
    return [log_thetapr, log_theta_prpr, pressure_prime]

def _ax_no_grav_wrapper(r_list, inputs_list, p_full, ed_full, theta_full, psd, nsd, fa15, eps, sig_pin, delta_sigma, p_steps):
    output = []
    theta_list, thetapr_list, pressure_list = inputs_list
    for ix, r_temp in enumerate(r_list):
        output.append(_axion_log_de_no_grav(r_temp, [theta_list[ix], thetapr_list[ix], pressure_list[ix]], p_full, ed_full, theta_full, psd, nsd, fa15, eps, sig_pin, delta_sigma, p_steps))
    return np_array(output).T

def bc_no_grav(ya, yb, theta0, theta_max): 
    return [ya[0] - log(theta0), yb[0] - log(theta_max), yb[2]]

def axion_dw_no_grav(p0, theta0, theta_max, eos_data, axion_data, num_steps = 500, len_r_eval = 1, eval_steps = 10**3, p_steps = 13000, rtol = 1e-5):
    pressure_full, energy_density_full, theta_full, psd, nsd = eos_data
    fa15, eps, sig_pin, delta_sigma = axion_data

    init_guess = vstack((np_log(linspace(theta0, theta_max, eval_steps)), - (theta0 - theta_max) / len_r_eval 
                         * ones(eval_steps) / linspace(theta0, theta_max, eval_steps), zeros(eval_steps)))

    solution = solve_bvp(lambda x, y: _ax_no_grav_wrapper(x, y, pressure_full, energy_density_full, theta_full, psd, nsd, fa15, eps, sig_pin, delta_sigma, p_steps),
        lambda xa, xb: bc_no_grav(xa, xb, theta0, theta_max), linspace(0, len_r_eval, eval_steps), init_guess, tol = rtol)

    #find the edge of the star where pressure goes below zero
    edge_index = -1
    #sample tov solution data to have num_steps points
    step_size = max([1, int(floor(edge_index / num_steps))])
    #convert radius to km
    radius_range = solution.x[0:edge_index:step_size] * SUN_SCHWARZ_RAD
    theta = solution.y[0][0:edge_index:step_size]
    pressure = solution.y[2][0:edge_index:step_size]

    ed = []
    for ix, theta_temp in enumerate(theta):
        t_ix = closest_ix(theta_temp, theta_full[::p_steps])
        shift = eps * PION_MASS**2 * F_PION**2 * (1 - ftheta_func(theta_temp))
        p_temp = [elt - shift for elt in pressure_full[t_ix * p_steps: (t_ix + 1) * p_steps]]
        ed_temp = [elt + shift for elt in  energy_density_full[t_ix * p_steps: (t_ix + 1) * p_steps]]
        eos = interp1d(p_temp, ed_temp, fill_value = 'extrapolate')
        ed.append(eos(pressure[ix]))

    return solution, radius_range, theta, pressure, ed    

def sd_of_ptheta(p, theta_list, eos_data, axion_data, p_steps = 13000):
    p_full, ed_full, theta_full, psd_full, nsd_full = eos_data
    fa15, eps, sig_pin, delta_sigma = axion_data
    nsd_temp = []
    psd_temp = []

    for ix, theta in enumerate(theta_list):
        t_ix = closest_ix(theta, theta_full[::p_steps])
        shift = eps * PION_MASS**2 * F_PION**2 * (1 - ftheta_func(theta))
        p_temp = [elt - shift for elt in p_full[t_ix * p_steps: (t_ix + 1) * p_steps]]
        psd = interp1d(p_temp, psd_full[t_ix * p_steps: (t_ix + 1) * p_steps])
        nsd = interp1d(p_temp, nsd_full[t_ix * p_steps: (t_ix + 1) * p_steps])
        if p[ix] > p_temp[0]:
            psd_temp.append(psd(p[ix]))
            nsd_temp.append(nsd(p[ix]))
        else:
            psd_temp.append(0)
            nsd_temp.append(0)
    return np_array(psd_temp), np_array(nsd_temp)

def relax_de(radius, inputs, r_space, psd, nsd, fa15, eps, sig_pin, delta_sigma):
    theta, thetapr = inputs
    psd_temp = interp1d(r_space, psd)(radius)
    nsd_temp = interp1d(r_space, nsd)(radius)
    ftheta = np_sqrt(1 - 4 * UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * np_sin(theta / 2)**2)
    ftheta_pr = - UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * np_sin(theta) / ftheta
    uf = ftheta_pr * 0.1973 * (-eps * F_PION**2 * PION_MASS**2 + sig_pin * (psd_temp + nsd_temp) 
                      + delta_sigma / ftheta**2 * (psd_temp - nsd_temp))
    return thetapr, uf / (fa15**2 * FA2_OVER_R2_TO_GEVFM3)

def relax_de_log(radius, inputs, r_space, psd, nsd, fa15, eps, sig_pin, delta_sigma):
    y, ypr = inputs
    theta = []
    thetapr = []
    for yix, y_temp in enumerate(y):
        if -50 < y_temp < 50:
            theta.append(pi / (1 + exp(y_temp)))
            thetapr.append(-pi * exp(y_temp) / (1 + exp(y_temp))**2 * ypr[yix])
        elif y_temp > 50:
            theta.append(pi / (1 + exp(50)))
            thetapr.append(-1e-6)
        elif y_temp < -50:
            theta.append(pi / (1 + exp(-50)))
            thetapr.append(-1e-6)
    theta = np_array(theta)
    thetapr = np_array(thetapr)

    psd_temp = interp1d(r_space, psd)(radius)
    nsd_temp = interp1d(r_space, nsd)(radius)
    ftheta = np_sqrt(1 - 4 * UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * np_sin(theta / 2)**2)
    ftheta_pr = - UP_MASS * DOWN_MASS / (UP_MASS + DOWN_MASS)**2 * np_sin(theta) / ftheta
    uf = ftheta_pr * 0.1973 * (-eps * F_PION**2 * PION_MASS**2 + sig_pin * (psd_temp + nsd_temp) 
                      + delta_sigma / ftheta**2 * (psd_temp - nsd_temp))
    y_prpr = uf / (fa15**2 * FA2_OVER_R2_TO_GEVFM3) * ypr / thetapr - 2 * thetapr * ypr / theta - ypr**2
    return ypr, y_prpr

def axion_dw_relax(eos_data, axion_data, len_r_eval = 1, eval_steps = 10**3, p_steps = 13000, max_iter = 10**3, tol = 1e-5, max_err = 1e-8):
    fa15, eps, sig_pin, delta_sigma = axion_data

    r_space = linspace(0, len_r_eval, eval_steps)
    theta_guess = concatenate((np_array([pi, pi]), pi / (1 + np_exp(linspace(-16, 16, eval_steps - 4))), np_array([0, 0])))

    max_diff = 1
    iter_counter = 0
    while max_diff > max_err and iter_counter < max_iter:
        p_guess = - fa15**2 * FA2_OVER_R2_TO_GEVFM3 / 2 * gradient(theta_guess, len_r_eval / (eval_steps - 1))**2
        psd_temp, nsd_temp = sd_of_ptheta(p_guess, theta_guess, eos_data, axion_data)
        bvp_guess = vstack((theta_guess, gradient(theta_guess, len_r_eval / (eval_steps - 1))))
        bvp_results_temp = solve_bvp(lambda x, y: relax_de(x, y, r_space, psd_temp, nsd_temp, fa15, eps, sig_pin, delta_sigma),
            lambda xa, xb: [xa[0] - pi, xb[0]], r_space, bvp_guess, tol = tol)
        theta_new = bvp_results_temp.y[0]
        max_diff = max(np_abs(theta_new - theta_guess))
        print(max_diff)
        theta_guess = theta_new
        iter_counter += 1
    print('test')

    if iter_counter < max_iter:
        print('successful relaxation')
    else:
        print('max iter reached')

    return r_space, p_guess, theta_guess

def axion_dw_relax_log(eos_data, axion_data, len_r_eval = 1, eval_steps = 10**3, p_steps = 13000, max_iter = 10**3, tol = 1e-5, max_err = 1e-8, mix_ratio = 0.5):
    fa15, eps, sig_pin, delta_sigma = axion_data

    r_space = linspace(0, len_r_eval, eval_steps)
    y_guess = linspace(-16, 16, eval_steps)

    y_guess_list = []
    p_guess_list = []
    psd_list = []
    nsd_list = []
    max_diff = 1
    iter_counter = 0
    while max_diff > max_err and iter_counter < max_iter:
        p_guess = - fa15**2 * FA2_OVER_R2_TO_GEVFM3 / 2 * gradient(pi / (1 + np_exp(y_guess)), len_r_eval / (eval_steps - 1))**2
        p_guess_list.append(p_guess)
        psd_temp, nsd_temp = sd_of_ptheta(p_guess, pi / (1 + np_exp(y_guess)), eos_data, axion_data)
        psd_list.append(psd_temp)
        nsd_list.append(nsd_temp)
        bvp_guess = vstack((y_guess, gradient(y_guess, len_r_eval / (eval_steps - 1))))
        bvp_results_temp = solve_bvp(lambda x, y: relax_de_log(x, y, r_space, psd_temp, nsd_temp, fa15, eps, sig_pin, delta_sigma),
            lambda xa, xb: [xa[0] + 16, xb[0] - 16], r_space, bvp_guess, tol = tol)
        theta_old = pi / (1 + np_exp(y_guess))
        theta_new = pi / (1 + np_exp(bvp_results_temp.y[0]))
        max_diff = max(np_abs(theta_old - theta_new))
        print(max_diff)
        y_guess_list.append(y_guess)
        y_guess = np_log(pi / (theta_old * (1 - mix_ratio) + mix_ratio * theta_new) - 1)
        iter_counter += 1

    if iter_counter < max_iter:
        print('successful relaxation')
    else:
        print('max iter reached')

    return SUN_SCHWARZ_RAD * r_space, p_guess, pi / (1 + np_exp(y_guess)), psd_temp, nsd_temp, y_guess_list, p_guess_list, psd_list, nsd_list