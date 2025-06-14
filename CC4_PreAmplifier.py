import numpy as np
import iminuit
import scipy.optimize as optimize
from scipy.optimize import fsolve

import sys
import io
from contextlib import redirect_stdout, redirect_stderr

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *


#=================================================================================================================================================================
#==== JFET CLASS & FUNCTIONS =====================================================================================================================================

class JFET:
    def __init__(self, W_L_ratio, a, N_d, E_g=1.12, alpha=5.4e-4, beta=655, eps_s=11.9, T_0=300):
        """
        Class that simulate JFETs with explicit temperature dependance

        Parameters:
        -----------
        W_L_ratio : float
            Ratio betweem channel width and channel lenght
        a : float
            Channel depth [m]
        N_d : float
            Donor doping concentration [cm^-3]
        E_g : float
            Bandgap energy [eV]
        alpha : float 
            Parameter to adjust E_g with temperature [eV/K]
        beta : float
            Parameter to adjust E_g with temperature [K]
        eps_s : float
            Dieletric constant of the semiconductor
        T_0 : float
            Reference temperature [K]
        """

        self.W_L_ratio = W_L_ratio
        self.a = a
        self.N_d = N_d
        self.E_g = E_g
        self.alpha = alpha
        self.beta = beta
        self.eps_s = eps_s
        self.T_0 = T_0

        self.k_b = 1.38e-23         # Boltzmann constant [J/K]
        self.h  = 6.626e-34         # Planck constant [J·s]
        self.q = 1.6e-19            # Electron charge [C]
        self.m_0 = 9.1095e-31       # Electron mass [kg]
        self.eps_0 = 8.85e-12       # Vacuum permittivity [F/m]

    def safe_power(self, x, p, tol=1e-8):
        """
        Safely compute x^p for both scalar and array inputs.
        
        - For integer exponents: Preserve sign (negative inputs allowed)
        - For fractional exponents: Clamp negative inputs to 0 to avoid NaNs
        
        Parameters:
        -----------
        x : scalar or array-like
            Input values
        p : float
            Exponent
        tol : float, optional
            Tolerance for checking if exponent is an integer
        
        Returns:
        --------
        scalar or ndarray
            Result of x^p with safety checks
        """
        x = np.asarray(x)
        p_rounded = np.round(p)
        is_integer = np.isclose(p, p_rounded, atol=tol)
        
        if is_integer:
            # Integer exponent: preserve sign for odd exponents
            p_int = int(p_rounded)
            abs_result = np.abs(x) ** p_int
            if p_int % 2 == 1:  # Odd exponent
                return np.sign(x) * abs_result
            else:  # Even exponent
                return abs_result
        else:
            # Fractional exponent: clamp negative values to 0
            x_clamped = np.maximum(x, 0)
            return x_clamped ** p
    
    def density_of_states_cond_band(self, T):
        """
        Calculate the number of states in the conduction band as function of temperature
        """
        M_c = 6
        m_l = 0.98*self.m_0
        m_t = 0.19*self.m_0
        m_de = (m_l * m_t**2)**(1/3)

        num = 2*np.pi * m_de * self.k_b * T
        den = self.h**2

        return (2* M_c*(num/den)**1.5)*1e-6  # result in cmˆ-3
    
    def energy_gap(self, T):
        """
        Calculate the energy gap as a function of temperature
        """
        E_g_0 = 1.169      

        return E_g_0 - (self.alpha * T**2) / (T + self.beta)

    def built_in_potential(self, T):
        """
        Calculate the built-in potential as a function of temperature
        """
        N_c = self.density_of_states_cond_band(T) 
        N_c_m3 = N_c * 1e6 # convert from cm^-3 to m^-3
        N_d_m3 = self.N_d * 1e6  # Convert from cm^-3 to m^-3
        E_g_T = self.energy_gap(T)
        E_g_T_J = E_g_T * self.q # convert in J
        ln_term = np.log(N_c_m3 / N_d_m3)

        return (E_g_T_J - self.k_b * T * ln_term)/self.q

    def pinch_off_voltage(self):
        """
        Calculate the pinch-off voltage for the JFET
        """
        N_d_m3 = self.N_d * 1e6  # Convert from cm^-3 to m^-3

        return (self.q * N_d_m3 * self.a**2) / (2 * self.eps_s * self.eps_0)

    
    def mobility(self, T):
        """
        Calculate electrons mobility as a function of temperature
        """       
        m_alpha = 1.5
        m_beta = 3.13
        mu_0ea = 4195  # cm^2/V s
        mu_0eb = 2153  # cm^2/V s

        rateo = T / self.T_0
        A = 1 / (mu_0ea * rateo**(-m_alpha))
        B = 1 / (mu_0eb * rateo**(-m_beta))

        return 1/(A+B)
    
    def full_channel_conductance(self, mu):
        """
        Calculate the full-channel conductance
        """
        N_d_m3 = self.N_d * 1e6  # Convert from cm^-3 to m^-3
        mu_m = mu *1e-4 # convert from cmˆ-2 to mˆ-2

        return self.W_L_ratio * self.q * N_d_m3 * self.a * mu_m

    def I_d(self, V_ds, V_gs, T, lambda_mod=0):
        """
        Calculate the drain-source current for given conditions

        Parameters:
        -----------
        V_ds : scalar or array
            Drain-source voltage [V]
        V_gs : scalar or array
            Gate-source voltage [V]
        T : float
            Temperature [K]
        lambda_mod : float, optional
            Channel-length modulation factor

        Returns:
        --------
        array
            Drain-Source current [A]
        """

        V_bi = self.built_in_potential(T)
        V_p = self.pinch_off_voltage()
        mu = self.mobility(T)
        G = self.full_channel_conductance(mu)

        V_T = V_bi - V_p            # Threshold voltage
        V_sat = V_p - V_bi + V_gs   # Saturation voltage

        V_ds = np.asarray(V_ds)
        V_gs = np.asarray(V_gs)

        I_d = np.zeros_like(V_ds)

        # Regions definition
        linear_region = V_gs >= V_T
        saturation_region = (V_ds >= V_sat) & linear_region

        # Linear region
        I_d[linear_region] = (
            G * (
                V_ds[linear_region]
                - (2 / (3 * np.sqrt(V_p))) * self.safe_power(V_bi + V_ds[linear_region] - V_gs[linear_region], 1.5)
                + (2 / (3 * np.sqrt(V_p))) * self.safe_power(V_bi - V_gs[linear_region], 1.5)
                )
            )
        # Saturation region
        I_d[saturation_region] = (
            G * (
                V_sat[saturation_region]
                - (2 / (3 * np.sqrt(V_p))) * self.safe_power(V_bi + V_sat[saturation_region] - V_gs[saturation_region], 1.5)
                + (2 / (3 * np.sqrt(V_p))) * self.safe_power(V_bi - V_gs[saturation_region], 1.5)
                )
            ) * (1 + lambda_mod * (V_ds[saturation_region] - V_sat[saturation_region]))

        return I_d
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

def infer_jfet_params(I_DSS, VTO, g_m, T, method='differential_evolution'):
    """
    Function that infers donor concentration and geometric channel parameters starting from JFET data sheet values
    
    Parameters:
    -----------
    I_DSS : float
        Saturation current @ V_gs = 0 [A]
    VTO : float
        Pinch off voltage [V]
    g_m : float
        Transconductance [S]
    T : float
        Nominal data sheet temperature [K]
    method : str, optional
        Optimization method to use. Options:
        - 'differential_evolution' (default): Uses SciPy's differential evolution
        - 'iminuit': Uses iMinuit for optimization (requires iminuit to be installed)
    
    Returns:
    --------
    dict
        Inferred parameters, including:
        - 'N_d': Donor concentration
        - 'W_L_ratio': Channel width and length ratio
        - 'a': Channel depth
        - 'covariance_matrix': Covariance matrix (only returned when method='iminuit')
    """
    
    k_b = 1.38e-23          # Boltzmann constant 
    h = 6.626e-34           # Planck constant
    q = 1.6e-19             # Electron charge [C]
    m_0 = 9.1095e-31        # Electron mass
    T_0 = 300               # Reference temperature [K]
    
    def mobility(T):
        m_alpha = 1.5
        m_beta = 3.13
        mu_0a = 4195  # cm^2/V s
        mu_0b = 2153  # cm^2/V s
        rateo = T / T_0
        A = 1/(mu_0a * rateo**(-m_alpha))
        B = 1/(mu_0b * rateo**(-m_beta))
        return 1/(A + B)
    
    def density_of_states_cond_band(T):
        M_c = 6
        m_l = 0.98*m_0
        m_t = 0.19*m_0
        m_de = (m_l * m_t**2)**(1/3) 
        num = 2*np.pi * m_de * k_b * T
        den = h**2

        return (2 * M_c * (num/den)**1.5) * 1e-6  # Convert from m^-3 to cm^-3
    
    def energy_gap(T):
        E_g = 1.12             # Bandgap energy [eV]
        alpha = 5.4e-4         # Parameter to adjust E_g with temperature [eV/K]
        beta = 655             # Parameter to adjust E_g with temperature [K]
        E_g_0 = 1.169
        return E_g_0 - (alpha * T**2) / (T + beta)
    
    def built_in_potential(T, N_d):
        N_c = density_of_states_cond_band(T) 
        N_c_m3 = N_c * 1e6     # Convert from cm^-3 to m^-3
        N_d_m3 = N_d * 1e6     # Convert from cm^-3 to m^-3
        E_g_T = energy_gap(T)
        E_g_T_J = E_g_T * q
        ln_term = np.log(N_c_m3 / N_d_m3)

        return (E_g_T_J - k_b * T * ln_term)/q
    
    def pinch_off_voltage(N_d, a):
        eps_0 = 8.85e-12        # Vacuum permittivity [F/m]
        eps_s = 11.9            # Relative permittivity of Si 
        N_d_m3 = N_d * 1e6      # Convert from cm^-3 to m^-3
        return (q * N_d_m3 * a**2) / (2 * eps_s * eps_0)
    
    def full_channel_conductance(mu, N_d, W_L_ratio, a):
        N_d_m3 = N_d * 1e6      # Convert from cm^-3 to m^-3
        mu_m = mu * 1e-4        # Convert from cm²/(V·s) to m²/(V·s)
        return W_L_ratio * q * N_d_m3 * a * mu_m
    
    def objective_function(x):
        N_d, W_L_ratio, a = x
        mu = mobility(T)  
        Psi_bi = built_in_potential(T, N_d)
        Psi_p = pinch_off_voltage(N_d, a)
        G_i = full_channel_conductance(mu, N_d, W_L_ratio, a)
        V_T = Psi_bi - Psi_p

        # Check if Psi_p is positive and greater than Psi_bi to avoid invalid sqrt
        if Psi_p <= 0 or Psi_bi <= 0 or Psi_bi >= Psi_p:
            return np.inf  # Return infinity to penalize invalid parameters

        f_1 = (g_m - G_i * (1 - np.sqrt(Psi_bi/Psi_p)))**2
        f_2 = (I_DSS - G_i * (Psi_p/3 - Psi_bi * (1 - 2/3 * np.sqrt(Psi_bi/Psi_p))))**2
        f_3 = (VTO - V_T)**2
        return f_1 + f_2 + f_3
    
    initial_guess = [1e16, 10, 0.5e-6]
    bounds = [
            (1e15, 1e19),           # N_d range 
            (1e-1, 100),             # W_L_ratio range 
            (0.01e-6, 10e-6)        # a range 
            ]
    
    
    if method == 'differential_evolution':
        result = optimize.differential_evolution(
            objective_function, 
            bounds=bounds,
            popsize=50,
            maxiter=1000, 
            tol=1e-9,    
            updating='deferred' if hasattr(optimize, 'DEStrategy') else 'immediate',
            )
        
        return {
            "N_d": result.x[0],
            "W_L_ratio": result.x[1],
            "a": result.x[2],
            }
    
    elif method == 'iminuit':
        def chi2(N_d, W_L_ratio, a):
            return objective_function([N_d, W_L_ratio, a])
        
        m = iminuit.Minuit(
            chi2, 
            N_d=initial_guess[0], 
            W_L_ratio=initial_guess[1], 
            a=initial_guess[2]
            )

        m.limits['N_d'] = bounds[0]
        m.limits['W_L_ratio'] = bounds[1]
        m.limits['a'] = bounds[2]
        m.errordef = iminuit.Minuit.LEAST_SQUARES
        
        m.migrad()
        
        if not m.valid:
            raise RuntimeError("Optimization with iminuit failed to converge.")
        
        covariance_matrix = m.covariance
        
        return {
            "N_d": m.values["N_d"],
            "W_L_ratio": m.values["W_L_ratio"],
            "a": m.values["a"],
            "covariance_matrix": covariance_matrix
            }
    
    else:
        raise ValueError(f"Unknown optimization method: {method}. Choose 'differential_evolution' or 'iminuit'.")


#=================================================================================================================================================================
#==== RC INTEGRATOR ==============================================================================================================================================

def integrate_signal(R_f, C_f, input_signal, dt, decay=True):
    tau = R_f * C_f             
    offset = np.mean(input_signal[:int(len(input_signal)*0.2)]) 
    output_signal = np.zeros_like(input_signal)
    effective_signal = input_signal - offset  

    if decay:
        decay_factor = np.exp(-dt /tau)
    else: decay_factor = 1

    for i in range(1, len(input_signal)):
        input_contribution = (effective_signal[i] + effective_signal[i-1]) * dt / (2 * C_f)
        output_signal[i] = output_signal[i-1] * decay_factor + input_contribution
    
    return -output_signal 