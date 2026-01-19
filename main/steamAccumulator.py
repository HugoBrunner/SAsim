#%% IMPORT PACKAGES

import time 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from CoolProp.CoolProp import PropsSI, PhaseSI

#%% CONSTANTS
FLUID   = 'Water'
P_CRIT  = 22063999.9 # Critical pressure of water

D_accu  = 2.8
L_accu  = 11.6 - 2.8
V_TOTAL = np.pi * D_accu**2 / 4 * L_accu + 4/3 * np.pi * (D_accu / 2)**3 # Accumulator total volume (m^3)

TAU_C   = 85      # Condensation relaxation time (s)
TAU_E   = TAU_C   # Evaporation relaxation time (s)
HA_21   = 5e4     # (ha)_21 (W/(m^3*K))

# Numerical constants
DP = 100.0      # dP (Pa)
DH = 1000.0     # dh (J/kg)

#%% THERMO PROPERTIES
def get_thermo_properties(p, h):
    
    """
    Calculate the thermo-physical properties of water for a given pressure and enthalpy.
    """
    
    try:
        T = PropsSI('T', 'P', p, 'H', h, FLUID)
        # --- DÉBUT CORRECTION ---
        rho = PropsSI('D', 'P', p, 'H', h, FLUID) # Densité (kg/m^3)
        v = 1.0 / rho                              # Volume spécifique (m^3/kg)
        # --- FIN CORRECTION ---
    except ValueError:
        return 1e9, 1e9, 1e9, 0.0, 0.0

    # Propriétés de saturation à cette pression
    if p < P_CRIT:
        h_sat_liq = PropsSI('H', 'P', p, 'Q', 0, FLUID)
        h_sat_vap = PropsSI('H', 'P', p, 'Q', 1, FLUID)
    else:
        h_sat_liq = h
        h_sat_vap = h+1000
    
    return T, v, rho, h_sat_liq, h_sat_vap

def calculate_partial_derivatives(p, h):
    
    """
    Calcultes the partial derivatives with finite differences.
    """
    
    try:
        # v(p + dp, h)
        rho_plus_p = PropsSI('D', 'P', p + DP, 'H', h, FLUID)
        v_plus_p = 1.0 / rho_plus_p
        
        # v(p - dp, h)
        rho_minus_p = PropsSI('D', 'P', p - DP, 'H', h, FLUID)
        v_minus_p = 1.0 / rho_minus_p
        
        # v(p, h + dh)
        rho_plus_h = PropsSI('D', 'P', p, 'H', h + DH, FLUID)
        v_plus_h = 1.0 / rho_plus_h
        
        # v(p, h - dh)
        rho_minus_h = PropsSI('D', 'P', p, 'H', h - DH, FLUID)
        v_minus_h = 1.0 / rho_minus_h
        
        # (dv/dp)_h
        dv_dp_h = (v_plus_p - v_minus_p) / (2 * DP)
        # (dv/dh)_p
        dv_dh_p = (v_plus_h - v_minus_h) / (2 * DH)
    except ValueError:
        return 0.0, 0.0
        
    return dv_dp_h, dv_dh_p

#%% STEAM ACCUMULATOR MODEL

def derivatives(t, y, M1_in, M1_out, H1_in, M2_in, H2_in, M2_out):
    
    """
    First order differential equations system.
    y = [M1, M2, p, h1, h2]
    """
    
    M1, M2, p, h1, h2 = y
    
    # FLuid properties
    T1, v1, rho1, h_prime, h_double_prime = get_thermo_properties(p, h1)
    T2, v2, rho2, _, _ = get_thermo_properties(p, h2)
    
    V1 = M1 * v1
    V2 = M2 * v2
    
    dv1_dp_h, dv1_dh_p = calculate_partial_derivatives(p, h1)
    dv2_dp_h, dv2_dh_p = calculate_partial_derivatives(p, h2)
    
    r = h_double_prime - h_prime
    
    if V1 + V2 > V_TOTAL * 1.05:
        return [0.0] * 5
        
    # Evaporation and condensation mass flow rates
    m_e = (rho1 * V1 * max(0.0, h1 - h_prime)) / (TAU_E * r)
    m_c = (rho1 * V1 * max(0.0, h_prime - h1)) / (TAU_C * r)
        
    m_PT1 = m_c - m_e
    m_PT2 = m_e - m_c
    
    # Heat transfer
    Q_21 = HA_21 * max(0.0, T2 - T1) * V1
        
    # Derative computations
    m_dot_1B = M1_in - M1_out
    m_dot_2B = M2_in - M2_out
    
    dM1_dt = m_dot_1B + m_PT1
    dM2_dt = m_dot_2B + m_PT2
    
    mh_dot_1B = M1_in * H1_in - M1_out * h1
    mh_dot_2B = M2_in * H2_in - M2_out * h2
    
    # Term A
    A = (h1 * dv1_dh_p - v1) * dM1_dt + (h2 * dv2_dh_p - v2) * dM2_dt
    
    # Term B
    B_1 = dv1_dh_p * (mh_dot_1B + m_PT1 * h_double_prime + Q_21)
    B_2 = dv2_dh_p * (mh_dot_2B + m_PT2 * h_double_prime - Q_21)
    B = B_1 + B_2

    # Term C (dp/dt denominator)
    C = M1 * (dv1_dp_h + dv1_dh_p * v1) + M2 * (dv2_dp_h + dv2_dh_p * v2)
    
    # dp/dt
    if abs(C) > 1e-10:
        dp_dt = (A - B) / C
    else:
        dp_dt = 0.0
        
    # dh1/dt
    if M1 != 0:
        dh1_dt = (1 / M1) * (mh_dot_1B + m_PT1 * h_double_prime + Q_21 + M1 * v1 * dp_dt - h1 * dM1_dt)
    else:
        dh1_dt = 0.0
        
    # dh2/dt
    if M2 != 0:
        dh2_dt = (1 / M2) * (mh_dot_2B + m_PT2 * h_double_prime - Q_21 + M2 * v2 * dp_dt - h2 * dM2_dt)
    else:
        dh2_dt = 0.0

    return [dM1_dt, dM2_dt, dp_dt, dh1_dt, dh2_dt]

def run_simulation(time_span, initial_conditions, boundary_flows, num_points=1000):
    
    """ 
    It runs the transient simulation for the given interval.
    """
    
    t_start, t_end = time_span
    t_points = np.linspace(t_start, t_end, num_points)
    
    solution = solve_ivp(
        fun=lambda t, y: derivatives(t, y, *boundary_flows), 
        t_span=time_span, 
        y0=initial_conditions, 
        t_eval=t_points, 
        method='RK45', 
        rtol=1e-5,  
        atol=1e-7
    )
    return solution

def calculate_derived_results(times, P_sim, H1_sim, H2_sim):
    
    """ 
    Calculates the derived results of the model (enthalpy difference, ...).
    """
    
    H_PRIME_SAT = np.array([PropsSI('H', 'P', p_val, 'Q', 0, FLUID) for p_val in P_sim])
    Delta_h_vap_sat = H2_sim - H_PRIME_SAT
    
    return H_PRIME_SAT, H2_sim - H1_sim, Delta_h_vap_sat

#%% WATER LEVEL

def wet_area_cylinder(H, D):
    
    """ 
    Calculates the filled cross-sectional area (A) at height H for the cylinder body.
    This formula is mathematically correct for H in [0, D].
    """
    
    R = D / 2.0
    if H <= 0: return 0.0
    if H >= D: return (np.pi * R**2)
        
    arg = (R - H) / R
    
    # Safety clamp for arccos
    arg = max(-1.0, min(1.0, arg))

    # Area of a circular segment
    term1 = R**2 * np.arccos(arg)
    # Term 2: Area of the triangle, calculated from R, H. (R-H) * sqrt(R^2 - (R-H)^2)
    term2 = (R - H) * np.sqrt(2 * R * H - H**2)
    
    return term1 - term2
    
def volume_hemispherical_ends(H, D):
    
    """
    Calculates the volume of liquid in the two hemispherical ends (one sphere) for a height H.
    Formula for a spherical cap: (1/3) * pi * H^2 * (3R - H).
    """
    
    R = D / 2.0
    V_total_sphere = (4.0/3.0 * np.pi * R**3) 
    
    if H <= 0: return 0.0
    if H >= D: return V_total_sphere 

    V_cap = (1.0/3.0) * np.pi * (H**2) * (3.0 * R - H)

    return V_cap

def total_liquid_volume(H, D, L):
    
    """ 
    Calculates the total liquid volume V1 for a height H,
    including the cylindrical body (length L) and the two hemispherical ends. 
    """
    
    V_cylinder_body = wet_area_cylinder(H, D) * L
    V_ends = volume_hemispherical_ends(H, D)
    
    return V_cylinder_body + V_ends

def find_height_H(V_liquid_target, D, L):
    
    """ 
    Uses fsolve to find the height H corresponding to the target volume V_liquid_target.
    
    """
    R = D / 2.0
    
    # Define the equation to solve: V_calculated(H) - V_target = 0
    def equation_to_solve(H):
        return total_liquid_volume(H[0], D, L) - V_liquid_target
    
    H_guess = R
    
    try:
        H_result = fsolve(equation_to_solve, H_guess, maxfev=2000)[0]
        # Ensure H is within the domain [0, D]
        return max(0.0, min(D, H_result))
        
    except Exception:
        # If fsolve fails (e.g., convergence error), return -1
        return -1.0

def calculate_accumulation_level(M1_sim, P_sim, H1_sim, D, L):
    
    """ 
    Calculates the liquid height (H) and density (rho1) over time. 
    """
    
    H_level_sim = []
    rho1_sim    = []
    
    for M1, p, h1 in zip(M1_sim, P_sim, H1_sim):
        try:
            _, _, rho1, _, _ = get_thermo_properties(p, h1)
        except:
            rho1 = 1000.0 

        V1_liquid = M1 / rho1 if rho1 > 0 else 0.0

        H = find_height_H(V1_liquid, D, L)
        
        H_level_sim.append(H)
        rho1_sim.append(rho1)
        
    return np.array(H_level_sim), np.array(rho1_sim)

#%% SIMULATION
P_INIT = 34e5
T_INIT = 250 + 273.15

STEAM_INIT = 'SH' # SAT or SH
V_PRIME_INIT = 1.0 / PropsSI('D', 'P', P_INIT, 'Q', 0, FLUID) 
V_DOUBLE_PRIME_INIT = 1.0 / PropsSI('D', 'P', P_INIT, 'Q', 1, FLUID)
H_PRIME_INIT = PropsSI('H', 'P', P_INIT, 'Q', 0, FLUID)
H_DOUBLE_PRIME_INIT = PropsSI('H', 'P', P_INIT, 'Q', 1, FLUID)
H_DOUBLE_PRIME_INIT_SH = PropsSI('H', 'P', P_INIT, 'T', T_INIT, FLUID)

alpha = 0.4

V1_INIT = V_TOTAL * alpha
V2_INIT = V_TOTAL * (1 - alpha)
M1_INIT = V1_INIT / V_PRIME_INIT
M2_INIT = V2_INIT / V_DOUBLE_PRIME_INIT
H1_INIT = H_PRIME_INIT
H2_INIT = H_DOUBLE_PRIME_INIT

y0 = [M1_INIT, M2_INIT, P_INIT, H1_INIT, H2_INIT]

T_END = 2100.0
t_charge_stop = 300.0
t_discharge_stop = 1400.0

M1_IN, M1_OUT, H1_IN = 0.0, 0.0, 0.0
if STEAM_INIT == 'SAT':
    H2_IN_CHARGE = H_DOUBLE_PRIME_INIT
else:
    H2_IN_CHARGE = H_DOUBLE_PRIME_INIT_SH
M2_IN_CHARGE = 20
M2_OUT_DISCHARGE = 8.0

sta = time.time()

# 1 - Charging
boundary_flows_charge = [M1_IN, M1_OUT, H1_IN, M2_IN_CHARGE, H2_IN_CHARGE, 0.0]
sol_charge = run_simulation((0, t_charge_stop), y0, boundary_flows_charge, num_points=int(t_charge_stop))
y_charge_end = sol_charge.y[:, -1]

# 2 - Discharging
boundary_flows_discharge = [M1_IN, M1_OUT, H1_IN, 0.0, 0.0, M2_OUT_DISCHARGE]
sol_discharge = run_simulation((t_charge_stop, t_discharge_stop), y_charge_end, boundary_flows_discharge, num_points=int(t_discharge_stop-t_charge_stop))
y_discharge_end = sol_discharge.y[:, -1]

# 2 - Stabilization
boundary_flows_stop = [M1_IN, M1_OUT, H1_IN, 0.0, 0.0, 0.0]
sol_stop = run_simulation((t_discharge_stop, T_END), y_discharge_end, boundary_flows_stop, num_points=int(T_END-t_discharge_stop))

# Final results
time_sim = np.concatenate((sol_charge.t, sol_discharge.t, sol_stop.t))
P_sim = np.concatenate((sol_charge.y[2, :], sol_discharge.y[2, :], sol_stop.y[2, :]))
H1_sim = np.concatenate((sol_charge.y[3, :], sol_discharge.y[3, :], sol_stop.y[3, :]))
T1_sim = PropsSI('T', 'P', P_sim, 'H', H1_sim, FLUID) - 273.15
H2_sim = np.concatenate((sol_charge.y[4, :], sol_discharge.y[4, :], sol_stop.y[4, :]))
T2_sim = PropsSI('T', 'P', P_sim, 'H', H2_sim, FLUID) - 273.15
M2_sim = np.concatenate((sol_charge.y[1, :], sol_discharge.y[1, :], sol_stop.y[1, :]))
M1_sim = np.concatenate((sol_charge.y[0, :], sol_discharge.y[0, :], sol_stop.y[0, :]))
H_sim, RHO1_sim = calculate_accumulation_level(M1_sim, P_sim, H1_sim, D_accu, L_accu)

H_PRIME_SAT, Delta_h, Delta_h_vap_sat = calculate_derived_results(time_sim, P_sim, H1_sim, H2_sim)

#%% PLOTS

plt.style.use('seaborn-v0_8-whitegrid')

# Graphe 1 : Pression dans l'Accumulateur
plt.figure(figsize=(12, 6))
plt.plot(time_sim, P_sim / 1e5, label='Modèle Non-Équilibre', color='blue', linewidth=2)
plt.axvline(x=t_charge_stop, color='red', linestyle='--', label='Début du Déchargement')
plt.axvline(x=t_discharge_stop, color='green', linestyle='--', label='Fin du Déchargement/Début Stabilisation')
plt.title('Pression dans l\'Accumulateur de Vapeur: Chargement puis Déchargement')
plt.xlabel('Temps (s)')
plt.ylabel('Pression (bar)')
plt.legend()
plt.grid(True)
plt.show()

# Graphe 2 : Écart d'Enthalpie (Indicateur de Non-Équilibre)
plt.figure(figsize=(12, 6))
plt.plot(time_sim, Delta_h_vap_sat / 1e3, label='$\Delta h_{2,sat} = h_2 - h\'_{sat}$ (Vapeur)', color='darkorange', linewidth=2)
plt.plot(time_sim, (H1_sim - H_PRIME_SAT) / 1e3, label='$\Delta h_{1,sat} = h_1 - h\'_{sat}$ (Liquide)', color='green', linestyle='--')
plt.axvline(x=t_charge_stop, color='red', linestyle='--')
plt.axvline(x=t_discharge_stop, color='green', linestyle='--')
plt.title('Écarts d\'Enthalpie Spécifique par rapport à l\'État de Saturation Liquide ($h\'_{sat}$)')
plt.xlabel('Temps (s)')
plt.ylabel('Écart d\'Enthalpie (kJ/kg)')
plt.legend()
plt.grid(True)
plt.show()

# Graphe 3 : Masse de la Vapeur (M2)
fig, ax1 = plt.subplots(figsize=(12, 6))
color_M1 = 'orange'
ax1.set_xlabel('Temps (s)')
ax1.set_ylabel('Masse d\'Eau ($M_1$) (kg)', color=color_M1)
ax1.plot(time_sim, M1_sim, label='Masse d\'Eau ($M_1$)', color=color_M1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color_M1)
ax1.grid(True)

ax2 = ax1.twinx()  
color_M2 = 'teal'
ax2.set_ylabel('Masse de Vapeur ($M_2$) (kg)', color=color_M2)
ax2.plot(time_sim, M2_sim, label='Masse de Vapeur ($M_2$)', color=color_M2, linewidth=2, linestyle='-')
ax2.tick_params(axis='y', labelcolor=color_M2)
ax1.axvline(x=t_charge_stop, color='red', linestyle='--')
ax1.axvline(x=t_discharge_stop, color='green', linestyle='--')
fig.suptitle('Masse de Vapeur/Eau dans l\'Accumulateur')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')
plt.show()

# Graphe 4 : Niveau dans l'Accumulateur
plt.figure(figsize=(12, 6))
plt.plot(time_sim, H_sim, label='Liquid Level (H)', color='purple', linewidth=2)
plt.axhline(y=D_accu, color='red', linestyle=':', label=f'Inner Diameter D = {D_accu:.2f} m')
plt.axvline(x=t_charge_stop, color='red', linestyle='--')
plt.axvline(x=t_discharge_stop, color='green', linestyle='--')
plt.title('Niveau d\'eau dans l\'accumulateur vapeur')
plt.xlabel('Temps (s)')
plt.ylabel('Niveau (m)')
plt.legend()
plt.grid(True)
plt.show()

end = time.time()
print(f'Temps pris par la simulation: {end-sta:.2f} seconds')