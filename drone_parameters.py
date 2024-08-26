import numpy as np
import pandas as pd

g = 9.81 # gravity

"""
parameters come from 
Geometric Tracking Control of a Quadrotor UAV on SE(3)
"""
m = 5    # kg
d = 0.315   # from drone center to motor center m
inertia = np.diag([0.0820, 0.0845, 0.1377])  # kgm2
inertia_inv = np.linalg.inv(inertia)
c_tau_f = 8.004e-4  # m

m_thrust_to_fm = np.array([[1.0, 1.0, 1.0, 1.0],
                           [0.0, -d, 0.0, d],
                           [d, 0.0, -d, 0.0],
                           [-c_tau_f, c_tau_f, -c_tau_f, c_tau_f]])
m_thrust_to_fm_inv = np.linalg.inv(m_thrust_to_fm)

f_motor_max = 50.0  # maximum possible thrust per motor [N] Thrust per motor: 200 - 800 grams for small drones
f_motor_min = 0.1   # minimum possible thrust per motor [N]

"""
drone rotors position vectors in body frame
"""
num_of_rotors = 4
p_px = np.array([d, 0, 0])
p_nx = np.array([-d, 0, 0])
p_py = np.array([0, d, 0])
p_ny = np.array([0, -d, 0])
rotor_position = [p_px, p_nx, p_py, p_ny]

rotor_radius = 0.2 # [m] 15inch diameter rotor

"""
control params
"""
k_x = 16
k_v = 5.6
k_r = 8.81
k_omega = 2.54


m_payload = 1.0 # kg
c_d = 1.5   # no unit [0.5-1.5]
area_frontal = 0.1  # m^2 [0.01-0.1]
rho_air = 1.225 # kg/m^3 air density