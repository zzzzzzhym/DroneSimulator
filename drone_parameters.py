import numpy as np

g = 9.81 # gravity

"""
parameters come from 
Geometric Tracking Control of a Quadrotor UAV on SE(3)
"""
m = 4.34    # kg
d = 0.315   # from drone center to motor center m
inertia = np.diag([0.0820, 0.0845, 0.1377])  # kgm2
inertia_inv = np.linalg.inv(inertia)
c_tau_f = 8.004e-4  # m

"""
control params
"""
k_x = 16
k_v = 5.6
k_r = 8.81
k_omega = 2.54

m_thrust_to_fm = np.array([[1.0, 1.0, 1.0, 1.0],
                           [0.0, -d, 0.0, d],
                           [d, 0.0, -d, 0.0],
                           [-c_tau_f, c_tau_f, -c_tau_f, c_tau_f]])
m_thrust_to_fm_inv = np.linalg.inv(m_thrust_to_fm)

m_payload = 1.0 # kg
c_d = 1.5   # no unit [0.5-1.5]
area_frontal = 0.1  # m^2 [0.01-0.1]
rho_air = 1.225 # kg/m^3