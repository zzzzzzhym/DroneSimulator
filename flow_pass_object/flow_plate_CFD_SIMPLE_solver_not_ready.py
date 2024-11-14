import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

class Air:
    def __init__(self):
        self.rho = 1.225
        self.mu = 1.7894e-5 # Dynamic viscosity (Gamma in the scheme)
        self.nu = self.mu / self.rho    # Kinematic viscosity

class SimpleSolver:
    def __init__(self, L, H, Nx, Ny, dx, dy, mu, rho):
        """Initialize the CFD simulation parameters and fields.

        Args:
            L (float): Length of the domain.
            H (float): Height of the domain.
            Nx (int): Number of grid points in the x-direction.
            Ny (int): Number of grid points in the y-direction.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
            u (float): Initial velocity in the x-direction.
            v (float): Initial velocity in the y-direction.
            Gamma (float): Diffusion coefficient.
            rho (float): Density of the fluid.
        """
        self.L, self.H = L, H
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy

        # regardless of the shape of the array, the first index is the x-direction and the second index is the y-direction
        self.u = np.zeros((Nx+1, Ny)) + 0  # Staggered in x (u at vertical cell faces)
        self.v = np.zeros((Nx, Ny+1))  # Staggered in y (v at horizontal cell faces)
        self.p = np.zeros((Nx, Ny))  # Pressure at cell centers
        self.Gamma = mu
        self.rho = rho
        self.epsilon = 1e-6


        self.momentum_epoch = 100
        self.pressure_epoch = 100
        self.simple_epoch = 200
        self.accuracy_trace = {'u': [], 'v': [], 'p': []}

        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def get_xy_coords(self, i, j):
        x = i * self.dx
        y = j * self.dy
        return x, y

    def apply_initial_guess(self):
        self.u = np.zeros((Nx+1, Ny)) + 11.0
        self.v = np.zeros((Nx, Ny+1)) 
        R = 30 # Radius of the cylinder
        U = 12.0 # Free-stream velocity

        for i in range(Nx):
            for j in range(Ny):
                x, y = self.get_xy_coords(i, j)
                x = -(self.L - x + R)
                y = y - self.H*0.5
                self.u[i, j], self.v[i, j] = get_velocity_field_cylinder(x, y, U, R)

    def set_velocity_boundary_conditions(self, u, v):
        """Apply boundary conditions to the velocity field."""
        # bottom
        u[:, 0] = u[:, 1]
        v[:, 0] = v[:, 1]
        # top
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        # left
        u[0, :] = 2.0
        v[0, :] = 0.0
        # right
        u[-1, :] = 0.0
        v[-1, :] = 0.0

    def set_pressure_boundary_conditions(self, p):
        # left
        p[0, :] = 0.0

    def set_pressure_correction_boundary_conditions(self, p):
        p[:, 0] = p[:, 1]   # bottom
        p[:, -1] = p[:, -2] # top

        p[0, :] = 0.0
        p[0, :] = p[1, :]   # left

        p[-1, :] = p[-2, :] # right

    def get_single_face_D_F_A_of_u(self, i, j):
        # assumes uniform grid spacing
        F_w = 0.5*self.rho*(self.u[i, j] + self.u[i-1, j])*self.dy
        F_e = 0.5*self.rho*(self.u[i+1, j] + self.u[i, j])*self.dy
        F_s = 0.5*self.rho*(self.v[i, j] + self.v[i-1, j])*self.dx
        F_n = 0.5*self.rho*(self.v[i, j+1] + self.v[i-1, j+1])*self.dx
        # assumes mu does not change        
        D_w = self.Gamma/self.dx*self.dy
        D_e = self.Gamma/self.dx*self.dy
        D_s = self.Gamma/self.dy*self.dx
        D_n = self.Gamma/self.dy*self.dx
        return F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n
    
    def get_field_D_F_A_of_u(self, u, v):
        """
        Calculate the fluxes and diffusion coefficients for the velocity field.

        Parameters:
        u (numpy.ndarray): The velocity field in the x-direction. Shape: (Nx+1, Ny)
        v (numpy.ndarray): The velocity field in the y-direction. Shape: (Nx, Ny+1)
        we need F at u[1:Nx, 1:Ny-1], thus F should have a shape of (Nx-1, Ny-2)
        F_w, F_e, F_s, F_n below are assigned to the location of u, F[i,j] maps to u[i+1, j+1]

        Returns:
        tuple: A tuple containing:
            - F_w (numpy.ndarray): The flux in the west direction. Shape: (Nx-1, Ny-2)
            - F_e (numpy.ndarray): The flux in the east direction. Shape: (Nx-1, Ny-2)
            - F_s (numpy.ndarray): The flux in the south direction. Shape: (Nx-1, Ny-2)
            - F_n (numpy.ndarray): The flux in the north direction. Shape: (Nx-1, Ny-2)
            - D_w (float): The diffusion coefficient in the west direction.
            - D_e (float): The diffusion coefficient in the east direction.
            - D_s (float): The diffusion coefficient in the south direction.
            - D_n (float): The diffusion coefficient in the north direction.
        """
        # assumes uniform grid spacing
        F_w = 0.5*self.rho*(u[1:-1, 1:-1] + u[:-2, 1:-1])*self.dy
        F_e = 0.5*self.rho*(u[2:, 1:-1] + u[1:-1, 1:-1])*self.dy
        F_s = 0.5*self.rho*(v[1:, 1:-2] + v[:-1, 1:-2])*self.dx
        F_n = 0.5*self.rho*(v[1:, 2:-1] + v[:-1, 2:-1])*self.dx
        # assumes mu does not change        
        D_w = self.Gamma/self.dx*self.dy
        D_e = self.Gamma/self.dx*self.dy
        D_s = self.Gamma/self.dy*self.dx
        D_n = self.Gamma/self.dy*self.dx
        return F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n

    def get_single_face_D_F_A_of_v(self, i, j):
        # assumes uniform grid spacing
        F_w = 0.5*self.rho*(self.u[i, j] + self.u[i, j-1])*self.dy
        F_e = 0.5*self.rho*(self.u[i+1, j] + self.u[i+1, j-1])*self.dy
        F_s = 0.5*self.rho*(self.v[i, j-1] + self.v[i, j])*self.dx
        F_n = 0.5*self.rho*(self.v[i, j] + self.v[i, j+1])*self.dx
        # assumes mu does not change with space
        D_w = self.Gamma/self.dx*self.dy
        D_e = self.Gamma/self.dx*self.dy
        D_s = self.Gamma/self.dy*self.dx
        D_n = self.Gamma/self.dy*self.dx
        return F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n
    
    def get_field_D_F_A_of_v(self, u, v):
        """
        Calculate the fluxes and diffusion coefficients for the velocity field.

        Parameters:
        u (numpy.ndarray): The velocity field in the x-direction. Shape: (Nx+1, Ny)
        v (numpy.ndarray): The velocity field in the y-direction. Shape: (Nx, Ny+1)
        we need F at v[1:Nx-1, 1:Ny], thus F should have a shape of (Nx-2, Ny-1)
        F_w, F_e, F_s, F_n below are assigned to the location of v, F[i,j] maps to v[i+1, j+1]        

        Returns:
        tuple: A tuple containing:
            - F_w (numpy.ndarray): The flux in the west direction. Shape: (Nx-2, Ny-1)
            - F_e (numpy.ndarray): The flux in the east direction. Shape: (Nx-2, Ny-1)
            - F_s (numpy.ndarray): The flux in the south direction. Shape: (Nx-2, Ny-1)
            - F_n (numpy.ndarray): The flux in the north direction. Shape: (Nx-2, Ny-1)
            - D_w (float): The diffusion coefficient in the west direction.
            - D_e (float): The diffusion coefficient in the east direction.
            - D_s (float): The diffusion coefficient in the south direction.
            - D_n (float): The diffusion coefficient in the north direction.
        """
        # assumes uniform grid spacing
        F_w = 0.5*self.rho*(u[1:-2, 1:] + u[1:-2, :-1])*self.dy
        F_e = 0.5*self.rho*(u[2:-1, 1:] + u[2:-1, :-1])*self.dy
        F_s = 0.5*self.rho*(v[1:-1, :-2] + v[1:-1, 1:-1])*self.dx
        F_n = 0.5*self.rho*(v[1:-1, 1:-1] + v[1:-1, 2:])*self.dx
        # assumes mu does not change with space
        D_w = self.Gamma/self.dx*self.dy
        D_e = self.Gamma/self.dx*self.dy
        D_s = self.Gamma/self.dy*self.dx
        D_n = self.Gamma/self.dy*self.dx
        return F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n

    def set_scheme_coefficients(self, F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n):
        aP, aW, aE, aS, aN = self.get_upwind_coefficients(F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n)
        # aP, aW, aE, aS, aN = self.get_hybrid_coefficients(F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n)
        return aP, aW, aE, aS, aN

    def get_upwind_coefficients(self, F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n):
        aW = D_w + np.maximum(F_w, 0) 
        aE = D_e + np.maximum(-F_e, 0)
        aS = D_s + np.maximum(F_s, 0)
        aN = D_n + np.maximum(-F_n, 0)
        aP = aW + aE + aS + aN + F_e - F_w + F_n - F_s

        min_aP = np.min(aP)
        if 0.0001*4*0.9 > min_aP:
            warnings.warn(f"Unstable solution: aP = {min_aP}", UserWarning) 
        aP = np.maximum(aP, self.epsilon)

        return aP, aW, aE, aS, aN
    
    def get_hybrid_coefficients(self, F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n):
        # West face
        aW = np.where(np.abs(F_w) <= 2 * D_w, D_w - 0.5 * F_w, D_w + np.maximum(F_w, 0))
        # East face
        aE = np.where(np.abs(F_e) <= 2 * D_e, D_e + 0.5 * F_e, D_e + np.maximum(-F_e, 0))
        # South face
        aS = np.where(np.abs(F_s) <= 2 * D_s, D_s - 0.5 * F_s, D_s + np.maximum(F_s, 0))
        # North face
        aN = np.where(np.abs(F_n) <= 2 * D_n, D_n + 0.5 * F_n, D_n + np.maximum(-F_n, 0))
        
        # Central coefficient
        aP = aW + aE + aS + aN + F_e - F_w + F_n - F_s

        return aP, aW, aE, aS, aN
    
    def get_coefficient_field_of_velocity(self, u, v):
        """a(i,j) maps to u(i+1, j+1), v(i+1, j+1), only the interior cells have a assigned"""
        F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n = self.get_field_D_F_A_of_u(u, v)
        aP_u, aW_u, aE_u, aS_u, aN_u = self.set_scheme_coefficients(F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n)
        F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n = self.get_field_D_F_A_of_v(u, v)
        aP_v, aW_v, aE_v, aS_v, aN_v = self.set_scheme_coefficients(F_w, F_e, F_s, F_n, D_w, D_e, D_s, D_n)       
            
        return aP_u, aW_u, aE_u, aS_u, aN_u, aP_v, aW_v, aE_v, aS_v, aN_v

    def solve_momentum_equation(self, u_init, v_init):

        u = u_init.copy()
        v = v_init.copy()
        
        accuracy_u = []
        accuracy_v = []
        for it in range(self.momentum_epoch):
            u_old = u.copy()
            v_old = v.copy()
            self.set_velocity_boundary_conditions(u, v)
            aP_u, aW_u, aE_u, aS_u, aN_u, aP_v, aW_v, aE_v, aS_v, aN_v = self.get_coefficient_field_of_velocity(u, v)

            min_aP_u = np.min(aP_u)
            min_aP_v = np.min(aP_v)
            if 0.0001*4*0.9 > min_aP_u or 0.0001*4*0.9 > min_aP_v:
                warnings.warn(f"Unstable solution: aP_u = {min_aP_u}, aP_v = {min_aP_v}", UserWarning)
            for i in range(1, Nx):
                for j in range(1, Ny-1):
                    u[i, j] = (aW_u[i-1, j-1] * u[i-1, j] + aE_u[i-1, j-1] * u[i+1, j] +
                                aS_u[i-1, j-1] * u[i, j-1] + aN_u[i-1, j-1] * u[i, j+1] + self.dy*(self.p[i-1, j] - self.p[i,j])) / aP_u[i-1, j-1]

            for j in range(1, Ny):
                for i in range(1, Nx-1):
                    v[i, j] = (aW_v[i-1, j-1] * v[i-1, j] + aE_v[i-1, j-1] * v[i+1, j] +
                                aS_v[i-1, j-1] * v[i, j-1] + aN_v[i-1, j-1] * v[i, j+1] + self.dx*(self.p[i, j-1] - self.p[i,j])) / aP_v[i-1, j-1]
            accuracy_per_step_u = np.linalg.norm(u - u_old, ord=np.inf)
            accuracy_u.append(accuracy_per_step_u)
            accuracy_per_step_v = np.linalg.norm(v - v_old, ord=np.inf)
            accuracy_v.append(accuracy_per_step_v)
            if accuracy_per_step_u < 1e-6 and accuracy_per_step_v < 1e-6:
                break
        return u, v, accuracy_u, accuracy_v

    def solve_pressure_field_coefficients(self, u, v):
        """accodring to eqn(6.32), cell (I, J) depends on the neighboring cells (I-1, J), (I+1, J), (I, J-1), (I, J+1), 
        only interior cells have assigned coefficients. So we only need the coeffs in cell p[1:Nx-1, 1:Ny-1]"""
        aP_u, aW_u, aE_u, aS_u, aN_u, aP_v, aW_v, aE_v, aS_v, aN_v = self.get_coefficient_field_of_velocity(u, v)
        # because rho does not change for the entire domain, the coeffs of eqn(6.32) has rho cancelled out
        # I, J corresponds to i,J and I,j, thus I from 1 to Nx-1, J from 1 to Ny-1:
        # a[I, J] ~ d[i, J] ~ a[i, J]
        # a[I, J] ~ d[I, j] ~ a[I, j]
        # b_prime[I, J] ~ u*[i, J] ~ v*[I, j]
        # so same as aP_u and aP_v, we map d[i, J]/d[I, j] to dA_u[i-1, j] / dA_v[i, j-1]
        d_u = self.dy/aP_u
        d_v = self.dx/aP_v
        dA_u = self.dy/aP_u*self.dy
        dA_v = self.dx/aP_v*self.dx
        b_prime = u[1:-2,1:-1]*self.dy - u[2:-1,1:-1]*self.dy + v[1:-1,1:-2]*self.dx - v[1:-1,2:-1]*self.dx
        a_E = dA_u[1:, :]
        a_W = dA_u[:-1, :]
        a_N = dA_v[:, 1:]
        a_S = dA_v[:, :-1]
        a_P = a_E + a_W + a_N + a_S
        a_P = np.maximum(a_P, self.epsilon)
        return a_E, a_W, a_N, a_S, a_P, b_prime, d_u, d_v

    def solve_pressure_correction(self, u, v):
        """Solve the pressure correction equation."""
        p_prime = np.zeros(self.p.shape)  # Initialize pressure correction field
        accuracy = []
        
        a_E, a_W, a_N, a_S, a_P, b_prime, d_u, d_v = self.solve_pressure_field_coefficients(u, v)
        for it in range(self.pressure_epoch):
            p_prime_old = p_prime.copy()
            self.set_pressure_correction_boundary_conditions(p_prime)
           
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    p_prime[i, j] = (a_W[i-1, j-1] * p_prime[i-1, j] + a_E[i-1, j-1] * p_prime[i+1, j] +
                                a_S[i-1, j-1] * p_prime[i, j-1] + a_N[i-1, j-1] * p_prime[i, j+1] + b_prime[i-1, j-1]) / a_P[i-1, j-1]

            accuracy_per_step = np.linalg.norm(p_prime - p_prime_old, ord=np.inf)
            accuracy.append(accuracy_per_step)
            if accuracy_per_step < 1e-6:
                break
        return p_prime, d_u, d_v,accuracy

    def apply_correction(self, u, v, p, p_prime, d_u, d_v):
        # apply relaxation
        alpha_p = 0.1
        alpha_u = 0.5
        alpha_v = 0.5
        p_new = p.copy()
        self.set_pressure_boundary_conditions(p_new)
        p_new = p_new + p_prime*alpha_p
        u_new = u.copy()
        u_new[1:-1, 1:-1] = u_new[1:-1, 1:-1] + d_u*(p[1:, 1:-1] - p[:-1, 1:-1])*alpha_u
        v_new = v.copy()
        v_new[1:-1, 1:-1] = v_new[1:-1, 1:-1] + d_v*(p[1:-1, 1:] - p[1:-1, :-1])*alpha_v
        return u_new, v_new, p_new

    def solve_other_properties(self):
        pass

    def solve_simple(self):
        self.apply_initial_guess()
        for it in range(self.simple_epoch):
            if it % 10 == 0:
                print(f"Simple solver iteration {it}")
                # save the current state
                np.save(os.path.join(self.current_dir, f"u_{it}.npy"), self.u)
                np.save(os.path.join(self.current_dir, f"v_{it}.npy"), self.v)
                np.save(os.path.join(self.current_dir, f"p_{it}.npy"), self.p)
            u, v, accuracy_u_momentum, accuracy_v_momentum = self.solve_momentum_equation(self.u, self.v)
            p_prime, d_u, d_v,accuracy_p_correction = self.solve_pressure_correction(u, v)
            u_new, v_new, p_new = self.apply_correction(u, v, self.p, p_prime, d_u, d_v)
            self.solve_other_properties()
            accuracy_u = np.linalg.norm(self.u - u_new, ord=np.inf)
            accuracy_v = np.linalg.norm(self.v - v_new, ord=np.inf)
            accuracy_p = np.linalg.norm(self.p - p_new, ord=np.inf)
            self.accuracy_trace['u'].append(accuracy_u)
            self.accuracy_trace['v'].append(accuracy_v)
            self.accuracy_trace['p'].append(accuracy_p)
            self.u, self.v, self.p = u_new, v_new, p_new
            print(f"Accuracy: u {accuracy_u}, v {accuracy_v}, p {accuracy_p}")
            if accuracy_u < 1e-6 and accuracy_v < 1e-6 and accuracy_p < 1e-6:
                break

    def plot_results(self):
        # velocity field
        plt.figure()
        x = np.linspace(0, self.L, self.Nx)
        y = np.linspace(0, self.H, self.Ny)
        x_coords, y_coords = np.meshgrid(x, y)
        plt.quiver(x_coords.T, y_coords.T, self.u[:-1, :], self.v[:, :-1], angles='xy', scale_units='xy')  # self.u/v[i,j] maps to the cell center coordinate (i+1, j+1)
        # pressure field
        plt.figure()
        plt.contourf(self.p)
        plt.colorbar()

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.accuracy_trace['u'])
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('u')
        plt.subplot(3, 1, 2)
        plt.plot(self.accuracy_trace['v'])
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('v')
        plt.subplot(3, 1, 3)
        plt.plot(self.accuracy_trace['p'])
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('p')
        plt.show()

def get_velocity_field_cylinder(x, y, U, R):
    """
    Calculate the velocity field for flow around a circular cylinder.
    (0, 0) is the center of the cylinder.
    Parameters:
        x (float or np.ndarray): x-coordinate(s)
        y (float or np.ndarray): y-coordinate(s)
        U (float): Free-stream velocity
        R (float): Radius of the cylinder
    Returns:
        u (np.ndarray): x-component of velocity
        v (np.ndarray): y-component of velocity
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    u_r = U * (1 - (R**2 / r**2)) * np.cos(theta)
    u_theta = -U * (1 + (R**2 / r**2)) * np.sin(theta)

    # Velocity components for flow around a circular cylinder in Cartesian coordinates:
    # u = U * (1 + (R^2 / r^2) * sin^2(theta) - (R^2 / r^2) * cos^2(theta))
    # v = -2 * U * (R^2 / r^2) * sin(theta) * cos(theta)
    u = (r > R) *u_r * np.cos(theta) - u_theta * np.sin(theta)
    v = (r > R) *u_r * np.sin(theta) + u_theta * np.cos(theta)    
    return u, v

def get_peclet_number(u, v, dx, dy, rho, Gamma):
    # Calculate the maximum absolute velocity in the x and y directions
    max_u = np.max(np.abs(u))
    max_v = np.max(np.abs(v))
    
    # Calculate the Peclet number in the x and y directions
    peclet_x = max_u * dy * rho / Gamma
    peclet_y = max_v * dx * rho / Gamma
    
    # Return the maximum Peclet number
    return max(peclet_x, peclet_y)

if __name__ == "__main__":
    # Domain parameters
    L, H = 1, 1
    Nx, Ny = 100, 100
    dx, dy = L / Nx, H / Ny
    # Flow parameters
    u, v = 1.0, 1.0  # m/s Velocity
    Gamma = 0.0001  # m^2/s Diffusion coefficient
    rho = 1.225 # kg/m^3 Density of air
    
    artificial_viscosity_scale = 100

    peclet = get_peclet_number(u, v, dx, dy, rho, Gamma*artificial_viscosity_scale)
    if peclet > 2:
        warnings.warn(f"Instability warning: Peclet number {peclet} > 2", UserWarning)

    solver = SimpleSolver(L, H, Nx, Ny, dx, dy, Gamma*artificial_viscosity_scale, rho)
    solver.solve_simple()
    print(solver.u)
    print(solver.v)
    print(solver.p)
    solver.plot_results()