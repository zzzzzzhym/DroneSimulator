import numpy as np
import matplotlib.pyplot as plt


def construct_coeff_a_upwind(u, v, dx, dy, rho, Gamma):
    # assume evenly spaced grid
    dx_WP = dx
    dx_PE = dx
    dy_SP = dy
    dy_NP = dy

    # weak assumption: u, v are constant in the cell
    u_w = u
    u_e = u
    v_s = v
    v_n = v

    # use definition of D and F in section 5.7
    A_w = (dy_SP + dy_NP)*0.5
    A_e = (dy_SP + dy_NP)*0.5
    A_s = (dx_WP + dx_PE)*0.5
    A_n = (dx_WP + dx_PE)*0.5
    
    D_w = Gamma/dx_WP*A_w
    D_e = Gamma/dx_PE*A_e
    F_w = rho*u_w*A_w
    F_e = rho*u_e*A_e

    D_s = Gamma/dy_SP*A_s
    D_n = Gamma/dy_NP*A_n
    F_s = rho*v_s*A_s
    F_n = rho*v_n*A_n

    aW = D_w + np.maximum(F_w, 0) 
    aE = D_e + np.maximum(-F_e, 0)
    aS = D_s + np.maximum(F_s, 0)
    aN = D_n + np.maximum(-F_n, 0)
    aP = aW + aE + aS + aN + F_e - F_w + F_n - F_s

    return aP, aW, aE, aS, aN

def solve_single_cell(phi, aP, aW, aE, aS, aN):
    Nx, Ny = phi.shape
    phi_old = phi.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            phi[i, j] = (aW * phi[i-1, j] + aE * phi[i+1, j] +
                         aS * phi[i, j-1] + aN * phi[i, j+1]) / aP
    return phi

def solve_convection_diffusion_problem(phi, u, v, dx, dy, rho, Gamma):
    Nx, Ny = phi.shape
    aP, aW, aE, aS, aN = construct_coeff_a_upwind(u, v, dx, dy, rho, Gamma)
    accuracy = []
    for it in range(1000):
        phi_old = phi.copy()
        phi = solve_single_cell(phi, aP, aW, aE, aS, aN)
        accuracy_per_step = np.linalg.norm(phi - phi_old, ord=2)
        accuracy.append(accuracy_per_step)
        if accuracy_per_step < 1e-6:
            break
    return phi, accuracy

if __name__ == "__main__":
    # Domain parameters
    L, H = 1.0, 1.0
    Nx, Ny = 50, 50
    dx, dy = L / Nx, H / Ny

    # Flow parameters
    u, v = 1.0, 1.0  # m/s Velocity
    Gamma = 0.0001  # m^2/s Diffusion coefficient
    rho = 1.225 # kg/m^3 Density of air

    # Initialize scalar field
    phi = np.zeros((Nx+1, Ny+1))

    # Boundary conditions
    phi[:, 0] = 1  # bottom boundary
    phi[0, :] = 0  # left boundary
    phi[:, -1] = 0  # top boundary
    phi[-1, :] = 0  # right boundary

    # Iterative solution
    phi, accuracy = solve_convection_diffusion_problem(phi, u, v, dx, dy, rho, Gamma)

    # Plot the solution
    plt.contourf(np.linspace(0, L, Nx+1), np.linspace(0, H, Ny+1), phi.T, levels=20, cmap='viridis')
    plt.colorbar(label='Scalar Field (phi)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Convection-Diffusion with Upwind Differencing')
    plt.axis('equal')
    #plot the accuracy
    plt.figure()
    plt.plot(accuracy)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Convergence of the solution')
    plt.show()