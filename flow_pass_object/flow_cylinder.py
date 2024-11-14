import numpy as np
import matplotlib.pyplot as plt

def velocity_potential_cylinder(x, y, U, R):
    """
    Calculate the velocity potential function for flow around a circular cylinder.
    Parameters:
        x (float or np.ndarray): x-coordinate(s)
        y (float or np.ndarray): y-coordinate(s)
        U (float): Free-stream velocity
        R (float): Radius of the cylinder
    Returns:
        phi (float or np.ndarray): Velocity potential at the given coordinate(s)
    
    Equation:
        phi = U * (r + (R^2 / r)) * cos(theta)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    phi = (r > R) * U * (r + (R**2 / r)) * np.cos(theta)
    
    return phi

def stream_function_cylinder(x, y, U, R):
    """
    Calculate the stream function for flow around a circular cylinder.
    Parameters:
        x (float or np.ndarray): x-coordinate(s)
        y (float or np.ndarray): y-coordinate(s)
        U (float): Free-stream velocity
        R (float): Radius of the cylinder
    Returns:
        psi (float or np.ndarray): Stream function at the given coordinate(s)
    
    Equation:
        psi = U * (r - (R^2 / r)) * sin(theta)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    psi =  (r > R) * U * (r - (R**2 / r)) * np.sin(theta)
    return psi

def velocity_field_cylinder(x, y, U, R):
    """
    Calculate the velocity field for flow around a circular cylinder.
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

def plot_flow_with_direction(U, R, x_range=(-120, -10), y_range=(-5, 5)):
    """
    Plot the velocity potential, streamlines, and velocity field (quiver) for flow around a circular cylinder.
    Parameters:
        U (float): Free-stream velocity
        R (float): Radius of the cylinder
        x_range (tuple): Range of x values for the plot
        y_range (tuple): Range of y values for the plot
        grid_size (int): Number of grid points in each direction
    """
    x = np.linspace(x_range[0], x_range[1], 600)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)

    # Calculate stream function and velocity components
    phi = velocity_potential_cylinder(X, Y, U, R)
    psi = stream_function_cylinder(X, Y, U, R)
    u, v = velocity_field_cylinder(X, Y, U, R)

    # plt.figure(figsize=(8, 6))
    # heatmap = plt.contourf(X, Y, phi, levels=200, cmap='plasma')  # Heat map with filled contours
    # plt.colorbar(heatmap, label="Velocity Potential")
    # plt.gca().add_patch(plt.Circle((0, 0), R, color='k', fill=False, linewidth=2, linestyle='--', label='Cylinder'))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Heat Map of Velocity Potential Around a Cylinder')
    # plt.axis('equal')

    # Plot streamlines using stream function
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, psi, levels=100, cmap="coolwarm")  # Contours for streamlines
    plt.colorbar(contour, label="Stream Function (ψ)")
    plt.gca().add_patch(plt.Circle((0, 0), R, color='k', fill=False, linewidth=2, linestyle='--'))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Streamlines Around a Circular Cylinder')
    plt.axis('equal')

    plt.show()


if __name__ == "__main__":
    U = 12.0  # Free-stream velocity
    R = 100.0  # Radius of the cylinder
    plot_flow_with_direction(U, R)
