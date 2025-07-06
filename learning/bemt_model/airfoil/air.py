class Air:
    rho = 1.225
    mu = 1.7894e-5 # Dynamic viscosity (Gamma in the scheme)
    nu = mu / rho    # Kinematic viscosity

if __name__ == "__main__":
    print(Air.rho)
    print(Air.mu)
    print(Air.nu)