import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from learning import model
from learning import trainer

class UnitDisturbance:
    """disturbance in a single dimension
    """
    def __init__(self, num_of_kernels, dt: float) -> None:
        self.disturbance = 0
        self.a_hat = np.zeros([num_of_kernels, 1])   # estimated a matrix
        self.p_cov = np.eye(num_of_kernels)     # covariance matrix, p_cov.shape = dim_of_a_hat*dim_of_a_hat
        self.lambda_reg = np.eye(num_of_kernels)*0.01
        self.dt = dt
        self.r = np.eye(1)  # measurement noise, r.shape = (measured disturbance dimension)^2
        self.q = np.eye(num_of_kernels)    # process noise, q.shape = p_cov.shape   
        self.model_scale = 10 # normalization factor for the model labels

    @staticmethod
    def update_adaptation(a_hat: np.ndarray, 
                            p: np.ndarray, 
                            phi_vec: np.ndarray, 
                            y_vec: float, 
                            s_vec: float, 
                            r: np.ndarray, 
                            q: np.ndarray, 
                            lambda_val: np.ndarray, 
                            dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Updates the linear parameter estimate (a_hat) and the covariance matrix (P)
        based on the composite adaptation law.

        Args:
            a_hat (numpy.ndarray): 1D. Current estimate of the linear parameter vector.
            p (numpy.ndarray): 2D. Current covariance matrix for the parameter estimate.
            phi (numpy.ndarray): 1D. Learned basis function representing the aerodynamic effect.
            y (float): Measured aerodynamic residual force.
            s (float): Composite tracking error, defined as s = dot(q) - dot(q_r).
            r (numpy.ndarray): 2D. Positive definite gain matrix for prediction error weighting.
            q (numpy.ndarray): 2D. Positive definite gain matrix for covariance update.
            lambda_val (np.ndarray): 2D. Damping gain for regularization.
            dt (float): Time step size for discretization.

        Returns:
            tuple:
                a_hat_new (numpy.ndarray): Updated linear parameter estimate.
                P_new (numpy.ndarray): Updated covariance matrix.

        Notes:
            - The method implements Equations 8 and 9 from the composite adaptation law.
            - Equation 8 updates the parameter estimate (a_hat) based on the prediction error,
            tracking error (s), and a regularization term.
            - Equation 9 updates the covariance matrix (P), accounting for process noise (Q)
            and measurement noise (R).
        """        
        # Equation 8 - Update for a_hat
        phi = np.atleast_2d(phi_vec)    # treat this as C matrix in y = Cx + D, do not need to transpose
        s = np.atleast_2d(s_vec).T
        y = np.atleast_2d(y_vec).T
        prediction_error = phi @ a_hat - y
        a_hat_dot = -lambda_val @ a_hat - p @ phi.T @ np.linalg.inv(r) @ prediction_error + p @ phi.T @ s
        a_hat_new = a_hat + a_hat_dot * dt

        # Equation 9 - Update for P
        P_dot = -2 * lambda_val * p + q - p @ phi.T @ np.linalg.inv(r) @ phi @ p
        P_new = p + P_dot * dt

        return a_hat_new, P_new     

    def step_estimator(self, kernel: np.ndarray, measured_disturbance: np.ndarray, tracking_error: np.ndarray) -> None:
        """
        Updates the disturbance estimator with the given kernel, measured disturbance, and tracking error.

        Args:
            kernel (np.ndarray): Kernel output.
            measured_disturbance (np.ndarray): The measured aerodynamic residual force.
            tracking_error (np.ndarray): The composite tracking error, defined as s = dot(q) - dot(q_r).
        """
        self.a_hat, self.p_cov = self.update_adaptation(self.a_hat, self.p_cov, kernel, measured_disturbance*self.model_scale, tracking_error, self.r, self.q, self.lambda_reg, self.dt)
        self.disturbance = self.get_disturbance(kernel)

    def get_disturbance(self, kernel: np.ndarray):
        disturbance = np.atleast_2d(kernel)@self.a_hat*(1/self.model_scale)
        return disturbance[0, 0]

class DisturbanceEstimator:
    def __init__(self, model_name: str, dt: float) -> None:
        self.phi, h, config = trainer.load_model(model_name)
        self.phi.eval()
        self.num_of_kernals = config.phi_net.dim_of_output
        self.dof_of_disturbance = 6
        self.dt = 0.01
        self.f_x = UnitDisturbance(self.num_of_kernals, self.dt)
        self.f_y = UnitDisturbance(self.num_of_kernals, self.dt)
        self.f_z = UnitDisturbance(self.num_of_kernals, self.dt)
        self.tq_x = UnitDisturbance(self.num_of_kernals, self.dt)
        self.tq_y = UnitDisturbance(self.num_of_kernals, self.dt)
        self.tq_z = UnitDisturbance(self.num_of_kernals, self.dt)

    def update_kernel(self, v: np.ndarray, q: np.ndarray, f: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            kernel = self.phi(torch.from_numpy(np.hstack([v, q, f])))
        return kernel.numpy()
    
    def step_disturbance(self,  v: np.ndarray, q: np.ndarray, f: np.ndarray, measured_disturbance: np.ndarray, tracking_error: np.ndarray) -> None:
        self.kernel = self.update_kernel(v, q, f)
        self.f_x.step_estimator(self.kernel, measured_disturbance[0], tracking_error[0])
        self.f_y.step_estimator(self.kernel, measured_disturbance[1], tracking_error[1])
        self.f_z.step_estimator(self.kernel, measured_disturbance[2], tracking_error[2])
        self.tq_x.step_estimator(self.kernel, measured_disturbance[3], tracking_error[3])
        self.tq_y.step_estimator(self.kernel, measured_disturbance[4], tracking_error[4])
        self.tq_z.step_estimator(self.kernel, measured_disturbance[5], tracking_error[5])


if __name__ == "__main__":
    dist = DisturbanceEstimator("test", 0.01)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    f = np.array([1.0, 1.0, 1.0, 1.0])
    f_x = []
    f_y = []
    f_z = []
    tq_x = []
    tq_y = []
    tq_z = []
    t = np.arange(0, 10, 0.01)
    for _ in t:
        dist.step_disturbance(v, q, f, y, s)
        f_x.append(dist.f_x.disturbance)
        f_y.append(dist.f_y.disturbance)
        f_z.append(dist.f_z.disturbance)
        tq_x.append(dist.tq_x.disturbance)
        tq_y.append(dist.tq_y.disturbance)
        tq_z.append(dist.tq_z.disturbance)

    fig, axs = plt.subplots(6,1,sharex=True)
    axs[0].plot(t, f_x)
    axs[1].plot(t, f_y)
    axs[2].plot(t, f_z)
    axs[3].plot(t, tq_x)
    axs[4].plot(t, tq_y)
    axs[5].plot(t, tq_z)
    plt.show()
    

