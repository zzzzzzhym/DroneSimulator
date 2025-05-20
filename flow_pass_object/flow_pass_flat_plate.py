import numpy as np
import warnings
import matplotlib.pyplot as plt

class FlowPassFlatPlate:
    class Interface:
        """This class provides interface functions to convert wall with arbitrary
        facing direction and wind with arbitrary direction to the nominal formulation
        of the flow pass a plate problem, and convert the solution back to the original
        problem.
        """
        def __init__(self, wall_normal: np.ndarray, wall_infinity_direction: np.ndarray, wall_origin: np.ndarray, wall_length: float):
            """
            Initialize the Converter class.

            Args:
                wall_normal (np.ndarray): Wall normal vector.
                wall_origin (np.ndarray): Origin point of the wall.
                wall_length (float): Length of the wall.
            """
            self.wall_origin = wall_origin
            self.wall_length = wall_length
            self.wall_normal = wall_normal
            self.wall_infinity_direction = FlowPassFlatPlate.Interface.correct_wall_infinity_direction(wall_normal, wall_infinity_direction)
            self.model = FlowPassFlatPlate.CoreFormulation(a=wall_length/4)
            self.transformation_matrix_inertial_to_model = None
            self.transformation_matrix_model_to_inertial = None
            self.u_free = None
            self.is_wind_constant_direction_set = False
        
        @staticmethod
        def correct_wall_infinity_direction(wall_normal: np.ndarray, wall_infinity_direction: np.ndarray):
            """correct the wall_infinity_direction to be in the same plane as wall_normal
            """
            wall_infinity_direction = wall_infinity_direction - np.dot(wall_infinity_direction, wall_normal) * wall_normal
            wall_infinity_direction_norm = np.linalg.norm(wall_infinity_direction)
            if wall_infinity_direction_norm < 1e-6:
                raise ValueError("The wall infinity direction is too close to wall normal")
            else:
                return wall_infinity_direction / wall_infinity_direction_norm

        def set_constant_wind(self, u_free: np.ndarray):
            """Compute transformation matrices before solution to save computation time"""
            self.transformation_matrix_inertial_to_model, self.transformation_matrix_model_to_inertial = self.get_rotation_matrix_between_inertial_and_model_frame(u_free)
            self.is_wind_constant_direction_set = True

        def get_solution(self, u_free: np.ndarray, location: np.ndarray, is_wind_constant_direction_set: bool = False):
            """Get the velocity at the location in the original frame"""
            if is_wind_constant_direction_set:
                if not self.is_wind_constant_direction_set:
                    self.set_constant_wind(u_free)
                transformation_matrix_inertial_to_model = self.transformation_matrix_inertial_to_model
                transformation_matrix_model_to_inertial = self.transformation_matrix_model_to_inertial
            else:
                transformation_matrix_inertial_to_model, transformation_matrix_model_to_inertial = self.get_rotation_matrix_between_inertial_and_model_frame(u_free)
            location_in_model = self.convert_wall_location_to_model_frame(location, transformation_matrix_inertial_to_model)
            u_free_in_model = FlowPassFlatPlate.Interface.convert_u_free_to_model_frame(u_free, transformation_matrix_inertial_to_model)
            u_free_norm, alpha = FlowPassFlatPlate.Interface.get_model_input(u_free_in_model)
            u_in_model, v_in_model = self.model.get_velocity(u_free_norm, alpha, location_in_model[0], location_in_model[1])
            velocity_in_inertial = self.convert_velocity_to_inertial_frame(np.array([u_in_model, v_in_model, u_free_in_model[2]]), transformation_matrix_model_to_inertial)
            return velocity_in_inertial

        def convert_wall_location_to_model_frame(self, location_original: np.ndarray, transformation_matrix_inertial_to_model):
            """convert wall original location to the nominal model of flowing pass a flat plate
            in the nominal model, wall is vertical with norm pointing to -y
            """
            location_in_model = transformation_matrix_inertial_to_model@(location_original - self.wall_origin)
            return location_in_model

        def convert_velocity_to_inertial_frame(self, velocity_in_model: np.ndarray, transformation_matrix_model_to_inertial):
            """convert velocity in the nominal model of flowing pass a flat plate to the original frame
            """
            velocity_in_inertial = transformation_matrix_model_to_inertial@velocity_in_model
            return velocity_in_inertial

        @staticmethod
        def convert_u_free_to_model_frame(u_free: np.ndarray, transformation_matrix_inertial_to_model):
            """convert free stream velocity to the nominal model of flowing pass a flat plate
            in the nominal model, wall is vertical with norm pointing to -y
            """
            u_free_in_model = transformation_matrix_inertial_to_model@u_free
            return u_free_in_model

        @staticmethod
        def get_model_input(u_free_in_model: np.ndarray):
            """Get the attack angle of the flow from the wall facing direction. This is a utility function for users to get the attack angle from the wall facing direction.
            
            Args:
                u_free (np.ndarray): Free stream velocity vector.
                wall_direction (np.ndarray): Wall facing direction vector.
            
            Returns:
                float: Attack angle in [rad].
            """
            u_free_norm = np.linalg.norm(u_free_in_model[:2])
            if u_free_norm < 1e-3:
                alpha = 0
            else:
                u_free_direction = u_free_in_model / u_free_norm
                alpha = np.arctan2(u_free_direction[1], u_free_direction[0])
            return u_free_norm, alpha

        def get_rotation_matrix_between_inertial_and_model_frame(self, u_free):
            """rotation matrix from wall frame to wind field model frame
            Assume the wall is vertical, and wind is horizontal in inertial frame
            """
            y_axis = -self.wall_normal
            z_axis = self.wall_infinity_direction
            x_axis = np.cross(y_axis, z_axis)
            transformation_matrix_inertial_to_model = np.array([x_axis, y_axis, z_axis])
            transformation_matrix_model_to_inertial = transformation_matrix_inertial_to_model.T
            return transformation_matrix_inertial_to_model, transformation_matrix_model_to_inertial
            
        def analyze_velocity_field(self, u_free=np.array([12.0, 0.0, 0.0]), field_dimension=5, resolution=20):
            field_dimension = field_dimension
            resolution = resolution
            X_range = np.linspace(-field_dimension, field_dimension, resolution)
            Y_range = np.linspace(-field_dimension, field_dimension, resolution)
            X_field, Y_field = np.meshgrid(X_range, Y_range)
            u_field = np.zeros_like(X_field)
            v_field = np.zeros_like(Y_field)
            self.calculate_velocity_field(u_free, X_range, Y_range, u_field, v_field)
            self.print_velocity_norm(u_field, v_field)
            self.plot_velocity_field(X_field, Y_field, u_field, v_field)

        def calculate_velocity_field(self, u_free, X_range, Y_range, u_field, v_field):
            for i, X in enumerate(X_range):
                for j, Y in enumerate(Y_range):
                    velocity_in_inertial = self.get_solution(u_free, np.array([X, Y, 0]), is_wind_constant_direction_set=True)
                    u_field[j, i] = velocity_in_inertial[0]
                    if np.abs(velocity_in_inertial[0]) < 1e-6:
                        print("bug")
                    v_field[j, i] = velocity_in_inertial[1]

        def plot_velocity_field(self, X_field, Y_field, u_field, v_field):
            plt.figure(figsize=(8, 8))
            plt.title("Velocity Field arrows for direction and magnitude")
            plt.quiver(X_field, Y_field, u_field, v_field, np.sqrt(u_field**2 + v_field**2), angles='xy', scale=500, cmap="viridis")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.colorbar(label="Velocity Magnitude")

            wall_on_xy = np.array([-self.wall_normal[1], self.wall_normal[0], 0])
            wall_on_xy_norm = np.linalg.norm(wall_on_xy)
            if wall_on_xy_norm > 1e-6:
                wall_on_xy = wall_on_xy /wall_on_xy_norm
                end_1 = self.wall_origin + 0.5 * self.wall_length * wall_on_xy
                end_2 = self.wall_origin - 0.5 * self.wall_length * wall_on_xy
                x_line = np.linspace(end_1[0], end_2[0], 100)
                y_line = np.linspace(end_1[1], end_2[1], 100)
                plt.plot(x_line, y_line, color="black", linestyle="--", label="Line from -2a to 2a")
            else:
                print("Wall normal is parallel to x-y plane, cannot plot the wall")

            plt.show()

        def print_velocity_norm(self, u_field, v_field):
            velocity_norm = np.sqrt(u_field**2 + v_field**2)
            print("Velocity Magnitude Field")
            for row in velocity_norm.T:
                print(" ".join(f"{value:6.1f}" for value in row))


    class CoreFormulation:
        """reference: Currie, Iain G., and I. G. Currie. Fundamental mechanics of fluids. CRC press, 2002.
        """
        def __init__(self, a=1.0):
            """
            Initialize the flow past a flat plate simulation.

            Args:
                U (float, optional): Free stream velocity in [m/s]. Defaults to 12.0.
                alpha (float, optional): Angle of attack in [rad]. Defaults to np.pi/3.
                a (float, optional): Semi-major axis length of the flat plate in [m]. This is 1/4 of the length of the wall. Defaults to 1.0.
                field_dimension (int, optional): The dimension of the field to simulate in [m]. Defaults to 5.
                resolution (int, optional): The resolution of the simulation grid. Defaults to 20.
            """
            self.a = a
            self.c = a  # conformal transformation (this is Joukowski transformation) parameter. In this case, it is the semi-major axis

        def get_velocity(self, U, alpha, X, Y):
            Z = X + 1j * Y
            z1 = 0.5 * Z + 0.5 * np.sqrt(Z**2 - 4 * self.c**2)
            z2 = 0.5 * Z - 0.5 * np.sqrt(Z**2 - 4 * self.c**2)
            if np.abs(z1) > self.c:
                z = z1
            elif np.abs(z2) > self.c:
                z = z2
            else:
                warnings.warn("Invalid z: expecting |z| > c. Possibly inquiring a point on the plate.")
                return 0, 0
            dWdz = (U * np.exp(-1j * alpha) - U * np.exp(1j * alpha) * self.a**2 / z**2) / (1 - self.a**2 / z**2)
            u = np.real(dWdz)
            v = -np.imag(dWdz)
            return u, v


if __name__ == "__main__":
    # Example usage
    wall_norm = np.array([0.0, 0.0, 1.0])   # facing x-axis
    wall_origin = np.array([0.0, 0.0, -0.01])
    wall_length = 8.0    
    flow = FlowPassFlatPlate.Interface(wall_norm, wall_origin, wall_length)
    u_free=np.array([10.0, 0.0, 0.0])
    flow.analyze_velocity_field(u_free)

