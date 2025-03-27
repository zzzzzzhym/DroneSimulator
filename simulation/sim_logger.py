import os
import numpy as np
import matplotlib.pyplot as plt

import simulator

def log_sim_result(result: simulator.Engine, file_name: str) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    upper_dir = os.path.dirname(current_dir)
    file_path = os.path.join(upper_dir, "data", "training", file_name)
    if not os.path.exists(file_path):
        headers, array = construct_csv_array(result)
        np.savetxt(file_path, array, delimiter=',', fmt='%.17f', header=headers, comments='')
        print("Sim data is written into:\n" + file_path)
    else:
        raise ValueError("File already exist:\n" + file_path)

def construct_csv_array(sim_data: simulator.Engine) -> tuple[str, np.ndarray]:
    headers = "vx, vy, vz, q_0, q_1, q_2, q_3, f_motor_0, f_motor_1, f_motor_2, f_motor_3, f_disturb_x, f_disturb_y, f_disturb_z"
    csv = np.hstack((sim_data.logger_np["v"], sim_data.logger_np["q"], sim_data.logger_np["f_motor"], sim_data.logger_np["f_disturb"]))
    return headers, csv

if __name__ == "__main__":
    sim_test = simulator.Engine()
    sim_test.run_simulation(120)
    log_sim_result(sim_test, "test_wall_effect_h_1_5r_dh_0.csv")  
    sim_test.make_plots()  
    plt.show()
