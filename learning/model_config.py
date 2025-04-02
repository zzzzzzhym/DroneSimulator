from dataclasses import dataclass

class ModelConfig:
    @dataclass
    class PhiNet:
        dim_of_layer0 = 50
        dim_of_layer1 = 60
        dim_of_layer2 = 50
        dim_of_input: int
        dim_of_output: int

    @dataclass
    class HNet:
        dim_of_layer0 = 20
        dim_of_input: int
        dim_of_output: int

    @dataclass
    class Trainer:
        learning_rate = 0.0005
        frequency_h = 2
        num_epochs = 1000
        alpha = 0.01    # weight of cross entropy loss
        gamma = 10.0    # normalization of a
        spectral_norm = 2.0    # normalization
        a_shot = 32    # batch size for a 
        phi_shot = 256    # batch size of phi net training
        is_dynamic_environment = True    # flag to adapt to environment with matrix a, False equivalent to a = ones(size)

    def __init__(self, num_of_conditions, dim_of_input=11, dim_of_feature=5, dim_of_label=3):
        """consider to change pwm to force input or directly use F + T without conversion
        (v_0 v_1 v_2 q_0 q_1 q_2 q_3 pwm_0 pwm_1 pwm_2 pwm_3)
        velocity (3) + quaternion (4) + input (4) = 11
        number of features (not necessarily the same as dim of disturbance forces)
        dim_of_label: dimension of label vector. Because we are trying to compare the disturbance force in x y z direction,
        the label vector is a 3-dim vector, in the paper, disturbance torque is not estimated.
        """
        self.phi_net = self.PhiNet(dim_of_input=dim_of_input, dim_of_output=dim_of_feature)
        self.h_net = self.HNet(dim_of_input=dim_of_feature, dim_of_output=num_of_conditions)
        self.trainer = self.Trainer
        self.num_of_conditions = num_of_conditions
        self.dim_of_label = dim_of_label

if __name__ == "__main__":
    config = ModelConfig(num_of_conditions=3, 
                         dim_of_input=11, 
                         dim_of_feature=3,
                         dim_of_label=3)