from dataclasses import dataclass

# num_of_conditions = 1

# phi_net = {}

# phi_net['dim_of_layer0'] = 10
# phi_net['dim_of_layer1'] = 6
# phi_net['dim_of_layer2'] = 3
# # consider to change pwm to force input or directly use F + T without conversion
# # velocity (3) + quaternion (4) + input (4) = 11
# phi_net['dim_of_input'] = 11   # v_0 v_1 v_2 q_0 q_1 q_2 q_3 pwm_0 pwm_1 pwm_2 pwm_3 
# # number of features (not necessarily the same as dim of disturbance forces)
# phi_net['dim_of_output'] = 3   # fa_x fa_y fa_z

# h_net = {}
# h_net['dim_of_layer0'] = 20
# # should be the same as output of phi net
# h_net['dim_of_input'] = phi_net['dim_of_output']
# # number of wind conditions
# h_net['dim_of_output'] = num_of_conditions

# training = {}
# training['learning_rate'] = 0.0005
# training['frequency_h'] = 2
# training['num_epochs'] = 2000
# training['alpha'] = 0.01    # weight of cross entropy loss
# training['gamma'] = 10.0    # normalization of a
# training['spectral_norm'] = 2.0    # normalization
# training['a_shot'] = 32    # batch size for a 
# training['phi_shot'] = 256    # batch size of phi net training
# training['SN'] = 2    # batch size of phi net training

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
        phi_shot = 512    # batch size of phi net training
        is_dynamic_environment = True    # flag to adapt to environment with matrix a, False equivalent to a = ones(size)

    def __init__(self, num_of_conditions, dim_of_input=11, dim_of_feature=3, dim_of_label=3):
        """consider to change pwm to force input or directly use F + T without conversion
        (v_0 v_1 v_2 q_0 q_1 q_2 q_3 pwm_0 pwm_1 pwm_2 pwm_3)
        velocity (3) + quaternion (4) + input (4) = 11
        number of features (not necessarily the same as dim of disturbance forces)"""
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