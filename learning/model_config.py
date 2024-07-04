num_of_conditions = 1

phi_net = {}

phi_net['dim_of_layer0'] = 50
phi_net['dim_of_layer1'] = 60
phi_net['dim_of_layer2'] = 50
# consider to change pwm to force input or directly use F + T without conversion
# velocity (3) + quaternion (4) + input (4) = 11
phi_net['dim_of_input'] = 11   # v_0 v_1 v_2 q_0 q_1 q_2 q_3 pwm_0 pwm_1 pwm_2 pwm_3 
# number of features (not necessarily the same as dim of disturbance forces)
phi_net['dim_of_output'] = 3   # fa_x fa_y fa_z

h_net = {}
h_net['dim_of_layer0'] = 20
# should be the same as output of phi net
h_net['dim_of_input'] = phi_net['dim_of_output']
# number of wind conditions
h_net['dim_of_output'] = num_of_conditions

training = {}
training['learning_rate'] = 0.0005
training['frequency_h'] = 2
training['num_epochs'] = 1000
training['alpha'] = 0.01    # weight of cross entropy loss
training['gamma'] = 10.0    # normalization of a
training['spectral_norm'] = 2.0    # normalization
training['a_shot'] = 32    # batch size for a 
training['phi_shot'] = 256    # batch size of phi net training
training['SN'] = 2    # batch size of phi net training



