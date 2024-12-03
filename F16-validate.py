import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import control as ctrl
import collimator as C

F16_model = C.load_model('F16_NonlinearModel_v2')
# F16 = C.load('F16_NonlinearModel_v2')


alt = 1500
vt = 250
trim_control = np.zeros(4)
trim_states = np.zeros(12)
trim_states[2] = alt
trim_states[6] = vt
weights = np.zeros(12)
weights[2] = 5
weights[3:6] = 10*np.ones(3)
weights[6] = 2
weights[7:12] = 10*np.ones(5)
def trim_cost(trim_values): 
    trim_control = trim_values[0:4]
    trim_states[4]  = trim_values[4]
    trim_states[7]  = trim_values[4]
    try:
        F16_model.set_parameters({'trim_control':trim_control, 'trim_states':trim_states})
        out = C.run_simulation(F16_model).to_pandas()
        states_dot = out[['states_dot_0.npos_dot','states_dot_0.epos_dot','states_dot_0.alt_dot',
            'states_dot_0.phi_dot','states_dot_0.theta_dot','states_dot_0.psi_dot',
             'states_dot_0.vt_dot','states_dot_0.alpha_dot','states_dot_0.beta_dot',
             'states_dot_0.P_dot','states_dot_0.Q_dot','states_dot_0.R_dot',
            ]].to_numpy()[-1]
    except:
        print('Inf')
        return np.Inf
    else:
        J = np.dot(np.multiply(states_dot,states_dot), weights)
        print(J)
        return J


trim_values_0 = np.array([20000, -1.5, 0, 0, 0.002])
bounds = optimize.Bounds([4000,-25,-21.5,-30,-0.1745],[90000,25,21.5,30,0.7854])
minimum = optimize.minimize(trim_cost, trim_values_0, method='Powell',bounds = bounds, 
                           options = {'maxfev': 20, 'disp': True, 'return_all':True})
print(minimum)


