import numpy as np
import control as ctrl

def seismograph(u):
    return general_seismograph(u, 0.0, 0)

def noisy_seismograph(u):
    n_delay = 0
    noise_variance = 0.1
    return general_seismograph(u, noise_variance, n_delay)

def stabilized_seismograph(u, k_1, k_2, noise_variance=0.0):
    Ts = 1.0/1000.0
    theta_C = np.array([-0.667101077, 6.51615437e-04, 189.236476, -221.343890, 127.566764])
    theta_C = np.hstack(((k_1+0.2)*theta_C[0:2], (-(k_2-0.3)**2+1.0)*theta_C[2:5]))
    
    C_z = S(theta_C)
    S_z = S()
    T_z = C_z*S_z / (1+C_z*S_z)
    
    if u.ndim == 1:
        number_of_inputs = 1
        length_of_input = u.shape[0]
    else:
        number_of_inputs = u.shape[0]
        length_of_input = u.shape[1]
    T = np.linspace(0.0, Ts*length_of_input, length_of_input)
    y = np.zeros((number_of_inputs, length_of_input))
    
    for k in range(number_of_inputs):
        if u.ndim > 1:
            u_k = u[k, :]
        else:
            u_k = u
        response = ctrl.forced_response(T_z, T, u_k, 0.0)
        y_k = response.y
        y[k, :] = y_k + np.random.normal(0.0, noise_variance, np.size(y_k))
    return y if u.ndim > 1 else y[0, :]

def noisy_stabilized_seismograph(u, k_1, k_2):
    return stabilized_seismograph(u, k_1, k_2, noise_variance=0.1)

def general_seismograph(u, noise_variance, n_delay, theta=None, Ts=1.0/1000.0):
    S_z = S(theta=theta)
    
    if u.ndim == 1:
        number_of_inputs = 1
        length_of_input = u.shape[0]
    else:
        number_of_inputs = u.shape[0]
        length_of_input = u.shape[1]
    
    T = np.linspace(0.0, Ts*length_of_input, length_of_input)
    y = np.zeros((number_of_inputs, length_of_input))
    
    for k in range(number_of_inputs):
        if u.ndim > 1:
            u_k = u[k, :]
        else:
            u_k = u
        response = ctrl.forced_response(S_z, T, u_k, 0.0)
        y_k = response.y
        y_k = np.roll(y_k, n_delay)
        #y_k[0:n_delay] = 0.0
        
        y[k, :] = y_k + np.random.normal(0.0, noise_variance, np.size(y_k))
    return y if u.ndim > 1 else y[0, :]

# theta = [a_1, a_2, b_0, b_1, b_2]
def S(theta=None):
    if theta is None:
        omega_res = 42.0*2*np.pi
        delta = 0.521
        Ts = 1.0/1000.0

        a_1 = 2*omega_res**2 - 8/Ts**2
        a_2 = 4/Ts**2 - 4*delta*omega_res/Ts + omega_res**2

        b_0 = omega_res**2
        b_1 = 2*omega_res**2
        b_2 = omega_res**2

        normalizer = 4/Ts**2 + 4*delta*omega_res/Ts + omega_res**2

        theta = np.array([a_1, a_2, b_0, b_1, b_2])/normalizer; # a_1, a_2, b_0, b_1, b_2        

    z = ctrl.tf('z')
    S = (theta[2] + theta[3]*(1/z) + theta[4]*(1/z**2)) / (1.0 + theta[0]*(1/z) + theta[1]*(1/z**2))
    return S
