import numpy as np
import control as ctrl

from seismograph import general_seismograph

def earthquake():
    N = 3500
    N_begin = 190
    N_end = 340
    
    L = 5
    L_factor = 4
    
    y = np.zeros(N)
    for k in range(0, N_end-N_begin):
        y_k = 0.0
        for l in range(0, L):
            y_k += np.imag(np.exp(2*L_factor*1j*np.pi*l*k/(N_end-N_begin)))
       
        gain = (-np.cos(2*np.pi*k/(N_end - N_begin)) + 1.0)/2.0
        y[k+N_begin] = gain * y_k
        
    return y

def measurement_1():
    n_delay = 983
    k = 0.7
    eq = earthquake()
    return k * general_seismograph(eq, 0.2, n_delay, Ts=1.0/100.0)

def measurement_2():
    n_delay = 3158
    k = 0.1
    eq = earthquake()
    return k * general_seismograph(eq, 0.2, n_delay, Ts=1.0/100.0)

def measurement_3():
    n_delay = 633
    k = 0.9
    eq = earthquake()
    return k * general_seismograph(eq, 0.2, n_delay, Ts=1.0/100.0)

