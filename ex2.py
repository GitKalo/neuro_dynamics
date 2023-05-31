##############
# Numerical integration of the Hidgkin-Huxley neuron model using Heun's method.
# 
# Written by Kaloyan Danovski (May 30th, 2023)
##############

import math
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(style='whitegrid')

# Define constants
c = 9 * math.pi
c_inv = 1 / c
e_na = 115
e_k = -12
v_rest = 10.6
g_na = 1080 * math.pi
g_k = 324 * math.pi
g_m = 2.7 * math.pi

I = 280

g_a = g_g = 10
e_a = 60
e_g = -20
a_a = 1.1
a_g = 5
b_a = 0.19
b_g = 0.3

# Define evolution functions
a_n = lambda v : (10-v) / (100 * (math.e**((10-v)/10) - 1))
b_n = lambda v : 0.125 * math.e**(-v/80)
a_m = lambda v : (25-v) / (10 * (math.e**((25-v)/10) - 1))
b_m = lambda v : 4 * math.e**(-v/18)
a_h = lambda v : 0.07 * math.e**(-v/20)
b_h = lambda v : 1 / (math.e**((30-v)/10) + 1)

dV = lambda v, n, m, h, I_syn : c_inv * (g_na * m**3 * h * (e_na-v) + g_k * n**4 * (e_k-v) + g_m * (v_rest-v) + I + I_syn)
dn = lambda v, n : a_n(v) * (1-n) - b_n(v) * n
dm = lambda v, m : a_m(v) * (1-m) - b_m(v) * m
dh = lambda v, h : a_h(v) * (1-h) - b_h(v) * h

# Synaptic currents
I_a = lambda v, r : g_a * r * (e_a - v)
I_g = lambda v, r : g_g * r * (e_g - v)
dr_a = lambda r, v_pre : a_a * f(v_pre) * (1-r) - b_a * r
dr_g = lambda r, v_pre : a_g * f(v_pre) * (1-r) - b_g * r
T_max = 1; k_p = 5; v_p = 62
f = lambda v_pre : T_max / (1 + math.e**(-(v_pre - v_p)/k_p))

# Dictionary with functions for calculating synaptic current based on type of coupling
I_syn = {
    None: lambda *_ : 0,
    'exc': I_a,
    'inh': I_g
}

r = {
    None: lambda *_ : 0,
    'exc': dr_a,
    'inh': dr_g
}

def run(neur1=None, neur2=None, 
        n_wrt=1000, n_step=10, ht=0.01, 
        v1=0, n1=0.4, m1=0.1, h1=0.6,
        v2=0, n2=0.4, m2=0.1, h2=0.6
    ) :
    '''
    Integrate the evolution equations for the coupled neurons.

    The neur1 and neur2 parameters specify the type of coupling between neurons. A value of None indicates
    no coupling, while values of 'exc' and 'inh' signfiy that the neuron is under the respective type of
    influence from the other neuron.
    '''
    ### Heun's method
    t = 0

    # Start with no open channels
    r1 = r2 = 0

    ts = np.empty(n_wrt)
    vs = np.empty((2, n_wrt))

    for i_wrt in range(n_wrt) :        # outer loop — record data
        for i_step in range(n_step) :   # inner loop — step updates
            # Integrate neuron 1
            k_r1 = ht * r[neur1](r1, v2)
            r1_new = r1 + 0.5 * (k_r1 + ht * r[neur1](r1 + k_r1, v2))

            k_v1 = ht * dV(v1, n1, m1, h1, I_syn[neur1](v1, r1))
            v1_new = v1 + 0.5 * (k_v1 + ht * dV(v1 + k_v1, n1, m1, h1, I_syn[neur1](v1, r1)))

            k_n1 = ht * dn(v1, n1)
            n1 = n1 + 0.5 * (k_n1 + ht * dn(v1, n1))
            k_m1 = ht * dm(v1, m1)
            m1 = m1 + 0.5 * (k_m1 + ht * dm(v1, m1))
            k_h1 = ht * dh(v1, h1)
            h1 = h1 + 0.5 * (k_h1 + ht * dh(v1, h1))

            r1 = r1_new
            v1 = v1_new

            # Integrate neuron 2
            k_r2 = ht * r[neur2](r2, v1)
            r2_new = r2 + 0.5 * (k_r2 + ht * r[neur2](r2 + k_r2, v1))

            k_v2 = ht * dV(v2, n2, m2, h2, I_syn[neur2](v2, r2))
            v2_new = v2 + 0.5 * (k_v2 + ht * dV(v2 + k_v2, n2, m2, h2, I_syn[neur2](v2, r2)))

            k_n2 = ht * dn(v2, n2)
            n2 = n2 + 0.5 * (k_n2 + ht * dn(v2, n2))
            k_m2 = ht * dm(v2, m2)
            m2 = m2 + 0.5 * (k_m2 + ht * dm(v2, m2))
            k_h2 = ht * dh(v2, h2)
            h2 = h2 + 0.5 * (k_h2 + ht * dh(v2, h2))

            r2 = r2_new
            v2 = v2_new

            # Increment time
            t += ht

        ts[i_wrt] = t
        
        # Record V
        vs[0,i_wrt] = v1
        vs[1,i_wrt] = v2
    
    return ts, vs

def run_random_init(neur1=None, neur2=None, n_wrt=1000) :
    return run(
        neur1, neur2,
        n_wrt=n_wrt, 
        v1=np.random.random_sample() * 50,
        v2=np.random.random_sample() * 50,
        n1=np.random.random_sample(),
        m1=np.random.random_sample(),
        h1=np.random.random_sample(),
        n2=np.random.random_sample(),
        m2=np.random.random_sample(),
        h2=np.random.random_sample()
    )

n_wrt = 2000

# Unidirectional coupling
ts, vs1 = run_random_init('exc', None, n_wrt)
ts, vs2 = run_random_init('inh', None, n_wrt)

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8,6))
axs[0].plot(ts, vs1[0], label=f'exc')
axs[0].plot(ts, vs1[1], label=f'None')
axs[0].set_ylabel('V')
axs[0].legend()
axs[1].plot(ts, vs2[0], label=f'inh')
axs[1].plot(ts, vs2[1], label=f'None')
axs[1].set_ylabel('V')
axs[1].legend()
# plt.axhline(v_rest, ls='--', color='gray', label='V_rest')
plt.xlabel('t')
# fig.suptitle("Unidirectional coupling")
plt.savefig('report_couple_uni.pdf')
plt.show()

# Bidirectional coupling
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16,9))
for i, (neur1, neur2) in enumerate([('exc', 'exc'), ('inh', 'inh'), ('exc', 'inh')]) :
    ts, vs = run_random_init(neur1, neur2, 2*n_wrt)
    axs[i].plot(ts, vs[0], label=f'{neur1}', lw=1)
    axs[i].plot(ts, vs[1], label=f'{neur2}', lw=1)
    axs[i].set_ylabel('V')
    axs[i].legend()
plt.xlabel('t')
# fig.suptitle("Bidirectional coupling")
plt.savefig('report_couple_bi.pdf')
plt.show()