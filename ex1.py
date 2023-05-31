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

# Define evolution functions
a_n = lambda v : (10-v) / (100 * (math.e**((10-v)/10) - 1))
b_n = lambda v : 0.125 * math.e**(-v/80)
a_m = lambda v : (25-v) / (10 * (math.e**((25-v)/10) - 1))
b_m = lambda v : 4 * math.e**(-v/18)
a_h = lambda v : 0.07 * math.e**(-v/20)
b_h = lambda v : 1 / (math.e**((30-v)/10) + 1)

dV = lambda v, n, m, h, I : c_inv * (g_na * m**3 * h * (e_na-v) + g_k * n**4 * (e_k-v) + g_m * (v_rest-v) + I)
dn = lambda v, n : a_n(v) * (1-n) - b_n(v) * n
dm = lambda v, m : a_m(v) * (1-m) - b_m(v) * m
dh = lambda v, h : a_h(v) * (1-h) - b_h(v) * h

def run(I, n_wrt=200, n_step=10, ht=0.01, v=0, n=0.4, m=0.1, h=0.6) :
    ### Heun's method
    t = 0

    ts = np.empty(n_wrt)
    vs = np.empty(n_wrt)
    ns = np.empty(n_wrt)
    ms = np.empty(n_wrt)
    hs = np.empty(n_wrt)
    i_na = np.empty(n_wrt)
    i_k = np.empty(n_wrt)
    i_l = np.empty(n_wrt)

    for i_wrt in range(n_wrt) :        # outer loop — record data
        for i_step in range(n_step) :   # inner loop — step updates
            # Calculate new value of V
            k_v = ht * dV(v, n, m, h, I)
            v_new = v + 0.5 * (k_v + ht * dV(v + k_v, n, m, h, I))

            # Calculate new values of n, m, h (with old v)
            k_n = ht * dn(v, n)
            n = n + 0.5 * (k_n + ht * dn(v, n))

            k_m = ht * dm(v, m)
            m = m + 0.5 * (k_m + ht * dm(v, m))

            k_h = ht * dh(v, h)
            h = h + 0.5 * (k_h + ht * dh(v, h))

            v = v_new

            # Increment time
            t += ht

        ts[i_wrt] = t
        
        # Record V
        vs[i_wrt] = v

        ns[i_wrt] = n
        ms[i_wrt] = m
        hs[i_wrt] = h

        # Record currents
        i_na[i_wrt] = g_na * (v - e_na)
        i_k[i_wrt] = g_k * (v - e_k)
        i_l[i_wrt] = g_m * (v_rest - v)
    
    return vs, ns, ms, hs, i_na, i_k, i_l, ts

n_wrt = 1000
# I = 280
vs_280, ns_280, ms_280, hs_280, i_na_280, i_k_280, i_l_280, ts_280 = run(280, n_wrt=n_wrt)
vs_350, ns_350, ms_350, hs_350, i_na_350, i_k_350, i_l_350, ts_350 = run(350, n_wrt=n_wrt)

# Plot V
plt.plot(ts_280, vs_280, label='I=280')
plt.plot(ts_350, vs_350, label='I=350')
# plt.axhline(v_rest, ls='--', color='gray', label='V_rest')
plt.xlabel('t')
plt.ylabel('V')
# plt.title("Potential")
plt.legend()
plt.savefig('report_v.pdf')
plt.show()

# Plot activation variables
lim = 150
plt.plot(ts_280[:lim], ns_280[:lim], label='n')
plt.plot(ts_280[:lim], ms_280[:lim], label='m')
plt.plot(ts_280[:lim], hs_280[:lim], label='h')
plt.xlabel('t')
# plt.title(f"Activation variables, up to t={ts_280[lim]:.1f}")
plt.legend()
plt.savefig('report_vars.pdf')
plt.show()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8,6))
plt.plot(ts_280, i_na_280, color=colors[0], label='I_Na')
plt.plot(ts_280, i_k_280, color=colors[1], label='I_K')
plt.plot(ts_280, i_l_280, color=colors[2], label='I_L')
plt.plot(ts_350, i_na_350, color=colors[0], ls='-.', label='I_Na')
plt.plot(ts_350, i_k_350, color=colors[1], ls='-.', label='I_K')
plt.plot(ts_350, i_l_350, color=colors[2], ls='-.', label='I_L')
plt.xlabel('t')
plt.ylabel('I (pA)')
# plt.title("Currents at I=280 (solid) and I=350 (dashed)")
# fig.tight_layout()
plt.legend()
plt.savefig('report_curr.pdf')
plt.show()

### Frequency - Current curve
def run_freq_curr(n_wrt=2000) :
    ### Forward pass
    i_start, i_end = 150, 350; Is = range(i_start, i_end)
    freq = np.zeros(abs(i_end - i_start))
    vs, ns, ms, hs, *_ = run(
        Is[0], 
        n_wrt=n_wrt, 
        v=np.random.random_sample() * 50,
        n=np.random.random_sample(),
        m=np.random.random_sample(),
        h=np.random.random_sample()
    )

    for i, I in enumerate(Is) :
        vs, ns, ms, hs, i_na, i_k, i_l, ts = run(I, n_wrt=n_wrt, v=vs[-1], n=ns[-1], m=ms[-1], h=hs[-1])
        
        # Get spike peak indeces
        spikes_idx = np.argwhere((vs[1:-1] > v_rest) & (vs[:-2] < vs[1:-1]) & (vs[1:-1] > vs[2:])).reshape(-1,)
        
        # Calculate frequency from last 2 spike interval
        if spikes_idx.size >= 2 :
            freq[i] = 1 / (spikes_idx[-1] - spikes_idx[-2])

    np.savetxt(f'freq_{i_start}_{i_end}_w{n_wrt:.0f}.txt', np.array((Is, freq)))

    ### Backward pass
    i_start, i_end = 350, 150; Is = list(range(i_end, i_start))[::-1]
    freq = np.zeros(abs(i_end - i_start))

    vs, ns, ms, hs, *_ = run(
        Is[0], 
        n_wrt=n_wrt, 
        v=np.random.random_sample() * 50,
        n=np.random.random_sample(),
        m=np.random.random_sample(),
        h=np.random.random_sample()
    )
    for i, I in enumerate(Is) :
        vs, ns, ms, hs, i_na, i_k, i_l, ts = run(I, n_wrt=n_wrt, v=vs[-1], n=ns[-1], m=ms[-1], h=hs[-1])
        
        # Get spike peak indeces
        spikes_idx = np.argwhere((vs[1:-1] > v_rest) & (vs[:-2] < vs[1:-1]) & (vs[1:-1] > vs[2:])).reshape(-1,)
        
        # Calculate frequency from last 2 spike interval
        if spikes_idx.size >= 2 :
            freq[i] = 1 / (spikes_idx[-1] - spikes_idx[-2])

    np.savetxt(f'freq_{i_start}_{i_end}_w{n_wrt:.0f}.txt', np.array((Is, freq)))

n_wrt = 2000
# run_freq_curr(n_wrt)

# Plot forward and backward freq - current curves
Is, freq_fwd = np.loadtxt('freq_150_350_w2000.txt')
_, freq_bwd = np.loadtxt('freq_350_150_w2000.txt')

plt.plot(Is, freq_fwd, label="150 -> 350")
plt.plot(Is[::-1], freq_bwd, label="350 -> 150")
plt.xlabel("I")
plt.ylabel("f")
# plt.title("Frequency - current curve")
plt.legend()
plt.savefig('report_freq_curr.pdf')
plt.show()