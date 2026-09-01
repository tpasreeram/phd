import time

import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *

import couplings as cp

# Global Variables
N = 3

class LambdaSystem:
    g = basis(3, 0) # |g>
    f = basis(3, 1) # |f>
    e = basis(3, 2) # |e>

    ge = g * e.dag() # |g><e|
    eg = e * g.dag() # |e><g|

    ef = e * f.dag() # |e><f|
    fe = f * e.dag() # |f><e|

    ee = e * e.dag() # |e><e|
    gg = g * g.dag() # |g><g|
    ff = f * f.dag() # |f><f|

# Creates list of local operators in the full Hilbert space
def local_ops(op):
    s_list = []
    for i in range(N):
        ops = [qeye(3)] * N
        ops[i] = op
        s_list.append(tensor(ops))
    return s_list

# Builds collective dissipative jump operators
def build_ops(gmat, op, tol=1e-8):
    N = len(gmat)

    evals, evecs = np.linalg.eigh(gmat)
    ops_list = local_ops(op)

    c_ops = []

    for i, val in enumerate(evals):
        if val > tol:
            L_i = sum(evecs[j, i] * ops_list[j] for j in range(N))
            c_ops.append(np.sqrt(2 * val) * L_i)
    return c_ops

# Simulation
def sim(atom, gmat, Omega, Delta, tlist, o):
    ops_ee = local_ops(atom.ee)
    ops_gg = local_ops(atom.gg)
    ops_ff = local_ops(atom.ff)
    ops_ef = local_ops(atom.ef)
    ops_fe = local_ops(atom.fe)

    # e_ops = [ops_ee[1], ops_gg[1], ops_ff[1]]

    tmp = [atom.f] * N
    tmp[0] = atom.f
    psi0 = tensor(tmp)

    c_ops = build_ops(gmat, atom.ge)

    e_ops = [c.dag() * c for c in c_ops]
    
    H = []
    H.append(Delta * (ops_ee[0] + ops_ee[1] + ops_ee[2]))

    # H.append([-(Omega / 2) * ops_ef[0], lambda t, args: np.exp(1j * args["omega1"] * t)])
    # H.append([-(Omega / 2) * ops_fe[0], lambda t, args: np.exp(-1j * args["omega1"] * t)]))
    #
    # H.append([-(Omega / 2) * ops_ef[1], lambda t, args: np.exp(1j * args["omega1"] * t)])
    # H.append([-(Omega / 2) * ops_fe[1], lambda t, args: np.exp(-1j * args["omega1"] * t)]))

    H.append([-(Omega / 2) * ops_ef[1], lambda t, args: np.exp(1j * args["omega"] * t)])
    H.append([-(Omega / 2) * ops_fe[1], lambda t, args: np.exp(-1j * args["omega"] * t)])

    H.append([-(Omega / 2) * ops_ef[2], lambda t, args: np.exp(1j * args["omega"] * t)])
    H.append([-(Omega / 2) * ops_fe[2], lambda t, args: np.exp(-1j * args["omega"] * t)])

    # H = sum((Delta * ops_ee[i] - (Omega/2) * (ops_ef + ops_fe)[i]) for i in range(N))
    

    args = {"omega": o * 2 * np.pi}
    opts = {"progress_bar":"tqdm", "matrix_form":1, "nsteps":1e6}

    result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, args=args, options=opts)
    return result

def main():
    atom = LambdaSystem()
    g1 = np.ones((N, N), dtype=complex) * 1

    Delta = 50
    Omega = 5
    
    results = {}
    tmax = 1e4
    tlist = np.linspace(0, tmax, 500)
    
    omegas = [1, 10, 50]
    for o in omegas:
        r = sim(atom, g1, Omega, Delta, tlist, o)
        results[o] = r
    
    for o, r in results.items():
        OD2 = (Omega / Delta)**2
        for i, val in enumerate(r.expect):
            plt.plot(tlist * OD2, val / OD2, label=f'o={o}')

    plt.xlabel('Time')
    plt.ylabel('Expectation value')
    plt.legend()
    plt.show()   
    
    # fig, ax = r.plot_expect()
if __name__ == "__main__":
    main()
