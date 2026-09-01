import time

import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *

import couplings as cp

# Global Variables
N = 1
Delta = 100
Omega = 1
tlist = np.linspace(0, 10, 500)

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
def sim(atom, gmat):
    observable = "emission"

    ops_ee = local_ops(atom.ee)
    ops_ef = local_ops(atom.ef)
    ops_fe = local_ops(atom.fe)
    ops_eg = local_ops(atom.eg)
    ops_ge = local_ops(atom.ge)

    c_ops = build_ops(gmat, atom.ge)
    c_ops = []

    fops = local_ops(atom.ee)
    if observable == "emissio":
        # e_ops = [c.dag() * c for c in c_ops]
        e_ops = [atom.gg, atom.ff]

    
    tmp = [atom.g] * N
    tmp[0] = atom.f
    psi0 = tensor(tmp)

    H = sum((Delta * ops_ee[i] - (Omega/2) * (ops_ef + ops_fe + ops_ge + ops_eg)[i]) for i 
            in range(N))
    opts = {"progress_bar":"enhanced", "matrix_form":1}
    result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options=opts)
    return result

def main():
    atom = LambdaSystem()
    g1 = np.ones((N, N))

    results = {
        "g1": sim(atom, g1),    
        # "g2": sim(atom, g2),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for g_name, r in results.items():
        I = np.sum(r.expect, axis=0)
        y = I
        # dydt = np.gradient(y, tlist)

        ax.plot(
            tlist,
            y,
            linewidth=4,
            label=g_name
        )
    
        ax.set_title("Emission", fontsize=16)
        ax.set_ylabel("Emission (log10)", fontsize=16)
        ax.set_xlabel("time", fontsize=16)
        ax.legend(fontsize=12)
    plt.show()
if __name__ == "__main__":
    main()
