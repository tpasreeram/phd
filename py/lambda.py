import time

import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *

import couplings as cp

# Global Variables
N = 6

# ----------------------------------------
# Lambda System
# ----------------------------------------
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
    id3 = qeye(3)

    ops_list = []
    for i in range(N):
        ops = [id3] * N
        ops[i] = op
        ops_list.append(tensor(ops))
    return ops_list

# Builds collective dissipative jump operators
def build_ops(gmat, op, tol=1e-8):
    evals, evecs = np.linalg.eigh(gmat)
    ops_list = local_ops(op)

    c_ops = []

    for i, val in enumerate(evals):
        if val > tol:
            L_i = sum(evecs[j, i] * ops_list[j] for j in range(N))
            c_ops.append(np.sqrt(2 * val) * L_i)
    return c_ops

# Simulation
def sim(atom, gmat, Omega, Delta, tlist):
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
    H = sum((Delta * ops_ee[i] - (Omega/2) * (ops_ef + ops_fe)[i]) for 
            i in range(N))
    opts = {"progress_bar":"tqdm", "matrix_form":1, "nsteps":1e6}
    # opts = {"progress_bar":"tqdm", "map":"loky", "num_cpus":12}
    # result = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=100, options=opts)
    result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options=opts)
    return result

# ---------------------------------------------------------------------------
# 3 level -> effective 2 level
# ---------------------------------------------------------------------------
class TwoLevelSystem:
    g = basis(2, 0) # |g>
    f = basis(2, 1) # |f>

    gf = g * f.dag() # |g><f|
    fg = f * g.dag() # |f><g|

    gg = g * g.dag() # |g><g|
    ff = f * f.dag() # |f><f|

def local_ops_elim(op):
    id2 = qeye(2)

    ops_list = []
    for i in range(N):
        ops = [id2] * N
        ops[i] = op
        ops_list.append(tensor(ops))
    return ops_list

def build_ops_elim(atom, gmat, Omega, Delta, tol=1e-8):
    evals, evecs = np.linalg.eigh(gmat)
    ops_list = local_ops_elim(atom.gf)

    c_ops = []

    for i, val in enumerate(evals):
        if val > tol:
            L_i = sum(evecs[j, i] * ops_list[j] for j in range(N))
            c_ops.append(np.sqrt(val / 2) * (Omega / Delta) * L_i)
    return c_ops

def get_ops_elim(atom, gmat, Omega, Delta, observable):
    c_ops = build_ops_elim(atom, gmat, Omega, Delta)

    if observable == "emission":
        e_ops = [c.dag() * c for c in c_ops]
    elif observable == "population":
        l_ops = local_ops_elim(atom.gf)
        e_ops = [l.dag() * l for l in l_ops]
    return c_ops, e_ops

def sim_elim(atom, gmat, Omega, Delta, tlist, observable):
    ops_gg = local_ops_elim(atom.gg)
    H = sum((Omega ** 2 / (4 * Delta)) * ops_gg[i] for i in range(N))

    tmp = [atom.f] * N
    rho = tensor(tmp)

    opts = {"progress_bar":"tqdm", "nsteps":1e6}
    c_ops, e_ops = get_ops_elim(atom, gmat, Omega, Delta, observable)
    result = mesolve(H, rho, tlist, c_ops, e_ops=e_ops, options=opts)
    return result

# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    atom = LambdaSystem()
    atom2 = TwoLevelSystem()

    g1 = np.ones((N, N), dtype=complex)

    Delta = [5]
    Omega = 5
    
    results = {}
    tmax = 1000
    tlist = np.linspace(0, tmax, 500)

    for d in Delta:
        # tmax = 50 / ((Omega / d) ** 2)
        r = sim(atom, g1, Omega, d, tlist)
        results[d] = (tlist, r)

    for d, (tlist, r) in results.items():
        OD2 = (Omega/d)**2
        r.plot_expect()

        # for i, expect_vals in enumerate(r.expect):
        #
        #     # cutoff = 1 / d
            #
            # t_shifted = (tlist - cutoff) * OD2

            # something = sim_elim(atom2, g1, Omega, Delta[0], tlist, "emission")
            # plt.plot(t_shifted, abs(expect_vals - something.expect[0]) / OD2, label=f'delta={d}')
            # plt.plot(tlist * OD2, expect_vals, label=f'delta={d}, c[{i}]')
            # plt.plot(tlist * OD2, something.expect[0] / OD2, label=f'delta={d}, c[{i}]')
    
    # something = sim_elim(atom2, g1, Omega, Delta[0], tlist, "emission")
    # plt.plot(tlist * (Omega / Delta[0])**2, something.expect[0])


    plt.xlabel('Time')
    plt.ylabel('Expectation value')
    plt.legend()
    plt.show()   
    
    # fig, ax = r.plot_expect()
if __name__ == "__main__":
    main()
