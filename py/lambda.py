import time

import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *

import couplings as cp
# Global Variables
N = 3

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
def sim(atom, gmat, Omega, Delta, tlist, rho):
    ops_ee = local_ops(atom.ee)
    ops_gg = local_ops(atom.gg)
    ops_ff = local_ops(atom.ff)
    ops_ef = local_ops(atom.ef)
    ops_fe = local_ops(atom.fe)


    psi0 = rho

    c_ops = build_ops(gmat, atom.ge)

    # e_ops = [sum(local_ops(atom.gg))]
    # e_ops = [ops_ee[1], ops_gg[1], ops_ff[1]]
    e_ops = [c.dag() * c for c in c_ops]

    H = sum((Delta * ops_ee[i] - (Omega/2) * (ops_ef[i] + ops_fe[i])) for 
            i in range(N))

    opts_me = {"progress_bar":"tqdm", "matrix_form":1, "nsteps":1e6}
    # opts_mc = {"progress_bar":"tqdm", "nsteps":1e8, "map":"loky", "num_cpus":12}
    #
    # mc = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=500, options=opts_mc)
    me = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options=opts_me)

    return me

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
            c_ops.append(np.sqrt(2 * val) * (Omega / (2 * Delta)) * L_i)
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

    opts = {"progress_bar":"tqdm", "nsteps":1e8}
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

    Delta = [1, 5, 10, 50, 100, 500]
    od = .05
    # Delta = [10]
    # Omega = 1

    
    results = {}
    # tmax = 10000
    # tlist = np.linspace(0, tmax, tmax)

    tmp = [atom.f] * N
    rho = tensor(tmp)
    for d in Delta:
        Omega = d * od
        tmax = max(int(10 * np.floor((d / Omega) ** 2)), 10)
        tlist = np.linspace(0, tmax, tmax)
        me = sim(atom, g1, Omega, d, tlist, rho)
        results[d] = (tlist, me)
    
    # n = len(Delta)
    # fig, axes = plt.subplots(n, 1, figsize=(8, 4*n), squeeze=False)
    # axes = axes.ravel()

    errs = {}
    # for ax, (d, (tlist, r)) in zip(axes, results.items()):
    for d, (tlist, r) in results.items():
        Omega = d * od
        OD2 = (Omega/d)**2
        something = sim_elim(atom2, g1, Omega, d, tlist, "emission")

        for i, expect_vals in enumerate(r.expect):
            # abs_err = np.maximum(np.abs(expect_vals - something.expect[0]), 1e-6)
            # plt.plot(tlist * OD2, abs_err, label=f'delta={d}')
            
            # ref = something.expect[0]
            # rel_err = np.abs(expect_vals - ref) / np.maximum(np.abs(ref), 1e-12)
            # rel_err = np.maximum(rel_err, 1e-6)
            #
            # plt.plot(tlist * OD2, rel_err, label=f'delta={d}')

            error = np.sqrt(np.mean(np.abs(something.expect[0] - expect_vals)**2))
            errs[d] = error

            # ax.plot(tlist*OD2, expect_vals / OD2, label=f"{Omega}/{d}")
            # ax.plot(tlist*OD2, something.expect[0] / OD2, label = "adiabatic elimination")
            # ax.set_title(f'delta={d}')
            # ax.legend()
   

    x = np.log10([a * od for a in Delta])
    y = np.log10(list(errs.values()))
    # x = Delta
    # y = list(errs.values())

    plt.plot(x, y, 'o')
    # m, b = np.polyfit(x, y, 1)
    #
    # y_fit = m * x + b
    # plt.plot(x, y_fit, '-')
    #
    # equation = f"y = {m:.3f}x + {b:.3f}"
    # plt.title(f'{equation}: o/d={od}')
    # plt.xlabel('Time')
    # plt.ylabel('Expectation value')

    # mc = sim(atom, g1, Omega, Delta, tlist, rho)
    #
    # s = sim_elim(atom2, g1, Omega, Delta, tlist, "emission")
    # plt.plot(tlist, s.expect[0], label=f'elim')
    # plt.plot(tlist, mc.expect[0], label=f'mesolve')
    # OD2 = (Omega / Delta) ** 2
    # plt.plot(tlist * OD2, (np.sum(mc.expect, axis=0)), label=f'mcsolve')
    #
    # tmp = [atom.g] * N
    # tmp[3] = atom.f
    # rho = tensor(tmp)
    #
    # mc = sim(atom, g2, Omega, Delta, tlist, rho)
    # plt.plot(tlist * OD2, (np.sum(mc.average_expect, axis=0)), label=f'mcsolve')

    # for i, traj in enumerate(mc.runs_expect[0]):
    #     plt.plot(tlist, traj, alpha=0.3, label=f'{i}')
    # plt.title(f'Omega/Delta = {od}')
    plt.tight_layout()
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
