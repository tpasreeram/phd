import time

import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *

import couplings as cp
# Global Variables
N = 4

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
def sim(atom, gmat, Omega, Delta, tlist, rho, spacing):
    ops_ee = local_ops(atom.ee)
    ops_gg = local_ops(atom.gg)
    ops_ff = local_ops(atom.ff)
    ops_ef = local_ops(atom.ef)
    ops_fe = local_ops(atom.fe)

    c_ops = build_ops(gmat, atom.ge)

    # e_ops = [sum(local_ops(atom.gg))]
    # e_ops = [ops_gg[0], ops_gg[1], ops_gg[2], ops_ee[0], ops_ee[1], ops_ee[2],
    #          ops_ff[0], ops_ff[1], ops_ff[2]]
    e_ops = [c.dag() * c for c in c_ops]
    # e_ops = [sum(local_ops(atom.gg)), sum(local_ops(atom.ee)), sum(local_ops(atom.ff))]

    H0 = sum(Delta * ops_ee[i] for i in range(N))
    H = [H0]

    H.append([-Omega/2 * ops_ef[0], lambda t, **kwargs: np.exp(-1j * kwargs["phi"][4]
                                                               * t)])
    H.append([-Omega/2 * ops_fe[0], lambda t, **kwargs: np.exp(1j * t * kwargs["phi"][4])])

    H.append([-Omega/2 * ops_ef[0], lambda t, **kwargs: np.exp(-1j *t* kwargs["phi"][0])])
    H.append([-Omega/2 * ops_fe[0], lambda t, **kwargs: np.exp(1j *t* kwargs["phi"][0])])

    H.append([-Omega/2 * ops_ef[1], lambda t, **kwargs: np.exp(-1j *t* kwargs["phi"][0])])
    H.append([-Omega/2 * ops_fe[1], lambda t, **kwargs: np.exp(1j *t* kwargs["phi"][0])])

# fix omega/delta, vary omega and delta in lambda.py to keep same ratio, make sure
# deltas correct, large delta    

    H.append([-Omega/2 * ops_ef[1], lambda t, **kwargs: np.exp(-1j *t* kwargs["phi"][1])])
    H.append([-Omega/2 * ops_fe[1], lambda t, **kwargs: np.exp(1j *t* kwargs["phi"][1])])

    H.append([-Omega/2 * ops_ef[2], lambda t, **kwargs: np.exp(-1j *t* kwargs["phi"][1])])
    H.append([-Omega/2 * ops_fe[2], lambda t, **kwargs: np.exp(1j *t* kwargs["phi"][1])])

    H.append([-Omega/2 * ops_ef[2], lambda t, **kwargs: np.exp(-1j *t* kwargs["phi"][2])])
    H.append([-Omega/2 * ops_fe[2], lambda t, **kwargs: np.exp(1j *t* kwargs["phi"][2])])

    H.append([-Omega/2 * ops_ef[3], lambda t, **kwargs: np.exp(-1j *t* kwargs["phi"][2])])
    H.append([-Omega/2 * ops_fe[3], lambda t, **kwargs: np.exp(1j *t* kwargs["phi"][2])])

    H.append([-Omega/2 * ops_ef[3], lambda t, **kwargs: np.exp(-1j *t* kwargs["phi"][3])])
    H.append([-Omega/2 * ops_fe[3], lambda t, **kwargs: np.exp(1j *t* kwargs["phi"][3])])

    # H.append([-Omega/2 * ops_ef[4], lambda t, **kwargs: np.exp(-1j * kwargs["phi"][3])])
    # H.append([-Omega/2 * ops_fe[4], lambda t, **kwargs: np.exp(1j * kwargs["phi"][3])])
    #
    # H.append([-Omega/2 * ops_ef[4], lambda t, **kwargs: np.exp(-1j * kwargs["phi"][4])])
    # H.append([-Omega/2 * ops_fe[4], lambda t, **kwargs: np.exp(1j * kwargs["phi"][4])])
    #
    # H.append([-Omega/2 * ops_ef[5], lambda t, **kwargs: np.exp(-1j * kwargs["phi"][4])])
    # H.append([-Omega/2 * ops_fe[5], lambda t, **kwargs: np.exp(1j * kwargs["phi"][4])])
    #
    # H.append([-Omega/2 * ops_ef[5], lambda t, **kwargs: np.exp(-1j * kwargs["phi"][6])])
    # H.append([-Omega/2 * ops_fe[5], lambda t, **kwargs: np.exp(1j * kwargs["phi"][6])])

    phi = np.arange(N + 1) * spacing

    args = { "phi" : [p + Delta for p in phi] }
    hamilt = QobjEvo(H, args)
    
    opts_me = {"progress_bar":"tqdm", "matrix_form":1, "nsteps":1e6}
    me = mesolve(hamilt, rho, tlist, c_ops, e_ops=e_ops, options=opts_me)
    return me

    # opts_mc = {"progress_bar":"tqdm", "nsteps":1e8, "map":"loky", "num_cpus":12}
    # mc = mcsolve(H, rho, tlist, c_ops, e_ops=e_ops, ntraj=100, args=args, options=opts_mc)
    # return mc

def sim2(atom, gmat, Omega, Delta, tlist, rho):
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

    # opts_me = {"progress_bar":"tqdm", "matrix_form":1, "nsteps":1e6}
    opts_mc = {"progress_bar":"tqdm", "nsteps":1e8, "map":"loky", "num_cpus":12}
    
    mc = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=1000, options=opts_mc)
    # me = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options=opts_me)

    return mc
# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    atom = LambdaSystem()
    g1 = np.ones((N, N), dtype=complex)
    g2 = cp.ssh(N, 0, 1, .3, .7)
    
    Delta = 100
    Omega = 1
    OD2 = (Omega / Delta) ** 2
        
    # spacing = [0, 0.1, 0.5, 1, 2, 5]
    spacing = [0]
    tmax = 1000
    tlist = np.linspace(0, tmax, tmax)

    tmp = [atom.g] * N
    tmp[2] = atom.f
    rho = tensor(tmp)

    OD2 = (Omega / Delta) ** 2
    for s in spacing:
        mc = sim(atom, g1, Omega, Delta, tlist, rho, s)
        plt.plot(tlist * OD2, mc.expect[0], label=f'{s} g')
        # plt.plot(tlist * OD2, mc.expect[1] / OD2, label=f'{s} e')
        # plt.plot(tlist * OD2, mc.expect[2] / OD2, label=f'{s} f')

    # tmp = [atom.f] * N
    # tmp[0] = atom.g
    # rho = tensor(tmp)
    #
    # mc = sim(atom, g2, Omega, Delta, tlist, rho)
    # plt.plot(tlist * OD2, mc.expect[0] / OD2, label=f'stuff')
    # mc2 = sim2(atom, g1, Omega, Delta, tlist, rho)
    # plt.plot(tlist * OD2, mc2.expect[0] / OD2, label=f'stuff2')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
