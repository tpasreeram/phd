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
Gamma = 1

tlist = np.linspace(0, 10000, 500)

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

atom = LambdaSystem()
H = (Delta * atom.ee - (Omega / 2) * (atom.ef + atom.fe + atom.ge + atom.eg))

psi0 = atom.g

c_ops = [] # [np.sqrt(Gamma) * atom.ge]

e_ops = [atom.gg, atom.ff, atom.ee]
opts = {"progress_bar":"tqdm", "matrix_form":1, "nsteps":1e6}
result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options=opts)

plt.plot(tlist, result.expect[0], label=r"$P_g$")
plt.plot(tlist, result.expect[1], label=r"$P_f$")
plt.plot(tlist, result.expect[2], label=r"$P_e$")

plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.show()
