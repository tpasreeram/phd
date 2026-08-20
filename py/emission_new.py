import numpy as np
import scipy
import matplotlib.pyplot as plt
from qutip import *
import couplings as cp
import time

# ------------------------------
# Excited State
# ------------------------------

def exc(N, n_exc, vals):
    occ = [0]*N
    for i in vals:
        occ[i] += 1
    psi = enr_fock([2]*N, excitations=n_exc, state=occ)
    return psi

_operator_cache = {}
def build_ops(gmat, N, n_exc, tol=1e-4):
    Gamma, alpha = np.linalg.eigh(gmat)

    key = ("a_ops", N, n_exc)
    if key in _operator_cache:
        a_ops = _operator_cache[key]
    else:
        a_ops = enr_destroy([2]*N, excitations=n_exc)
        _operator_cache[key] = a_ops

    c_ops = []
    for nu in range(N):
        if Gamma[nu] > tol:
            O_nu = sum(alpha[i, nu] * a_ops[i] for i in range(N))
            c_ops.append(np.sqrt(2 * Gamma[nu]) * O_nu)

    return c_ops, a_ops

def get_ops(g_name, gmat, N, n_exc, observable):
    key = (g_name, observable, n_exc)
    if key in _operator_cache:
        return _operator_cache[key]

    c_ops, a_ops = build_ops(gmat, N, n_exc, 1e-10)
    
    if observable == "emission":
        e_ops = [c.dag() * c for c in c_ops]
    elif observable == "population":
        e_ops = [a.dag() * a for a in a_ops]

    _operator_cache[key] = (c_ops, e_ops)
    return c_ops, e_ops

# ------------------------------
# Simulation
# ------------------------------

def sim(N, n_exc, rho, g_name, gmat, tlist, observable="emission"):
    c_ops, e_ops = get_ops(g_name, gmat, N, n_exc, observable)
    H = 0 * c_ops[0]
    
    opts = {}#{"progress_bar":"tqdm"}

    result = mesolve(H, rho, tlist, c_ops, e_ops=e_ops, options=opts)
    return result

def extract_modes(result):
    I = np.sum(result.expect, axis=0)
    e = 1e-12
    return np.log10((I + e) / (I[0] + e))

# ------------------------------
# Plotting
# ------------------------------

def plot(result, tlist, obs):
    if obs == "emission":
        grouped = {}

        for label, value in result.items():
            r_name, g_name = [s.strip() for s in label.split("|")]
            if g_name not in grouped:
                grouped[g_name] = {}
            grouped[g_name][r_name] = value
        g_names = list(grouped.keys())
        n = len(g_names)
    
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(5*n, 4),
            sharex=True,
            sharey=True,
        )
        if n==1:
            axes = [axes]
        for ax, g_name in zip(axes, g_names):
            for r_name, result_m in grouped[g_name].items():
                y = extract_modes(result_m)
                ax.plot(tlist, y, label=r_name)
            ax.tick_params(labelleft=True, labelbottom=True)
            ax.set_title(g_name)
            ax.set_xlabel("time")
            ax.set_ylabel("normalized emission (log10)")
            ax.legend()
    elif obs == "population":
        labels = list(result.keys())
        n = len(labels)
    
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5*ncols,4*nrows),
            sharex=True,
            sharey=True,
        )
        axes = np.array(axes).flatten()
        for ax, label in zip(axes, labels):
            r = result[label]
            for i, site in enumerate(r.expect):
                ax.plot(tlist, np.log10(site), label=f"site {i}")
            # ax.tick_params(labelleft=True, labelbottom=True)
            ax.set_xlabel("time")
            ax.set_ylabel("population")
            ax.set_title(label)
            ax.legend()
        for ax in axes[n:]:
            ax.remove()
    # plt.tight_layout()
    plt.show()

def get_rho(N, n_exc):
    return {
        # "rho0": exc(N, n_exc, range(N)),
        "rho1": exc(N, n_exc, [0]),
        "rho2": exc(N, n_exc, [4]),
        # "9 & 10": exc(N, n_exc, [8, 9]),
        # "10 & 11": exc(N, n_exc, [9, 10]),
        # "6 & 14": exc(N, n_exc, [5, 13]),
    }

def get_gmat(N):
    return {
        # "N=20, 0.2/0.8":    cp.ssh(N, 0, 1, .2, .8),
        # "N=20, 0.8/0.2":    cp.ssh(N, 0, 1, .8, .2),
        "trivial":          cp.ssh(N, 0, 1, .7, .3),
        "topological":      cp.ssh(N, 0, 1, .3, .7),
    }


def main():
    N = 8
    n_exc = 1
    tlist = np.linspace(0, 5, 500)
    obs = "emission"


    rhos_config = get_rho(N, n_exc)

    gmat_config = get_gmat(N)
    
    states = ["rho1", "rho2"]
    couplings = ["trivial", "topological"]
    results = {}


    
    for g_name in couplings:
        g = gmat_config[g_name]
        for r_name in states:
            rho = rhos_config[r_name]
            label = f"{r_name} | {g_name}"
            results[label] = sim(N, n_exc, rho, g_name, g, tlist, obs)
        
        # eval_k, mat = get_mode_matrix(g_name, g, N, n_exc=2, mode_index=0)

        # plt.imshow(np.abs(mat)**2, origin="lower")
        # plt.colorbar(label="|c_ij|^2")
        # plt.xlabel("j")
        # plt.ylabel("i")
        # plt.title(f"Mode 0, eigenvalue={eval_k:.3g}")
        # plt.show()
    plot(results, tlist, obs)

if __name__ == "__main__":
    main()
