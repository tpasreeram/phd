import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from functools import lru_cache
from qutip import *
import couplings as cp

# ------------------------------
# Operators
# ------------------------------
@lru_cache(maxsize=None)
def local_ops(N):
    sp = sigmap()
    id2 = qeye(2)

    sp_list = []
    for i in range(N):
        ops = [id2] * N
        ops[i] = sp
        sp_list.append(tensor(ops))
    return sp_list

def build_ops(gmat, N, tol=1e-10):
    Gamma, alpha = np.linalg.eigh(gmat)
    ops_list = local_ops(N)
    c_ops = []
    for nu in range(N):
        if Gamma[nu] > tol:
            O_nu = sum(alpha[i, nu] * ops_list[i] for i in range(N))
            c_ops.append(np.sqrt(2 * Gamma[nu]) * O_nu)
    return c_ops

_operator_cache = {}
def get_ops(g_name, gmat, N, observable):
    key = (g_name, observable)
    if key in _operator_cache:
        return _operator_cache[key]

    c_ops = build_ops(gmat, N)

    if observable == "emission":
        e_ops = [c.dag() * c for c in c_ops]
    elif observable == "population":
        l_ops = local_ops(N)
        e_ops = [l.dag() * l for l in l_ops]
    
    _operator_cache[key] = (c_ops, e_ops)

    return c_ops, e_ops


# ------------------------------
# Excited State
# ------------------------------

def exc(N, vals):
    states = [basis(2,0) for _ in range(N)]
    # psi0 = np.zeros(N)
    for i in vals:
        states[i] = basis(2,1)
        # psi0[i] = 1
    out = tensor(states)
    
    # psi0 /= np.linalg.norm(psi0)
    # return psi0
    return out

# ------------------------------
# Simulation
# ------------------------------

def sim(N, rho, g_name, gmat, tlist, observable="emission"):
    H = 0 * tensor([qeye(2)] * N)
    opts = {}#{"progress_bar":"tqdm"}
    c_ops, e_ops = get_ops(g_name, gmat, N, observable)
    result = mesolve(H, rho, tlist, c_ops, e_ops=e_ops, options=opts)
    return result

def extract_modes(result):
    I = np.sum(result.expect, axis = 0)
    return np.log10(I / I[0])

def single_excitation(N, rho, gmat, tlist, observable="emission"):
    result = np.zeros((N, len(tlist)), dtype=complex)
    for k, t in enumerate(tlist):
        U = scipy.linalg.expm(- 1/2 * gmat * t)
        result[:, k] = U @ rho
    
    populations = np.abs(result)**2
    if observable == "emission":
        return np.sum(populations, axis=0)
    elif observable == "population":
        return populations
    return result


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
                # ax.plot(tlist, np.log10(result_m), label=r_name)
            # ax.tick_params(labelleft=True, labelbottom=True)
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
            # for i, site in enumerate(r):
                ax.plot(tlist, site, label=f"site {i}")
            # ax.tick_params(labelleft=True, labelbottom=True)
            ax.set_xlabel("time")
            ax.set_ylabel("population")
            ax.set_title(label)
            # ax.legend()
        for ax in axes[n:]:
            ax.remove()
    # plt.tight_layout()
    plt.show()

# ------------------------------
# Configs
# ------------------------------

def get_rho(N):
    return {
        "rho0": exc(N, range(N)),
        "rho1": exc(N, [0]),
        "rho2": exc(N, [4]),
        "rho4": exc(N, [0, 1]),
        "rho5": exc(N, [2, 3]),
        "rho6": 1/np.sqrt(2) * (exc(N, [0]) + exc(N, [3])),
    }

def get_gmat(N):
    return {
        # "full par":     cp.chain_par(N, 1, 0.1),
        # "full perp":    cp.chain_perp(N, 1, 0.1),
        # "nn chain":     cp.nn(N, 1, 0.1, 0, 1),
        # "nn ring":      cp.nn(N, 1, 0.1, 1, 1),
        # "nnn chain":    cp.nnn(N, 1, 0.1, 0, 1),
        # "nnn ring":     cp.nnn(N, 1, 0.1, 1, 1),
        "ssh chain0":   cp.ssh(N, 0, 1, .3, 0.7),
        "ssh chain1":   cp.ssh(N, 0, 1, .7, 0.3),
        "ssh ring":     cp.ssh(N, 1, 1, 0.5, 0.4),
    }

CONFIG=dict(
    N = 8,  # Number of atoms

    tmax = 5,
    nt = 500,   # number of timesteps

    states = ["rho1", "rho2"],
    couplings = ["ssh chain0", "ssh chain1"],
    
    observable = "emission",
    save_fig = False,
)

# ------------------------------
# Main
# ------------------------------

def main():
    N = CONFIG["N"]
    tlist = np.linspace(0, CONFIG["tmax"], CONFIG["nt"])
    obs = CONFIG["observable"]

    rhos_config = get_rho(N)
    gmat_config = get_gmat(N)

    results = {}

    start = time.time()
    for g_name in CONFIG["couplings"]:
        g = gmat_config[g_name]
        for r_name in CONFIG["states"]:
            rho = rhos_config[r_name]
            label = f"{r_name} | {g_name}"
            results[label] = sim(N, rho, g_name, g, tlist, obs)
            # results[label] = single_excitation(N, rho, g, tlist, obs)
    end = time.time()
    plot(results, tlist, obs)
if __name__ == "__main__":
    main()
