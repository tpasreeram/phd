import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time
import couplings as cp

# ------------------------------
# Operators
# ------------------------------

def build_jump_vecs(gmat, tol=1e-10):
    Gamma, alpha = np.linalg.eigh(gmat)
    return [
        np.sqrt(2 * Gamma[nu]) * alpha[:, nu]
        for nu in range(len(Gamma))
        if Gamma[nu] > tol
    ]

def build_Heff(jump_vecs, N):
    decay = sum(np.outer(v, v.conj()) for v in jump_vecs)
    return -0.5j * decay  # no coherent H in your case

_operator_cache = {}
def get_ops(g_name, gmat, N, observable):
    key = (g_name, observable)
    if key in _operator_cache:
        return _operator_cache[key]

    jump_vecs = build_jump_vecs(gmat)
    H_eff = build_Heff(jump_vecs, N)

    if observable == "emission":
        # C_nu†C_nu in single-exc block = |v_nu><v_nu|
        e_ops = [np.outer(v.conj(), v) for v in jump_vecs]
    elif observable == "population":
        # projector onto site i = just reading out |psi[i]|^2
        e_ops = None  # handled directly in sim

    _operator_cache[key] = (H_eff, e_ops)
    return H_eff, e_ops


# ------------------------------
# Initial States
# ------------------------------

def exc(N, vals):
    """
    Single-excitation state vector in N-dim subspace.
    vals: list of site indices to excite.
    If multiple sites given, equal superposition (normalized).
    """
    psi = np.zeros(N, dtype=complex)
    for i in vals:
        psi[i] = 1.0
    norm = np.linalg.norm(psi)
    return psi / norm


# ------------------------------
# Simulation
# ------------------------------

class Result:
    """Minimal stand-in for qutip Result, keeps plot() working."""
    def __init__(self, expect):
        self.expect = expect  # list of 1D arrays, one per e_op


def sim(N, psi0, g_name, gmat, tlist, observable="emission"):
    H_eff, e_ops = get_ops(g_name, gmat, N, observable)

    # Diagonalize H_eff once, evolve cheaply at each t
    eigenvalues, R = np.linalg.eig(H_eff)
    coeffs = np.linalg.solve(R, psi0)

    if observable == "emission":
        # <psi(t)|C†C|psi(t)> for each mode nu
        expect = [np.zeros(len(tlist)) for _ in e_ops]
        for k, t in enumerate(tlist):
            psi_t = R @ (coeffs * np.exp(eigenvalues * t))
            for nu, e_op in enumerate(e_ops):
                expect[nu][k] = np.real(psi_t.conj() @ e_op @ psi_t)

    elif observable == "population":
        # |psi_i(t)|^2 per site — no matmul needed
        expect = [np.zeros(len(tlist)) for _ in range(N)]
        for k, t in enumerate(tlist):
            psi_t = R @ (coeffs * np.exp(eigenvalues * t))
            pops = np.abs(psi_t)**2
            for i in range(N):
                expect[i][k] = pops[i]

    return Result(expect)


def extract_modes(result):
    I = np.sum(result.expect, axis=0)
    return np.log10(I / I[0])


# ------------------------------
# Plotting (unchanged)
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
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]
        for ax, g_name in zip(axes, g_names):
            for r_name, result_m in grouped[g_name].items():
                y = extract_modes(result_m)
                ax.plot(tlist, y, label=r_name)
            ax.set_title(g_name)
            ax.set_xlabel("time")
            ax.set_ylabel("normalized emission (log10)")
            ax.legend()

    elif obs == "population":
        labels = list(result.keys())
        n = len(labels)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)
        axes = np.array(axes).flatten()
        for ax, label in zip(axes, labels):
            r = result[label]
            for i, site in enumerate(r.expect):
                ax.plot(tlist, site, label=f"site {i}")
            ax.set_xlabel("time")
            ax.set_ylabel("population")
            ax.set_title(label)
        for ax in axes[n:]:
            ax.remove()
    plt.show()


# ------------------------------
# Configs
# ------------------------------

def get_rho(N):
    return {
        "rho0": exc(N, range(N)),
        "rho1": exc(N, [0]),
        "rho2": exc(N, [4]),
        "rho4": exc(N, [0, 1]),   # superposition of sites 0 and 1
        "rho5": exc(N, [2, 3]),
        "rho6": exc(N, [0, 3]),   # previously 1/sqrt(2)*(|0>+|3>), same thing now
    }

def get_gmat(N):
    return {
        "ssh chain0": cp.ssh(N, 0, 1, .1, 0.9),
        "ssh chain1": cp.ssh(N, 0, 1, .3, 0.7),
        "ssh chain2": cp.ssh(N, 0, 1, .5, 0.5),
        "ssh chain3": cp.ssh(N, 0, 1, .7, 0.3),
        "ssh ring":   cp.ssh(N, 1, 1, 0.5, 0.4),
    }

CONFIG = dict(
    N = 8,
    tmax = 5,
    nt = 500,
    states = ["rho1", "rho2"],
    couplings = ["ssh chain0", "ssh chain3", "ssh chain2", "ssh chain1"],
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
            psi0 = rhos_config[r_name]
            label = f"{r_name} | {g_name}"
            results[label] = sim(N, psi0, g_name, g, tlist, obs)
    end = time.time()
    print(f"elapsed: {end - start:.3f}s")
    plot(results, tlist, obs)

if __name__ == "__main__":
    main()
