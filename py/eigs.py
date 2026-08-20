import numpy as np
import couplings as cp
import scipy
import matplotlib.pyplot as plt


def ssh_eigs(N, periodic, t0, t1, t2):
    diag = np.full(N, t0)
    off = np.empty(N - 1)
    off[0::2] = t1
    off[1::2] = t2
    
    if periodic == 0:
        vals, vecs = scipy.linalg.eigh_tridiagonal(diag, off)
    else:
        mat = np.zeros((N, N))
        np.fill_diagonal(mat, t0)

        mat += np.diag(off, 1)
        mat += np.diag(off, -1)
        t = t1 if (N-1)%2 == 0 else t2
        mat[0, -1] = mat[-1, 0] = t
        vals, vecs = scipy.linalg.eigh(mat)

    idx = np.argsort(np.imag(vals))
    vals = vals[idx]
    vecs = vecs[:, idx]

    return vals, vecs

def dssh_eigs(N, periodic, gamma, Gamma1, Gamma2):
    GammaR = gamma + Gamma1 + Gamma2
    igr = -1j * GammaR

    off = np.empty(N - 1, dtype=complex)
    off[0::2] = 1j * Gamma1
    off[1::2] = 1j * Gamma2
    
    mat = np.zeros((N, N), dtype=complex)
    np.fill_diagonal(mat, igr)
    mat += np.diag(off, 1)
    mat += np.diag(off, -1)
    
    if periodic == 1:
        t = (1j * Gamma1) if (N - 1) % 2 == 0 else (1j * Gamma2)
        mat[-1, 0] = mat[0, -1] = t

    vals, vecs = scipy.linalg.eig(mat)
    idx = np.argsort(np.imag(vals))
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs

CONFIG=dict(
    N = 2000,
    ns = 100,
    
    t0 = 1,
    t1 = np.linspace(0, 4, 100),
    t2 = 2,

    Gamma1 = np.linspace(0, 1, 100),
    Gamma2 = 1 - np.linspace(0, 1, 100),
    gamma = 0,
)

def plot_eig(N, x0, x1, x2):
    lx1 = len(x1)
    eigs_obc = np.empty((lx1, N))
    # eigs_pbc = np.empty((lx1, N))

    for i in range(lx1):
        eigs_obc[i], p = ssh_eigs(N, 0, x0, x1[i], x2[i])
        # eigs_pbc[i], q = dssh_eigs(N, 1, x0, x1[i], x2[i])

    eigs_obc = (np.array(eigs_obc))
    # eigs_pbc = np.imag(np.array(eigs_pbc))

    for j in range(eigs_obc.shape[1]):
        plt.plot(x1, eigs_obc[:, j], 'r.', markersize=4)
        # plt.plot(x1, eigs_pbc[:, j], 'g.', markersize=4)
    # plt.axvline(x=2, color='k', linestyle='-', linewidth=2)
    ticks = np.arange(0, 1.01, 0.1)

    # Default labels
    labels = [f"{t:.1f}" for t in ticks]

    # Change labels at 0.3 and 0.7
    idx03 = np.where(np.isclose(ticks, 0.2))[0][0]
    idx07 = np.where(np.isclose(ticks, 0.8))[0][0]
    labels[idx03] = "A"
    labels[idx07] = "B"
    plt.xticks(ticks, labels)
    plt.show()

def compute_ipr(vecs):
    vecs = vecs / np.linalg.norm(vecs, axis=0)
    return np.sum(np.abs(vecs)**4, axis=0)

def plot_ipr(N, x0, x1, x2):
    lx1 = len(x1)
    ipr_obc = np.empty(((lx1), N))
    ipr_pbc = np.empty(((lx1), N))

    for i in range(lx1):
        vals_obc, vecs_obc = ssh_eigs(N, 0, x0, x1[i], x2[i])
        vals_pbc, vecs_pbc = ssh_eigs(N, 1, x0, x1[i], x2[i])

        ipr_obc[i] = compute_ipr(vecs_obc)
        ipr_pbc[i] = compute_ipr(vecs_pbc)
    
    plt.plot(x1, np.log10(np.mean(ipr_obc, axis=1)), 'r.', markersize=6)
    plt.axvline(x=0.3, color='k',linestyle='--',linewidth=2)
    plt.axvline(x=0.7, color='k',linestyle='--',linewidth=2)
    plt.xlabel("Gamma_1", fontsize=14)
    plt.ylabel("Inverse Participation", fontsize=14)
    # fig, axes = plt.subplots(10,5, figsize=(15,20))
    # for j, ax in enumerate(axes.flatten()):
        # ax.plot(x1, np.log10(ipr_obc[:, j]), 'r.', markersize=3)
        # ax.plot(x1, np.log10(ipr_pbc[:, j]), 'g.', markersize=3)
        # ax.axhline(y=np.log10(1/50), color='k',linestyle='-',linewidth=1)
    plt.show()
    return 0

def plot_realspace_profiles(N, gamma, Gamma1, Gamma2, state_indeces=[0,24,25,49]):
    G1_slices =  [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8] # np.linspace(0.1, 0.3, 10) #
    G2_slices =  [0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2] # 1 - np.linspace(0.1, 0.3, 10) #
    gamma_slices = [gamma] * len(G1_slices)

    n_states = len(state_indeces)
    n_rows = len(G1_slices)

    sites  = np.arange(N)
    fig, axes = plt.subplots(
        n_rows, n_states,
        figsize=(3 * n_states, 2.5 * n_rows),
        sharex=True
    )
    
    picks  = np.linspace(0, N - 1, n_states, dtype=int)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (g, G1, G2) in enumerate(zip(gamma_slices, G1_slices, G2_slices)):
        vals, vecs = ssh_eigs(N, 0, g, G1, G2)

        for col, idx in enumerate(state_indeces):
            ax  = axes[row, col]
            psi = np.abs(vecs[:, idx]) ** 2
            psi = psi / psi.sum()

            ax.bar(sites, psi, width=0.8, alpha=0.8)
            ax.set_ylim(0, None)
            ax.axvline(x=N / 2 - 0.5, lw=0.8, ls='--')

            if row == 0:
                E = vals[idx]
                ax.set_title(f"state {idx}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Γ₁={G1:.2f}", fontsize=7)
            if row == n_rows - 1:
                ax.set_xlabel("site n", fontsize=8)

    fig.suptitle(f"Real-space profiles |ψₙ|²  (OBC, N={N})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("realspace_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: realspace_profiles.png")

def main():
    N = 50#CONFIG["N"]

    Gamma1 = np.linspace(0., 1., 100)
    Gamma2 = 1 - np.linspace(0., 1., 100)
    gamma = 1
    plot_eig(N, gamma, Gamma1, Gamma2)
    # plot_ipr(N, gamma, Gamma1, Gamma2)
    # plot_realspace_profiles(N, gamma, Gamma1, Gamma2)

if __name__ == "__main__":
    main()
