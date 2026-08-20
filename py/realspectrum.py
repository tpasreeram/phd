import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# ── model ─────────────────────────────────────────────────────────────────────

def dssh_eigs(N, periodic, gamma, Gamma1, Gamma2):
    """Returns (eigenvalues [complex], eigenvectors) sorted by Im(E)."""
    GammaR = gamma + Gamma1 + Gamma2
    igr    = -1j * GammaR
    off    = np.empty(N - 1, dtype=complex)
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
    return vals[idx], vecs[:, idx]


# ── 1. Real-space profiles ─────────────────────────────────────────────────────

def plot_realspace_profiles(N, gamma, Gamma1, Gamma2, n_states=6):
    """
    Plot |ψₙ|² for selected eigenstates at each (gamma, Gamma1, Gamma2) point.

    Parameters
    ----------
    N       : int   — system size
    gamma   : float or array — onsite loss rate(s)
    Gamma1  : array — intra-cell coupling values  (one per row)
    Gamma2  : array — inter-cell coupling values  (same length as Gamma1)
    n_states: int   — how many eigenstates to show per row
    """
    Gamma1 = np.atleast_1d(Gamma1)
    Gamma2 = np.atleast_1d(Gamma2)
    gamma  = np.broadcast_to(gamma, Gamma1.shape)

    assert len(Gamma1) == len(Gamma2), "Gamma1 and Gamma2 must have the same length"

    n_rows = len(Gamma1)
    sites  = np.arange(N)
    picks  = np.linspace(0, N - 1, n_states, dtype=int)

    fig, axes = plt.subplots(
        n_rows, n_states,
        figsize=(3 * n_states, 2.5 * n_rows),
        sharex=True
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (g, G1, G2) in enumerate(zip(gamma, Gamma1, Gamma2)):
        vals, vecs = dssh_eigs(N, 0, g, G1, G2)

        for col, idx in enumerate(picks):
            ax  = axes[row, col]
            psi = np.abs(vecs[:, idx]) ** 2
            psi = psi / psi.sum()

            ax.bar(sites, psi, color='steelblue', width=0.8, alpha=0.8)
            ax.set_ylim(0, None)
            ax.axvline(x=N / 2 - 0.5, color='gray', lw=0.8, ls='--')

            if row == 0:
                E = vals[idx]
                ax.set_title(f"state {idx}\nE={E.real:.2f}+{E.imag:.2f}i", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"γ={g:.2f}, Γ₁={G1:.2f}\nΓ₂={G2:.2f}\n|ψₙ|²", fontsize=7)
            if row == n_rows - 1:
                ax.set_xlabel("site n", fontsize=8)

    fig.suptitle(f"Real-space profiles |ψₙ|²  (OBC, N={N})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("realspace_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: realspace_profiles.png")


# ── 2. Complex energy spectrum ─────────────────────────────────────────────────

def plot_complex_spectrum(N, gamma, Gamma1, Gamma2):
    """
    Plot OBC vs PBC eigenvalues in the complex plane for each
    (gamma, Gamma1, Gamma2) point.

    Parameters
    ----------
    N      : int
    gamma  : float or array
    Gamma1 : array — one value per subplot
    Gamma2 : array — same length as Gamma1
    """
    Gamma1 = np.atleast_1d(Gamma1)
    Gamma2 = np.atleast_1d(Gamma2)
    gamma  = np.broadcast_to(gamma, Gamma1.shape)

    assert len(Gamma1) == len(Gamma2), "Gamma1 and Gamma2 must have the same length"

    n_slices = len(Gamma1)
    cols     = min(3, n_slices)
    rows     = int(np.ceil(n_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (g, G1, G2) in enumerate(zip(gamma, Gamma1, Gamma2)):
        ax = axes[i]
        vals_obc, _ = dssh_eigs(N, 0, g, G1, G2)
        vals_pbc, _ = dssh_eigs(N, 1, g, G1, G2)

        ax.scatter(vals_pbc.real, vals_pbc.imag, c='green', s=25, alpha=0.7, label='PBC', zorder=3)
        ax.scatter(vals_obc.real, vals_obc.imag, c='red',   s=25, alpha=0.7, label='OBC', zorder=4)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        ax.set_title(f"γ={g:.2f}, Γ₁={G1:.2f}, Γ₂={G2:.2f}", fontsize=9)
        ax.set_xlabel("Re(E)", fontsize=8)
        ax.set_ylabel("Im(E)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    for ax in axes[n_slices:]:
        ax.set_visible(False)

    fig.suptitle(f"Complex spectrum: OBC (red) vs PBC (green)  [N={N}]", fontsize=13)
    plt.tight_layout()
    plt.savefig("complex_spectrum.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: complex_spectrum.png")


# ── 3. Spectrum evolution vs Γ₁ ───────────────────────────────────────────────

def plot_spectrum_evolution(N, gamma, Gamma1, Gamma2):
    """
    Track Im(E) and |Re(E)| of all OBC eigenvalues along the
    (Gamma1, Gamma2) parameter path.

    Parameters
    ----------
    N      : int
    gamma  : float or array
    Gamma1 : array — parameter axis (x-axis of the plot)
    Gamma2 : array — paired values, same length as Gamma1
    """
    Gamma1 = np.atleast_1d(Gamma1)
    Gamma2 = np.atleast_1d(Gamma2)
    gamma  = np.broadcast_to(gamma, Gamma1.shape)

    assert len(Gamma1) == len(Gamma2), "Gamma1 and Gamma2 must have the same length"

    im_all, re_all = [], []
    for g, G1, G2 in zip(gamma, Gamma1, Gamma2):
        vals, _ = dssh_eigs(N, 0, g, G1, G2)
        im_all.append(vals.imag)
        re_all.append(np.abs(vals.real))

    im_all = np.array(im_all)   # shape (n_points, N)
    re_all = np.array(re_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for j in range(N):
        sc = ax.scatter(Gamma1, im_all[:, j], c=re_all[:, j],
                        cmap='plasma', s=4, vmin=0, vmax=re_all.max())
    plt.colorbar(sc, ax=ax, label="|Re(E)|")
    ax.set_xlabel("Γ₁"); ax.set_ylabel("Im(E)")
    ax.set_title("Imaginary spectrum vs Γ₁  (OBC)")

    ax = axes[1]
    for j in range(N):
        ax.plot(Gamma1, re_all[:, j], '.', ms=2, color='purple', alpha=0.4)
    ax.set_xlabel("Γ₁"); ax.set_ylabel("|Re(E)|")
    ax.set_title("|Re(E)| vs Γ₁  — nonzero = complex eigenvalues (OBC)")

    plt.tight_layout()
    plt.savefig("spectrum_evolution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: spectrum_evolution.png")


# ── run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N      = 50
    Gamma1 = np.linspace(0, 1, 100)
    Gamma2 = 1 - Gamma1
    gamma  = 0.0   # scalar — broadcast to all points

    # For sliced plots, pick a handful of representative points
    slice_idx = np.linspace(0, len(Gamma1) - 1, 6, dtype=int)
    G1_slices = Gamma1[slice_idx]
    G2_slices = Gamma2[slice_idx]

    print("─── 1. Real-space profiles ───")
    plot_realspace_profiles(N, gamma, G1_slices, G2_slices, n_states=6)

    print("─── 2. Complex spectrum ───")
    plot_complex_spectrum(N, gamma, G1_slices, G2_slices)

    print("─── 3. Spectrum evolution ───")
    plot_spectrum_evolution(N, gamma, Gamma1, Gamma2)
