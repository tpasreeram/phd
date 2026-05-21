import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from qutip import enr_destroy, enr_fock
import couplings as cp

# ── helpers ──────────────────────────────────────────────────────────────────

def make_two_exc_basis(N: int, n_exc: int = 2):
    return list(itertools.combinations(range(N), 2))   # i < j, qubit sites


def build_heff(gmat: np.ndarray, N: int, n_exc: int):
    a_ops = enr_destroy([2] * N, excitations=n_exc)

    H_eff = None
    for i in range(N):
        for j in range(N):
            if abs(gmat[i, j]) < 1e-15:
                continue
            term = (-1j / 2.0) * gmat[i, j] * a_ops[i].dag() * a_ops[j]
            H_eff = term if H_eff is None else H_eff + term

    if H_eff is None:
        # zero matrix in the right space
        H_eff = 0 * a_ops[0].dag() * a_ops[0]

    return H_eff, a_ops


def fock_state(N: int, n_exc: int, occ: list):
    return enr_fock([2] * N, excitations=n_exc, state=occ)


def decompose_into_two_exc_basis(eigenstate_vec: np.ndarray, N: int, n_exc: int,
                                  basis_pairs: list, a_ops):
    # Build a reference ket for each basis pair to read off its ENR index.
    # enr_fock returns a ket whose dense array is a unit vector; argmax gives
    # the index of the '1' entry in the ENR basis ordering.
    A = np.zeros((N, N), dtype=complex)
    for (i, j) in basis_pairs:
        occ = [0] * N
        occ[i] += 1
        occ[j] += 1
        ref = fock_state(N, n_exc, occ).full().ravel()   # unit vector in ENR space
        idx = int(np.argmax(np.abs(ref)))                # index of the '1'
        amp = eigenstate_vec[idx]
        A[i, j] = amp
        A[j, i] = amp                                    # symmetric for plotting
    return A


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse_heff(gmat: np.ndarray, N: int, n_exc: int = 2,
                 n_states: int | None = None,
                 sort_by: str = "decay"):
    H_eff, a_ops = build_heff(gmat, N, n_exc)
    basis_pairs   = make_two_exc_basis(N, n_exc)

    # Dense diagonalisation (H_eff is non-Hermitian)
    H_dense = H_eff.full()
    evals, evecs_cols = np.linalg.eig(H_dense)

    if sort_by == "decay":
        order = np.argsort(np.imag(evals))   # most negative Im(E) first
    else:
        order = np.argsort(np.real(evals))
    evals     = evals[order]
    evecs_cols = evecs_cols[:, order]

    if n_states is not None:
        evals      = evals[:n_states]
        evecs_cols = evecs_cols[:, :n_states]

    M = len(evals)

    # Build qutip kets and project.
    # ENR space has a flat dimension (e.g. 22 for N=6, n_exc=2), NOT 2^N.
    # Use [[enr_dim], [1]] so QuTiP sees a simple column vector of the right size.
    intensities = np.zeros((M, N, N))

    for k in range(M):
        vec = evecs_cols[:, k]
        vec /= np.linalg.norm(vec)
        A   = decompose_into_two_exc_basis(vec, N, n_exc, basis_pairs, a_ops)
        intensities[k] = np.abs(A) ** 2

    return evals, intensities, basis_pairs


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_eigenstate_heatmaps(evals, intensities, N: int,
                              ncols: int = 4,
                              cmap: str = "inferno",
                              title_prefix: str = "Eigenstate",
                              threshold: float = 1e-12):
    weights = intensities.reshape(len(intensities), -1).sum(axis=1)
    keep    = weights > threshold

    evals = np.asarray(evals)[keep]
    intensities = np.asarray(intensities)[keep]

    M     = len(evals)
    ncols = min(ncols, M)
    nrows = (M + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.5 * ncols, 3.2 * nrows),
                              squeeze=False)
    fig.suptitle("Two-excitation decomposition of H_eff eigenstates",
                 fontsize=14, y=1.01)


    for k in range(M):
        r, c  = divmod(k, ncols)
        ax    = axes[r][c]

        vmax = intensities[k].max() or 1.0

        im    = ax.imshow(intensities[k], origin="lower",
                          cmap=cmap, vmin=0, vmax=vmax,
                          aspect="equal")
        E     = evals[k]
        label = (f"{title_prefix} {k}\n"
                 f"E = {E.real:.3g}{E.imag:+.3g}i")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("site j", fontsize=7)
        ax.set_ylabel("site i", fontsize=7)
        ax.set_xticks(ticks=[0, 9, 19], labels=['1', '10', '20'])
        ax.set_yticks(ticks=[0, 9, 19], labels=['1', '10', '20'])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=r"$|c_{ij}|^2$")

    # hide unused axes
    for k in range(M, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    return fig


def plot_dominant_state_overview(evals, intensities, N: int,
                                  cmap: str = "inferno"):
    dom_idx = np.argmax(intensities, axis=0)    # (N, N)
    dom_val = np.max(intensities, axis=0)       # (N, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(dom_idx, origin="upper", cmap="tab20",
                     vmin=0, vmax=len(evals)-1, aspect="equal")
    ax1.set_title("Dominant eigenstate index per (i,j)", fontsize=11)
    ax1.set_xlabel("site j"); ax1.set_ylabel("site i")
    fig.colorbar(im1, ax=ax1, label="eigenstate index")

    im2 = ax2.imshow(dom_val, origin="upper", cmap=cmap, aspect="equal")
    ax2.set_title(r"Max $|c_{ij}|^2$ over eigenstates", fontsize=11)
    ax2.set_xlabel("site j"); ax2.set_ylabel("site i")
    fig.colorbar(im2, ax=ax2, label=r"$\max_k |c_{ij}^{(k)}|^2$")

    plt.tight_layout()
    return fig


# ── example / entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── define your system here ──────────────────────────────────────────────
    N     = 20         # number of sites
    n_exc = 2          # two-excitation sector
    # ─────────────────────────────────────────────────────────────────────────

    for i in np.arange(.25, .2501, 0.0001):
        gmat = cp.ssh(N, 0, 1, i, 1 - i)
        print("Building H_eff and diagonalising …")
        evals, intensities, basis_pairs = analyse_heff(
            gmat, N, n_exc=n_exc, sort_by="decay"
        )
    
        # print(f"Found {len(evals)} eigenstates")
        # for k, E in enumerate(evals):
        #     print(f"  [{k:2d}]  E = {E.real:+.6f} {E.imag:+.6f}i  "
        #         f"  decay rate = {-2*E.imag:.6f}")
        
        fig1 = plot_eigenstate_heatmaps(evals, intensities, N, ncols=4)
        fig1.savefig(f"heff_eigenstates_{i:.2f}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: heff_eigenstates_{i:.2f}.png")
    
        # fig2 = plot_dominant_state_overview(evals, intensities, N)
        # fig2.savefig(f"heff_dominant_{i:.2f}.png", dpi=150, bbox_inches="tight")
        # print(f"Saved: heff_dominant_{i:.2f}.png")

        # plt.show()
