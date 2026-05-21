import numpy as np
import scipy

# ------------------------------
# Coupling matrices
# ------------------------------

def chain_perp(N, k0, a):
    g = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                g[i,i] = .5
            else:
                r = abs(i-j) * a * 2 * np.pi
                kr = k0 * r
                g[i,j] = (3/4) * (
                    np.sin(kr)/kr
                    + np.cos(kr)/(kr**2)
                    - np.sin(kr)/(kr**3)
                )
    return g

def chain_par(N, k0, a):
    g = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                g[i,i] = .5
            else:
                r = abs(i-j) * a * 2 * np.pi
                kr = k0 * r
                g[i,j] = (3/2) * (
                    np.sin(kr) / kr ** 3 - np.cos(kr) / kr ** 2
                )
    return g

# ------------------------------
# Interaction matrices
# ------------------------------

def nn(N, k0, a, periodic, orientation):
    B = np.zeros((N, N))
    g = np.zeros((N, N))
    if periodic == 0: 
        B = (
            np.eye(N) +
            np.diag(np.ones(N-1), k=1) +
            np.diag(np.ones(N-1), k=-1)
        )
        if orientation == 1:
            A = chain_par(N, k0, a)
        else:
            A = chain_perp(N, k0, a)
        g = np.where(B == 1, A, 0)
    else:
        for i in range(N):
            B[i, i] = 1
            B[i, (i+1) % N] = 1
            B[i, (i-1) % N] = 1
        if orientation == 1:
            A = chain_par(N, k0, a)
        else:
            A = chain_perp(N, k0, a)
        g = np.where(B == 1, A, 0)
    return g

def nnn(N, k0, a, periodic, orientation):
    B = np.zeros((N, N))
    g = np.zeros((N, N))
    if periodic == 0: 
        B = (
            np.eye(N) +
            np.diag(np.ones(N-1), k=1) +
            np.diag(np.ones(N-1), k=-1) +
            np.diag(np.ones(N-2), k=2) +
            np.diag(np.ones(N-2), k=-2)
        )
        if orientation == 1:
            A = chain_par(N, k0, a)
        else:
            A = chain_perp(N, k0, a)
        g = np.where(B == 1, A, 0)
    else:
        for i in range(N):
            B[i, i] = 1
            B[i, (i+1) % N] = 1
            B[i, (i-1) % N] = 1
            B[i, (i+2) % N] = 1
            B[i, (i-2) % N] = 1
        if orientation == 1:
            A = chain_par(N, k0, a)
        else:
            A = chain_perp(N, k0, a)
        g = np.where(B == 1, A, 0)
    return g

def ssh(N, periodic, t0, t1, t2):
    g = np.zeros((N, N), dtype=complex)
    np.fill_diagonal(g, t0)

    for i in range(N - 1):
        if i % 2 == 0:
            t = t1
        else:
            t = t2
        g[i,i+1]=g[i+1,i]=t

    if periodic == 1:
        t = t1 if (N-1) % 2 == 0 else t2
        g[0, N - 1] = g[N - 1, 0] = t
    return g
