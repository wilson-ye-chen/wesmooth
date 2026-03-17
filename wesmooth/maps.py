import numpy as np

def err_map_shift(x, s=1):
    b = np.random.normal(0, s)
    return x + b

def err_map_sine(x, s=2, a=0.3, k=3):
    c = np.random.normal(0, s, size=(k, 1))
    c = np.tile(c, (1, len(x)))
    x = np.tile(x, (k, 1))
    y = x - a * np.sin(np.pi * (x - c)) / (np.pi)
    w = np.random.uniform(size=(1, k))
    w = w / np.sum(w)
    return np.dot(w, y)

def err_map_gp(
    x,
    lengthscale=0.05,
    variance=0.5,
    beta=0.5,     # decay in w(t) = rho * exp(-beta*|t|)
    rho=0.5,      # must be in (0,1) to guarantee monotonicity
    jitter=1e-10,
):
    """
    Samples a smooth monotone function f on a 1D grid x (sorted),
    with:
        - E[f(x)] = x
        - strictly increasing (a.s.)
        - variance bounded as |x| grows

    Construction:
        u ~ GP(m, k_RBF)
        m(x) = -0.5 * k(x,x)   -> E[exp(u(x))] = 1
        f(x) = x + ∫_0^x w(t)*(exp(u(t)) - 1) dt
        w(t) = rho * exp(-beta*|t|),  0 < rho < 1
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D array with at least 2 points.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0,1).")
    if beta <= 0:
        raise ValueError("beta must be > 0.")

    # --- RBF kernel ---
    def rbf(a, b):
        a = a[:, None]
        b = b[None, :]
        sqdist = (a - b) ** 2
        return variance * np.exp(-0.5 * sqdist / lengthscale**2)

    # Ensure 0 is included for integration anchor
    if np.any(x == 0.0):
        x_aug = x
        idx0 = int(np.where(x_aug == 0.0)[0][0])
        keep_mask = None
    else:
        x_aug = np.sort(np.unique(np.concatenate([x, [0.0]])))
        idx0 = int(np.where(x_aug == 0.0)[0][0])
        keep_mask = np.isin(x_aug, x)

    # --- Sample GP ---
    K = rbf(x_aug, x_aug)
    mu = -0.5 * np.diag(K)  # ensures E[exp(u)] = 1
    L = np.linalg.cholesky(K + jitter * np.eye(len(x_aug)))
    u = mu + L @ np.random.randn(len(x_aug))

    # --- Build integrand ---
    w = rho * np.exp(-beta * np.abs(x_aug))
    integrand = w * (np.exp(u) - 1.0)

    # --- Integrate from 0 using trapezoids ---
    r = np.zeros_like(x_aug)

    # Right side (x >= 0)
    for i in range(idx0 + 1, len(x_aug)):
        dx = x_aug[i] - x_aug[i - 1]
        r[i] = r[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * dx

    # Left side (x <= 0)
    for i in range(idx0 - 1, -1, -1):
        dx = x_aug[i + 1] - x_aug[i]
        r[i] = r[i + 1] - 0.5 * (integrand[i] + integrand[i + 1]) * dx

    f_aug = x_aug + r

    if keep_mask is None:
        return f_aug
    else:
        return f_aug[keep_mask]

## EOF
