"""Shared SILO pipeline: cross-entropy warm start, L-BFGS polish, sequential
thresholding with refit on the surviving support.  Faithful to the MATLAB
scripts (CEopt + fminunc + sparsify), with the optimizer settings made
explicit so the protocol is reproducible seed by seed.
"""
import numpy as np
from scipy.optimize import minimize


def ce_warm_start(obj, dim, rng, lb=-5.0, ub=5.0, n_samp=200, n_elite=20,
                  iters=80, tol=1e-10):
    """Cross-entropy method with diagonal Gaussian proposal (CEopt-like)."""
    mu = np.zeros(dim)
    sig = np.full(dim, (ub - lb) / 2)
    best_x, best_f = mu.copy(), np.inf
    for _ in range(iters):
        X = mu + sig * rng.standard_normal((n_samp, dim))
        X = np.clip(X, lb, ub)
        f = np.array([obj(x) for x in X])
        f = np.where(np.isfinite(f), f, np.inf)
        idx = np.argsort(f)[:n_elite]
        if f[idx[0]] < best_f:
            best_f, best_x = f[idx[0]], X[idx[0]].copy()
        mu = X[idx].mean(0)
        sig = 0.9 * X[idx].std(0) + 0.1 * sig
        if sig.max() < tol:
            break
    return best_x


def lbfgs(obj, x0, maxiter=3000):
    res = minimize(obj, x0, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12,
                            "maxfun": 200000})
    return res.x


def sparsify(obj_factory, x0, w, maxiter=3000):
    """Threshold at max|x|/w, refit on the surviving support, threshold again.
    obj_factory(support) returns an objective on the reduced coordinates."""
    x0 = np.array(x0, dtype=float)
    A = np.max(np.abs(x0))
    thresh = A / w
    x0[np.abs(x0) < thresh] = 0.0
    support = np.flatnonzero(x0)
    if support.size == 0:
        return x0, thresh, support
    obj_r = obj_factory(support)
    xr = lbfgs(obj_r, x0[support], maxiter)
    x = np.zeros_like(x0)
    x[support] = xr
    x[np.abs(x) < thresh] = 0.0
    return x, thresh, np.flatnonzero(x)


def run_pipeline(make_obj, dim, rng, stages, w, ce_kwargs=None,
                 x_init=None):
    """stages: list of dicts of penalty weights.  Stage 0 is used for the CE
    warm start and first polish; later stages are threshold-and-refit steps.
    make_obj(weights, support=None) -> objective on full or reduced coords."""
    ce_kwargs = ce_kwargs or {}
    obj0 = make_obj(stages[0])
    if x_init is None:
        x = ce_warm_start(obj0, dim, rng, **ce_kwargs)
    else:
        x = np.array(x_init, dtype=float)
    x = lbfgs(obj0, x)
    hist = [x.copy()]
    for st in stages[1:]:
        x, thresh, supp = sparsify(lambda s: make_obj(st, support=s), x, w)
        hist.append(x.copy())
    return x, hist


def embed(support, dim):
    def f(xr):
        y = np.zeros(dim)
        y[support] = xr
        return y
    return f


def refit_pure(make_pure_obj, eta, maxiter=3000):
    """Final polish: minimize the pure compatibility loss (no penalties) on the
    support of eta, then zero entries that fall below 1e-12 * max|eta|."""
    eta = np.array(eta, dtype=float)
    support = np.flatnonzero(eta)
    if support.size == 0:
        return eta
    obj = make_pure_obj(support)
    xr = lbfgs(obj, eta[support], maxiter)
    x = np.zeros_like(eta)
    x[support] = xr
    x[np.abs(x) < 1e-12 * np.max(np.abs(x))] = 0.0
    return x
