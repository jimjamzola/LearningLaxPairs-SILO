"""Inviscid Burgers: symbolic regression on the Clairaut identity (paper 4.4)
    2 s g_u + f_u (g_s + u) = 0,   s = S_x,
with f(u) polynomial of degree N1 and g(s,u) of bidegree (N2, N3).
Port of Hopf.m (random init, L-BFGS, threshold-refit with w = 100).
Classification uses Proposition 4.1: for f = a u + lambda every solution is
    g = 2 s^3/(3a) - u s + Phi(s^2 - a u),
so we read off deg f and deg Phi.
"""
import sys, json, time
import numpy as np
from multiprocessing import Pool
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("SILO_OUT", os.path.join(HERE, os.pardir, "output"))
os.makedirs(OUT, exist_ok=True)
from common import run_pipeline, embed, refit_pure

N1, N2, N3 = 3, 4, 2
NPTS = 30
DIM = (N1 + 1) + (N2 + 1) * (N3 + 1)


def split(eta):
    xi = eta[:N1 + 1]
    zeta = eta[N1 + 1:].reshape(N2 + 1, N3 + 1, order="F")   # zeta[j,k] s^j u^k
    return xi, zeta


def residual(eta, s, u):
    xi, zeta = split(eta)
    fu = sum(j * xi[j] * u ** (j - 1) for j in range(1, N1 + 1))
    gu = sum(k * zeta[j, k] * s ** j * u ** (k - 1) for j in range(N2 + 1) for k in range(1, N3 + 1))
    gs = sum(j * zeta[j, k] * s ** (j - 1) * u ** k for j in range(1, N2 + 1) for k in range(N3 + 1))
    return 2 * s * gu + fu * (gs + u)


def make_objective(s, u, lam, support=None):
    emb = embed(support, DIM) if support is not None else (lambda x: x)

    def obj(x):
        eta = emb(x)
        xi, zeta = split(eta)
        r = residual(eta, s, u)
        nz, nx = np.max(np.abs(zeta)), np.max(np.abs(xi))
        if nz < 1e-12 or nx < 1e-12:
            return 1e6          # MATLAB: 0/0 = NaN; the trivial solutions are excluded
        return np.mean(r ** 2) / (nz * nx) + lam * np.sum(np.abs(eta))
    return obj


def classify(eta, s, u):
    xi, zeta = split(eta)
    r = residual(eta, s, u)
    scale = max(np.max(np.abs(xi)), 1e-300)
    res = float(np.mean(r ** 2))
    tol = 1e-7 * scale
    nzf = np.flatnonzero(np.abs(xi) > tol)
    degf = int(nzf.max()) if nzf.size else -1
    out = {"residual": res, "deg_f": degf, "nnz": int(np.sum(eta != 0))}
    if res > 1e-10 * (1 + np.max(np.abs(zeta)) ** 2) * scale ** 2:
        out["label"] = "nonconverged"
        return out
    if degf <= 0:
        out["label"] = "degenerate (f constant, g = g(s))" if degf == 0 else "trivial (f = 0)"
        return out
    if degf >= 2:
        out["label"] = "deg f >= 2"
        return out
    a, lam = xi[1], xi[0]
    # h = g - g0 must be Phi(xi) with xi = s^2 - a u, Phi of degree <= 2
    h = zeta.copy()
    h[3, 0] -= 2.0 / (3 * a)
    h[1, 1] -= -1.0
    # basis: 1, xi, xi^2 expanded in (s^j u^k)
    basis = []
    for d in range(3):
        B = np.zeros_like(zeta)
        # (s^2 - a u)^d
        from math import comb
        for i in range(d + 1):
            B[2 * i, d - i] += comb(d, i) * (-a) ** (d - i)
        basis.append(B.ravel())
    X = np.stack(basis, axis=1)
    c, *_ = np.linalg.lstsq(X, h.ravel(), rcond=None)
    fit_res = np.linalg.norm(X @ c - h.ravel()) / (np.linalg.norm(zeta) + 1e-300)
    out["phi_coeffs"] = c.tolist()
    out["phi_fit_defect"] = float(fit_res)
    nzp = np.flatnonzero(np.abs(c) > 1e-6 * np.max(np.abs(zeta)))
    degphi = int(nzp.max()) if nzp.size else 0
    out["deg_phi"] = degphi
    out["a"], out["lambda"] = float(a), float(lam)
    if fit_res > 1e-5:
        out["label"] = "f linear, g not of Prop-4.1 form"
    else:
        out["label"] = "f linear, deg Phi = %d" % degphi
    return out


STAGES = [dict(lam=1e-6), dict(lam=0.0), dict(lam=0.0)]
W_THRESH = 100.0


def one_run(seed):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, NPTS)
    u = rng.uniform(-1, 1, NPTS)

    def make_obj(st, support=None):
        return make_objective(s, u, st["lam"], support=support)

    t = time.time()
    x0 = rng.uniform(0, 1, DIM)
    eta, hist = run_pipeline(make_obj, DIM, rng, STAGES, W_THRESH, x_init=x0)
    eta = refit_pure(lambda supp: make_objective(s, u, 0.0, support=supp), eta)
    info = classify(eta, s, u)
    info.update(seed=int(seed), time=time.time() - t, eta=eta.tolist())
    return info


def one_run_logged(args):
    info = one_run(args)
    with open(os.path.join(OUT, "progress_burgers.jsonl"), "a") as fh:
        fh.write(json.dumps(info) + "\n")
    return info


if __name__ == "__main__":
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    with Pool(2) as pool:
        results = list(pool.imap_unordered(one_run_logged, range(nseeds)))
    json.dump(results, open(os.path.join(OUT, "results_burgers.json"), "w"), indent=1)
    labels = {}
    for r in results:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    print(labels)
