"""Free Schrodinger equation: SILO search with two nondegeneracy barriers.

Affine ansatz  f = a1(k) q + a2(k) mu,  g = a3(k) q + a4(k) q_x + a5(k) mu,
a_j(k) = z_{0j} + z_{1j} k + z_{2j} k^2  (15 complex = 30 real parameters).
Clairaut compatibility gives the algebraic constraints (paper eq. 3.4)
    i a1 - a4 = 0,  a2 a3 - a5 a1 = 0,  a2 a4 - a3 = 0.
Barriers on the Gram matrix G = Z^H Z (Z is 5x3):
    det  : r/2 |det G|^{-2}
    pinv : r/2 tr(G^{-1})
Pipeline follows SchrodingerSILO3.m (CE warm start, polish, two threshold-refit
passes with w = 200).
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

NK = 100


def Z_of(eta):
    z = eta[0::2] + 1j * eta[1::2]
    return z.reshape(5, 3, order="F")       # Z[j, l] = z_{lj}: coefficient of k^l in a_{j+1}


def constraint_residual(Z, k):
    kk = np.stack([np.ones_like(k), k, k ** 2])       # (3, N)
    a = Z @ kk                                        # (5, N)
    a1, a2, a3, a4, a5 = a
    F = np.abs(1j * a1 - a4) ** 2 + np.abs(a2 * a3 - a5 * a1) ** 2 + np.abs(a2 * a4 - a3) ** 2
    return F.mean()


def make_objective(k, barrier, r, s, support=None, dim=30):
    emb = embed(support, dim) if support is not None else (lambda x: x)

    def obj(x):
        eta = emb(x)
        Z = Z_of(eta)
        J = (1 - r) / 2 * constraint_residual(Z, k)
        if r > 0:
            G = Z.conj().T @ Z
            sv = np.linalg.svd(Z, compute_uv=False)
            if barrier == "det":
                D = np.prod(sv ** 2)
                J += r / 2 / (D ** 2 + 1e-300)
            elif barrier == "pinv":
                J += r / 2 * np.sum(1.0 / (sv ** 2 + 1e-300))
            elif barrier == "det_n":      # scale-invariant: det G / (tr G / 3)^3 in (0, 1]
                D = np.prod(sv ** 2) / (np.mean(sv ** 2) ** 3 + 1e-300)
                J += r / 2 / (D ** 2 + 1e-300)
            elif barrier == "pinv_n":     # scale-invariant: tr(G^-1) tr(G) / 9 >= 1
                J += r / 2 * np.sum(1.0 / (sv ** 2 + 1e-300)) * np.sum(sv ** 2) / 9
            if barrier in ("det_n", "pinv_n"):   # the loss is homogeneous, so fix the scale
                J += (np.sum(sv ** 2) - 3.0) ** 2
        return J + s * np.sum(np.abs(eta))
    return obj


def poly_deg(c, tol):
    """degree of polynomial with coefficient vector c (index = power), -1 if zero"""
    nz = np.flatnonzero(np.abs(c) > tol)
    return int(nz.max()) if nz.size else -1


def classify(eta, k):
    Z = Z_of(eta)
    res = constraint_residual(Z, k)
    scale = np.max(np.abs(Z)) + 1e-300
    tol = 1e-6 * scale
    a = [Z[j, :] for j in range(5)]
    degs = [poly_deg(a[j], tol) for j in range(5)]
    sv = np.linalg.svd(Z, compute_uv=False)
    out = {"residual": float(res), "degrees": degs,
           "sv": sv.tolist(), "rank_Z": int(np.sum(sv > 1e-8 * sv[0])),
           "nnz": int(np.sum(eta != 0))}
    if res > 1e-10 * scale ** 4:
        label = "nonconverged"
    elif degs[0] < 0:
        label = "field-free (a1=0)"
    elif degs[1] < 0:
        label = "zero-parameter (nonspectral)"
    elif degs[1] == 0:
        label = "collapsed M(k) family (a2 const)"
    elif degs[1] == 1:
        label = "UTM class (a2 linear in k)"
        out["a1_degree"] = degs[0]
    else:
        label = "other"
    out["label"] = label
    return out


STAGES = [dict(r=0.1, s=1e-3), dict(r=1e-4, s=0.0), dict(r=1e-6, s=0.0)]
W_THRESH = 200.0


def one_run(args):
    barrier, seed = args
    rng = np.random.default_rng(seed)
    k = rng.uniform(-1, 1, NK)

    def make_obj(st, support=None):
        return make_objective(k, barrier, st["r"], st["s"], support=support)

    t = time.time()
    eta, hist = run_pipeline(make_obj, 30, rng, STAGES, W_THRESH,
                             ce_kwargs=dict(lb=-3, ub=3, iters=60))
    eta = refit_pure(lambda supp: make_objective(k, barrier, 0.0, 0.0, support=supp), eta)
    info = classify(eta, k)
    info.update(barrier=barrier, seed=int(seed), time=time.time() - t, eta=eta.tolist())
    return info


def one_run_logged(args):
    info = one_run(args)
    with open(os.path.join(OUT, "progress_schrodinger.jsonl"), "a") as fh:
        fh.write(json.dumps(info) + "\n")
    return info


if __name__ == "__main__":
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    barriers = sys.argv[2:] if len(sys.argv) > 2 else ["det", "pinv"]
    jobs = [(b, s) for b in barriers for s in range(nseeds)]
    with Pool(2) as pool:
        results = list(pool.imap_unordered(one_run_logged, jobs))
    json.dump(results, open(os.path.join(OUT, "results_schrodinger.json"), "w"), indent=1)
    for b in barriers:
        rs = [r for r in results if r["barrier"] == b]
        labels = {}
        for r in rs:
            labels[r["label"]] = labels.get(r["label"], 0) + 1
        print(b, labels)
