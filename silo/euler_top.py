"""Euler top: SILO seed protocol + exact rigidity at the classical pair.

Ansatz (paper eq. 2.5): L_ij = sum_k xi_ijk Omega_k, P_ij = sum_k zeta_ijk Omega_k.
Loss (paper eq. 2.6, 2.9): relative compatibility residual + sparsity, optionally
skew-symmetry (hypothesis-class restriction) and the state-support penalty J_full.

Protocols
  A  unconstrained 54 params, no J_full
  B  skew-symmetric 18 params, no J_full
  C  skew-symmetric 18 params, with J_full          (paper eq. 2.9)
  D  unconstrained 54 params, with J_full
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

I = np.array([1.2, 2.3, 3.1])
NPTS = 100
E = np.zeros((3, 3, 3))                      # hat map basis
E[0] = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
E[1] = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
E[2] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]


def omega_dot(W):
    W1, W2, W3 = W[:, 0], W[:, 1], W[:, 2]
    return np.stack([(I[1] - I[2]) / I[0] * W2 * W3,
                     (I[2] - I[0]) / I[1] * W1 * W3,
                     (I[0] - I[1]) / I[2] * W1 * W2], axis=1)


def blocks_from_eta(eta, skew):
    """Return coefficient blocks A_k (for L) and B_k (for P), shape (3,3,3)."""
    if skew:
        M = eta[:9].reshape(3, 3)            # L = hat(M Omega): A_k = hat(M[:,k])
        N = eta[9:18].reshape(3, 3)
        A = np.einsum("ik,ijl->kjl", M, E)   # A[k] = sum_i M[i,k] E[i]
        B = np.einsum("ik,ijl->kjl", N, E)
    else:
        A = eta[:27].reshape(3, 3, 3)        # A[k][i,j] = xi_{ijk}
        B = eta[27:54].reshape(3, 3, 3)
    return A, B


def make_objective(W, skew, delta, s, mu=1.0, support=None, dim=None):
    Wd = omega_dot(W)
    denom_norm_target = 1.0
    emb = embed(support, dim) if support is not None else (lambda x: x)

    def obj(x):
        eta = emb(x)
        A, B = blocks_from_eta(eta, skew)
        L = np.einsum("kij,nk->nij", A, W)
        P = np.einsum("kij,nk->nij", B, W)
        Ld = np.einsum("kij,nk->nij", A, Wd)
        C = L @ P - P @ L
        num = np.mean(np.sum((Ld - C) ** 2, axis=(1, 2)))
        den = np.mean(np.sum(Ld ** 2, axis=(1, 2))) + 1e-300
        loss = num / den
        nL = eta[:27].copy() if not skew else eta[:9]
        normpen = mu * (np.sum(nL ** 2) - denom_norm_target) ** 2
        J = (1 - delta) * loss + normpen + s * np.sum(np.abs(eta))
        if delta > 0:
            jf = 1.0
            for k in range(3):
                jf *= np.sum(A[k] ** 2) * np.sum(B[k] ** 2)
            J += delta / (jf + 1e-300)
        return J
    return obj


def pure_loss(eta, W, skew):
    return make_objective(W, skew, 0.0, 0.0, mu=0.0)(eta)


def classify(eta, W, skew, tol=1e-6):
    A, B = blocks_from_eta(eta, skew)
    res = pure_loss(eta, W, skew)
    out = {"residual": float(res)}
    # state rank in the linear coefficient basis
    Amat = A.reshape(3, 9).T
    sv = np.linalg.svd(Amat, compute_uv=False)
    state_rank = int(np.sum(sv > 1e-8 * max(sv[0], 1e-300)))
    out["state_rank"] = state_rank
    out["state_rank_P"] = int(np.sum(np.linalg.svd(B.reshape(3, 9).T, compute_uv=False) > 1e-8 * max(sv[0], 1e-300)))
    skewness = max(np.linalg.norm(A[k] + A[k].T) for k in range(3)) / (np.linalg.norm(A) + 1e-300)
    skewP = max(np.linalg.norm(B[k] + B[k].T) for k in range(3)) / (np.linalg.norm(B) + 1e-300)
    out["skew_defect"] = float(max(skewness, skewP))
    is_skew = out["skew_defect"] < 1e-6
    converged = res < 1e-8
    classical = False
    in_family = False
    if converged and is_skew and state_rank == 3:
        # L = hat(M Omega), P = hat(N Omega).  Exact solution family (see
        # euler_rigidity.py): M = R diag(m), N = R(n0 + lam diag(m)) with
        # q = m*m in the pencil alpha*I^2 + beta*I of conserved quadratics.
        # Signature: M^T M = diag(q) with q in the pencil, M^T N diagonal.
        M = np.stack([np.array([A[k][2, 1], A[k][0, 2], A[k][1, 0]]) for k in range(3)], axis=1)
        N = np.stack([np.array([B[k][2, 1], B[k][0, 2], B[k][1, 0]]) for k in range(3)], axis=1)
        G = M.T @ M
        q = np.diag(G)
        e0 = np.linalg.norm(G - np.diag(q)) / np.linalg.norm(G)
        X = np.stack([I ** 2, I], axis=1)
        ab, *_ = np.linalg.lstsq(X, q, rcond=None)
        e1 = np.linalg.norm(X @ ab - q) / np.linalg.norm(q)
        H = M.T @ N
        e2 = np.linalg.norm(H - np.diag(np.diag(H))) / (np.linalg.norm(H) + 1e-300)
        in_family = (e0 < 1e-5) and (e1 < 1e-5) and (e2 < 1e-5)
        ab = ab / np.linalg.norm(ab)
        out["pencil_alpha_beta"] = [float(ab[0]), float(ab[1])]
        out["family_defect"] = float(max(e0, e1, e2))
        classical = in_family and abs(ab[1]) < 1e-5
    if not converged:
        label = "nonconverged"
    elif state_rank < 3:
        label = "state-degenerate"
    elif not is_skew:
        label = "non-skew"
    elif classical:
        label = "classical-skeleton"
    elif in_family:
        label = "pencil-family"
    else:
        label = "other-skew"
    out["label"] = label
    out["nnz"] = int(np.sum(eta != 0))
    return out


PROTOCOLS = {
    "A": dict(skew=False, jfull=False),
    "B": dict(skew=True, jfull=False),
    "C": dict(skew=True, jfull=True),
    "D": dict(skew=False, jfull=True),
}
DELTA, S, W_THRESH = 0.1, 1e-2, 100.0


def one_run(args):
    proto, seed = args
    cfg = PROTOCOLS[proto]
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1, 1, size=(NPTS, 3))
    skew = cfg["skew"]
    dim = 18 if skew else 54
    d0 = DELTA if cfg["jfull"] else 0.0
    stages = [dict(delta=d0, s=S),
              dict(delta=d0 * 1e-2, s=0.0),
              dict(delta=d0 * 1e-5, s=0.0)]

    def make_obj(st, support=None):
        return make_objective(W, skew, st["delta"], st["s"], support=support, dim=dim)

    t = time.time()
    eta, hist = run_pipeline(make_obj, dim, rng, stages, W_THRESH,
                             ce_kwargs=dict(lb=-3, ub=3, iters=60))
    eta = refit_pure(lambda supp: make_objective(W, skew, 0.0, 0.0, mu=0.0, support=supp, dim=dim), eta)
    info = classify(eta, W, skew)
    info.update(protocol=proto, seed=int(seed), time=time.time() - t, eta=eta.tolist())
    return info


def one_run_logged(args):
    info = one_run(args)
    with open(os.path.join(OUT, "progress_euler.jsonl"), "a") as fh:
        fh.write(json.dumps(info) + "\n")
    return info


if __name__ == "__main__":
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    jobs = [(p, s) for p in "ABC" for s in range(nseeds)] + [("D", s) for s in range(min(nseeds, 20))]
    with Pool(2) as pool:
        results = list(pool.imap_unordered(one_run_logged, jobs))
    json.dump(results, open(os.path.join(OUT, "results_euler.json"), "w"), indent=1)
    for p in "ABCD":
        rs = [r for r in results if r["protocol"] == p]
        labels = {}
        for r in rs:
            labels[r["label"]] = labels.get(r["label"], 0) + 1
        conv = [r["residual"] for r in rs if r["label"] != "nonconverged"]
        print(p, PROTOCOLS[p], labels,
              "median residual %.2e max %.2e" % (np.median(conv), np.max(conv)) if conv else "")
