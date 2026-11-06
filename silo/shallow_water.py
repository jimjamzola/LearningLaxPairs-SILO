"""Shallow water: SILO in the truncated Laurent ansatz (paper eq. 5.7-5.10),
port of ShallowWater.m with a multi-seed protocol.

L(p;u,rho) = sum_{m=-M-}^{M+} c_m(u,rho) p^m,  c_1 == 1,
B = (1/2 L^2)_{>0}, residuals R1, R2 of paper (5.8), penalty (5.10).
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

A_SIGN = 1.0
BOX = dict(umin=-1, umax=1, rhomin=0.1, rhomax=1.5, pmin=0.25, pmax=2.0)


class Ansatz:
    def __init__(self, Mminus, Mplus, Nu, Nrho):
        self.Mminus, self.Mplus, self.Nu, self.Nrho = Mminus, Mplus, Nu, Nrho
        self.mvec = np.arange(-Mminus, Mplus + 1)
        self.block = (Nu + 1) * (Nrho + 1)
        self.free_blocks = [i for i, m in enumerate(self.mvec) if m != 1]
        self.dim = len(self.free_blocks) * self.block

    def theta(self, eta):
        T = np.zeros((len(self.mvec), self.Nu + 1, self.Nrho + 1))
        T[self.mvec == 1, 0, 0] = 1.0
        for n, i in enumerate(self.free_blocks):
            T[i] = eta[n * self.block:(n + 1) * self.block].reshape(self.Nu + 1, self.Nrho + 1, order="F")
        return T

    def eta_of_theta(self, T):
        eta = np.zeros(self.dim)
        for n, i in enumerate(self.free_blocks):
            eta[n * self.block:(n + 1) * self.block] = T[i].ravel(order="F")
        return eta

    def classical_eta(self):
        T = np.zeros((len(self.mvec), self.Nu + 1, self.Nrho + 1))
        T[self.mvec == 1, 0, 0] = 1.0
        T[self.mvec == 0, 1, 0] = 1.0
        T[self.mvec == -1, 0, 1] = A_SIGN
        return self.eta_of_theta(T)


def residuals(T, ans, u, rho, p):
    """Vectorized R1, R2 and (L_u, L_rho) at sample points."""
    Nu, Nrho = ans.Nu, ans.Nrho
    U = u[:, None] ** np.arange(Nu + 1)                  # (N, Nu+1)
    Rh = rho[:, None] ** np.arange(Nrho + 1)
    dU = np.zeros_like(U); dU[:, 1:] = np.arange(1, Nu + 1) * u[:, None] ** np.arange(Nu)
    dR = np.zeros_like(Rh); dR[:, 1:] = np.arange(1, Nrho + 1) * rho[:, None] ** np.arange(Nrho)
    c = np.einsum("ni,mij,nj->nm", U, T, Rh)             # (N, nblocks)
    cu = np.einsum("ni,mij,nj->nm", dU, T, Rh)
    cr = np.einsum("ni,mij,nj->nm", U, T, dR)
    mvec = ans.mvec
    P = p[:, None] ** mvec                               # (N, nblocks)
    L = (c * P).sum(1); Lu = (cu * P).sum(1); Lr = (cr * P).sum(1)
    Lp = (c * mvec * p[:, None] ** (mvec - 1)).sum(1)
    # B = (1/2 L^2)_{>0}: convolution of coefficient rows
    nb = len(mvec)
    expconv = np.arange(2 * mvec[0], 2 * mvec[-1] + 1)
    cc = np.zeros((len(u), 2 * nb - 1)); ccu = np.zeros_like(cc); ccr = np.zeros_like(cc)
    for i in range(nb):
        for j in range(nb):
            cc[:, i + j] += 0.5 * c[:, i] * c[:, j]
            ccu[:, i + j] += cu[:, i] * c[:, j]
            ccr[:, i + j] += cr[:, i] * c[:, j]
    keep = expconv >= 1
    nv = expconv[keep]
    Pn = p[:, None] ** nv
    Bu = (ccu[:, keep] * Pn).sum(1); Br = (ccr[:, keep] * Pn).sum(1)
    Bp = (cc[:, keep] * nv * p[:, None] ** (nv - 1)).sum(1)
    R1 = -u * Lu - rho * Lr + (Bp * Lu - Bu * Lp)
    R2 = -A_SIGN * Lu - u * Lr + (Bp * Lr - Br * Lp)
    return R1, R2, Lu, Lr


def make_objective(ans, samples, r, s, support=None):
    u, rho, p = samples.T
    emb = embed(support, ans.dim) if support is not None else (lambda x: x)

    def obj(x):
        eta = emb(x)
        T = ans.theta(eta)
        R1, R2, Lu, Lr = residuals(T, ans, u, rho, p)
        loss = np.mean(R1 ** 2 + R2 ** 2)
        degen = np.mean(Lu ** 2) * np.mean(Lr ** 2)
        return (1 - r) * loss + r / (degen + 1e-14) + s * np.sum(np.abs(eta))
    return obj


def pure_loss(ans, eta, samples):
    return make_objective(ans, samples, 0.0, 0.0)(eta)


def classify(ans, eta, samples):
    T = ans.theta(eta)
    res = pure_loss(ans, eta, samples)
    Tc = ans.theta(ans.classical_eta())
    out = {"residual": float(res), "nnz": int(np.sum(eta != 0))}
    scale = max(np.max(np.abs(T)), 1e-300)
    if res > 1e-10:
        out["label"] = "nonconverged"
        return out
    if np.max(np.abs(T[:, 1:, :])) < 1e-6 * scale and np.max(np.abs(T[:, :, 1:])) < 1e-6 * scale:
        out["label"] = "field-free L = l(p)"
        return out
    d = T - Tc
    out["deviation_from_BD"] = float(np.max(np.abs(d)))
    if np.max(np.abs(d)) < 1e-5 * scale:
        out["label"] = "Brunelli-Das"
        return out
    # constant shift only?
    d2 = d.copy(); d2[ans.mvec == 0, 0, 0] = 0
    if np.max(np.abs(d2)) < 1e-5 * scale:
        out["label"] = "Brunelli-Das + constant shift c0"
        out["shift"] = float(d[ans.mvec == 0, 0, 0][0])
        return out
    out["label"] = "other converged"
    return out


W_THRESH = 100.0


def one_run(args):
    cfg, seed = args
    ans = Ansatz(*cfg["window"])
    rng = np.random.default_rng(seed)
    N = 250
    u = rng.uniform(BOX["umin"], BOX["umax"], N)
    rho = rng.uniform(BOX["rhomin"], BOX["rhomax"], N)
    p = np.where(rng.random(N) > 0.5, 1, -1) * rng.uniform(BOX["pmin"], BOX["pmax"], N)
    samples = np.stack([u, rho, p], 1)
    r0 = cfg["r"]
    stages = [dict(r=r0, s=1e-3), dict(r=r0 * 1e-2, s=0.0), dict(r=r0 * 1e-5, s=0.0)]

    def make_obj(st, support=None):
        return make_objective(ans, samples, st["r"], st["s"], support=support)

    t = time.time()
    eta, hist = run_pipeline(make_obj, ans.dim, rng, stages, W_THRESH,
                             ce_kwargs=dict(lb=-5, ub=5, iters=cfg.get("ce_iters", 60)))
    eta = refit_pure(lambda supp: make_objective(ans, samples, 0.0, 0.0, support=supp), eta)
    info = classify(ans, eta, samples)
    info.update(config=cfg["name"], seed=int(seed), time=time.time() - t, eta=eta.tolist())
    return info


CONFIGS = {
    "base": dict(name="base", window=(2, 1, 2, 2), r=0.1, nseeds=40),
    "base_nopenalty": dict(name="base_nopenalty", window=(2, 1, 2, 2), r=0.0, nseeds=10),
    "enl3": dict(name="enl3", window=(3, 3, 3, 3), r=0.1, nseeds=8),
    "enl5": dict(name="enl5", window=(5, 5, 3, 3), r=0.1, nseeds=3, ce_iters=40),
}

def one_run_logged(args):
    info = one_run(args)
    with open(os.path.join(OUT, "progress_shallow.jsonl"), "a") as fh:
        fh.write(json.dumps(info) + "\n")
    return info


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS)
    # Each argument is a config name, optionally followed by :N to run seeds 0..N-1
    # instead of the config default (e.g. "base:28").
    spec = [(a.split(":")[0], int(a.split(":")[1]) if ":" in a else None) for a in names]
    names = [n for n, _ in spec]
    jobs = [(CONFIGS[n], s) for n, k in spec for s in range(k if k is not None else CONFIGS[n]["nseeds"])]
    with Pool(2) as pool:
        results = list(pool.imap_unordered(one_run_logged, jobs))
    json.dump(results, open(os.path.join(OUT, "results_shallow_%s.json" % "_".join(names)), "w"), indent=1)
    for n in names:
        rs = [r for r in results if r["config"] == n]
        labels = {}
        for r in rs:
            labels[r["label"]] = labels.get(r["label"], 0) + 1
        print(n, labels)
