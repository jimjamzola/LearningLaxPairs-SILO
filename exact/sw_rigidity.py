"""Exact formal rigidity of the Brunelli-Das pair in polynomial-Laurent ansatze.

Residual (L and B independent unknowns):
    R(L,B) = L_t - {L, B},   L_t = L_u u_t + L_rho rho_t on shallow water.
rho mode:  R1 = -u L_u - rho L_rho - (L_p B_u - L_u B_p)
           R2 = -a L_u - u L_rho  - (L_p B_rho - L_rho B_p)
us  mode (s = sqrt(a rho), so s^2 = a rho):
           R1 = -u L_u - (s/2) L_s - (L_p B_u - L_u B_p)
           R2 = -2 s L_u - u L_s   - (L_p B_s - L_s B_p)
R is affine in L for fixed B and linear in B for fixed L, so
    R(eta_cl + v) = J v + Q(v, v),   Q(v,w) = -1/2 ({v_L, w_B} + {w_L, v_B}).
Everything is exact over Q (python Fractions + sympy DomainMatrix).

Beyond the per-direction obstruction ladder of ShallowWaterRigidity.m, this
script computes the second-order obstruction as a quadratic map on the
non-gauge tangent space N' and analyzes its zero cone exactly, then treats
third order on that cone.
"""
import sys, time, itertools
from fractions import Fraction as Fr
import sympy as sp
from sympy.polys.matrices import DomainMatrix
from sympy import QQ

A_SIGN = 1


# ---------- exact Laurent polynomials: dict {(j,k,m): Fraction} ----------
def padd(p, q, c=1):
    r = dict(p)
    for key, v in q.items():
        r[key] = r.get(key, Fr(0)) + c * v
        if r[key] == 0:
            del r[key]
    return r


def pmul(p, q):
    r = {}
    for (j1, k1, m1), v1 in p.items():
        for (j2, k2, m2), v2 in q.items():
            key = (j1 + j2, k1 + k2, m1 + m2)
            r[key] = r.get(key, Fr(0)) + v1 * v2
    return {k: v for k, v in r.items() if v != 0}


def pscale(p, c):
    return {k: c * v for k, v in p.items() if c * v != 0}


def d_u(p):
    return {(j - 1, k, m): j * v for (j, k, m), v in p.items() if j > 0}


def d_v(p):
    return {(j, k - 1, m): k * v for (j, k, m), v in p.items() if k > 0}


def d_p(p):
    return {(j, k, m - 1): m * v for (j, k, m), v in p.items() if m != 0}


U = {(1, 0, 0): Fr(1)}
V = {(0, 1, 0): Fr(1)}
P = {(0, 0, 1): Fr(1)}
ONE = {(0, 0, 0): Fr(1)}



def residual(L, B, mode):
    Lu, Lv, Lp = d_u(L), d_v(L), d_p(L)
    Bu, Bv, Bp = d_u(B), d_v(B), d_p(B)
    if mode == "rho":
        T1 = padd(pscale(pmul(U, Lu), -1), pscale(pmul(V, Lv), -1))
        T2 = padd(pscale(Lu, -A_SIGN), pscale(pmul(U, Lv), -1))
    else:
        T1 = padd(pscale(pmul(U, Lu), -1), pscale(pmul(V, Lv), Fr(-1, 2)))
        T2 = padd(pscale(pmul(V, Lu), -2), pscale(pmul(U, Lv), -1))
    R1 = padd(T1, padd(pmul(Lp, Bu), pscale(pmul(Lu, Bp), -1)), -1)
    R2 = padd(T2, padd(pmul(Lp, Bv), pscale(pmul(Lv, Bp), -1)), -1)
    return R1, R2


class Ansatz:
    def __init__(self, name, mL, mB, NuL, NvL, NuB, NvB, mode):
        self.name, self.mode = name, mode
        self.Lkeys = [(j, k, m) for m in mL for j in range(NuL + 1) for k in range(NvL + 1)]
        self.Bkeys = [(j, k, m) for m in mB for j in range(NuB + 1) for k in range(NvB + 1)]
        self.nL, self.nB = len(self.Lkeys), len(self.Bkeys)
        self.dim = self.nL + self.nB
        self.mL, self.mB, self.NuL, self.NvL, self.NuB, self.NvB = mL, mB, NuL, NvL, NuB, NvB

    def LB(self, vec):
        L = {self.Lkeys[i]: vec[i] for i in range(self.nL) if vec[i] != 0}
        B = {self.Bkeys[i]: vec[self.nL + i] for i in range(self.nB) if vec[self.nL + i] != 0}
        return L, B

    def vec(self, L, B):
        v = [Fr(0)] * self.dim
        for key, c in L.items():
            assert key in self.Lkeys, ("L term outside ansatz", key)
            v[self.Lkeys.index(key)] = c
        for key, c in B.items():
            assert key in self.Bkeys, ("B term outside ansatz", key)
            v[self.nL + self.Bkeys.index(key)] = c
        return v

    def fits(self, L, B):
        return all(k in self.Lkeys for k in L) and all(k in self.Bkeys for k in B)

    def classical(self):
        if self.mode == "rho":
            L = {(0, 0, 1): Fr(1), (1, 0, 0): Fr(1), (0, 1, -1): Fr(A_SIGN)}
        else:
            L = {(0, 0, 1): Fr(1), (1, 0, 0): Fr(1), (0, 2, -1): Fr(1)}
        B = {(0, 0, 2): Fr(1, 2), (1, 0, 1): Fr(1)}
        return L, B


class Model:
    """Exact linearization data at eta_cl."""

    def __init__(self, ans):
        self.ans = ans
        self.Lc, self.Bc = ans.classical()
        R1, R2 = residual(self.Lc, self.Bc, ans.mode)
        assert not R1 and not R2, "classical pair is not a solution"
        self.rowidx = {}
        # register every monomial the bilinear part can produce, so that rows
        # of Q outside the range of J are represented (as zero rows of J)
        for i in range(ans.nL):
            for j in range(ans.nB):
                v = [Fr(0)] * ans.dim; v[i] = Fr(1)
                w = [Fr(0)] * ans.dim; w[ans.nL + j] = Fr(1)
                self.Q(v, w)
        # Jacobian columns: R(cl + e_q) exactly (affine in each block)
        cols = []
        for q in range(ans.dim):
            e = [Fr(0)] * ans.dim
            e[q] = Fr(1)
            dL, dB = ans.LB(e)
            R1, R2 = residual(padd(self.Lc, dL), padd(self.Bc, dB), ans.mode)
            cols.append(self.rows(R1, R2))
        self.nrows = len(self.rowidx)
        self.J = self.to_dm(cols, self.nrows)

    def rows(self, R1, R2):
        out = {}
        for comp, R in ((1, R1), (2, R2)):
            for key, v in R.items():
                rk = (comp,) + key
                if rk not in self.rowidx:
                    self.rowidx[rk] = len(self.rowidx)
                out[self.rowidx[rk]] = v
        return out

    def to_dm(self, cols, nrows):
        M = [[QQ(0)] * len(cols) for _ in range(nrows)]
        for q, col in enumerate(cols):
            for r, v in col.items():
                M[r][q] = QQ(v.numerator, v.denominator)
        return DomainMatrix(M, (nrows, len(cols)), QQ)

    def Q(self, v, w):
        """Symmetric bilinear part: -1/2({v_L,w_B} + {w_L,v_B}) as a row-dict."""
        vL, vB = self.ans.LB(v)
        wL, wB = self.ans.LB(w)

        def br(L, B):
            R1 = padd(pmul(d_p(L), d_u(B)), pscale(pmul(d_u(L), d_p(B)), -1))
            R2 = padd(pmul(d_p(L), d_v(B)), pscale(pmul(d_v(L), d_p(B)), -1))
            return R1, R2
        a1, a2 = br(vL, wB)
        b1, b2 = br(wL, vB)
        R1 = pscale(padd(a1, b1), Fr(-1, 2))
        R2 = pscale(padd(a2, b2), Fr(-1, 2))
        return self.rows(R1, R2)

    def vecdm(self, rowdict):
        col = [[QQ(0)] for _ in range(self.nrows)]
        for r, v in rowdict.items():
            col[r][0] = QQ(v.numerator, v.denominator)
        return DomainMatrix(col, (self.nrows, 1), QQ)


def dm_cols(dm):
    M = dm.to_Matrix()
    return [list(M[:, i]) for i in range(M.shape[1])]


def hodograph_flux(F, mode):
    """Given a conserved density F(u,v) (v = rho or s), return the flux G with
    F_t = G_x on shallow water (a = A_SIGN), or None if F is not conserved."""
    u, v = sp.symbols("u v")
    if mode == "rho":
        ut, vt = -(u * sp.Symbol("ux") + A_SIGN * sp.Symbol("vx")), None
        # F_t = -(u F_u + rho F_rho) u_x - (a F_u + u F_rho) rho_x
        Gu = -(u * sp.diff(F, u) + v * sp.diff(F, v))
        Gv = -(A_SIGN * sp.diff(F, u) + u * sp.diff(F, v))
    else:
        # s = sqrt(a rho): s_t = -(s/2) u_x - u s_x,  u_t = -u u_x - 2 s s_x
        Gu = -(u * sp.diff(F, u) + (v / 2) * sp.diff(F, v))
        Gv = -(2 * v * sp.diff(F, u) + u * sp.diff(F, v))
    if sp.simplify(sp.diff(Gu, v) - sp.diff(Gv, u)) != 0:
        return None
    G = sp.integrate(Gu, u)
    G = G + sp.integrate(sp.expand(Gv - sp.diff(G, v)), v)
    assert sp.simplify(sp.diff(G, u) - Gu) == 0 and sp.simplify(sp.diff(G, v) - Gv) == 0
    return sp.expand(G)


def sym_to_dict(expr):
    """sympy polynomial in u, v, p (Laurent in p) -> {(j,k,m): Fraction}."""
    u, v, p = sp.symbols("u v p")
    expr = sp.expand(expr)
    out = {}
    for term in sp.Add.make_args(expr):
        c, rest = term.as_coeff_Mul()
        pw = rest.as_powers_dict()
        j, k, m = int(pw.get(u, 0)), int(pw.get(v, 0)), int(pw.get(p, 0))
        out[(j, k, m)] = out.get((j, k, m), Fr(0)) + Fr(int(c.p), int(c.q))
    return {k: c for k, c in out.items() if c != 0}


def whitham_densities(mode, maxdeg=6):
    """Polynomial conserved densities of shallow water up to given degree
    (Kupershmidt-Manin / Whitham hierarchy), computed by solving the EPD
    condition on the polynomial ansatz."""
    u, v = sp.symbols("u v")
    out = []
    for d in range(0, maxdeg + 1):
        # homogeneous by weight: u has weight 1, rho weight 2 (s weight 1)
        monos = []
        if mode == "rho":
            monos = [u ** i * v ** k for i in range(d + 1) for k in range(d + 1) if i + 2 * k == d]
        else:
            monos = [u ** i * v ** k for i in range(d + 1) for k in range(d + 1) if i + k == d]
        if not monos:
            continue
        cs = sp.symbols("q0:%d" % len(monos))
        F = sum(c * m for c, m in zip(cs, monos))
        if mode == "rho":
            Gu = -(u * sp.diff(F, u) + v * sp.diff(F, v)); Gv = -(A_SIGN * sp.diff(F, u) + u * sp.diff(F, v))
        else:
            Gu = -(u * sp.diff(F, u) + (v / 2) * sp.diff(F, v)); Gv = -(2 * v * sp.diff(F, u) + u * sp.diff(F, v))
        cond = sp.Poly(sp.expand(sp.diff(Gu, v) - sp.diff(Gv, u)), u, v).coeffs()
        sol = sp.solve(cond, cs, dict=True)
        Fs = sp.expand(F.subs(sol[0])) if sol else F
        free = [c for c in cs if Fs.has(c)]
        for c in free:
            Fi = sp.expand(Fs.subs({cc: (1 if cc == c else 0) for cc in free}))
            if Fi != 0:
                out.append(Fi)
    return out


def gauge_vectors(ans):
    """Tangents of exact symmetry families of the dispersionless Lax equation
    through eta_cl that fit the ansatz:
      (a) L -> L + eps L^n              (n >= 0)
      (b) B -> B + eps L^n              (centralizer)
      (c) spectral rescaling  L(mu p)/mu, B(mu p)/mu
      (d) canonical shifts  p -> p + eps F(u,v),  B -> B + eps (F B_p - G)
          for every hodograph pair (S_x, S_t) = (F, G), F_t = G_x  (B += eps(F B_p + G)).
    """
    u, v, p = sp.symbols("u v p")
    Lc, Bc = ans.classical()
    G, names = [], []
    Lpow = dict(ONE)
    for n in range(0, 12):
        if ans.fits(Lpow, {}):
            G.append(ans.vec(Lpow, {})); names.append("L+=L^%d" % n)
        if ans.fits({}, Lpow):
            G.append(ans.vec({}, Lpow)); names.append("B+=L^%d" % n)
        Lpow = pmul(Lpow, Lc)
        if not any(k in ans.Lkeys or k in ans.Bkeys for k in Lpow):
            break
    dL = padd(pmul(P, d_p(Lc)), Lc, -1)
    dB = padd(pmul(P, d_p(Bc)), Bc, -1)
    G.append(ans.vec(dL, dB)); names.append("rescaling")
    Lsym = p + u + (A_SIGN * v if ans.mode == "rho" else v ** 2) / p
    Bsym = p ** 2 / 2 + u * p
    for F in whitham_densities(ans.mode):
        Gf = hodograph_flux(F, ans.mode)
        assert Gf is not None
        dL = sym_to_dict(F * sp.diff(Lsym, p))
        dB = sym_to_dict(F * sp.diff(Bsym, p) + Gf)
        if ans.fits(dL, dB):
            G.append(ans.vec(dL, dB)); names.append("p+=eps(%s)" % F)
    return G, names


def analyze(ans, verbose=True):
    t0 = time.time()
    M = Model(ans)
    J = M.J
    rank = J.rank()
    nullity = ans.dim - rank
    print("[%s] dim %d (L %d, B %d), residual rows %d, rank J = %d, nullity %d  (%.1fs)"
          % (ans.name, ans.dim, ans.nL, ans.nB, M.nrows, rank, nullity, time.time() - t0))
    G, gnames = gauge_vectors(ans)
    Gdm = DomainMatrix([[QQ(x.numerator, x.denominator) for x in g] for g in G], (len(G), ans.dim), QQ).transpose()
    assert (J * Gdm).is_zero_matrix, "a gauge tangent is not in ker J"
    gdim = Gdm.rank()
    print("  gauge tangents: %s -> dim %d" % (", ".join(gnames), gdim))
    # kernel basis and non-gauge complement
    K = J.nullspace().transpose()            # columns span ker J
    Kcols = dm_cols(K)
    Gm = Gdm.to_Matrix()
    cur = Gm
    Nprime = []
    for c in Kcols:
        cand = cur.row_join(sp.Matrix(c))
        if cand.rank() > cur.rank():
            Nprime.append(c); cur = cand
    print("  non-gauge tangent directions: %d" % len(Nprime))
    # cokernel: left null space rows
    Lc = J.transpose().nullspace()          # rows annihilate columns of J
    Lc_M = Lc.to_Matrix()
    print("  coker dim = %d" % Lc_M.shape[0])

    # second-order obstruction as quadratic map on N'
    n = len(Nprime)
    Np = [[Fr(int(x.p), int(x.q)) for x in col] for col in Nprime]
    Qij = {}
    for i in range(n):
        for j in range(i, n):
            Qij[(i, j)] = M.vecdm(M.Q(Np[i], Np[j])).to_Matrix()
    cs = sp.symbols("c0:%d" % n)
    ob2 = sp.zeros(Lc_M.shape[0], 1)
    for (i, j), q in Qij.items():
        coef = cs[i] * cs[j] if i == j else 2 * cs[i] * cs[j]
        ob2 += coef * (Lc_M * q)
    quadrics = [sp.expand(e) for e in ob2 if sp.expand(e) != 0]
    print("  second-order obstruction: %d nonzero quadratic forms in %d variables" % (len(quadrics), n))
    # per-direction ladder (as in the MATLAB script) for reference
    per_dir = []
    for i in range(n):
        v = sp.Matrix([0] * n); v[i] = 1
        per_dir.append(all(q.subs({cs[k]: v[k] for k in range(n)}) == 0 for q in quadrics))
    print("  basis directions unobstructed at order 2:", [i for i, ok in enumerate(per_dir) if ok])
    # zero cone of the quadratic map
    Gb = sp.groebner(quadrics, *cs, order="grevlex")
    print("  Groebner basis of the order-2 obstruction ideal:", Gb.exprs)
    return dict(model=M, Nprime=Np, cs=cs, quadrics=quadrics, groebner=Gb, Lc=Lc_M,
                nullity=nullity, gauge_dim=gdim, n_excess=n, rank=rank)


ANSATZE = {
    "I": Ansatz("I: (u,rho) bidegree (2,2), L in p^[-3,2], B in p^[0,3]", range(-3, 3), range(0, 4), 2, 2, 2, 2, "rho"),
    "II": Ansatz("II: (u,rho) bidegree (3,3), same p-windows", range(-3, 3), range(0, 4), 3, 3, 3, 3, "rho"),
    "III": Ansatz("III: (u,s) bidegree (2,4), same p-windows", range(-3, 3), range(0, 4), 2, 4, 2, 4, "us"),
}

if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] in ("I3", "II3")):
    names = sys.argv[1:] if len(sys.argv) > 1 else ["I", "III", "II"]
    out = {}
    for nm in names:
        out[nm] = analyze(ANSATZE[nm])


def particular_solver(J):
    """Return (row_idx, piv_cols, Jsub_inv) so that for b in range(J) the vector
    x with x[piv] = Jsub_inv * b[rows], zero elsewhere, solves J x = b."""
    Jm = J.to_Matrix()
    rref, piv = Jm.rref()
    piv = list(piv)
    # independent rows: rref of J^T gives pivot rows
    _, rowpiv = Jm.T.rref()
    rows = list(rowpiv)
    Jsub = Jm.extract(rows, piv)
    assert Jsub.rank() == len(piv)
    return rows, piv, Jsub.inv()


def third_order(res, cone_vars):
    """Order-3 analysis on the order-2 zero cone, assumed (after Groebner) to be
    the coordinate subspace spanned by the directions in cone_vars."""
    M, Np, cs, Lc = res["model"], res["Nprime"], res["cs"], res["Lc"]
    J = M.J
    Jm = J.to_Matrix()
    rows, piv, Jinv = particular_solver(J)
    ndim = M.ans.dim
    # kernel basis (full, includes gauge)
    K = J.nullspace().transpose().to_Matrix()
    nk = K.shape[1]
    ws = sp.symbols("w0:%d" % nk)
    cvars = [cs[i] for i in cone_vars]
    # v(c) as exact vector of linear forms
    v = sp.zeros(ndim, 1)
    for i in cone_vars:
        v += cs[i] * sp.Matrix(Np[i])
    # Q(v,v) as vector of quadratic forms: expand bilinearly on basis
    Qvv = sp.zeros(M.nrows, 1)
    for a in cone_vars:
        for b in cone_vars:
            Qab = M.vecdm(M.Q(Np[a], Np[b])).to_Matrix()
            Qvv += cs[a] * cs[b] * Qab
    # particular Delta2 (quadratic in c)
    rhs = -Qvv
    D2 = sp.zeros(ndim, 1)
    xp = Jinv * rhs.extract(rows, [0])
    for i, pc in enumerate(piv):
        D2[pc] = xp[i]
    assert (Jm * D2 - rhs).applyfunc(sp.expand).is_zero_matrix, "second-order solve failed on the cone"
    # Delta2 = D2 + K w ; order-3 RHS = -2 Q(v, Delta2); solvability: Lc * Q(v, Delta2) = 0
    # Q(v, D2): bilinear, expand v over basis and D2 over unit vectors
    def Qvec_of(x_sym):
        """Q(v, x) where x is a symbolic column vector, v = sum c_i Np[i]."""
        out = sp.zeros(M.nrows, 1)
        for a in cone_vars:
            for q in range(ndim):
                if x_sym[q] == 0:
                    continue
                e = [Fr(0)] * ndim; e[q] = Fr(1)
                out += cs[a] * x_sym[q] * M.vecdm(M.Q(Np[a], e)).to_Matrix()
        return out
    t = Lc * Qvec_of(D2)                                   # cubic in c
    Kw = K * sp.Matrix(ws)
    Mw = Lc * Qvec_of(Kw)                                  # bilinear in (c, w)
    eqs = [sp.expand(e) for e in (t + Mw) if sp.expand(e) != 0]
    print("  order-3 system: %d equations in %d cone variables + %d kernel variables" % (len(eqs), len(cvars), nk))
    # eliminate w: Groebner with lex order, w first
    t0 = time.time()
    Gb = sp.groebner(eqs, *ws, *cvars, order="lex")
    elim = [g for g in Gb.exprs if not any(g.has(w) for w in ws)]
    print("  eliminated ideal in cone variables (%.1fs):" % (time.time() - t0), elim)
    return elim, Gb


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "I3":
    res = analyze(ANSATZE["I"])
    third_order(res, [0, 1, 2])

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "II3":
    res = analyze(ANSATZE["II"])
    third_order(res, [1])
