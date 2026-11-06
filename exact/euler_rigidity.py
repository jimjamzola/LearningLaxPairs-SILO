"""Exact infinitesimal rigidity of the classical Euler-top skeleton.

Ansatz: L = sum_k Omega_k A_k, P = sum_k Omega_k B_k with constant 3x3 A_k, B_k.
Compatibility as a polynomial identity in Omega:
    sum_k A_k Ddot_k(Omega) = [L(Omega), P(Omega)]     (54 scalar equations)
Skew ansatz: A_k = hat(M[:,k]), B_k = hat(N[:,k]) (18 unknowns, 18 equations).

We compute the Jacobian of the polynomial system at the classical pair
(M = diag(I), N = Id) over Q and compare its nullity with the dimension of the
gauge orbit through that point.  Nullity == gauge dimension means the classical
skeleton is infinitesimally rigid modulo gauge in that hypothesis class.
"""
import sympy as sp
import itertools

I1, I2, I3 = sp.Rational(6, 5), sp.Rational(23, 10), sp.Rational(31, 10)
Ivals = [I1, I2, I3]
w = sp.symbols("w1:4")
wdot = [(I2 - I3) / I1 * w[1] * w[2], (I3 - I1) / I2 * w[0] * w[2], (I1 - I2) / I3 * w[0] * w[1]]


def hat(v):
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def equations(A, B):
    L = sum((w[k] * A[k] for k in range(3)), sp.zeros(3))
    P = sum((w[k] * B[k] for k in range(3)), sp.zeros(3))
    Ld = sum((wdot[k] * A[k] for k in range(3)), sp.zeros(3))
    R = sp.expand(Ld - (L * P - P * L))
    eqs = []
    monos = [w[i] * w[j] for i in range(3) for j in range(i, 3)]
    for i in range(3):
        for j in range(3):
            poly = sp.Poly(R[i, j], *w)
            for m in monos:
                eqs.append(poly.coeff_monomial(m))
    return eqs


def analyze_skew():
    Ms = sp.Matrix(3, 3, sp.symbols("m1:10"))
    Ns = sp.Matrix(3, 3, sp.symbols("n1:10"))
    A = [hat(Ms[:, k]) for k in range(3)]
    B = [hat(Ns[:, k]) for k in range(3)]
    eqs = equations(A, B)
    vars_ = list(Ms) + list(Ns)
    Jac = sp.Matrix(eqs).jacobian(vars_)
    subs = {}
    for i in range(3):
        for j in range(3):
            subs[Ms[i, j]] = Ivals[i] if i == j else 0
            subs[Ns[i, j]] = 1 if i == j else 0
    J0 = Jac.subs(subs)
    eq0 = [e.subs(subs) for e in eqs]
    assert all(sp.simplify(e) == 0 for e in eq0)
    rank = J0.rank()
    print("Skew ansatz: %d equations, %d unknowns, Jacobian rank %d, nullity %d"
          % (len(eqs), len(vars_), rank, len(vars_) - rank))
    # gauge tangents: L -> cL (1), P -> P + lam L (1), rotation conjugation (3)
    Mc, Nc = sp.diag(*Ivals), sp.eye(3)
    G = []
    G.append(list(Mc) + [0] * 9)                       # scale L
    G.append([0] * 9 + list(Mc))                       # P += lam L
    for k in range(3):
        Rk = hat([1 if i == k else 0 for i in range(3)])
        G.append(list(Rk * Mc) + list(Rk * Nc))        # d/dt exp(tR) M
    G = sp.Matrix(G).T
    assert (J0 * G).is_zero_matrix
    print("  gauge subspace dimension:", G.rank())
    return len(vars_) - rank, G.rank()


def analyze_full():
    Asym = [sp.Matrix(3, 3, sp.symbols("a%d_1:10" % k)) for k in range(3)]
    Bsym = [sp.Matrix(3, 3, sp.symbols("b%d_1:10" % k)) for k in range(3)]
    eqs = equations(Asym, Bsym)
    vars_ = [x for A in Asym for x in A] + [x for B in Bsym for x in B]
    Jac = sp.Matrix(eqs).jacobian(vars_)
    Mc, Nc = sp.diag(*Ivals), sp.eye(3)
    A0 = [hat(Mc[:, k]) for k in range(3)]
    B0 = [hat(Nc[:, k]) for k in range(3)]
    subs = {}
    for k in range(3):
        for i in range(9):
            subs[Asym[k][i]] = A0[k][i]
            subs[Bsym[k][i]] = B0[k][i]
    J0 = Jac.subs(subs)
    rank = J0.rank()
    print("Unconstrained ansatz: %d equations, %d unknowns, Jacobian rank %d, nullity %d"
          % (len(eqs), len(vars_), rank, len(vars_) - rank))
    # gauge tangents: gl(3) conjugation mod scalars (8), L scale (1),
    # P += lam L (1), P += sum_k c_k Omega_k Id (3)
    G = []
    Eij = [sp.Matrix(3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0) for i in range(3) for j in range(3)]
    for X in Eij[:-1]:
        G.append([x for k in range(3) for x in (X * A0[k] - A0[k] * X)] +
                 [x for k in range(3) for x in (X * B0[k] - B0[k] * X)])
    G.append([x for k in range(3) for x in A0[k]] + [0] * 27)
    G.append([0] * 27 + [x for k in range(3) for x in A0[k]])
    for k in range(3):
        v = [0] * 54
        for i in range(3):
            v[27 + 9 * k + 4 * i] = 1
        G.append(v)
    G = sp.Matrix(G).T
    assert (J0 * G).is_zero_matrix, "gauge tangent not in kernel"
    print("  gauge subspace dimension:", G.rank())
    return len(vars_) - rank, G.rank()


if __name__ == "__main__":
    ns, gs = analyze_skew()
    nf, gf = analyze_full()
    print("RESULT skew: nullity %d vs gauge %d -> %s" % (ns, gs, "rigid" if ns == gs else "NOT rigid"))
    print("RESULT full: nullity %d vs gauge %d -> %s" % (nf, gf, "rigid" if nf == gf else "NOT rigid"))
