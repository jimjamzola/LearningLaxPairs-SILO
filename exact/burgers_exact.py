"""Exact classification of polynomial solutions of the Clairaut identity
    2 s g_u + f'(u) (g_s + u) = 0
within the Hopf.m ansatz (deg f <= 3, g of bidegree (4,2) in (s,u)).

Result: every solution with f' != 0 has f linear, f = a u + lambda, and
    g = 2 s^3/(3a) - u s + Phi(s^2 - a u),   Phi polynomial of degree <= 2.
(The general statement, for any polynomial degrees, follows from the
s-degree recursion in the write-up.)
"""
import sympy as sp

s, u = sp.symbols("s u")
xi = sp.symbols("xi0:4")
zeta = sp.symbols("zeta0:15")
f = sum(xi[j] * u ** j for j in range(4))
g = sum(zeta[5 * k + j] * s ** j * u ** k for j in range(5) for k in range(3))
R = sp.expand(2 * s * sp.diff(g, u) + sp.diff(f, u) * (sp.diff(g, s) + u))
eqs = sp.Poly(R, s, u).coeffs()
print("scalar equations:", len(eqs), " unknowns:", 4 + 15)

# Case analysis on deg f.  For each fixed leading behaviour, the equations are
# linear in zeta; we solve them and see what they force on xi.
for degf in (3, 2, 1):
    sub = {xi[j]: 0 for j in range(degf + 1, 4)}
    eqs_d = [sp.expand(e.subs(sub)) for e in eqs]
    eqs_d = [e for e in eqs_d if e != 0]
    sol = sp.solve(eqs_d, list(zeta) + list(xi[1:degf + 1]), dict=True)
    print("\ndeg f <= %d (xi_%d generic):" % (degf, degf))
    for so in sol:
        forced = {k: v for k, v in so.items() if k in xi}
        print("  forced on xi:", forced if forced else "none")
        gsol = sp.factor(g.subs(so))
        if not forced:
            print("  general g:", sp.collect(sp.expand(g.subs(so)), s))
    if degf == 1:
        so = [x for x in sol if x.get(xi[1], 1) != 0][0]
        a, lam = so.get(xi[1], xi[1]), xi[0]
        gsol = sp.expand(g.subs(so))
        base = 2 * s ** 3 / (3 * a) - u * s
        h = sp.expand(gsol - base)
        # h should be a polynomial in xi_ = s^2 - a u of degree <= 2
        c0, c1, c2 = sp.symbols("c0 c1 c2")
        X = s ** 2 - a * u
        diff = sp.expand(h - (c0 + c1 * X + c2 * X ** 2))
        csol = sp.solve(sp.Poly(diff, s, u).coeffs(), [c0, c1, c2], dict=True)
        print("  h = g - g0 is Phi(s^2 - a u) with Phi coefficients:", csol)
