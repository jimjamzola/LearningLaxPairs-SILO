"""Exact solution set of the Clairaut constraints (paper eq. 3.4) within the
ansatz a_j(k) = z_{0j} + z_{1j} k + z_{2j} k^2.

    i a1 - a4 = 0,   a2 a3 - a5 a1 = 0,   a2 a4 - a3 = 0     (identically in k)

Hand derivation: a4 = i a1, a3 = i a1 a2, and a1 (i a2^2 - a5) = 0.
So either a1 = 0 (field-free) or a5 = i a2^2.  Writing A = a1, B = a2,
degree constraints force deg B <= 1 and deg A + deg B <= 2.
This script confirms the decomposition with sympy and evaluates rank Z on each
component.
"""
import sympy as sp

k = sp.symbols("k")
z = sp.symbols("z0:15")   # z[3*j + l] = coefficient of k^l in a_{j+1}
a = [sum(z[3 * j + l] * k ** l for l in range(3)) for j in range(5)]
a1, a2, a3, a4, a5 = a
polys = [sp.expand(sp.I * a1 - a4), sp.expand(a2 * a3 - a5 * a1), sp.expand(a2 * a4 - a3)]
eqs = []
for p in polys:
    P = sp.Poly(p, k)
    eqs += [c for c in P.all_coeffs()]
eqs = [e for e in eqs if e != 0]
print("number of scalar equations:", len(eqs))

G = sp.groebner(eqs, *z, order="grevlex")
print("Groebner basis size:", len(G.exprs))

# Verify the claimed components generate the same variety:
# C0: a1 = 0 (z0=z1=z2=0), then a3 = a4 = 0 forced by eq 3 and eq 1.
# C1: a5 = i a2^2, a4 = i a1, a3 = i a1 a2, with deg constraints.
A0, A1, A2 = sp.symbols("A0 A1 A2")
B0, B1 = sp.symbols("B0 B1")
Apoly = A0 + A1 * k + A2 * k ** 2
Bpoly = B0 + B1 * k
sol = {a1: Apoly, a2: Bpoly, a3: sp.I * Apoly * Bpoly, a4: sp.I * Apoly, a5: sp.I * Bpoly ** 2}
for name, extra in (("stratum (iii) A quadratic, B const", {B1: 0}), ("stratum (iv) A linear, B linear", {A2: 0})):
    solx = {aj: sp.expand(v.subs(extra)) for aj, v in sol.items()}
    subsz = {z[3 * j + l]: sp.Poly(solx[a[j]], k).coeff_monomial(k ** l) for j in range(5) for l in range(3)}
    print(name, "-> constraint residuals:", [sp.simplify(p.subs(subsz)) for p in polys])

print("\nStrata of the solution set (exact):")
print(" (i)  a1 = a3 = a4 = 0, a2 and a5 arbitrary: field-free auxiliary system.")
print(" (ii) B = 0: a2 = a5 = a3 = 0, a4 = i a1, A arbitrary quadratic -> zero-parameter pair mu_x = A q, mu_t = i A q_x.")
print(" (iii) B = b0 nonzero constant: a2 = b0, a5 = i b0^2, a3 = a4 = i A (up to b0), A quadratic -> collapsed family (3.5).")
print(" (iv) B = b0 + b1 k, b1 != 0: A linear -> UTM class; gauge mu -> A(k) mu and k -> (k - b0)/b1 reduce it to the classical pair.")

# Rank of Z on each stratum
def Zmat(subsd):
    rows = []
    for j in range(5):
        pj = sp.Poly(sp.expand(sol[a[j]].subs(subsd)), k)
        rows.append([pj.coeff_monomial(k ** l) for l in range(3)])
    return sp.Matrix(rows)

print("\nrank Z on stratum (iii), generic quadratic A, B = b0:",
      Zmat({B1: 0}).rank(simplify=True))
print("rank Z on stratum (iv), generic linear A, B = b0 + b1 k:",
      Zmat({A2: 0}).rank(simplify=True))
print("rank Z on stratum (ii), generic quadratic A, B = 0:",
      Zmat({B0: 0, B1: 0}).rank(simplify=True))
Zcl = Zmat({A0: 1, A1: 0, A2: 0, B0: 0, B1: sp.I})
print("classical Z:", Zcl.T, " det(Z^H Z) =", sp.simplify((Zcl.H * Zcl).det()),
      " tr((Z^H Z)^-1) =", sp.simplify((Zcl.H * Zcl).inv().trace()))
