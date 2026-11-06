"""Summarize the jsonl progress files into LaTeX table fragments and a text
summary used by the report."""
import json, collections, numpy as np, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Directory holding progress_*.jsonl; table_rows.tex is written to the same place.
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, os.pardir, "results")


def load(tag):
    fn = f"{D}/progress_{tag}.jsonl"
    if not os.path.exists(fn):
        return []
    return [json.loads(l) for l in open(fn)]


def fmt(x):
    return "%.1e" % x if x > 0 else "0"


out = []
# ---------------- Euler ----------------
rs = load("euler")
out.append("%% Euler top")
names = {"A": "unconstrained, no $J_{\\rm full}$", "B": "skew, no $J_{\\rm full}$",
         "C": "skew $+$ $J_{\\rm full}$ (eq.~2.9)", "D": "unconstrained $+$ $J_{\\rm full}$"}
for p in "ABCD":
    sub = [r for r in rs if r["protocol"] == p]
    if not sub:
        continue
    c = collections.Counter(r["label"] for r in sub)
    conv = [r["residual"] for r in sub if r["label"] != "nonconverged"]
    pf = [r["pencil_alpha_beta"] for r in sub if r["label"] in ("pencil-family", "classical-skeleton")]
    ratios = [b / a for a, b in pf]
    line = "%s & %d & %d & %d & %d & %d & %d & %s" % (
        names[p], len(sub), c.get("state-degenerate", 0), c.get("non-skew", 0),
        c.get("pencil-family", 0) + c.get("classical-skeleton", 0), c.get("classical-skeleton", 0),
        c.get("nonconverged", 0) + c.get("other-skew", 0),
        fmt(max(conv)) if conv else "--")
    if ratios:
        line += " & $%.2f$ [$%.2f$, $%.2f$]" % (np.median(ratios), np.min(ratios), np.max(ratios))
    else:
        line += " & --"
    out.append(line + r" \\")
# ---------------- Schrodinger ----------------
rs = load("schrodinger")
out.append("%% Schrodinger")
for b in ("det", "pinv", "det_n", "pinv_n"):
    sub = [r for r in rs if r["barrier"] == b]
    if not sub:
        continue
    c = collections.Counter(r["label"] for r in sub)
    conv = [r["residual"] for r in sub if r["label"] != "nonconverged"]
    utm = [r for r in sub if r["label"].startswith("UTM")]
    a1const = sum(1 for r in utm if r.get("a1_degree", 9) == 0)
    minsv = [r["sv"][-1] for r in utm]
    out.append("%s & %d & %d & %d & %d & %d & %d & %d & %s \\\\" % (
        {"det": "determinant", "pinv": "pseudoinverse", "det_n": "determinant, scale-invariant", "pinv_n": "pseudoinverse, scale-invariant"}[b], len(sub),
        c.get("zero-parameter (nonspectral)", 0), c.get("collapsed M(k) family (a2 const)", 0),
        len(utm), a1const, c.get("field-free (a1=0)", 0),
        c.get("nonconverged", 0) + c.get("other", 0), fmt(max(conv)) if conv else "--"))
# ---------------- Burgers ----------------
rs = load("burgers")
out.append("%% Burgers")
if rs:
    c = collections.Counter(r["label"] for r in rs)
    conv = [r["residual"] for r in rs if r["label"] != "nonconverged"]
    out.append("%d & %d & %d & %d & %d & %d & %s \\\\" % (
        len(rs), c.get("f linear, deg Phi = 0", 0), c.get("f linear, deg Phi = 1", 0),
        c.get("f linear, deg Phi = 2", 0), c.get("degenerate (f constant, g = g(s))", 0),
        c.get("nonconverged", 0) + c.get("deg f >= 2", 0) + c.get("f linear, g not of Prop-4.1 form", 0),
        fmt(max(conv)) if conv else "--"))
    out.append("%% burgers labels: " + str(dict(c)))
# ---------------- Shallow water ----------------
rs = load("shallow")
out.append("%% Shallow water")
cfgnames = {"base": "$(2,1,2,2)$, penalty on", "base_nopenalty": "$(2,1,2,2)$, penalty off",
            "enl3": "$(3,3,3,3)$, penalty on", "enl5": "$(5,5,3,3)$, penalty on"}
for cfg in ("base", "base_nopenalty", "enl3", "enl5"):
    sub = [r for r in rs if r["config"] == cfg]
    if not sub:
        continue
    import shallow_water as sw
    for r in sub:   # relabel field-free outputs produced before the classifier update
        ans = sw.Ansatz(*{"base": (2, 1, 2, 2), "base_nopenalty": (2, 1, 2, 2), "enl3": (3, 3, 3, 3), "enl5": (5, 5, 3, 3)}[cfg])
        T = ans.theta(np.array(r["eta"])); scale = max(np.max(np.abs(T)), 1e-300)
        if r["label"] != "nonconverged" and np.max(np.abs(T[:, 1:, :])) < 1e-6 * scale and np.max(np.abs(T[:, :, 1:])) < 1e-6 * scale:
            r["label"] = "field-free L = l(p)"
    c = collections.Counter(r["label"] for r in sub)
    conv = [r["residual"] for r in sub if r["label"] != "nonconverged"]
    out.append("%s & %d & %d & %d & %d & %d & %s \\\\" % (
        cfgnames[cfg], len(sub), c.get("Brunelli-Das", 0), c.get("field-free L = l(p)", 0),
        c.get("other converged", 0) + c.get("Brunelli-Das + constant shift c0", 0), c.get("nonconverged", 0),
        fmt(max(conv)) if conv else "--"))
    out.append("%% shallow %s labels: %s" % (cfg, dict(c)))

txt = "\n".join(out)
print(txt)
open(f"{D}/table_rows.tex", "w").write(txt + "\n")
