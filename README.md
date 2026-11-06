# LearningLaxPairs-SILO

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23992657.svg)](https://doi.org/10.5281/zenodo.23992657)

Code and archived run logs for the SILO computations in

> J. Adriazola, G. Biondini, W. Zhu, and P. G. Kevrekidis,
> *Learning Lax Pairs: Revisiting the Classical Paradigm*,
> Studies in Applied Mathematics (2026). arXiv:2607.01493.

The scripts reproduce the multi-seed searches reported in Appendix A.4
("Numerical protocols and sensitivity") for the Euler top (Section 2), the free
Schrödinger equation (Section 3), the inviscid Burgers equation (Section 4) and the
shallow water system (Section 5). The KdV search (Section 6) was not rerun for this
paper. Its settings are reported in the original SILO paper.

## Contents

```
silo/                    SILO searches (numerical)
  common.py              shared pipeline: cross-entropy warm start, L-BFGS-B,
                         threshold-and-refit continuation, pure-loss refit
  euler_top.py           Euler top, protocols A-D
  schrodinger.py         free Schrödinger, determinant and pseudoinverse barriers
  burgers.py             inviscid Burgers, Clairaut identity
  shallow_water.py       shallow water, truncated Laurent ansatz
  summarize.py           tallies progress_*.jsonl into table_rows.tex
exact/                   exact symbolic checks (sympy), independent of the searches
  euler_rigidity.py      infinitesimal deformations of the classical Euler-top
                         pair modulo gauge (one direction survives, the skew pencil)
  schrodinger_exact.py   exact solution set of the Clairaut constraints (3.4)
  burgers_exact.py       exact polynomial classification behind Proposition 4.1
  sw_rigidity.py         exact rigidity computation for the shallow-water ansatz
results/                 archived logs of the runs reported in the paper
  progress_euler.jsonl, progress_schrodinger.jsonl,
  progress_burgers.jsonl, progress_shallow.jsonl
  table_rows.tex         output of summarize.py on these logs
run_paper.sh             the exact commands that produced the logs
```

Each line of a `progress_*.jsonl` file is one run. It records the seed, the
protocol or configuration, the final coefficient vector `eta`, the raw compatibility
residual, the classification label and the wall time.

## Requirements

Python 3.10 or later with NumPy, SciPy and SymPy (`pip install -r requirements.txt`).
The archived logs were produced in September 2026 with NumPy 2.x and SciPy 1.x.

## Reproducing the paper

Regenerate the tallies from the archived logs:

```
python silo/summarize.py results
```

This rewrites `results/table_rows.tex`, and the result is identical to the archived
file. To rerun the searches from scratch, run

```
bash run_paper.sh
```

New runs are written to `output/`, or to the directory named by the environment
variable `SILO_OUT`. Running a script with no arguments uses its built-in default
seed counts, which differ from the paper. Use the commands in `run_paper.sh`.

| Paper | Command | Runs |
|---|---|---|
| Euler top, protocols A, B, C (seeds 0-59) and D (seeds 0-19) | `python silo/euler_top.py 60` | 200 |
| Schrödinger, determinant and pseudoinverse barriers (seeds 0-49) | `python silo/schrodinger.py 50 det pinv` | 100 |
| Burgers (seeds 0-59) | `python silo/burgers.py 60` | 60 |
| Shallow water, base window with penalty (seeds 0-27) | `python silo/shallow_water.py base:28` | 28 |
| Shallow water, base window without penalty (seeds 0-9) | `python silo/shallow_water.py base_nopenalty` | 10 |
| Shallow water, window (3,3,3,3) (seeds 0 and 1) | `python silo/shallow_water.py enl3:2` | 2 |

Euler protocol D (unconstrained, with the state-support penalty) is the
"non-skew search with the support penalty" in Appendix A.4.

## Notes on the archived logs

**Exploratory Schrödinger runs.** `progress_schrodinger.jsonl` also contains 60
runs with scale-invariant variants of the two barriers (`det_n`, `pinv_n`,
seeds 0-29, `python silo/schrodinger.py 30 det_n pinv_n`). They were exploratory
and are not reported in the paper. `summarize.py` lists them as the last two
Schrödinger rows.

**Shallow-water labels.** In `progress_shallow.jsonl` the 13 base-window runs
whose Lax function is independent of $(u,\rho)$ carry the label `other converged`,
because the classifier was refined after those runs. `summarize.py` relabels them
as `field-free L = l(p)` from the stored coefficients, which gives the counts in
Appendix A.4 (15 Brunelli-Das, 13 field-free).

**Residuals.** The maximum residuals quoted in Appendix A.4 are the logged maxima
rounded up to two significant figures.

**Reproducibility across machines.** Every run seeds its own generator
(`numpy.random.default_rng(seed)`), so a run is deterministic on a given machine.
The Euler top, Burgers and shallow-water searches are well conditioned. In a fresh
environment (NumPy 2.5, SciPy 1.18, October 2026), rerunning the full Euler batch
reproduced all 200 logged coefficient vectors bit for bit, and sample Burgers and
shallow-water seeds did the same.
The Schrödinger searches are not stable in this sense. The rank barriers
$|\det Z^\dagger Z|^{-2}$ and $\operatorname{tr}(Z^\dagger Z)^{-1}$ can be weakened by
inflating the scale of $Z$, and the resulting trajectories amplify last-bit
differences in the linear algebra. The same seed run with different OpenBLAS CPU
kernels converges to different coefficient vectors, and sometimes to a different
class. Tallies from a rerun therefore differ seed by seed from the archived
batch, while the same classes appear in similar proportions. For example, one
rerun in October 2026 gave 13 UTM-type, 2 with $a_2$ constant, 31 $q$-independent,
1 other and 3 SVD failures for the determinant barrier, and 12 UTM-type,
33 $q$-independent and 5 other for the pseudoinverse barrier. The archived log is
the record of the batch reported in the paper.

## Citation

If you use this code, please cite the paper above and this archive,
Zenodo, doi:[10.5281/zenodo.23992657](https://doi.org/10.5281/zenodo.23992657) (see `CITATION.cff`).

## License

MIT. See `LICENSE`.
