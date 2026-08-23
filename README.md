# Personal repository for my TT algos

Started in julia, may pick up MATLAB in the future. I will try implementing iTensor later.
The Glauber work is in Python; the julia files are the earlier TT algorithm implementations.

## Layout

```
src/          the Python solvers and scripts (flat: they import each other by name)
tests/        pytest tests  (conftest.py puts src/ on sys.path)
julia/        the original julia TT algorithms
prototypes/   older / exploratory code: tt_cross, python_versions, dmrg_algo
papers/       summer_project/ (master) and glauber_tt/ (self-contained bundle)
results/      one folder per sweep: results/<run>/<run>.csv + its plots  (gitignored)
figures/      figures kept for the write-up
notes/        scratch notes and PDFs  (gitignored)
```

Run scripts from the repo root, e.g. `python3 src/glauber_solve.py 4 12`.
A sweep writes to `results/<run>/<run>.csv`, and `python3 src/plot_results.py`
writes its PNGs into that same folder.

`check_tex_sync.py` (repo root) verifies the `papers/glauber_tt/` bundle stays in
sync with the master paper: the bundle ships byte-identical copies of the solver
modules so it runs standalone, which is why `src/` is flat rather than a package.

## Julia files (julia/)

- tt-algos
  - Contains TT-SVD, TT-Round, TT-Direct-Sum. This has already been tested and made sure that they work.
- tt-cross
  - Contains the ACA, Max-vol, and TT-Cross. **tt-cross doesn't work**
- tt-test
  - Ghetto testing file. It's not elegant but it works :)
- Kressner Cross Algorithm
   - Theres some Golab Kahn, Summation Algo, and some cool things. Did this in NumPy for practice (`src/kressner.py`).
