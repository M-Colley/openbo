# FPBI / upstream BoTorch NSGA-II comparison

`scripts/benchmark_fpbi.py` runs a small, reproducible front-generation
comparison. It uses actual BoTorch `GenericDeterministicModel` objects and
OpenBO's `sample_model_fpbi`; `--fpbi-api direct` exercises `sample_fpbi`
instead. NSGA-II uses upstream BoTorch's `MultiOutputPosteriorMean` and
`optimize_with_nsgaii`, not a reimplementation.

Install the optional comparator in the existing environment:

```powershell
uv pip install --python .venv\Scripts\python.exe --no-cache pymoo
.venv\Scripts\python.exe scripts/benchmark_fpbi.py --output-dir benchmark_results/fpbi
.venv\Scripts\python.exe -m pytest tests/test_fpbi_benchmark.py -q
```

The default predeclared protocol has three seeds and four analytic cases:
convex and concave two-objective fronts, a constrained two-objective front,
and a three-objective sphere front with a radial nuisance variable. It runs
on CPU in double precision with one Torch thread. The protocol is written
before measured runs; an existing results directory is protected against
overwrite. No GP fitting or acquisition optimization is being benchmarked.

Two comparator arms separate output coverage from effort. The first uses a
fixed 32-member, 20-generation NSGA-II run and a common output cap
`m + n_steps**(m-1)`. The second rounds the measured FPBI point-evaluation
count to the nearest NSGA-II generation, with a 300-generation guard. Both
arms report actual evaluation counts and their ratio to FPBI. The output cap
does not ensure identical retained cardinality or equal optimization work.

Objective evaluations count **decision vectors**, including finite
differences and repeated diagnostic evaluations. A population batch counts
every vector; posterior calls are counted separately. Generation runtime
excludes imports, warmup, common output capping, reference construction,
metrics and serialization. Bounds and parameter constraints are independently
checked before scoring. The scripts preserve FPBI's per-attempt failures and
nominal ray success counts; upstream NSGA-II does not expose comparable
local-solver convergence diagnostics through this wrapper.

Quality uses theoretical objective ranges for scaling, a normalized cost
reference `(1.1,...,1.1)` and analytic continuous-front hypervolume. IGD uses a
fixed dense reference sampled uniformly in the one-dimensional decision
coordinate for 2D cases, and uniformly on the positive sphere surface for
3D. All methods receive the same deterministic greedy HV output cap. Raw
fronts and uncapped metrics remain available.

Outputs include the recorded protocol, installed versions/API signature,
source hashes and Git state, per-run JSON/CSV with raw fronts and diagnostics,
a median/min/max summary, a report and a static comparison figure. The
comparison is diagnostic: four small analytic problems and three seeds do
not establish global superiority. Cheap vectorized objectives, local
extrema, different numbers of posterior calls, and hardware can materially
change timing and front coverage.

The upstream signature and constraint conventions are documented in the
[BoTorch utilities API](https://botorch.readthedocs.io/en/stable/utils.html).
