# Selector before/after benchmark

`scripts/benchmark_selector.py` loads two saved selector modules and times
their complete `knee_region` and `select_preferred` functions in one
interpreter. It retains validation, normalization and each module's original
Pareto processing. It does not replace the upstream Pareto dispatch with a
different kernel to improve a result.

```powershell
python scripts/benchmark_selector.py --before path/to/before.py --after path/to/after.py --output-dir benchmark_results/selector
python -m pytest tests/test_selector_benchmark.py -q
```

Default cases `(batch,n,m)` are `(1,64,2)`, `(1,256,3)`, `(32,64,3)`,
`(32,256,3)`, `(32,256,5)` and `(1,1024,3)`. The predeclared settings are
seed 17, float32/float64, CPU/CUDA, one Torch CPU thread, two warmups and five
timed repetitions. CUDA is required unless `--devices cpu` is supplied.
Before/after execution order is counterbalanced across configurations.

CPU float64 random inputs and positive batched preferences are cast and
transferred outside timing, ensuring each implementation receives the same
input. The timer includes the full function and synchronizes GPU execution
immediately before and after each call. Equality checks and serialization
are excluded. Exact index and mask mismatches are reported without silently
relaxing tolerances for near-threshold cases.

The output folder contains source snapshots and hashes, the protocol saved
before timing, environment information, all repetitions, median/min/max
timings, speedup ratios, CUDA live allocator peaks and a report. A speedup
below one is a regression and is retained. CUDA peak memory is neither total
process/device memory nor reserved allocator cache; CPU memory is not measured.
Five repetitions describe one machine under current conditions and do not
establish general performance claims.
