# `mdance-cli` quickstart

`mdance-cli` is the command-line front end to the MDANCE C++ library. Before it existed,
the project built as a static library with no entry point — you either wrote your own C++
program against the headers, or edited a test driver and rebuilt. The CLI replaces that
with a CSV-in → JSON-out batch interface covering clustering, similarity analysis,
representative-frame prediction, and frame selection.

---

## 1. Build

```shell
cmake -S . -B build
cmake --build build -j
```

The binary lands at `build/cli/mdance-cli`. It is built by default (`-DBUILD_CLI=ON`);
`cmake --install build` puts it in `bin/`.

Run it with no arguments to print the full usage text — that text is always authoritative
for the build you have in front of you, since some options are compile-gated (see the
Gotchas section below).

## 2. Input format

A **headerless CSV of plain numbers**: one row per frame, one column per feature.

```
17.98,20.11,19.42,...
18.03,20.09,19.47,...
```

- No header row, no index column, no missing values. Non-finite values and ragged rows are
  rejected with the offending row number.
- `--natoms` is the atom count used to normalize MSD. For a trajectory of Cartesian
  coordinates that is `ncols / 3` — the bundled `tests/data/sim.csv` is 6001 × 150, so
  `--natoms 50`. For non-MD data leave it at the default `1`.

**Label files** (`--labels`, `--initial-labels`) are one integer per line, or two columns
`frame_id,label` — the last column is used either way. They must hold exactly one entry
per frame: HELM rejects a count that disagrees with the coordinate file rather than
truncating, since a mismatch means the two files came from different trajectories. The
label *values* are arbitrary integers — they need not be `0..K-1`, contiguous, or
non-negative, so a partition containing eQUAL's `-1` frames can be fed straight back in.

## 3. The five modes

Mode is chosen by a flag; only clustering takes `--algorithm`. The same five modes are
reachable from the [C API](c-api.md) and the [VMD/Tcl extension](vmd-tcl.md) without going
through files.

| Mode | Selected by | Purpose |
|---|---|---|
| Clustering | `--algorithm kmeans\|helm\|equal` (also `divine`, if built) | Partition frames, report scores |
| Analysis | `--analysis` | Extended similarity (iSIM), globally and per cluster |
| PRIME | `--prime` | Predict the representative / "native-like" frame |
| Selection | `--select` | Pick representative, diverse, or outlier frames |

Common options: `--input`, `--output`, `--natoms`, `--metric`.

Valid `--metric` values: `MSD` (default), `BUB`, `Fai`, `Gle`, `Ja`, `JT`, `RT`, `RR`,
`SM`, `SS1`, `SS2`. PRIME accepts only `RR` and `SM`.

### KMeans (NANI)

```shell
mdance-cli --algorithm kmeans \
           --input tests/data/sim.csv --output kmeans.json \
           --natoms 50 --nclusters 10 --metric MSD \
           --kinit CompSim --percentage 10
```

`--kinit` — `StratAll` (default), `StratReduced`, `CompSim`, `DivSelect`, `KmeansPP`,
`Random`, `VanillaKmeansPP`. `--percentage` (default 10) sizes the high-density region
used by the `CompSim` and `DivSelect` seeders.

These exact settings reproduce the Python MDANCE reference labels bit-for-bit on
`sim.csv` (all 6001 frames), so they are a good sanity check that your build is sound.

### HELM

Agglomerative merging that starts from an existing partition, so it needs
`--initial-labels`:

```shell
mdance-cli --algorithm helm \
           --input tests/data/sim.csv --output helm.json \
           --natoms 50 --nclusters 6 --metric MSD \
           --initial-labels tests/data/sim_labels_helm.csv \
           --merge-scheme Inter
```

`--merge-scheme` — `Intra`, `Inter` (default), `Half`. Optional trimming:
`--trim-start` (flag), `--min-samples` (default 0.01), `--trim-val`, `--trim-k`.

You can stop on merge cost instead of on a target count, with `--eps` — but the two are
mutually exclusive, and `--nclusters` defaults to 10, so you must explicitly zero it:

```shell
mdance-cli --algorithm helm --input tests/data/sim.csv --output helm.json \
           --natoms 50 --initial-labels tests/data/sim_labels_helm.csv \
           --eps 5 --nclusters 0
```

Merging then continues only while the merge distance stays below `--eps`. Passing `--eps`
without `--nclusters 0` fails with *"You must provide either nClusters or eps, but not
both."*

HELM is the only mode that adds a `zMatrix` to the output — the linkage matrix, one row
per merge, suitable for drawing a dendrogram.

### eQUAL

Radial/threshold clustering. The cluster count **emerges from `--threshold`**; `--nclusters`
is ignored. `--threshold` is required and has no default.

```shell
mdance-cli --algorithm equal \
           --input tests/data/sim.csv --output equal.json \
           --natoms 50 --metric MSD --threshold 8 \
           --seed-method medoid --n-seeds 1
```

Threshold is in the same units as the metric (for MSD, mean square deviation), so it is
dataset-specific — sweep it. On `sim.csv`:

| `--threshold` | clusters | frames clustered |
|---|---|---|
| 8 | 7 | 6001/6001 |
| 10 | 4 | 6001/6001 |
| 12 | 2 | 6000/6001 |
| 14 | 1 | 6000/6001 |

`--seed-method` — `medoid` (default) or `comp_sim`. `--n-seeds` is a count when ≥ 1, a
fraction of frames when in (0,1). Optional rejection: `--check-sim` with `--sim-threshold`,
and `--reject-lowd` with `--min-samples`.

eQUAL can leave frames unassigned — those get label `-1`, and are excluded from the
CH/DB score computation.

### Analysis (iSIM)

```shell
mdance-cli --analysis \
           --input tests/data/sim.csv --output isim.json \
           --natoms 50 --metric MSD --labels labels.csv
```

Without `--labels` you get one global `isim` number. With them you also get `clusterISIM`
(per-cluster compactness) and `clusterOutliers` (the least-central frame of each cluster,
as an index into the original matrix).

Lower iSIM = more compact. A partition is doing real work when every per-cluster value
sits well below the global one — on `sim.csv` with the KMeans labels above, global iSIM is
17.65 against per-cluster values of 3.79–10.32.

### PRIME

Predicts the representative frame from an existing partition. Requires `--labels`, and
`--metric` must be `RR` or `SM`.

```shell
mdance-cli --prime \
           --input tests/data/sim.csv --output prime.json \
           --natoms 50 --metric RR --labels labels.csv \
           --trim-frac 0.1 --weighted
```

Returns seven frame indices from different prediction schemes — `pairwise`, `union`,
`medoid`, `outlier`, plus the `medoidAll` / `medoidC0` / `medoidC0Trimmed` baselines.
`--trim-frac` drops that fraction of the most-central rows of the largest cluster before
predicting; `--weighted` weights each cluster's score by its population.

> **On continuous data, trust the medoid baselines over the four predictors.**
>
> `medoidAll`, `medoidC0` and `medoidC0Trimmed` are ordinary comp-sim medoids and behave
> as you would expect — on `sim.csv`, `medoidAll` is frame 527, which is exactly the
> argmin of the complementary-similarity landscape.
>
> The four predictors (`pairwise`, `union`, `medoid`, `outlier`) are different. They score
> candidate frames with the *two-object* RR/SM index, which for values in [0,1] reduces to
> `sim(x,y) = (1/p)·Σⱼ max(0, xⱼ + yⱼ − 1)`. That hinge only activates on columns where
> both frames sit in the upper tail, so the score rewards a frame for having **many extreme
> coordinates**, not for being central. On `sim.csv` this hands every predictor to frame
> 754 — the frame with the most high-tail columns in the dataset (26, against a mean of
> 6.2) — no matter which cluster it is scored against.
>
> The symptom is output that will not move: identical for `--metric RR` and `SM`, and
> unchanged for `--trim-frac` anywhere from 0 to 0.7.
>
> **Do not try to fix this by pre-normalizing.** PRIME already applies global min-max
> normalization internally, and that is idempotent — normalizing first changes nothing.
> The behaviour is intrinsic to applying a binary-fingerprint similarity to continuous
> coordinates, and it is faithful to the algorithm as specified: an independent
> reimplementation of the scorers reproduces the C++ output frame-for-frame.

### Frame selection

```shell
mdance-cli --select --method diversity --param 1 \
           --input tests/data/sim.csv --output sel.json \
           --natoms 50 --metric MSD
```

| `--method` | `--param` means | Returns |
|---|---|---|
| `diversity` (default) | percentage (≥1) or fraction (0,1) | A maximally diverse subset |
| `outliers` | count (≥1) or fraction (0,1) | The *n* least-central frames |
| `repsample` | count (≥1) or fraction (0,1) | Stratified sample; `--nbins` (default 10) |
| `medoid` | — | The single most central frame |
| `outlier` | — | The single least central frame |

`--param 1` on 6001 frames gives 60 frames (1%). Use `medoid` to pick a structure to show
or dock; `outliers` to find frames worth inspecting; `repsample` to thin a long trajectory
while preserving its spread.

## 4. Output

JSON at `--output`. **Progress goes to stderr, so stdout stays clean for piping.**

Clustering runs produce:

```json
{
  "algorithm": "kmeans",
  "nFrames": 6001,
  "nClusters": 10,
  "labels": [0, 4, 4, 9, 2],
  "clusterSizes": [464, 691, 498, 609, 674],
  "representatives": [2629, 546, 4092, 2633, 4794],
  "clusterMSD": [6.64, 8.9703, 7.0198, 3.7906, 7.8466],
  "scores": { "calinskiHarabasz": 842.799, "daviesBouldin": 1.99442 }
}
```

(Arrays are shown truncated: `labels` has one entry per frame, and the three per-cluster
arrays have one entry per cluster.)

- `labels[i]` is the cluster of frame `i`; `-1` means unassigned (eQUAL only).
- `representatives[c]` is the medoid **frame index** of cluster `c` — feed these straight
  back to your trajectory viewer.
- `clusterMSD[c]` is that cluster's internal spread; it equals the `clusterISIM` you would
  get from `--analysis` on the same labels.
- Higher Calinski-Harabasz is better; **lower** Davies-Bouldin is better. A score that is
  not defined for the partition — Calinski-Harabasz needs at least two clusters — is
  reported as `0`. Any value that cannot be represented in JSON is written as `null`
  rather than as a bare `nan`, which would make the file unparseable.

The three analysis modes emit their own smaller objects, each tagged with an `"analysis"`
field (`"isim"`, `"prime"`, or `"select"`).

## 5. Recipes

**Cluster, then inspect the representatives:**

```shell
mdance-cli --algorithm kmeans --input traj.csv --output k.json \
           --natoms 50 --nclusters 10
python3 -c "import json; print(json.load(open('k.json'))['representatives'])"
```

**Chain clustering into analysis** — the labels array is not a file, so extract it first:

```shell
python3 -c "
import json; d=json.load(open('k.json'))
open('labels.csv','w').write('\n'.join(map(str,d['labels'])))"

mdance-cli --analysis --input traj.csv --output isim.json \
           --natoms 50 --labels labels.csv
```

**Pick a threshold for eQUAL** by sweeping until the cluster count stabilizes:

```shell
for t in 4 6 8 10 12; do
  mdance-cli --algorithm equal --input traj.csv --output /dev/null \
             --natoms 50 --threshold $t 2>&1 | grep found
done
```

## 6. Gotchas

**Misspelled options are silently ignored.** The parser stores any `--key value` pair
without validating the key, so `--n-clusters 6` (instead of `--nclusters`) does not error —
it silently leaves the default of 10 in place. If a run ignores your settings, check your
spelling against the usage text first.

**Unknown options eat the next token.** Booleans are recognized from a fixed list
(`--refine`, `--trim-start`, `--analysis`, `--check-sim`, `--reject-lowd`, `--prime`,
`--weighted`, `--select`); everything else consumes the following argument as its value.
So a stray `--verbose` before `--input traj.csv` swallows `--input` and you get a confusing
"Missing required argument" further along.

**DIVINE is usually not available at all.** Its sources (`src/cluster/divine.cpp`) live on
a separate dev branch rather than here, so `--algorithm divine` fails with *"not available
in this build"*. `BUILD_DIVINE` auto-detects whether those sources are present, and forcing
it on without them is a configure-time error, not a fallback:

```
CMake Error at src/CMakeLists.txt:90 (message):
  BUILD_DIVINE=ON but src/cluster/divine.cpp is not present in this tree.
```

To get DIVINE you need the sources first; only then does `-DBUILD_DIVINE=ON` do anything.
The usage text lists `divine` only when it is actually compiled in, so checking
`mdance-cli` with no arguments tells you what your build supports.

**KMeans needs enough frames to seed from.** `--kinit StratAll` (the default), `CompSim`,
`StratReduced` and `DivSelect` all draw their seeds from `--percentage`% of the frames, so
a short trajectory can yield fewer seeds than you asked for clusters. That is refused:

```
Error: initialization produced only 0 of the 3 requested centers; raise the percentage
of frames the seeder samples (--percentage) or ask for fewer clusters.
```

Raise `--percentage` (up to 100) or lower `--nclusters`. Asking for more clusters than
there are frames is refused outright.

**`--natoms` changes your numbers.** It is the MSD normalization divisor. Leaving it at the
default `1` on a 50-atom trajectory inflates every MSD, iSIM, and eQUAL threshold by 50×.
Thresholds are not portable between runs with different `--natoms`.

**eQUAL's `--nclusters` is ignored.** Cluster count is a consequence of `--threshold`.

**HELM's `--eps` needs `--nclusters 0`.** They are mutually exclusive termination criteria
and `--nclusters` has a non-zero default, so `--eps` alone always errors out.

**`--align uni|kron`** parses but is unimplemented and raises an error; use `none`.
Likewise, the sklearn-based eQUAL seeds (`greedy`, `vanilla`, `mini_batch_kmeans`) are
rejected with an explanatory message — use `medoid` or `comp_sim`.

## 7. Verifying your build

```shell
cd build && ctest
```

`mdance_tests` is the gtest suite; `validate_equal_prime` drives this CLI and cross-checks
eQUAL, PRIME, and frame selection against an independent NumPy reference. The validator is
skipped rather than failed when NumPy is not importable, so confirm it actually ran if you
are relying on it.
