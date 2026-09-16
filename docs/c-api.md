# C API (`libmdance`)

`capi/mdance_capi.h` is a flat C interface over the C++ library. It exists so that hosts
that cannot consume C++ headers — the Tcl extension in `tcl/`, and any Python `ctypes`,
Fortran or Rust caller — can reach the same algorithms the CLI uses. Everything crosses
the boundary as plain `double*` / `int*` buffers and opaque handles; no C++ types, no
exceptions, no Eigen.

## Building and linking

```shell
cmake -S . -B build            # BUILD_SHARED is ON by default
cmake --build build -j
```

That produces `build/capi/libmdance.so` (`.dylib` on macOS, `mdance.dll` on Windows).

```shell
cc myprog.c -I/path/to/CPP-MDANCE/capi \
            -L/path/to/CPP-MDANCE/build/capi -lmdance -o myprog
```

The library is built with hidden visibility, so only the `MDANCE_API` functions listed
here are exported. If you move `libmdance.so` away from the build tree, the loader needs
to be told where it went (`LD_LIBRARY_PATH`, or an `-rpath` at link time).

## The three conventions

**1. Coordinates are row-major.** `coords` is a flat buffer of `nframes * ncols` doubles
with each frame contiguous — frame `i`, column `j` at `coords[i * ncols + j]`. This is
the layout you get from a C 2-D array, from NumPy's default, and from flattening a
per-frame loop. `natoms` is separate: it is the MSD normalization divisor, typically
`ncols / 3` for Cartesian coordinates.

**2. Every call returns a handle you must free — including on error.** The functions
never return `NULL` for an algorithm failure; they return a handle whose `*_error()`
accessor is non-`NULL`. So the shape of every call is the same:

```c
mdance_result_t* r = mdance_kmeans(coords, nframes, ncols, natoms, 10, "MSD", "StratAll", 10);
const char* err = mdance_result_error(r);
if (err) {
    fprintf(stderr, "mdance: %s\n", err);
} else {
    const int* labels = mdance_result_labels(r);   /* nframes entries */
    /* ... */
}
mdance_result_free(r);   /* required on both paths */
```

Bad input is reported the same way rather than crashing: a `NULL` or empty `coords`,
`nframes <= 0`, an unknown metric name, a `k` larger than the frame count, or a HELM call
with no initial labels all come back as an error string.

**3. Returned pointers belong to the handle.** `mdance_result_labels()` and friends point
into the handle's own storage. They are valid until `mdance_result_free()`; copy anything
you need to keep. Accessors return `NULL` (or `0`) for arrays a given algorithm does not
produce — `mdance_result_zmatrix()` is non-`NULL` only for HELM.

## Clustering

Four entry points, all returning `mdance_result_t*`:

| Function | Cluster count | Extra inputs |
|---|---|---|
| `mdance_kmeans` | `nclusters` | `kinit`, `percentage` |
| `mdance_helm` | `nclusters` (or `eps`) | `initial_labels` (one per frame) |
| `mdance_equal` | emerges from `threshold` | `seed_method`, `n_seeds`, … |
| `mdance_divine` | `nclusters` | `split`, `anchors`, `refine`, … |

`mdance_divine` is compile-gated: in a build without `BUILD_DIVINE` it returns a handle
carrying *"DIVINE is not available in this build of libmdance"* rather than failing to
link, so callers need no `#ifdef`.

String parameters take the same spellings as the CLI (`"MSD"`, `"StratAll"`, `"Inter"`,
`"medoid"`, …); passing `NULL` selects the documented default. `n_seeds` and `min_samples`
use the dual-typing convention: a value in (0,1) is a fraction of `nframes`, a value ≥ 1 is
a literal count.

Read results with the `mdance_result_*` accessors: `nframes`, `nclusters`, `labels`,
`cluster_sizes`, `cluster_msd`, `representatives` (the medoid frame index of each cluster,
`-1` if empty), `ch_score`, `db_score`, and the HELM-only `zmatrix` / `zmatrix_rows` /
`zmatrix_cols` triple, which is row-major like the input.

`labels` has one entry per frame; `-1` means unassigned, which only eQUAL produces. The
per-cluster arrays have `nclusters` entries.

### HELM initial labels

HELM merges an existing partition, so `initial_labels` must hold exactly `nframes`
entries — a mismatch is rejected rather than truncated, because it means the labels and
the coordinates came from different trajectories. The label *values* are arbitrary
integers: they do not have to be `0..K-1`, contiguous, or non-negative, so a partition
carrying eQUAL's `-1` frames can be fed straight in.

## Analysis, PRIME and selection

Three smaller families, each with its own handle type and its own `_error` / `_free`:

- **`mdance_analysis`** — extended similarity (iSIM) over the whole ensemble, plus, when
  `labels` is non-`NULL`, per-cluster compactness (`cluster_isim`) and each cluster's
  least-central frame (`cluster_outliers`). Pass `labels = NULL, nlabels = 0` for the
  global number alone.
- **`mdance_prime`** — representative/"native-like" frame prediction from an existing
  partition. Requires `labels`; `metric` must be `"RR"` or `"SM"`. Returns seven frame
  indices through separate accessors (`pairwise`, `union`, `medoid`, `outlier`, and the
  `medoid_all` / `medoid_c0` / `medoid_c0_trimmed` baselines), each `-1` when undefined.
  The caveat about PRIME's predictors on continuous data applies here exactly as it does
  to the CLI — see the [quickstart](mdance-cli-quickstart.md#prime).
- **`mdance_select`** — frame selection with no clustering. `method` is one of
  `"diversity"`, `"outliers"`, `"repsample"`, `"medoid"`, `"outlier"`; `param` is a
  percentage or a count/fraction depending on the method, and `nbins` applies to
  `repsample` only. Read the answer with `mdance_select_count()` and
  `mdance_select_indices()`.

## Scores that are not defined

`ch_score` and `db_score` are `0` when the partition cannot support them — a single
cluster leaves the Calinski-Harabasz ratio undefined. They are always finite doubles, so
you can serialize them without a special case.

## Thread safety

The handles carry no shared state and none of the entry points touch globals, so separate
handles can be used from separate threads. A single handle is not synchronized; do not
free one while another thread is reading its arrays. The algorithms themselves are
single-threaded.
