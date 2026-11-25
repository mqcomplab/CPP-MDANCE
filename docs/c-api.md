# C API (`libmdance`)

`capi/mdance_capi.h` is a C interface to the C++ library, for any program that can call
C: the Tcl extension in `tcl/`, Python through `ctypes`, Fortran, Rust, or a C or C++
application. Only plain C types and opaque handles cross it: no C++ types, no
exceptions, no Eigen.

## Building and linking

```shell
cmake -S . -B build            # BUILD_SHARED is ON by default
cmake --build build -j
cmake --install build --prefix /opt/mdance     # lib/, bin/ and include/mdance_capi.h
```

The build makes `build/capi/libmdance.so` (`libmdance.dylib` on macOS, `libmdance.dll`
and its import library `libmdance.dll.a` with MinGW).

```shell
cc myprog.c -I/opt/mdance/include -L/opt/mdance/lib -lmdance -o myprog
```

The library exports only the `MDANCE_API` functions. With MinGW it carries the C++
runtime inside it (`MDANCE_STATIC_RUNTIME`, on by default), so it needs no
`libstdc++-6.dll` beside it. A program can also link the static `mdance_capi_static`
target with `MDANCE_STATIC` defined.

Release builds run on any x86-64 CPU. `-DMDANCE_NATIVE=ON` optimizes for the CPU of the
machine that builds it.

## Conventions

- **Errors.** Nothing aborts, exits or prints.
  - A call that returns a result object (`mdance_result_t`, `mdance_analysis_t`,
    `mdance_prime_t`, `mdance_select_t`) always returns one. Its `*_error()` is the
    message when the call failed, else `NULL`. Free it in both cases.
  - A call that makes a dataset, options or a control returns `NULL` on failure.
  - A call that returns an `int` returns 0 on success and −1 on failure.
  - For both kinds, `mdance_last_error()` is the message (per thread).
  - Out of memory is an error like the others.
- **Pointers into a result** (labels, sizes, …) are valid until it is freed.
- **Frames** are rows: 0 … nframes − 1, in the order you gave them. Every index a result
  holds is a row, or −1 for none.
- **Threads.** Calls on different threads run concurrently; nothing is global.
  - A dataset can be read by several calls at once. It must not be changed or freed
    while a call reads it.
  - `*_free(NULL)` does nothing.

## Datasets

```c
mdance_dataset_t* d = mdance_dataset_new(nframes, natoms);       /* 3 columns an atom */
for (int f = 0; f < nframes; ++f)
    if (mdance_dataset_set_frame_float(d, f, xyz[f], 3 * natoms) != 0)
        fprintf(stderr, "%s\n", mdance_last_error());
```

| Function | Purpose |
|---|---|
| `mdance_dataset_new(nframes, natoms)`, `mdance_dataset_new_cols(nframes, ncols, natoms)` | An empty dataset, every frame still to be set. |
| `mdance_dataset_from_rows(rows, nframes, ncols, natoms)` | A copy of a row-major buffer. |
| `mdance_dataset_set_frame` (`double`), `mdance_dataset_set_frame_float` | Set one frame's values. |
| `mdance_dataset_set_frame_atoms_float(d, f, pos, natoms_in_pos, atoms, n)` | Set a frame from a program's array of every atom's x y z, taking the listed atoms. This is VMD's timestep layout. |
| `mdance_dataset_read_dcd(path, atoms, n, first, last, stride)` | Read a DCD (CHARMM or X-PLOR, either byte order), checked record by record. |
| `mdance_dataset_superpose(d, fit, nfit, ref)` | Least-RMSD fit of every frame onto frame `ref` by the `fit` atoms, applied to all the atoms. |
| `mdance_dataset_get_frame`, `_nframes`, `_ncols`, `_natoms`, `_nset`, `_free` | Read a frame back, the sizes, how many frames are set, and free. |

Values are checked as they come in: NaN and infinities are refused, naming the frame and
column. A call on a dataset with a frame not set is refused.

## Options

```c
mdance_options_t* o = mdance_options_new();
mdance_options_set(o, "k", "10");
mdance_options_set(o, "kinit", "StratAll");
mdance_options_set_ints(o, "labels", labels, nframes);     /* lists of whole numbers */
mdance_options_set_double(o, "threshold", 2.0);
```

A call reads its options and checks every one. These errors name the option: a name the
call does not take, a number that is not one (hex, `inf` and `nan` are refused, and
`010` is ten), a value out of range, or a choice not offered. A call leaves the options
unchanged, so one options object can serve many calls.

## Controls: cancel and progress

```c
mdance_control_t* c = mdance_control_new();
/* on another thread: */ mdance_control_cancel(c);
/* while it runs:     */ mdance_control_progress(c), mdance_control_stage(c);
```

A cancelled call fails with the error `cancelled`, within a fraction of a second. Pass
the control to the call; pass `NULL` for none.

## Calls on a dataset

```c
mdance_result_t*   mdance_cluster(d, "kmeans" | "divine" | "helm" | "equal", o, c);
mdance_analysis_t* mdance_analyze(d, o, c);
mdance_prime_t*    mdance_prime_run(d, o, c);
mdance_select_t*   mdance_select_run(d, o, c);
```

| Call | Options (defaults) |
|---|---|
| kmeans | `k` (required), `metric` MSD, `kinit` StratAll, `percentage` 10, `seed` −1 (−1: from the clock; read it back with `mdance_result_seed`) |
| divine | `k` (required; 0: until `end_mode`), `metric`, `split` WeightedMSD, `anchors` NANI, `kinit` StratAll, `refine` 0, `threshold` 0, `end_mode` k, `percentage` 10 |
| helm | `labels` (required: one starting cluster per frame, at most 5000 distinct), `k`, `eps` −1, `metric`, `merge_scheme` Inter, `trim_start` 0, `min_samples` 0.01, `trim_val` 0, `trim_k` 0 |
| equal | `threshold` (required), `metric`, `seed_method` medoid, `n_seeds` 1, `check_sim` 0, `sim_threshold` 0, `reject_lowd` 0, `min_samples` 10, `percentage` 10, `align` none |
| analyze | `metric` MSD, `labels` (optional) |
| prime_run | `labels` (required), `metric` RR, `trim_frac` 0, `weighted` 0 |
| select_run | `method` diversity, `metric` MSD, `param` 10, `nbins` 10 |

Labels are one per frame: a cluster number in 0 … nframes − 1, or −1 for none.

Results:

| Accessor | Gives |
|---|---|
| `mdance_result_labels`, `_cluster_sizes`, `_representatives`, `_cluster_msd` | Per frame and per cluster arrays. |
| `_ch_score`, `_db_score` | The Calinski–Harabasz and Davies–Bouldin scores. |
| `_zmatrix*` | HELM's merges. |
| `_seed` | The seed k-means used. |
| `mdance_analysis_*`, `mdance_prime_*`, `mdance_select_*` | As in version 1. |

## The first interface

`mdance_kmeans`, `mdance_divine`, `mdance_helm`, `mdance_equal`, `mdance_analysis`,
`mdance_prime` and `mdance_select` are kept. Each takes a row-major `coords` buffer and
positional arguments, and returns the same objects. They now go through the same checks:
- Coordinates that are not finite are refused.
- So are values out of range: a percentage over 100 and labels past the frames
  (`analysis` and `prime`).
- An unknown DIVINE `end_mode` is refused; it used to be read as `k`.

## Scores that are not defined

With one cluster the Calinski–Harabasz score is 0/0: it is reported as 0. The scores of
eQUAL are computed over the clustered frames only (its −1 frames are noise, not a
cluster).
