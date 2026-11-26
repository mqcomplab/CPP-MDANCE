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
| kmeans | `k` (required), `metric` MSD, `kinit` StratAll, `percentage` 10, `seed` −1 (−1: from the clock; read it back with `mdance_result_seed`), `tol` 0 (0: until no label changes; 1e-4: Python MDANCE's stop, scikit-learn's `KMeans` default) |
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
| `_seed` | The seed k-means used, for the seedings that draw one (Random, KmeansPP, VanillaKmeansPP); −1 otherwise. |
| `mdance_analysis_*`, `mdance_prime_*`, `mdance_select_*` | As in version 1. |

## Scans: every k at once, on threads

```c
int ks[] = {2, 3, 4, 5, 6, 7, 8};
mdance_options_set(o, "threads", "0");        /* 0 or absent: one a processor */
mdance_scan_t* s = mdance_scan_run(d, "kmeans", ks, 7, o, c);
for (int i = 0; i < mdance_scan_count(s); ++i) {
    const mdance_result_t* r = mdance_scan_result(s, i);   /* k = mdance_scan_k(s, i) */
    if (mdance_result_error(r)) continue;                   /* this k failed; the others did not */
    /* ... */
}
mdance_scan_free(s);
```

`mdance_scan_run` makes the same clustering at every k of a list, the values run at once
on threads. It takes `"kmeans"`, `"divine"` or `"helm"`, and that algorithm's options
without `k`, plus `threads`:
- **The answers.** Each k's result is the one `mdance_cluster` gives with it, to the last
  digit. The threads change the time only.
- **Memory.** The threads share the dataset, which is not copied.
- **Order.** The values of k are run largest first, since they take longest; the results
  keep the list's order.
- **Errors.** A k that fails, for example too few frames to seed it, has the error in its
  own result, and the others are made. `mdance_scan_error` is the scan's own failure: no
  dataset, an algorithm without k, `k` among the options, or "cancelled".
- **Controls.** A control cancels every k still running or waiting. Its progress is the
  share of the values done, its stage `"scan"`.

On 4 cores, a scan of k = 2…20 takes 2.4–2.5 times less time than the same values one
after another (see [Performance](performance.md)).

## The first interface

`mdance_kmeans`, `mdance_divine`, `mdance_helm`, `mdance_equal`, `mdance_analysis`,
`mdance_prime` and `mdance_select` are kept. Each takes a row-major `coords` buffer and
positional arguments, returns the same objects, and runs through the same checks as the
calls on a dataset.

## Compatibility with version 1

The interface itself is compatible:
- every version 1 function is still there, with the same name, arguments and results;
- a program built against the version 1 header compiles and links unchanged;
- a program built against the version 1 library runs with this one.

The datasets, options and controls are additions. What changed is behaviour: some calls
version 1 accepted are now errors, two results are different, and a few details around
them changed.

### Calls that are now errors

| Call | Accepted by version 1 | Version 2 |
|---|---|---|
| all | coordinates that are NaN or infinite, processed as numbers | refused, naming the frame and column |
| all | `natoms` 0 or negative (infinite or NaN values in some paths) | refused |
| `mdance_kmeans`, `mdance_divine`, `mdance_equal` | any `percentage` | 0 to 100, for every seeding: `KmeansPP` and `Random`, which do not use it, refuse 200 too |
| `mdance_divine` | any `end_mode` other than `"points"`, read as `"k"` (`"K"`, `"x"`) | `"k"` or `"points"`; `NULL` is still `"k"` |
| `mdance_divine` | any `threshold` | 0 to 1 |
| `mdance_helm` | `nclusters` above the frames | 0 to nframes |
| `mdance_helm` | any `eps` | −1 (none), or above 0 as a float (−0.99999999 is −1, as in `mdance-cli`) |
| `mdance_helm` | `trim_k` negative or not whole; `trim_val` negative | `trim_k` a whole number ≥ 0; `trim_val` ≥ 0 |
| `mdance_helm` | any number of starting clusters | 5 000 at most: each pair's distance is kept |
| `mdance_equal` | `threshold`, `n_seeds` or `min_samples` below 0 | refused |
| `mdance_equal` | `align` `""`, read as `"none"` | refused; `NULL` is still `"none"` |
| `mdance_analysis`, `mdance_prime` | a label below −1 (ignored), or past the frames (allocations of gigabytes, or a crash) | refused: a label is −1 (none) or 0 to nframes − 1. Label arrays shorter or longer than the frames are still padded with −1 or cut, as before |
| `mdance_prime` | any `trim_frac` | in [0, 1) |
| `mdance_select` | a `param` below 0 (taken as 0) | refused |

These inputs crashed version 1 and are now errors as well: a CompSim or StratReduced
seeding from fewer frames than k, a percentage over 100 there, DIVINE's refinement of
identical frames, and labels of 2³¹ − 1.

### Results that differ

- **The RR diversity pick** (`mdance_select`, `"diversity"`, metric RR) returned the
  number of columns as a frame, repeatedly. Every candidate's value was below the
  starting −1. It now returns distinct frames.
- **HELM with as many starting clusters as `nclusters`, or fewer,** returns them
  unmerged. With fewer, version 1 merged them all into one cluster.

Every other result is the same to the last digit, for a library built with the same
flags (see [Performance](performance.md)). A build with `-march=native`, version 1's
forced default and now `MDANCE_NATIVE`, differs from a portable one in the last digits
of some MSDs: the fused multiply-add instructions round differently.

### Other changes

- **Error messages are worded otherwise.** They now name the call and the option, for
  example `kmeans: option percentage: 200 is outside 0 .. 100`. Code that matches a
  message's text should be checked. DIVINE's "not available in this build" is
  unchanged.
- **Nothing is printed.** Version 1 wrote a line to standard output at every k-means
  call ("Initializing random number generator with: …"), two lines at every DIVINE
  split, and warnings to standard error.
- **The `*_error()` accessors given `NULL`** return "out of memory", so a result that
  could not be made reads as failed. Version 1 never returned `NULL`; it ended the
  process when out of memory.
- **The k-means random seed** is per call. Before, it was one generator for the whole
  process, which concurrent calls shared. It is drawn from the clock as before, and it
  is never negative.
- **`MDANCE_API` on Windows** is `dllimport` for a program that uses the header. A
  program that compiles `mdance_capi.cpp` into itself defines `MDANCE_BUILDING` (to
  export the functions) or `MDANCE_STATIC` (neither). A program that links the DLL
  needs nothing.
- **Release builds are portable.** They no longer use `-march=native` unless
  `MDANCE_NATIVE=ON`.
- **With MinGW the library carries its C++ runtime** (`MDANCE_STATIC_RUNTIME`, on by
  default), so `libstdc++-6.dll` and its siblings are no longer needed beside it.

## Scores that are not defined

With one cluster the Calinski–Harabasz score is 0/0: it is reported as 0. The scores of
eQUAL are computed over the clustered frames only (its −1 frames are noise, not a
cluster).
