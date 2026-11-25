# VMD / Tcl extension

`tcl/mdance_tcl.cpp` makes the [C API](c-api.md) a loadable Tcl extension, so MDANCE
runs inside a VMD session on the trajectory VMD has loaded, and the frame numbers that
come back are rows of what you gave it. It loads into plain `tclsh` too.

Version 2 (this one) adds **datasets** (coordinates loaded once, from VMD's own DCD
writer or frame by frame, and clustered as often as you like) and **jobs** (any call on a
thread of its own while VMD keeps drawing). The first version's calls work as they did.

## Building

The extension is off by default: it needs Tcl's headers and stub library.

```shell
cmake -S . -B build -DBUILD_TCL=ON
cmake --build build -j
```

This makes `build/tcl/mdance_tcl.so` (`.dll` on Windows, `.dylib` on macOS) and a
`pkgIndex.tcl` beside it. If configuring fails with *"Tcl stub library (libtclstub8.6)
not found"*, install Tcl's development package (`apt install tcl-dev`, `brew install
tcl-tk`, or MSYS2's `mingw-w64-x86_64-tcl`).

The extension is **one file**. libmdance is linked into it, and with MinGW so is the C++
runtime, so it needs nothing beside it and cannot pick up another program's
`libstdc++-6.dll`. It links Tcl's *stub* library, not Tcl itself, so it uses the Tcl of
whatever program loads it. A second Tcl in VMD's process aborts it with `alloc: invalid
block`.

`cmake --install build` puts the extension and its `pkgIndex.tcl` in
`<prefix>/lib/mdance2.0`. Set `MDANCE_TCL_INSTALL_DIR` to put them elsewhere, for example
in a folder on VMD's `auto_path`.

Release builds are portable: they run on any x86-64 CPU. `-DMDANCE_NATIVE=ON` optimizes
for the CPU of the machine that builds it (AVX2 and FMA where it has them). That build is
faster, but runs only on CPUs that have the same instruction sets.

## Loading

```tcl
lappend auto_path /path/to/build/tcl
package require mdance          ;# 2.0
# or: load /path/to/build/tcl/mdance_tcl.so Mdance
::mdance::info                  ;# version 2.0.0 api 2 divine 1 threaded 1
```

## Datasets

A dataset holds the coordinates in the library: one frame per row, read and checked once.
It is a command:

```tcl
set d [::mdance::data create $nframes $natoms]
$d frame 0 [$sel get {x y z}]            ;# {x y z} triples, or a flat list of numbers
$d info                                  ;# nframes natoms ncols nset busy
$d get 0                                 ;# a frame's values
$d superpose ?-fit {atom ...}? ?-reference frame?
$d delete
```

Every value is checked as it comes in: a coordinate that is NaN or infinite is refused,
naming the frame and atom. A call on a dataset with a frame never set is refused.

### From VMD, the fast way: a DCD

VMD writes a selection's frames itself, in C. The library reads the file. No Tcl list is
made of the coordinates at all:

```tcl
set sel [atomselect top "name CA"]
set tmp [file join $env(TMPDIR) frames[pid].dcd]
animate write dcd $tmp beg 0 end -1 skip 1 sel $sel waitfor all top
set d [::mdance::data dcd $tmp]          ;# ?-atoms {i ...}? ?-first f? ?-last l? ?-stride s?
file delete $tmp
```

The reader takes CHARMM and X-PLOR DCDs in either byte order, as VMD writes them. It
checks every record: a file cut short, a record of the wrong length, or frames or atoms
it does not hold are errors.

### From a list

```tcl
set d [::mdance::data list $coords $nframes $natoms]    ;# a flat list, frame after frame
```

### Superposition

`$d superpose` fits every frame onto the reference frame (frame 0 by default). It uses
the rotation and translation of least RMSD of the `-fit` atoms (all of them by default,
each weighing the same), applies them to all the frame's atoms, and computes it in C by
Horn's quaternion method.

## Calls

Every call takes a dataset or, as in version 1, a flat list with its counts:

| Call | On a dataset | Version 1 form |
|---|---|---|
| k-means (NANI) | `::mdance::kmeans $d k ?opt?` | `::mdance::kmeans coords nframes natoms k ?opt?` |
| DIVINE | `::mdance::divine $d k ?opt?` | `… coords nframes natoms k ?opt?` |
| HELM | `::mdance::helm $d k labels ?opt?` | `… coords nframes natoms k labels ?opt?` |
| eQUAL | `::mdance::equal $d -threshold r ?opt?` | `… coords nframes natoms ?opt?` |
| iSIM | `::mdance::analysis $d ?-labels l? ?-metric m?` | as eQUAL |
| PRIME | `::mdance::prime $d -labels l ?opt?` | as eQUAL |
| Frame picks | `::mdance::select $d -method m ?opt?` | as eQUAL |

The options:

| Call | Options (defaults) |
|---|---|
| kmeans | `-metric MSD` `-kinit StratAll` `-percentage 10` `-seed -1` |
| divine | `-metric MSD` `-split WeightedMSD` `-anchors NANI` `-kinit StratAll` `-refine` (a flag) `-threshold 0` `-end-mode k` `-percentage 10` |
| helm | `-metric MSD` `-merge-scheme Inter` `-eps -1` `-trim-start` (a flag) `-min-samples 0.01` `-trim-val 0` `-trim-k 0` |
| equal | `-threshold` (required) `-metric MSD` `-seed-method medoid` `-n-seeds 1` `-percentage 10` `-min-samples 10` `-check-sim` (a flag) `-sim-threshold 0` `-reject-lowd` (a flag) `-align none` |
| analysis | `-metric MSD` `-labels {…}` |
| prime | `-labels {…}` (required) `-metric RR` `-trim-frac 0` `-weighted` (a flag) |
| select | `-method diversity` (diversity, outliers, repsample, medoid or outlier) `-metric MSD` `-param 10` `-nbins 10` |

The library checks every option before anything runs, and these are errors that name the
option: an option the call does not take, one given twice, a value that is not a number,
or a value out of range (k above the frames, a percentage over 100, a label past the
frames). In version 1 a misspelt option was ignored and a bad number silently became the
default. Numbers are decimal: `010` is ten.

`-seed` makes k-means' Random and k-means++ seedings repeatable. With `-seed -1` the seed
comes from the clock. The result's `seed` key says which seed was used, so passing it
back gives the same clusters.

The results are the same dicts as in version 1:
- The clusterings return `algorithm`, `nFrames`, `nClusters`, `labels`, `clusterSizes`,
  `representatives`, `clusterMSD`, `score_calinskiHarabasz`, `score_daviesBouldin`, and
  `zMatrix` (HELM) and `seed` (k-means).
- `analysis` returns `isim`, `nClusters`, `clusterISIM` and `clusterOutliers`.
- `prime` returns `pairwise`, `union`, `medoid`, `outlier`, `medoidAll`, `medoidC0`,
  `medoidC0Trimmed` and `nClusters`.
- `select` returns a list of frames.

### Scans

```tcl
set rs [::mdance::scan kmeans $d -ks {2 3 4 5 6 7 8} -kinit StratAll ?-threads n?]
foreach r $rs {
    if {[dict exists $r error]} { puts "k [dict get $r k]: [dict get $r error]"; continue }
    puts "k [dict get $r k]: CH [dict get $r score_calinskiHarabasz]"
}
```

`::mdance::scan kmeans|divine|helm $d -ks {k …}` makes the clustering at every k, the
values run at once on threads. By default it uses one thread a processor; `-threads n`
sets another count. Its options are the algorithm's; HELM's starting clusters are
`-labels {…}`.

It returns a list with one dict a k, in the list's order:
- the clustering's dict, with its `k`, which equals the single call's;
- or `k` and `error` for a value that failed, while the others are made.

It runs as a job too: `::mdance::job start scan kmeans $d -ks {…}`, and Cancel stops
every value still running.

The coordinates' list is read as a list only. Version 1 made its text, about 19 bytes a
number, kept with the list until VMD quit. Its options parser read every argument as
text, the coordinates included.

## Jobs: long calls that leave VMD responsive

A call blocks Tcl until it ends. In VMD that freezes the window. A job runs the same
call on a thread of its own:

```tcl
set j [::mdance::job start -command done kmeans $d 10 -kinit StratAll]
proc done {j} {
    if {[catch {::mdance::job result $j} r]} { puts "failed: $r"; return }
    puts "clusters: [dict get $r clusterSizes]"
}
```

| Command | What it does |
|---|---|
| `::mdance::job start ?-command script? call args…` | Starts the call and returns the job's name. |
| `::mdance::job status $j` | Returns `state running`, `done`, `error` or `cancelled`, plus `progress` (0 to 1) and `stage`. |
| `::mdance::job cancel $j` | Stops the call within a fraction of a second; it then ends with the error `cancelled`. |
| `::mdance::job wait $j` | Blocks until the job ends. |
| `::mdance::job result $j` | Returns the call's result, or raises its error, and forgets the job. |
| `::mdance::job forget $j` | Cancels the job and forgets it. |
| `::mdance::job names` | Lists the jobs. |

`-command` runs the script with the job's name appended, from the event loop, once the
job ends. It needs a threaded Tcl, which VMD's and ActiveTcl's are. Elsewhere, poll
`job status`.

While a job reads a dataset, the dataset cannot be changed: `frame` and `superpose` are
refused. It can be deleted; it is freed once the job ends.

## Using the result in VMD

The library numbers frames as rows of the dataset. If you gave it frames `first`,
`first+step`, …, then row `i` is VMD frame `first + i*step`:

```tcl
set r [::mdance::kmeans $d 5]
set big [lindex [lsort -integer -decreasing -indices [dict get $r clusterSizes]] 0]
animate goto [expr {$first + [lindex [dict get $r representatives] $big] * $step}]
```

## Errors

Every failure is a Tcl error with the library's message. Nothing the extension does
aborts, exits, or writes to VMD's console. An input that used to crash VMD is now an
error. These include:
- a CompSim seeding with too few frames;
- a percentage over 100;
- DIVINE refinement on identical frames;
- a label of 2³¹ − 1;
- coordinates that are not finite.

## Migrating from version 1

The calls, their positional arguments and the result keys are the same. A call that was
right in version 1 gives the same answer in version 2, with two exceptions, both fixes:
- the RR diversity pick returned the number of columns as a frame, repeatedly;
- HELM with as many starting clusters as k, or fewer, merged fewer into one cluster; it
  now leaves them as they are.

The library underneath has the same changes as the C interface, each listed in the
[C API's compatibility section](c-api.md#compatibility-with-version-1). These include
the values it now refuses: a percentage outside 0–100, coordinates that are not
finite, labels outside −1 … nframes − 1, and DIVINE's `end_mode` other than `k` or
`points`. The extension adds its own changes, which matter to a script that worked by
accident:

| Version 1 | Version 2 |
|---|---|
| An option it did not know, or a misspelt one, was ignored, and the call ran with the default | an error that lists the call's options |
| An option given twice: the first was used | an error |
| A value that was not a number (`-percentage abc`) silently became the default | an error naming the option |
| A flag was found anywhere among the arguments, so `-refine 1` worked: the `1` was ignored | a flag takes no value; the `1` is read as an option and refused |
| An option without its value at the end was ignored | an error |
| Coordinates whose count is not a multiple of the frames: the rest was dropped | refused |
| `-labels` for `analysis` and `prime` of any length: cut or padded | one label per frame |
| Every call printed a k-means seed or DIVINE's splits to VMD's console | nothing is printed |

Also:
- `::mdance::version` returns `2.0.0`;
- the result dicts have the same keys, plus `seed` for k-means;
- `package require mdance_native`, the first version's name, still works.
