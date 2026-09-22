# VMD / Tcl extension

`tcl/mdance_tcl.cpp` wraps the [C API](c-api.md) as a loadable Tcl extension, so MDANCE
runs *inside* a VMD session against the trajectory already in memory. No CSV export, no
round trip through the filesystem, and the frame indices that come back are the frame
indices VMD is showing you.

It is a normal Tcl extension, so it also loads into plain `tclsh` — which is the easiest
way to try it.

## Building

The Tcl extension is **off by default**, because it needs Tcl development headers:

```shell
cmake -S . -B build -DBUILD_TCL=ON
cmake --build build -j
```

That produces `build/tcl/mdance_tcl.so`. If configuration fails with *"Tcl stub library
(libtclstub8.6) not found"*, install the Tcl dev package (`apt install tcl-dev`,
`brew install tcl-tk`).

The extension links against the Tcl **stub** library rather than the Tcl runtime, so it
binds to whatever interpreter loads it. That is what makes it safe inside VMD: linking the
full runtime would pull a second Tcl into the process, which aborts with
`alloc: invalid block`.

## Loading

```tcl
load /path/to/CPP-MDANCE/build/tcl/mdance_tcl.so mdance
puts [::mdance::version]      ;# -> 1.0
```

The second argument is the prefix Tcl uses to find the entry point (`Mdance_Init`); pass
it explicitly rather than relying on Tcl deriving it from the filename.

`mdance_tcl.so` depends on `libmdance.so` next to it in `build/capi/`. The build embeds a
path to it, so loading from the build tree just works. If you relocate either file, set
`LD_LIBRARY_PATH` (or `DYLD_LIBRARY_PATH`) to the directory holding `libmdance.so` before
starting VMD.

Loading also registers the package, so a later `package require mdance_native` is
satisfied. There is no `pkgIndex.tcl`, so `load` has to come first.

## Getting coordinates out of VMD

Every command takes the trajectory as one flat list of doubles, frame-major: all of frame
0's values, then all of frame 1's. Flatten an `atomselect` over the frames:

```tcl
set sel     [atomselect top "name CA"]
set natoms  [$sel num]
set nframes [molinfo top get numframes]

set coords {}
for {set f 0} {$f < $nframes} {incr f} {
    $sel frame $f
    foreach xyz [$sel get {x y z}] {
        lappend coords {*}$xyz
    }
}
```

That gives `nframes * natoms * 3` values. The commands derive the column count themselves
as `[llength $coords] / $nframes`; `natoms` is passed separately because it is the MSD
normalization divisor, not a shape.

**Align first.** MDANCE compares coordinates directly, so a trajectory that has not been
fit to a reference clusters on rigid-body motion rather than on conformation. Use VMD's
`measure fit` / `$sel move` before extracting.

## Clustering

```tcl
set res [::mdance::kmeans $coords $nframes $natoms 10 -metric MSD -kinit CompSim]

puts "sizes:  [dict get $res clusterSizes]"
puts "reps:   [dict get $res representatives]"
puts "CH:     [dict get $res score_calinskiHarabasz]"
```

| Command | Signature |
|---|---|
| `::mdance::kmeans` | `coords nframes natoms nclusters ?-metric MSD? ?-kinit StratAll? ?-percentage 10?` |
| `::mdance::helm` | `coords nframes natoms nclusters initial_labels ?-metric MSD? ?-merge-scheme Inter? ?-eps -1? ?-trim-start? ?-min-samples 0.01? ?-trim-val 0? ?-trim-k 0?` |
| `::mdance::equal` | `coords nframes natoms -threshold <f> ?-metric MSD? ?-seed-method medoid? ?-n-seeds 1? ?-percentage 10? ?-check-sim? ?-sim-threshold 0? ?-reject-lowd? ?-min-samples 10? ?-align none?` |
| `::mdance::divine` | `coords nframes natoms nclusters ?-metric MSD? ?-split WeightedMSD? ?-anchors NANI? ?-kinit StratAll? ?-refine? ?-threshold 0? ?-end-mode k? ?-percentage 10?` |

All four return a dict with `algorithm`, `nFrames`, `nClusters`, `labels` (one per frame),
`clusterSizes`, `representatives`, `clusterMSD`, `score_calinskiHarabasz` and
`score_daviesBouldin`. HELM adds `zMatrix`, a list of rows.

`::mdance::helm` takes the initial partition as its fifth positional argument — a list of
one integer per frame. The values are arbitrary: they need not be `0..K-1` or even
non-negative, so an eQUAL result containing `-1` can be fed straight back in.

`::mdance::divine` is compile-gated and errors with *"DIVINE is not available in this
build"* unless the library was built with `BUILD_DIVINE`.

## Analysis, PRIME and selection

```tcl
set labels [dict get $res labels]

# Extended similarity, globally and per cluster
set a [::mdance::analysis $coords $nframes $natoms -metric MSD -labels $labels]
puts "isim [dict get $a isim], per-cluster [dict get $a clusterISIM]"

# Representative-frame prediction
set p [::mdance::prime $coords $nframes $natoms -metric RR -labels $labels -trim-frac 0.1]
puts "medoidAll [dict get $p medoidAll]"

# Frame selection with no clustering; returns a flat LIST, not a dict
set frames [::mdance::select $coords $nframes $natoms -method repsample -param 20 -nbins 10]
```

`::mdance::analysis` returns `isim`, `nClusters`, `clusterISIM`, `clusterOutliers`;
`-labels` is optional and without it you get the global `isim` alone. `::mdance::prime`
returns `pairwise`, `union`, `medoid`, `outlier`, `medoidAll`, `medoidC0`,
`medoidC0Trimmed` and `nClusters`, requires `-labels`, and accepts only `RR` or `SM` —
read the caveat on its predictors in the
[quickstart](mdance-cli-quickstart.md#prime) before trusting them on continuous
coordinates. `::mdance::select` is the odd one out: it returns a plain list of frame
indices.

## Using the result in VMD

`representatives` and the PRIME/selection outputs are frame indices into the trajectory you
extracted, so they feed straight back into VMD:

```tcl
# Jump to the medoid of the largest cluster
animate goto [lindex [dict get $res representatives] 0]

# Keep only a representative subsample
set keep [::mdance::select $coords $nframes $natoms -method repsample -param 50]
```

To colour by cluster, write the label into a per-atom field (`user` is the usual choice)
frame by frame, then colour the representation by `User`:

```tcl
set all [atomselect top "all"]
for {set f 0} {$f < $nframes} {incr f} {
    $all frame $f
    $all set user [lindex $labels $f]
}
```

## Error handling

Failures come back as ordinary Tcl errors with the message as the result, so `catch` is
all you need:

```tcl
if {[catch {::mdance::equal $coords $nframes $natoms -threshold 8} res]} {
    puts "mdance failed: $res"
}
```

## Gotchas

**Unknown options are ignored.** Options are matched by name wherever they appear; a
misspelled one is not an error, it just leaves the default in place. If a run ignores your
settings, check the spelling against the table above.

**`nframes` must match the data you flattened.** The column count is derived as
`[llength $coords] / $nframes`, so an `nframes` that is too large silently reshapes the
trajectory into the wrong matrix and clusters nonsense. Zero or negative is rejected.

**k-means needs enough frames to seed from.** The `StratAll`, `CompSim`, `StratReduced`
and `DivSelect` seeders draw from `-percentage`% of the frames, so on a short trajectory
they can come up with fewer seeds than clusters and the call errors with *"initialization
produced only N of the K requested centers"*. Raise `-percentage` (up to 100) or ask for
fewer clusters.

**`-natoms` changes your numbers.** It is the MSD divisor, so thresholds and iSIM values
are not comparable between runs that used different values. For a Cartesian selection it
is `[$sel num]`.
