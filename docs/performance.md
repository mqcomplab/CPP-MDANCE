# Performance

This page compares four things: CPP-MDANCE before this work ("stock", upstream main
3a2f073), CPP-MDANCE now, MDANCE (the Python package, 0.5.0), and the VMD Tcl extension
before and after. Every comparison makes the same calls on the same data. The programs
are in `bench/`, so anyone can repeat it.

## How it was measured

- **Data.** Two sets:
  - `tests/data/sim.csv`: 6001 frames of 50 atoms (150 columns).
  - The tau peptide trajectory of MDANCE's examples, heavy atoms: 6002 frames of 99 atoms
    (297 columns), written from VMD.
- **Machine.** Windows 11, 4 cores. GCC 13.2 (MinGW-w64), Eigen 3.4, VMD 2.0.0. Python
  3.12 with numpy 2.5.5, scikit-learn 1.9.1 and MDANCE 0.5.0.
- **One run at a time.** Each run started only once total CPU had averaged under 30% for
  20 s. Background load at the starts was 11–28%, from the desktop's own programs.
  Differences under about 15% are within that noise.
- **The C programs.**
  - `bench/mdance_bench.c` times every C-API call. It writes each call's answers (labels,
    sizes, representatives, MSDs, scores, Z matrix, picks) at full precision, so two
    builds can be compared.
  - `bench/mdance_profile.cpp` times the phases inside a k-means call.
  - `bench/vmd_bench.tcl` runs inside VMD.
- **The Python program.** `bench/mdance_py_bench.py` makes the same calls with MDANCE.
  After each clustering it also computes what every C-API result carries: each cluster's
  medoid and MSD, plus scikit-learn's CH and DB scores.
- **Threads.** CPP-MDANCE runs on one thread. MDANCE runs with numpy's and
  scikit-learn's default threads, which here means all 4 cores.
- **Builds.** Stock is shown two ways:
  - as shipped, with `-march=native` forced into Release;
  - with the portable flags that are now the default (`-O3`).

  Ours is shown both ways too: portable, and with `-DMDANCE_NATIVE=ON`.

## Are the answers the same?

### Stock against ours, same flags

All 45 calls on sim.csv and the 44 on tau were compared: labels, sizes, representatives,
cluster MSDs, scores, HELM's Z matrix, PRIME's frames, every pick. All are identical to
the last digit, except one fix:
- **The RR diversity pick.** Stock returned the number of columns as a "frame", over and
  over: `2031,150,150,150,…` on sim.csv. Ours returns 600 distinct frames.

A third data set (tau Cα, 12 atoms) agrees too: 44 of 45, with the same single
difference.

Native builds differ from portable ones in the last digits of two cluster MSDs. The
reason is that FMA instructions round differently. Stock-native and ours-native agree
with each other on every call except the same RR fix.

### MDANCE (Python) against ours

| Call | Agreement (both data sets) |
|---|---|
| k-means, all four NANI seedings | the same labels |
| HELM (k 5 from 40) | the same labels |
| eQUAL (medoid seeds) | the same labels |
| DIVINE NANI, OutlierPair, SplinterPair | the same partitions, numbered otherwise (ARI 1.0) |
| DIVINE OutlierPair with refinement | ARI 0.78 (sim), 0.84 (tau) |

The refined OutlierPair split differs in stock CPP-MDANCE as well: it is a difference
between CPP-MDANCE and MDANCE, not one this work made. It is worth a look upstream.

## The numbers

### sim.csv (6001 frames × 50 atoms), in milliseconds

| Call | MDANCE (Python) | Stock (native) | Stock (portable) | **Now (portable)** | Now (native) |
|---|---:|---:|---:|---:|---:|
| k-means k=10, StratAll | 552 | 335 | 390 | **236** | 206 |
| k-means k=10, CompSim | 208 | 340 | 350 | **169** | 158 |
| k-means k=10, DivSelect | 1 131 | 2 365 | 2 755 | **427** | 388 |
| k-means scan k = 2…20 | **1 583** | 5 426 | 5 762 | 4 391 | 4 036 |
| HELM k=5 from 40 | 138 | 52 | 58 | **47** | 51 |
| DIVINE k=10, NANI | 509 | 379 | 377 | **330** | 322 |
| DIVINE k=10, SplinterPair | 1 245 | 258 | 312 | **200** | 210 |
| eQUAL, medoid seeds | 34 865 | 4 720 | 5 088 | **1 262** | 1 428 |
| eQUAL, comp_sim seeds | > 1 800 000 (stopped at 30 min) | 317 076 | 307 904 | **11 082** | 8 014 |
| PRIME (RR) | 371 671 | 2 181 | 2 153 | **1 735** | 1 696 |
| Diversity pick, 10% (MSD) | 53 238 | 2 185 | 2 389 | **345** | 255 |
| Medoid, outliers, rep. sample | 16–25 | 17–21 | 15–16 | **12–13** | 12–14 |

### tau heavy atoms (6002 frames × 99 atoms), in milliseconds

| Call | MDANCE (Python) | Stock (native) | Stock (portable) | **Now (portable)** | Now (native) |
|---|---:|---:|---:|---:|---:|
| k-means k=10, StratAll | 301 | 632 | 606 | **435** | 460 |
| k-means k=10, DivSelect | 1 238 | 5 615 | 5 876 | **1 581** | 1 770 |
| k-means scan k = 2…20 | **3 784** | 13 359 | 11 905 | 8 576 | 11 158 |
| DIVINE k=10, SplinterPair | 2 034 | 663 | 719 | **451** | 461 |
| eQUAL, medoid seeds | 279 374 | 65 692 | 69 117 | **20 697** | 21 224 |
| eQUAL, comp_sim seeds | not run | not run (over an hour) | not run (over an hour) | **245 774** | 183 462 |
| PRIME (RR) | not run | 4 310 | 4 611 | **3 753** | 4 047 |
| Diversity pick, 10% (MSD) | 55 979 | 4 145 | 4 578 | **691** | 685 |

How these compare:
- **Against stock:**
  - eQUAL with comp_sim seeds: 28 times faster.
  - eQUAL with medoid seeds, the diversity pick and DivSelect seeding: 3.3–7 times
    faster.
  - The rest: 1.1–2 times faster, with the same answers.
- **Against MDANCE (Python):** CPP-MDANCE is faster at everything except plain k-means.
  - PRIME: 210 times faster.
  - Diversity pick: 80–150 times.
  - eQUAL: 14–28 times.
  - DIVINE: 1.5–6 times.
- **Where MDANCE wins: the k-means scan,** 1.6 s against 4.4 s. Its Lloyd iterations
  are scikit-learn's, which run on all 4 cores with an optimized math library. Ours run
  on one core with Eigen's matrix product.

### Inside VMD (tau, 6001 frames × 99 atoms), in milliseconds

**Stock** is the first Tcl extension, used the way the plugin uses it today: `$sel get`
for every frame, made into one flat list, and that list passed to every call. **Now** is
the new extension: VMD writes the frames to a DCD, the library reads them into a dataset
once, and every call takes the dataset.

| Step | Stock | Now |
|---|---:|---:|
| Frames into the library | 292 (`$sel get`, list) | 444 (DCD) |
| k-means k=10 | 948 | 589 |
| k-means scan k = 2…20 | 13 909 | 9 198 |
| HELM | 115 | 81 |
| DIVINE NANI | 810 | 626 |
| eQUAL, medoid seeds | 68 517 | 20 116 |
| PRIME | 4 371 | 3 619 |
| Diversity pick | 4 530 | 627 |

**The DCD route is not faster at this size.** VMD's `animate write` plus the library's
reader cost 444 ms, against 292 ms for `$sel get`. Its gain is memory, and it grows with
size:
- The coordinates never become a Tcl list. Tcl keeps about 120 bytes a number of such a
  list, until VMD quits.
- The first extension also made the list's text, about 19 bytes a number, because its
  option parser read the coordinates as a string. That text also stayed.
- At 100 000 frames × 1 000 atoms, those Tcl copies would be tens of gigabytes.

**A long call as a job:** eQUAL with comp_sim seeds took 235 s as a job. Meanwhile Tcl's
event loop ran 15 223 times, once every 15 ms, so VMD kept drawing and answering. Before,
the same call froze VMD for its whole length.

## What made it slow, and what was done

The work was profiled with `mdance_profile` and by reading the code.

- **Diversity selection** (k-means' DivSelect and CompSim seeding, eQUAL's comp_sim
  seeds, the diversity pick). Each step scored every candidate through
  `extendedComparison`. That built a 2-row matrix and two vectors per candidate and read
  the candidate's row through a column-major array: a cache miss per value.
  - Now each step scores all the candidates with one matrix-vector product, using
    $nS + n|x|^2 - |c|^2 - 2c\cdot x - |x|^2$ for the MSD.
  - That score is bounded with its rounding error and that of the original sum. Only the
    candidates within that bound of the best are computed the original way, in the same
    order. So the frame chosen is the one stock chose.
  - Allocations are gone and the rows are read contiguously.
- **eQUAL** (96 s per call in VMD before). Two costs:
  - Each iteration computed the complementary similarity of every active frame, then
    sorted all of them to take the first.
  - Each seed was compared with each frame by allocating a pair matrix.

  Now:
  - The first rows of the order come from a bounded algebraic estimate. Only the rows that
    can reach the top are computed exactly, and a full sort is the fallback whenever a tie
    is possible.
  - The radial step decides each pair by a matrix product with an error bound. Only pairs
    near the threshold go through the original formula.
- **calculateCompSim.** It made three temporaries the size of the data: 2 GB for 100 000
  frames of 300 atoms. It now sweeps the columns into one accumulator per row, with no
  temporary.
  - Its summation order is now explicit, and the same for a row wherever the row is.
    Eigen grouped rows into packets and summed the last rows otherwise.
  - This is the one change of arithmetic: last-digit differences in complementary
    similarities. Every label, pick and representative compared above is unchanged.
- **k-means.**
  - Its centres were summed through strided rows at every iteration; now column by column.
  - Each frame's squared norm is computed once rather than at every iteration.
  - The data are moved rather than copied twice.
  - What remains is Eigen's matrix product, in 45 Lloyd iterations at k = 10. That
    product is where MDANCE's scikit-learn wins (see below).
- **PRIME.** The pairwise scorer allocated a vector for every pair of frames; now it uses
  one buffer and contiguous rows.
- **DIVINE.** It rescored every cluster from a copy of its frames at every split: O(N·D)
  a split, O(N²·D) to the end. Each cluster's score is now kept until its frames change.
- **The results** (representatives and cluster MSDs) scanned every label once per
  cluster; now they are bucketed in one pass.
- **The Tcl extension.**
  - Its option parser turned every argument into text, the coordinates included.
  - Every call converted and copied the list again; a scan converted it 19 times.
  - Datasets now hold the coordinates once, and jobs keep VMD responsive.

## What is still slow

- **The k-means scan.** A scan is up to 19 independent k-means runs; ours run one at a
  time on one core. Options:
  - run the k's of a scan on threads: they share the dataset;
  - or use Elkan's or Hamerly's bounds in the Lloyd iterations.

  Either should close most of the distance to scikit-learn. But k-means with the same
  seeds must keep giving the same labels, which the golden-output comparison would check.
- **eQUAL with comp_sim seeds on large sets.** It is 28 times faster, but still minutes on
  6000 frames of 99 atoms: every iteration diversity-selects 10% of all the frames again.
  Seeds whose neighbourhood was already computed could be remembered between iterations.
- **PRIME's pairwise scorer.** It is O(|c0| · N · D) by construction.
- **HELM.** Its distance matrix is rebuilt at every merge: O(k³) for k starting clusters.
  It is fast for the tens to hundreds of starting clusters in use; starts are capped at
  5 000.

## Repeating it

```shell
cmake -S . -B build -DBUILD_BENCH=ON -DBUILD_TCL=ON
cmake --build build -j
build/bench/mdance_bench tests/data/sim.csv 50 answers.txt     # timings to stdout
python bench/mdance_py_bench.py tests/data/sim.csv 50 answers_py.txt
vmd -dispdev text -e bench/vmd_bench.tcl -args build/tcl/mdance_tcl.dll system.pdb traj.dcd "noh" new
```

Comparing two builds' `answers.txt` with `diff` shows any call whose answers changed.
`BENCH_SKIP_COMPSIM=1` leaves out eQUAL with comp_sim seeds, which takes tens of minutes
on the stock library.
