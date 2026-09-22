# CPP-MDANCE
A c++ implementation of MDANCE, a flexible n-ary clustering package for all applications 

## Getting Started with MDance

Before you begin, make sure you have **Eigen** installed.

### Step 1: get source code

To build MDance from  source, first clone the GitHub repository:
```shell
git clone https://github.com/mqcomplab/CPP-MDANCE.git
```
Then navigate into the directory:
```shell
cd CPP-MDANCE
```
### Step 2: Configure with CMake

Run CMake to generate the build configuration files:
```shell
cmake -S . -B build
```
(Optional) You can choose from three different build types:
1. **Release (Default)**
2. **Debug**
3. **RelWithDebInfo**
To specify a different build type:
```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=<BuildType>
```
### Step 3: Build MDance
Compile MDance by running:
```shell
cmake --build build
```
### Step 4: Run Tests
Navigate to `build/tests/` folder:
```shell
cd build/tests/
```
Use `ctest` to execute the tests:
```shell
ctest
```
An example output is:
```shell
    Start 1: mdance_tests
1/1 Test #1: mdance_tests .....................   Passed    2.71 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   2.71 sec
```

<span style="color:red">TODO:</span> add instructions for installation and figure out how to make CPP-MDANCE easily portable.

### Step 5: Run a clustering job

The build produces `build/cli/mdance-cli`, a command-line front end covering clustering,
similarity analysis, representative-frame prediction, and frame selection:

```shell
./build/cli/mdance-cli --algorithm kmeans \
                       --input tests/data/sim.csv --output result.json \
                       --natoms 50 --nclusters 10
```

See **[docs/mdance-cli-quickstart.md](docs/mdance-cli-quickstart.md)** for the input
format, all five modes, the output schema, and common pitfalls.

The same algorithms are reachable without files: **[docs/c-api.md](docs/c-api.md)** covers
the `libmdance` C interface, and **[docs/vmd-tcl.md](docs/vmd-tcl.md)** covers the Tcl
extension that runs MDANCE inside a VMD session against the loaded trajectory.

## Important files

### Algorithms
- `src/cluster/KMeansRex/KMeans.cpp`: **NANI** (k-means with n-ary initialization)
- `src/cluster/helm.cpp`: **HELM** hierarchical merging
- `src/cluster/equal.cpp`: **eQUAL** radial/threshold clustering
- `src/cluster/prime.cpp`: **PRIME** representative-frame prediction
- DIVINE lives on a separate dev branch; `BUILD_DIVINE` auto-detects whether
  `src/cluster/divine.cpp` is present and stays off when it is not.

### Supporting code
- `src/tools/`: BTS, extended-similarity (esim), type definitions and cluster scores
- `cli/`: the `mdance-cli` front end (argument parsing, CSV input, JSON output)
- `capi/`: C API for the shared library, used by the VMD/Tcl integration
- `tcl/`: Tcl extension for VMD

### Tests and docs
- `tests/*.cpp`: GoogleTest suite; run everything with `ctest` from `build/`
- `tests/validate_equal_prime.py`: drives `mdance-cli` and cross-checks eQUAL, PRIME and
  frame selection against an independent NumPy reference (skipped if NumPy is missing)
- `tests/data/`: datasets used by both suites
- `docs/`: Sphinx documentation sources (`make -C docs html`) — CLI quickstart, C API
  reference, VMD/Tcl guide
