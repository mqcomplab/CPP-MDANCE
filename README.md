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

## Important files
- `src/cluster/KmeansRex/KmeansRexCore.cpp`: Has **NANI** implementation
- `src/cluster/divine.cpp`: Has **DIVINE** implementation
- `src/tools`: Has supporting functions, such as BTS, type definitions, and score calculations.
- <span style="color:red">TODO:</span> implement HELM
- `tests/runTests.sh`: Bash script for testing code by comoparing output to that of the Python library
   - `tests/data`: stores datasets
   - `tests/results`: Stores the results of the test. The results themselves are stored in the `*Results.txt` files, while the time and any error messages are stored in `*Time.txt` files
