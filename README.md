# CPP-MDANCE
A c++ implementation of MDANCE, a flexible n-ary clustering package for all applications 

308

## Getting Started with MDance

Before you begin, make sure you have **Eigen** installed.

### Step 1: get source code

To build MDance from  source, first clone the GitHub repository:
```
git clone https://github.com/mqcomplab/CPP-MDANCE.git
```
Then navigate into the directory:
```
cd CPP-MDANCE
```
### Step 2: Configure with CMake

Run CMake to generate the build configuration files:
```
cmake -S . -B build
```
(Optional) You can choose from three different build types:
1. **Release (Default)**
2. **Debug**
3. **RelWithDebInfo**
To specify a different build type:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=<BuildType>
```
### Step 3: Build MDance
Compile MDance by running:
```
cmake --build build
```
### Step 4: Run Tests
Navigate to the test folder:
```
cd tests
```
Use the shell script `runTests.sh` to execute the tests:
```
./runTests.sh
```
The expected output is:
```
Coppying test executable from build directory...
Calculating Python results...
Calculating C++ results...
Comparing results...

NANI
-------------------------------------------
test [1/1]
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
