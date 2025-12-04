# MDance tests

The tests we have for KMeans work as follows: 
1. run KMeans in C++ (this is the executable generated from `parseData.cpp`)
2. run KMeans in Python (the Python script is: `parseData.py`)
3. compare results with `parseTests.cpp`

When building MDance with `CMake`, the test executables are built in `build/tests`. Therefore, `runTests.sh` copies these executables to the `tests` folder, since this is where the input data is. 

**Note:** having unit tests with `GTest` or `Catch2` that only rely on the C++ implementation would be preferable. These may be added later down the line if there is a need for them. 