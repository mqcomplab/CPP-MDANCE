# CPP-MDANCE
A c++ implementation of MDANCE, a flexible n-ary clustering package for all applications 

## Dependencies

You need to have Eigen installed.   
<span style="color:red">TODO:</span> add instructions for installation and figure out how to make CPP-MDANCE easily portable.

## Important files
- `src/cluster/KmeansRex/KmeansRexCore.cpp`: Has **NANI** implementation
- `src/cluster/divine.cpp`: Has **DIVINE** implementation
- `src/tools`: Has supporting functions, such as BTS, type definitions, and score calculations.
- <span style="color:red">TODO:</span> implement HELM
- `tests/runTests.sh`: Bash script for testing code by comoparing output to that of the Python library
   - `tests/data`: stores datasets
   - `tests/results`: Stores the results of the test. The results themselves are stored in the `*Results.txt` files, while the time and any error messages are stored in `*Time.txt` files