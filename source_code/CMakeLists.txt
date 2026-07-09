cmake_minimum_required(VERSION 3.10)
project(test_calc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(Python_FIND_VIRTUALENV FIRST)
set(Python3_ROOT_DIR "/home/honestkids/miniforge3/envs/mdance")
set(Python3_EXECUTABLE "/home/honestkids/miniforge3/envs/mdance/bin/python3.1")
set(Python3_INCLUDE_DIR "/home/honestkids/miniforge3/envs/mdance/include/python3.14")
set(Python3_LIBRARY "/home/honestkids/miniforge3/envs/mdance/lib/libpython3.14.so")
message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES = ${Python3_LIBRARIES}")
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_executable(test_calc
    test_calc.cpp
)


target_include_directories(test_calc
    PRIVATE 
        ${EIGEN3_INCLUDE_DIR}
        "../src/" # relative path, can be changed to absolute if needed
)

target_link_directories(test_calc
    PRIVATE
        "../build/src"
)

target_link_libraries(test_calc PRIVATE
    mdance_divine
    mdance_helm
    mdance_kmeans
    mdance_tools
    python_bridge
    Eigen3::Eigen
    Python3::Python
    pybind11::embed
)
set(CMAKE_CXX_FLAGS "-g -O0" CACHE STRING "Debug flags" FORCE)





