cmake_minimum_required(VERSION 3.9)

project(MDance,
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT GENERATOR_IS_MULTI_CONFIG AND CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "Setting build type to 'Release' as none was specified."
    )
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(MDance STATIC
    src/cluster/divine.cpp 
    src/cluster/KMeansRex/KMeans.cpp 
    src/cluster/KMeansRex/KMeansRexCore.cpp 
    src/cluster/KMeansRex/mersenneTwister2002.c
    src/tools/bts.cpp 
    src/tools/esim.cpp 
    # src/tools/scores.cpp
    )

find_package(Eigen3 REQUIRED)

target_link_libraries(MDance 
    PRIVATE
        Eigen3::Eigen)

target_compile_options(MDance PRIVATE 
    $<$<CONFIG:Release>:-O3 -march=native -mtune=native>
    $<$<CONFIG:Debug>:-O0 -g>)
