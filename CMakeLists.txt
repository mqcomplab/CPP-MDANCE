cmake_minimum_required(VERSION 3.15)
project(MDance VERSION 1.0 LANGUAGES CXX)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# Compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug flags" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "Release flags" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2 -DNDEBUG" CACHE STRING "RelWithDebInfo flags" FORCE)

# Enable testing at the top level so `ctest` run from the build root discovers
# the tests registered in subdirectories (must precede add_subdirectory).
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

option(BUILD_CLI "Build mdance-cli command-line tool" ON)
if(BUILD_CLI)
    add_subdirectory(cli)
endif()

# eQUAL / PRIME / frame-selection validation. It drives the built CLI and
# cross-checks against an independent NumPy reference, so it is registered here
# rather than in tests/, where the mdance-cli target does not exist yet.
if(BUILD_CLI)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        # The validator imports numpy. Registering it unconditionally would turn
        # a missing optional dependency into a FAILING test rather than an absent
        # one, which reads as "the code is broken".
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                        RESULT_VARIABLE _mdance_numpy_rc
                        OUTPUT_QUIET ERROR_QUIET)
        if(_mdance_numpy_rc EQUAL 0)
            add_test(NAME validate_equal_prime
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_equal_prime.py)
            set_tests_properties(validate_equal_prime PROPERTIES
                ENVIRONMENT "MDANCE_CLI=$<TARGET_FILE:mdance-cli>")
        else()
            message(STATUS
                "numpy not importable by ${Python3_EXECUTABLE}; skipping the "
                "validate_equal_prime test (pip install numpy to enable it).")
        endif()
    endif()
endif()