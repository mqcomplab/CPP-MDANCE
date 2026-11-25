cmake_minimum_required(VERSION 3.15)
project(MDance VERSION 1.0 LANGUAGES CXX)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# The build types keep CMake's own flags (Release: -O3 -DNDEBUG), which a
# user can change. -march=native used to be forced into Release: a library
# built so runs only on CPUs with the builder's instruction sets, and one
# handed to another machine (a VMD plugin shipped with VMD) dies of an
# illegal instruction there. It is an option now, for a build that stays on
# the machine it was built on.
option(MDANCE_NATIVE "Optimize for this machine's CPU (-march=native): not portable" OFF)
if(MDANCE_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
endif()

# Enable testing at the top level so `ctest` run from the build root discovers
# the tests registered in subdirectories (must precede add_subdirectory).
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

option(BUILD_CLI "Build mdance-cli command-line tool" ON)
option(BUILD_BENCH "Build the benchmark programs in bench/" OFF)
if(BUILD_CLI)
    add_subdirectory(cli)
endif()

# A module loaded into another program (VMD) brings its own C++ runtime
# when built with MinGW: libstdc++, libgcc and winpthread linked in, so that
# no DLL of another version, already loaded by the host or found first on
# the PATH, is used in its place.
option(MDANCE_STATIC_RUNTIME "Link the C++ runtime statically into the shared library and the Tcl extension (MinGW)" ON)
function(mdance_static_runtime target)
    if(MINGW AND MDANCE_STATIC_RUNTIME)
        target_link_options(${target} PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic)
    endif()
endfunction()

option(BUILD_SHARED "Build shared library with C API" ON)
if(BUILD_SHARED OR BUILD_TCL)
    add_subdirectory(capi)
endif()

option(BUILD_TCL "Build Tcl extension for VMD" OFF)
if(BUILD_BENCH)
    add_subdirectory(bench)
endif()
if(BUILD_TCL)
    add_subdirectory(tcl)
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