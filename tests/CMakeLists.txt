cmake_minimum_required(VERSION 3.15)

# GoogleTest, handled like Eigen in src/CMakeLists.txt: prefer an installed
# copy, fetch one when there is none, and if neither is possible skip the test
# target rather than failing the configure -- someone who only wants to build
# the library should not be required to have a test framework.
find_package(GTest QUIET)
option(MDANCE_FETCH_GTEST "Download GoogleTest when none is installed" ON)
if(NOT GTest_FOUND AND MDANCE_FETCH_GTEST)
    message(STATUS "GoogleTest not found; fetching v1.15.2")
    include(FetchContent)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(googletest)
endif()

# Target names differ by source: a modern FindGTest and the fetched build both
# provide GTest::gtest_main, the fetched build also the bare gtest_main, and an
# older FindGTest only GTest::Main.
if(TARGET GTest::gtest_main)
    set(MDANCE_GTEST_LIBS GTest::gtest GTest::gtest_main)
elseif(TARGET gtest_main)
    set(MDANCE_GTEST_LIBS gtest gtest_main)
elseif(TARGET GTest::Main)
    set(MDANCE_GTEST_LIBS GTest::GTest GTest::Main)
else()
    message(STATUS
        "GoogleTest unavailable (and MDANCE_FETCH_GTEST=OFF); skipping the test "
        "target. The library still builds.")
    return()
endif()

# NOTE: do not name the target "test" -- that name is reserved by CMake once
# CTest support is enabled, and it also shadows the conventional `make test`.
add_executable(mdance_gtests test_utils.cpp
                    test_main.cpp
                    test_kmeans.cpp
                    test_helm.cpp
                    test_bts.cpp
                    test_tools.cpp
                    test_equal.cpp
                    test_prime.cpp
)
target_link_libraries(mdance_gtests PRIVATE
                    ${MDANCE_GTEST_LIBS}
                    mdance)
# GoogleTest 1.15 requires C++14; the rest of this project is C++17.
target_compile_features(mdance_gtests PRIVATE cxx_std_17)

# The fixtures locate their data as
#   current_path().parent_path().parent_path() / "tests" / "data"
# (see the comment in test_kmeans.cpp), which only resolves when the working
# directory is exactly two levels below the source root -- true for an in-source
# build/tests/, false for every out-of-source build directory. Running the suite
# from tests/data makes that expression resolve for any build layout.
add_test(NAME mdance_tests COMMAND mdance_gtests
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)
