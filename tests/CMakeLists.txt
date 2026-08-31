cmake_minimum_required(VERSION 3.15)

find_package(GTest REQUIRED)

# NOTE: do not name the target "test" -- that name is reserved by CMake once
# CTest support is enabled, and it also shadows the conventional `make test`.
add_executable(mdance_gtests test_utils.cpp
                    test_main.cpp
                    test_kmeans.cpp
                    test_helm.cpp
)
target_link_libraries(mdance_gtests PRIVATE
                    GTest::GTest
                    GTest::Main
                    mdance)

add_test(NAME mdance_tests COMMAND mdance_gtests)
