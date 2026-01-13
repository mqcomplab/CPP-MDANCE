cmake_minimum_required(VERSION 3.15)

find_package(GTest REQUIRED)

add_executable(test test_utils.cpp 
                    test_main.cpp
<<<<<<< HEAD
                    test_kmeans.cpp
                    test_helm.cpp
)
target_link_libraries(test PRIVATE 
                    GTest::GTest
                    GTest::Main
                    mdance)
                    
enable_testing()
add_test(NAME mdance_tests COMMAND test)