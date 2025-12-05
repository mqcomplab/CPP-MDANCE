cmake_minimum_required(VERSION 3.15)

add_executable(mdance_tests 
    parseData.cpp)

target_link_libraries(mdance_tests PRIVATE mdance)

add_executable(parser 
    parseTests.cpp)