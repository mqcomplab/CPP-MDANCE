# Benchmark programs (BUILD_BENCH): mdance_bench times every C-API call on a
# CSV and writes their answers for comparison between builds; mdance_profile
# times the phases inside a k-means call.
enable_language(C)
add_executable(mdance_bench mdance_bench.c)
target_link_libraries(mdance_bench PRIVATE mdance_shared)
add_executable(mdance_profile mdance_profile.cpp)
target_link_libraries(mdance_profile PRIVATE mdance_tools mdance_kmeans)
target_compile_features(mdance_profile PRIVATE cxx_std_17)
