cmake_minimum_required(VERSION 3.15)

enable_language(C CXX)

# The code uses the Eigen 3.4 slicing API (Eigen::all, Eigen::seq), so 3.3 is
# not enough even though it used to configure successfully.
find_package(Eigen3 3.4 REQUIRED)

# Create library for tools
add_library(mdance_tools STATIC
    tools/bts.cpp
    tools/cluster.cpp
    tools/hc_utils.cpp
    tools/esim.cpp
    tools/scores.cpp
)
target_include_directories(mdance_tools PUBLIC
    tools)
target_link_libraries(mdance_tools PUBLIC Eigen3::Eigen)

# create library for kmeans
add_library(mdance_kmeans STATIC 
    cluster/KMeansRex/KMeans.cpp
    cluster/KMeansRex/mersenneTwister2002.c
)
set_source_files_properties(cluster/KMeansRex/mersenneTwister2002.c PROPERTIES 
    LANGUAGE C
)
target_include_directories(mdance_kmeans PUBLIC cluster/KMeansRex)
target_link_libraries(mdance_kmeans PUBLIC mdance_tools)


# create library for helm
add_library(mdance_helm STATIC 
    cluster/helm.cpp
)
target_include_directories(mdance_helm PUBLIC 
    cluster
)
target_link_libraries(mdance_helm PUBLIC mdance_tools)

# create main mdance library
add_library(mdance INTERFACE)
target_link_libraries(mdance 
    INTERFACE 
        mdance_tools 
        mdance_kmeans 
        mdance_helm
)