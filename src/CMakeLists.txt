cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 3.3 REQUIRED)

# Create library for tools
add_library(mdance_tools STATIC
    tools/bts.cpp 
    tools/cluster.cpp 
    tools/esim.cpp 
    # FIXME: scores.cpp does not compile
    # this is not a core problem right now
    tools/scores.cpp
)
target_include_directories(mdance_tools PUBLIC 
    tools
    ${EIGEN3_INCLUDE_DIRS})

# create library for kmeans
add_library(mdance_kmeans STATIC 
    cluster/KMeansRex/KMeans.cpp 
    cluster/KMeansRex/KMeansRexCore.cpp
    cluster/KMeansRex/mersenneTwister2002.c 
)
target_include_directories(mdance_kmeans PUBLIC cluster/KMeansRex)
target_link_libraries(mdance_kmeans PUBLIC mdance_tools)

# create library for helm
add_library(mdance_helm STATIC 
    cluster/helm.cpp
)
target_link_libraries(mdance_helm PUBLIC mdance_tools)

# create divine library
add_library(mdance_divine STATIC 
    cluster/divine.cpp
)
target_link_libraries(mdance_divine PUBLIC mdance_tools)

# create main mdance library
add_library(mdance INTERFACE)
target_link_libraries(mdance 
    INTERFACE 
        mdance_tools 
        mdance_kmeans 
        mdance_helm 
        mdance_divine
)