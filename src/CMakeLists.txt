cmake_minimum_required(VERSION 3.15)

enable_language(C CXX)

# The code uses the Eigen 3.4 slicing API (Eigen::all, Eigen::seq), so 3.3 is
# not enough even though it used to configure successfully.
#
# Asking for 3.4 is still not sufficient on its own: Eigen's config file
# declares SameMajorVersion compatibility, so a 3.4 request is REJECTED by
# Eigen 5.x even though the code builds and passes against it. Ask for the real
# floor first, so a 3.x install is still held to it, then fall back to whatever
# Eigen is installed. If there is none at all, fetch one -- Eigen is header
# only, so populate the sources and declare the imported target here rather than
# running Eigen's own CMake, which would graft its test suite onto this build.
# Set MDANCE_FETCH_EIGEN=OFF to require an installed Eigen instead.
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 QUIET)
endif()
option(MDANCE_FETCH_EIGEN "Download Eigen when none is installed" ON)
if(NOT Eigen3_FOUND)
    if(NOT MDANCE_FETCH_EIGEN)
        message(FATAL_ERROR
            "Eigen was not found and MDANCE_FETCH_EIGEN=OFF. Install Eigen "
            "(apt install libeigen3-dev / brew install eigen) or set "
            "MDANCE_FETCH_EIGEN=ON to download it.")
    endif()
    message(STATUS "Eigen not found; fetching Eigen 3.4.0")
    include(FetchContent)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        # Pointing SOURCE_SUBDIR at a header directory with no CMakeLists.txt
        # makes FetchContent populate the sources and skip add_subdirectory,
        # which would otherwise add 72 targets and 957 ctest entries here.
        SOURCE_SUBDIR Eigen)
    FetchContent_MakeAvailable(eigen)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")
    endif()
endif()

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

# create equal library
add_library(mdance_equal STATIC
    cluster/equal.cpp
)
target_link_libraries(mdance_equal PUBLIC mdance_tools)

# create prime library
add_library(mdance_prime STATIC
    cluster/prime.cpp
)
target_link_libraries(mdance_prime PUBLIC mdance_tools)

# create main mdance library
add_library(mdance INTERFACE)
target_link_libraries(mdance 
    INTERFACE 
        mdance_tools
        mdance_kmeans
        mdance_helm
        mdance_equal
        mdance_prime
)