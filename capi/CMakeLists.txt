add_library(mdance_shared SHARED
    mdance_capi.cpp
)
target_link_libraries(mdance_shared PRIVATE mdance)
target_include_directories(mdance_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(mdance_shared PROPERTIES
    OUTPUT_NAME mdance
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(mdance_shared PRIVATE cxx_std_17)
