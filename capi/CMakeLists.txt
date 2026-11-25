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
# the header's MDANCE_API exports the symbols while the library is built and
# imports them in a program that uses it
target_compile_definitions(mdance_shared PRIVATE MDANCE_BUILDING)

# The same interface as a static library, for a module that should carry
# libmdance inside it (the Tcl extension: one file to install, nothing to
# find at load time). MDANCE_STATIC makes MDANCE_API empty.
add_library(mdance_capi_static STATIC
    mdance_capi.cpp
)
target_link_libraries(mdance_capi_static PUBLIC mdance)
target_include_directories(mdance_capi_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(mdance_capi_static PUBLIC MDANCE_STATIC)
target_compile_features(mdance_capi_static PRIVATE cxx_std_17)
set_target_properties(mdance_capi_static PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

mdance_static_runtime(mdance_shared)
install(TARGETS mdance_shared
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
install(FILES mdance_capi.h DESTINATION include)
