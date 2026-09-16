find_package(TCL REQUIRED)

add_library(mdance_tcl SHARED
    mdance_tcl.cpp
)

# Locate the Tcl stub library. FindTCL exposes TCL_STUB_LIBRARY on some CMake
# versions; otherwise derive it from the resolved Tcl library location (handles
# the macOS framework layout, where the stub lib lives under Versions/8.6).
if(NOT TCL_STUB_LIBRARY)
    get_filename_component(_tcl_lib_dir "${TCL_LIBRARY}" DIRECTORY)
    find_library(TCL_STUB_LIBRARY
        NAMES tclstub8.6 tclstub86 tclstub
        HINTS
            "${TCL_LIBRARY}/Versions/8.6"
            "${_tcl_lib_dir}"
            "${_tcl_lib_dir}/Versions/8.6")
endif()
if(NOT TCL_STUB_LIBRARY)
    message(FATAL_ERROR
        "Tcl stub library (libtclstub8.6) not found near ${TCL_LIBRARY}; "
        "cannot build the loadable Tcl extension.")
endif()

# Link the stub library, NOT the full Tcl runtime: the extension binds to the
# host interpreter's Tcl at load time (see USE_TCL_STUBS in mdance_tcl.cpp).
target_link_libraries(mdance_tcl PRIVATE mdance_shared ${TCL_STUB_LIBRARY})
target_include_directories(mdance_tcl PRIVATE ${TCL_INCLUDE_PATH})
target_compile_features(mdance_tcl PRIVATE cxx_std_17)
set_target_properties(mdance_tcl PROPERTIES
    PREFIX ""
    OUTPUT_NAME mdance_tcl
)
