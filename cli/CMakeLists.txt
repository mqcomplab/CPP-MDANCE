add_executable(mdance-cli main.cpp csv_io.cpp json_output.cpp)
target_link_libraries(mdance-cli PRIVATE mdance)
target_compile_features(mdance-cli PRIVATE cxx_std_17)
install(TARGETS mdance-cli DESTINATION bin)
