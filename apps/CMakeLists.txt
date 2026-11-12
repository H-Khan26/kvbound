add_executable(kvb-bench kvb-bench.cpp)
target_link_libraries(kvb-bench PRIVATE kvbound::cuda kvb_warnings)
set_target_properties(kvb-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
