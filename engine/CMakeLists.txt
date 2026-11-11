add_library(kvbound_core STATIC
  core/dtype.cpp
)
add_library(kvbound::core ALIAS kvbound_core)
target_include_directories(kvbound_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvbound_core PUBLIC cxx_std_20)
target_link_libraries(kvbound_core PRIVATE kvb_warnings)
