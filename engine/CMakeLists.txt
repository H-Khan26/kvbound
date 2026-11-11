add_library(kvbound STATIC
  core/dtype.cpp
  core/error.cpp
  io/json.cpp
  io/mapped_file.cpp
  io/model_config.cpp
  io/safetensors.cpp
)
add_library(kvbound::kvbound ALIAS kvbound)
target_include_directories(kvbound PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvbound PUBLIC cxx_std_20)
target_link_libraries(kvbound PRIVATE kvb_warnings)
