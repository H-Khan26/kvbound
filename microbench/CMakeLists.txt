# Standalone hardware-characterization binaries. Each prints one JSON object to stdout;
# `python -m kvbound.microbench` runs them and writes the result records.
foreach(name bandwidth_sweep pointer_chase gemm_peak)
  add_executable(kvb-${name} ${name}.cu)
  target_link_libraries(kvb-${name} PRIVATE CUDA::cudart)
  set_target_properties(kvb-${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
target_link_libraries(kvb-gemm_peak PRIVATE CUDA::cublasLt)
