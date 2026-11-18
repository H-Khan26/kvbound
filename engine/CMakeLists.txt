# CPU-only core: file formats, the op program, memory planning, sampling and cache bookkeeping.
# Everything here builds and is tested without a GPU.
add_library(kvbound STATIC
  core/dtype.cpp
  io/json.cpp
  io/mapped_file.cpp
  io/model_config.cpp
  io/safetensors.cpp
  io/weight_format.cpp
  kv/block_allocator.cpp
  kv/kv_layout.cpp
  memory/planner.cpp
  model/llama/program.cpp
  model/whisper/program.cpp
  runtime/sampler.cpp
  runtime/step.cpp
  sched/scheduler.cpp
)
add_library(kvbound::kvbound ALIAS kvbound)
target_include_directories(kvbound PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvbound PUBLIC cxx_std_20)
target_link_libraries(kvbound PRIVATE kvb_warnings)

if(KVB_ENABLE_CUDA)
  # The GPU runtime: kernels, device weights, KV pool and the engine.
  add_library(kvbound_cuda STATIC
    core/device.cpp
    kernels/attention.cu
    kernels/blas.cpp
    kernels/elementwise.cu
    kernels/gemv.cu
    kernels/quantized.cu
    kernels/reductions.cu
    kernels/rope_kv_write.cu
    kv/kv_pool.cpp
    model/llama/llama_model.cpp
    model/llama/weights.cpp
    runtime/engine.cpp
    runtime/generate.cpp
    runtime/serve.cpp
  )
  add_library(kvbound::cuda ALIAS kvbound_cuda)
  target_link_libraries(kvbound_cuda PUBLIC kvbound CUDA::cudart CUDA::cublas PRIVATE kvb_warnings)
endif()
