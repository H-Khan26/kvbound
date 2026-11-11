cmake_minimum_required(VERSION 3.27)
project(kvbound VERSION 0.1.0 LANGUAGES CXX)

option(KVB_ENABLE_CUDA "Build CUDA kernels and the GPU runtime" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(KVB_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  # Line info maps Nsight Compute source views back to .cu lines.
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>")
  message(STATUS "kvbound: CUDA ${CUDAToolkit_VERSION}, archs ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Warning flags shared by every kvbound target.
add_library(kvb_warnings INTERFACE)
if(MSVC)
  target_compile_options(kvb_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
else()
  target_compile_options(kvb_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

include(CTest)

add_subdirectory(engine)
if(KVB_ENABLE_CUDA)
  add_subdirectory(microbench)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()
