# Python extension _kvb: the engine and its kernels, for pytest and the evals. It is built into
# build/<preset>/python and loaded by kvbound.native.

if(NOT Python_EXECUTABLE)
  # Build against the project's uv environment rather than whichever Python is on PATH.
  foreach(candidate .venv/Scripts/python.exe .venv/bin/python)
    if(EXISTS ${PROJECT_SOURCE_DIR}/${candidate})
      set(Python_EXECUTABLE ${PROJECT_SOURCE_DIR}/${candidate})
    endif()
  endforeach()
endif()
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG b775c42f2eb3cac13efc5bc266766066306898a6  # v2.9.2
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(_kvb NB_STATIC kvb_module.cpp)
target_link_libraries(_kvb PRIVATE kvbound::cuda)
set_target_properties(_kvb PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
