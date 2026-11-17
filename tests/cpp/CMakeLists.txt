# Every *_test.cpp in this directory becomes one GoogleTest executable registered with CTest.
file(GLOB KVB_CPP_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
if(NOT KVB_CPP_TESTS)
  return()
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

foreach(src IN LISTS KVB_CPP_TESTS)
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE kvbound::kvbound kvb_warnings GTest::gtest_main)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE
    KVB_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/fixtures/tiny-llama"
    KVB_WHISPER_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/fixtures/tiny-whisper")
  gtest_discover_tests(${name})
endforeach()
