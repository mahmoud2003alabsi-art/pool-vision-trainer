cmake_minimum_required(VERSION 3.14)
project(pool_vision_trainer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-O2 -Wall -Wextra -Wno-unused-parameter)
endif()

# Dependency-free core: geometry + solver (used by tests too).
add_library(poolcore
  src/geom.cpp
  src/solver.cpp
)
target_include_directories(poolcore PUBLIC include)

# Vision layer: stb image I/O, detection, rendering, synth, JSON.
add_library(poolvision
  src/vision.cpp
  src/render.cpp
  src/synth.cpp
  src/jsonw.cpp
  src/stb_impl.cpp
)
target_include_directories(poolvision PUBLIC include third_party third_party/stb)
target_link_libraries(poolvision PUBLIC poolcore)

add_executable(pool-trainer src/main.cpp)
target_link_libraries(pool-trainer PRIVATE poolvision)

add_executable(pool_tests tests/test_solver.cpp)
target_link_libraries(pool_tests PRIVATE poolcore)

enable_testing()
add_test(NAME solver_tests COMMAND pool_tests)
