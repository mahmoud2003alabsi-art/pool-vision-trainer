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

# Dependency-free core: geometry + solver + physics (used by tests too).
add_library(poolcore
  src/geom.cpp
  src/solver.cpp
  src/physics.cpp
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

add_executable(pool_tests_physics tests/test_physics.cpp)
target_link_libraries(pool_tests_physics PRIVATE poolcore)

# Windows-only live aim overlay (Win32 + GDI; reuses poolvision).
if(WIN32)
  add_executable(pool-overlay WIN32 overlay/pool_overlay.cpp)
  target_include_directories(pool-overlay PRIVATE include)
  target_link_libraries(pool-overlay PRIVATE poolvision user32 gdi32 shell32)
  target_compile_definitions(pool-overlay PRIVATE
    UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Windows/MinGW: link statically so the exe runs without MinGW runtime DLLs
# (libstdc++-6.dll, libgcc_s_seh-1.dll, libwinpthread-1.dll).
if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_link_options(pool-trainer PRIVATE
    -static -static-libgcc -static-libstdc++)
  target_link_options(pool_tests PRIVATE
    -static -static-libgcc -static-libstdc++)
  target_link_options(pool_tests_physics PRIVATE
    -static -static-libgcc -static-libstdc++)
  if(TARGET pool-overlay)
    target_link_options(pool-overlay PRIVATE
      -static -static-libgcc -static-libstdc++)
  endif()
endif()

enable_testing()
add_test(NAME solver_tests COMMAND pool_tests)
add_test(NAME physics_tests COMMAND pool_tests_physics)

add_executable(pool_tests_menu tests/test_menu.cpp)
target_include_directories(pool_tests_menu PRIVATE overlay)
add_test(NAME menu_tests COMMAND pool_tests_menu)

add_executable(pool_tests_vision tests/test_vision.cpp)
target_link_libraries(pool_tests_vision PRIVATE poolvision)
add_test(NAME vision_tests COMMAND pool_tests_vision)

add_executable(pool_tests_stab tests/test_stabilize.cpp)
target_link_libraries(pool_tests_stab PRIVATE poolvision)
add_test(NAME stabilize_tests COMMAND pool_tests_stab)
