cmake_minimum_required(VERSION 3.1)
project(Manifold)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")  # enable assert
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context -Wsign-compare")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")
endif()

option(BUILD_PERFORMANCE_TEST "More subdivisition for performance test" OFF)
option(BUILD_OPENMP "Enable support for OpenMP" OFF)
option(BUILD_LOG "Enable verbose log" OFF)

include_directories(src)
include_directories(3rd_party/eigen/)
include_directories(3rd_party/libigl/include)

if(BUILD_PERFORMANCE_TEST)
    add_definitions(-DPERFORMANCE_TEST)
endif(BUILD_PERFORMANCE_TEST)

if(BUILD_OPENMP)
    add_definitions(-DWITH_OMP)
endif(BUILD_OPENMP)

if(BUILD_LOG)
    add_definitions(-DLOG_OUTPUT)
endif(BUILD_LOG)

set(
    manifold_SRC
    src/GridIndex.h
    src/Intersection.h
    src/Intersection.cc
    src/IO.h
    src/IO.cc
    src/main.cc
    src/Manifold.h
    src/Manifold.cc
    src/MeshProjector.cc
    src/MeshProjector.h
    src/Octree.h
    src/Octree.cc
    src/Parser.h
    src/Parser.cc
    src/types.h
)

add_executable(
    manifold
    ${manifold_SRC}
)

target_link_libraries(
    manifold
)
