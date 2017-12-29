cmake_minimum_required(VERSION 3.2.0)

project(seeta_fd_lib)

# Build options
option(USE_OPENMP      "Set to ON to build use openmp"  ON)
option(USE_SSE         "Set to ON to build use SSE"  ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -g -ggdb")

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
set(LIBRARY_OUTPUT_PATH ../../)
else()
set(LIBRARY_OUTPUT_PATH ../../Release)
endif (CMAKE_SYSTEM_NAME MATCHES "Windows")


# Platform
MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGTEST_USE_OWN_TR1_TUPLE=1")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGTEST_USE_OWN_TR1_TUPLE=1")
endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")

# Use SSE
if (USE_SSE)
    add_definitions(-DUSE_SSE)
    message(STATUS "Use SSE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()

# Use OpenMP
if (USE_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        message(STATUS "Use OpenMP")
        add_definitions(-DUSE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

# Set LIBRARY_OUTPUT_PATH

# Add header file
include_directories(include)

# Add source files
aux_source_directory(./src SRC_LIST)
aux_source_directory(./src/io SRC_LIST)
aux_source_directory(./src/util SRC_LIST)
aux_source_directory(./src/classifier SRC_LIST)
aux_source_directory(./src/feat SRC_LIST)

# Build shared library
add_library(seeta_fd_lib SHARED ${SRC_LIST})
