cmake_minimum_required(VERSION 3.2.0)

project(seeta_fa_lib)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -g -ggdb")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

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

# Add header file
include_directories(include)

# Add libarary directories
link_directories(../library)

# Add source file
aux_source_directory(./src SRC_LIST)

# Link library
add_library(seeta_fa_lib SHARED ${SRC_LIST})
