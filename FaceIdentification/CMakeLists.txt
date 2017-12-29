cmake_minimum_required (VERSION 3.2.0)

project (seeta_fi_lib)

set (VERSION_MAJOR 4)
set (VERSION_MINOR 5)

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

# Add header file
include_directories(include)

# Add source files
aux_source_directory(./src SRC_LIST)
aux_source_directory(./tools TOOLS_LIST)

# Link libaray
add_library(seeta_fi_lib SHARED ${SRC_LIST} ${TOOLS_LIST})
# set_target_properties(seeta_fi_lib PROPERTIES 
#   VERSION ${VERSION_MAJOR}.${VERSION_MINOR} 
#   SOVERSION ${VERSION_MAJOR}.${VERSION_MINOR}) 
