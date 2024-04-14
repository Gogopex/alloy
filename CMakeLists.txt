cmake_minimum_required(VERSION 3.10)

# Set your project name
project(Alloy)

# Specify the C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_EXTENSIONS OFF)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
file(GLOB SOURCES "src/*.c")

# Define the executable
add_executable(Alloy ${SOURCES})

# Find the Metal framework (macOS specific)
find_library(METAL_LIBRARY Metal)

# Link against the CMT library and the Metal framework
target_link_libraries(Alloy ${PROJECT_SOURCE_DIR}/lib/libcmt_lib.a ${METAL_LIBRARY})
