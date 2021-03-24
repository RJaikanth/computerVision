cmake_minimum_required(VERSION 3.1)
project(EdgeDetection)

find_package(OpenCV REQUIRED)

# Set Output Dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(edgeDet src/edges.cpp)
target_link_libraries(edgeDet ${OpenCV_LIBS})

