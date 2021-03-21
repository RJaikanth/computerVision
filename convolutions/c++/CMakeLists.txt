cmake_minimum_required(VERSION 3.1)
project(Convolutions)

# Set Output Dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Add 
find_package(OpenCV REQUIRED)


add_executable(conv src/convolutions.cpp)
target_link_libraries(conv ${OpenCV_LIBS})

