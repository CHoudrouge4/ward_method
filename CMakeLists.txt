cmake_minimum_required(VERSION 3.10)

project(wards)

include_directories(include)

include_directories("/home/hussein/projects/flann/flann/src/cpp/")

file(GLOB SOURCES "cpp/*.cc")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O3")

add_executable(wards.exe ${SOURCES})
