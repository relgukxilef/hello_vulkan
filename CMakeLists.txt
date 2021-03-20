cmake_minimum_required(VERSION 3.5)

project(vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

link_libraries(gdi32 user32 kernel32 glfw3 vulkan-1)

add_subdirectory(source)
