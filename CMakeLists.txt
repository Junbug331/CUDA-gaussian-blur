cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_language(CUDA)
project(CUDA-gaussian-blur)


find_package(spdlog REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/spdlog/install/lib/cmake/spdlog)
if(spdlog_FOUND)
    message(STATUS "spdlog found - ${spdlog_DIR}")
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/spdlog/install/include)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found - ${OpenCV_DIR}")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

set(RESOURCES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/res" CACHE PATH "RESOURCES_PATH")
add_definitions(-DRESOURCES_PATH="${RESOURCES_PATH}")

add_subdirectory(cuda_module)

set(SRC_FILES
        src/main.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE
        spdlog::spdlog
        ${OpenCV_LIBS}
        cuda_gaussian_blur
)

add_custom_target(copy-compile-commands ALL
        ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_CURRENT_LIST_DIR}
        )
