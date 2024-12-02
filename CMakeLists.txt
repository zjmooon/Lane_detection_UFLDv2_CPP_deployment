cmake_minimum_required(VERSION 3.16)

# CMake may not find CUDACXX, specify it explicitly
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(lane_detection LANGUAGES CXX CUDA)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR "/usr/src/tensorrt")

# # ARM ABI of the target platform
# set(TEGRA_ARMABI "aarch64-linux-gnu")
# X86_64 ABI of the target platform
set(TEGRA_ARMABI "x86_64-linux-gnu")

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_library(NVINFER nvinfer)

set(TensorRT_NV_DIR "/usr/lib/x86_64-linux-gnu/")
find_library(TensorRT_NVPARSERS_LIBRARY NAMES nvparsers HINTS ${TensorRT_NV_DIR})
set(TensorRT_NVPARSERS_LIBRARY "/usr/lib/x86_64-linux-gnu/libnvparsers.so")

message("The value of CUDA_INCLUDE_DIRS is: ${CUDA_INCLUDE_DIRS}")
message("The value of CUDA_LIBRARIES is: ${CUDA_LIBRARIES}")
message("The value of TensorRT_DIR is: ${TensorRT_DIR}")
message("The value of TensorRT_INCLUDE_DIRS is: ${TensorRT_INCLUDE_DIRS}")
message("The value of TensorRT_LIBRARIES is: ${TensorRT_LIBRARIES}")
message("The value of OpenCV_INCLUDE_DIRS is: ${OpenCV_INCLUDE_DIRS}")

add_library(lanemode src/lane_detection_UFLDv2.cpp ${TensorRT_DIR}/samples/common/logger.cpp)

cuda_add_library(yuv2rgb src/yuv2rgb.cu src/yuv2rgb.cuh)

target_compile_features(lanemode PUBLIC cxx_std_17)

target_compile_options(lanemode PUBLIC -Ofast)

target_include_directories(lanemode PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
            ${TensorRT_DIR}/samples/common /usr/include/${TEGRA_ARMABI} include src)

target_link_libraries(lanemode PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} 
                ${TensorRT_LIBRARIES} yuv2rgb cuda)

# genrate executable 
add_executable(lane_detection lane_detection_api_sample.cpp)

target_link_libraries(lane_detection lanemode)

set_target_properties(lane_detection lanemode yuv2rgb PROPERTIES CUDAARCH 87)