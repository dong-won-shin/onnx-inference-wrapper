cmake_minimum_required(VERSION 3.10)
project(ONNXRuntimeExample)

set(CMAKE_CXX_STANDARD 17)

# Set your onnxruntime path
set(ONNXRUNTIME_DIR "/home/dongwonshin/onnxruntime-linux-x64-gpu-1.15.1")

# Set project root directory
set(PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}")

# Define compile-time constant for C++ code
add_compile_definitions(
    PROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}"
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})
link_directories(${ONNXRUNTIME_DIR}/lib)

# Set executable and link libraries
add_library(onnx_helper src/OnnxHelper.cc)
add_library(inference_wrapper src/InferenceWrapper.cc)
add_library(deeplabv3_wrapper src/DeepLabV3InferenceWrapper.cc)
add_library(flownets_wrapper src/FlowNetsInferenceWrapper.cc)

add_executable(deeplabv3_inference_example src/examples/deeplabv3_inference_example.cpp)
add_executable(flownets_inference_example src/examples/flownets_inference_example.cpp)

target_link_libraries(inference_wrapper onnx_helper onnxruntime ${OpenCV_LIBS})
target_link_libraries(deeplabv3_wrapper inference_wrapper onnx_helper onnxruntime ${OpenCV_LIBS})
target_link_libraries(flownets_wrapper inference_wrapper onnx_helper onnxruntime ${OpenCV_LIBS})

target_link_libraries(deeplabv3_inference_example deeplabv3_wrapper inference_wrapper onnx_helper onnxruntime ${OpenCV_LIBS})
target_link_libraries(flownets_inference_example flownets_wrapper inference_wrapper onnx_helper onnxruntime ${OpenCV_LIBS})
