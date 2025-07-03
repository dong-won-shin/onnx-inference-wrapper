# ONNX Inference Wrapper

A C++ library for easy ONNX model inference with OpenCV integration. This project provides a flexible and extensible framework for running deep learning models using ONNX Runtime with GPU acceleration support.

## Overview

This repository contains a modular C++ wrapper for ONNX model inference that includes:

- **InferenceWrapper**: Base class providing common ONNX Runtime functionality
- **Easy-to-use API**: Simple interface for model initialization and inference
- **GPU Support**: CUDA execution provider integration
- **OpenCV Integration**: Seamless image processing and visualization

## Features

- 🚀 **High Performance**: GPU-accelerated inference using ONNX Runtime
- 🔧 **Modular Design**: Extensible architecture for different model types
- 🖼️ **OpenCV Integration**: Easy image preprocessing and postprocessing
- 🛡️ **Error Handling**: Robust error handling and validation

## Prerequisites

Before you begin, ensure you have the following installed:

- **CMake** (version 3.10 or higher)
- **C++17** compatible compiler
- **OpenCV** (version 4.0 or higher)
- **Eigen3** library
- **ONNX Runtime** (GPU version recommended)

### Installing Onnxruntime Dependencies

```bash
# Download ONNX Runtime (GPU version)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.15.1.tgz
```


## Quick Start

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd onnx_inference_wrapper
```

### Step 2: Configure the Build

Update the ONNX Runtime path in `CMakeLists.txt`:

```cmake
# Set your onnxruntime path
set(ONNXRUNTIME_DIR "/path/to/your/onnxruntime-linux-x64-gpu-1.15.1")
```
### Step 3: Prepare onnx model
For example,
```
cd model/deeplabv3
sh deepabv3_onnx_download.sh

cd model/flownets
sh flownets_onnx_donwload.sh
```

### Step 3: Build the Project

```bash
mkdir build
cd build
cmake ..
make -j4
```

### Step 4: Run the Example

```bash
./deeplabv3_inference_example
./flownets_inference_example
```

- deeplabv3 demo  
![deeplabv3_demo](./asset/deeplabv3.gif)

- flownets demo  
![flownets_demo](./asset/flownets.gif)


## Using Your Own ONNX Model

please check [deeplabv3_inference_example.cpp](src/examples/deeplabv3_inference_example.cpp) or [flownets_inference_example.cpp](src/examples/flownets_inference_example.cpp)