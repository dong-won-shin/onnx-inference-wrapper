#ifndef ONNX_HELPER_H
#define ONNX_HELPER_H

#include <iostream>
#include <vector>
#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct TensorInfo
{
  std::string name;
  std::vector<int64_t> shape;
  int type;
};

class OnnxHelper
{
public:
  static void GetModelInfo(
    Ort::Session& session,
    std::vector<TensorInfo>& inputInfos,
    std::vector<TensorInfo>& outputInfos);

  static void PrintModelInfo(
    const std::vector<TensorInfo>& inputInfos,
    const std::vector<TensorInfo>& outputInfos);

  static void Str2Ort(const std::string& modelFilePath, std::vector<ORTCHAR_T>& modelFileOrt);

  template<typename T>
  static Ort::Value CreateGPUTensor(const std::vector<std::int64_t>& shape, T* gpuData, int dataLength)
  {
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<T>(memoryInfo, gpuData, dataLength, shape.data(), shape.size());
  }
};

#endif