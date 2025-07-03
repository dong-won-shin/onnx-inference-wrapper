#ifndef INFERENCE_WRAPPER_H
#define INFERENCE_WRAPPER_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "OnnxHelper.h"


class InferenceWrapper {
public:
    InferenceWrapper();
    virtual ~InferenceWrapper();

    // Initialize the model with ONNX file path
    virtual bool initialize(const std::string& modelPath, int deviceId = 0);
    
    // Perform inference on input image (pure virtual function)
    virtual cv::Mat infer(const cv::Mat& inputImage) = 0;
    
    // Get input dimensions
    virtual std::pair<int, int> getInputDimensions() const;
    
    // Check if model is initialized
    virtual bool isInitialized() const;

protected:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> ortEnv_;
    std::unique_ptr<Ort::Session> ortSession_;
    
    // Model information
    std::vector<TensorInfo> inputInfos_;
    std::vector<TensorInfo> outputInfos_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    
    // Model state
    bool isInitialized_;
    int inputHeight_;
    int inputWidth_;
    
    // Common preprocessing method
    virtual cv::Mat preprocessImage(const cv::Mat& inputImage);
    
    // Common inference execution
    std::vector<Ort::Value> executeInference(const std::vector<Ort::Value>& inputTensors);
    
    // Get output tensor data
    float* getOutputTensorData(const std::vector<Ort::Value>& outputTensors, int outputIndex = 0);
};

#endif // INFERENCE_WRAPPER_H 