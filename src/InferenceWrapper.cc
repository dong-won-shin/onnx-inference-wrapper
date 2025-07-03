#include "InferenceWrapper.h"

InferenceWrapper::InferenceWrapper() 
    : isInitialized_(false), inputHeight_(0), inputWidth_(0) {
}

InferenceWrapper::~InferenceWrapper() {
    // Cleanup is handled by unique_ptr
}

bool InferenceWrapper::initialize(const std::string& modelPath, int deviceId) {
    try {
        // Convert the modelPath to ONNX compatible path
        std::vector<ORTCHAR_T> modelFileOrt;
        OnnxHelper::Str2Ort(modelPath, modelFileOrt);

        // Create ONNX runtime environment
        ortEnv_ = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "inference_wrapper"));

        // Configure session options
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(16);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Register CUDA Execution Provider
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = deviceId;
        cudaOptions.arena_extend_strategy = 0;
        cudaOptions.gpu_mem_limit = SIZE_MAX;
        cudaOptions.do_copy_in_default_stream = 1;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

        // Create session
        ortSession_ = std::unique_ptr<Ort::Session>(
            new Ort::Session(*ortEnv_, modelFileOrt.data(), sessionOptions));

        // Get model information
        OnnxHelper::GetModelInfo(*ortSession_, inputInfos_, outputInfos_);
        OnnxHelper::PrintModelInfo(inputInfos_, outputInfos_);

        // Extract input and output names
        for (const auto& inputInfo : inputInfos_) {
            inputNames_.push_back(inputInfo.name.c_str());
        }
        for (const auto& outputInfo : outputInfos_) {
            outputNames_.push_back(outputInfo.name.c_str());
        }

        // Set input dimensions
        if (!inputInfos_.empty()) {
            inputHeight_ = static_cast<int>(inputInfos_[0].shape[2]);
            inputWidth_ = static_cast<int>(inputInfos_[0].shape[3]);
        }

        isInitialized_ = true;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing model: " << e.what() << std::endl;
        isInitialized_ = false;
        return false;
    }
}

std::pair<int, int> InferenceWrapper::getInputDimensions() const {
    return std::make_pair(inputHeight_, inputWidth_);
}

bool InferenceWrapper::isInitialized() const {
    return isInitialized_;
}

cv::Mat InferenceWrapper::preprocessImage(const cv::Mat& inputImage) {
    // Default preprocessing: convert to float and normalize to [0, 1]
    cv::Mat imgFloat;
    inputImage.convertTo(imgFloat, CV_32FC3);
    imgFloat = imgFloat / 255.0;

    // Convert to blob format (NCHW)
    cv::dnn::blobFromImage(imgFloat, imgFloat, 1.0, cv::Size(), cv::Scalar(), true, false, CV_32F);

    return imgFloat;
}

std::vector<Ort::Value> InferenceWrapper::executeInference(const std::vector<Ort::Value>& inputTensors) {
    if (!isInitialized_) {
        throw std::runtime_error("Model not initialized. Call initialize() first.");
    }

    // Run inference
    auto outputTensors = ortSession_->Run(
        Ort::RunOptions{nullptr},
        inputNames_.data(),
        inputTensors.data(),
        inputTensors.size(),
        outputNames_.data(),
        outputNames_.size()
    );

    return outputTensors;
}

float* InferenceWrapper::getOutputTensorData(const std::vector<Ort::Value>& outputTensors, int outputIndex) {
    if (outputIndex >= outputTensors.size()) {
        throw std::runtime_error("Output index out of range");
    }
    return const_cast<float*>(outputTensors[outputIndex].GetTensorData<float>());
} 