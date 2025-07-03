#include "inference_wrapper/flownets_inference_wrapper.h"

FlowNetsInferenceWrapper::FlowNetsInferenceWrapper() 
    : inputHeight_(0), inputWidth_(0), outputHeight_(0), outputWidth_(0) {
}

bool FlowNetsInferenceWrapper::initialize(const std::string& modelPath, int deviceId) {
    // Initialize base class
    if (!InferenceWrapper::initialize(modelPath, deviceId)) {
        std::cerr << "Failed to initialize base inference wrapper" << std::endl;
        return false;
    }

    // Get input and output dimensions from model info
    const auto& inputInfos = inputInfos_;
    const auto& outputInfos = outputInfos_;

    if (inputInfos.empty() || outputInfos.empty()) {
        std::cerr << "Invalid model info: missing input or output information" << std::endl;
        return false;
    }

    // FlowNets expects input shape: [batch, channels, height, width]
    // where channels = 6 (3 for img1 + 3 for img2)
    if (inputInfos[0].shape.size() != 4) {
        std::cerr << "Invalid input shape: expected 4D tensor" << std::endl;
        return false;
    }

    inputHeight_ = static_cast<int>(inputInfos[0].shape[2]);
    inputWidth_ = static_cast<int>(inputInfos[0].shape[3]);

    // Output shape: [batch, channels, height, width]
    // where channels = 2 (flow_x, flow_y)
    if (outputInfos[0].shape.size() != 4) {
        std::cerr << "Invalid output shape: expected 4D tensor" << std::endl;
        return false;
    }

    outputHeight_ = static_cast<int>(outputInfos[0].shape[2]);
    outputWidth_ = static_cast<int>(outputInfos[0].shape[3]);

    std::cout << "FlowNets initialized successfully" << std::endl;
    std::cout << "Input dimensions: " << inputHeight_ << "x" << inputWidth_ << std::endl;
    std::cout << "Output dimensions: " << outputHeight_ << "x" << outputWidth_ << std::endl;

    return true;
}

cv::Mat FlowNetsInferenceWrapper::infer(const cv::Mat& img1, const cv::Mat& img2) {
    if (!isInitialized()) {
        throw std::runtime_error("FlowNets model not initialized");
    }

    if (img1.empty() || img2.empty()) {
        throw std::runtime_error("Input images cannot be empty");
    }

    // Preprocess images
    std::vector<float> inputData = preprocess(img1, img2);

    // Create input tensor
    std::vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(
        OnnxHelper::CreateGPUTensor<float>(
            inputInfos_[0].shape, 
            reinterpret_cast<float*>(inputData.data()), 
            inputData.size()
        )
    );

    // Run inference
    auto outputTensors = executeInference(inputTensors);

    if (outputTensors.empty()) {
        throw std::runtime_error("Inference failed: no output tensors");
    }

    // Get output data
    float* opticalFlowData = getOutputTensorData(outputTensors, 0);

    // Visualize optical flow
    return visualizeOpticalFlow(opticalFlowData, outputHeight_, outputWidth_, 
                               img1.rows, img1.cols);
}

std::pair<int, int> FlowNetsInferenceWrapper::getInputDimensions() const {
    return {inputHeight_, inputWidth_};
}

std::pair<int, int> FlowNetsInferenceWrapper::getOutputDimensions() const {
    return {outputHeight_, outputWidth_};
}

std::vector<float> FlowNetsInferenceWrapper::preprocess(const cv::Mat& img1, const cv::Mat& img2) {
    // Resize images to model input size
    cv::Mat img1Resized, img2Resized;
    cv::resize(img1, img1Resized, cv::Size(inputWidth_, inputHeight_));
    cv::resize(img2, img2Resized, cv::Size(inputWidth_, inputHeight_));

    // Convert to float
    cv::Mat img1f, img2f;
    img1Resized.convertTo(img1f, CV_32FC3);
    img2Resized.convertTo(img2f, CV_32FC3);

    // Prepare input tensor: [batch, channels, height, width] = [1, 6, H, W]
    std::vector<float> inputData(inputHeight_ * inputWidth_ * 6);

    for (int i = 0; i < inputHeight_; i++) {
        for (int j = 0; j < inputWidth_; j++) {
            // Copy from [H, W, 3] to [3, H, W] format for both images
            // Image 1: channels 0, 1, 2
            inputData[0 * inputHeight_ * inputWidth_ + i * inputWidth_ + j] = 
                img1f.at<cv::Vec3f>(i, j)[0] / 255.0f - mean_[0];
            inputData[1 * inputHeight_ * inputWidth_ + i * inputWidth_ + j] = 
                img1f.at<cv::Vec3f>(i, j)[1] / 255.0f - mean_[1];
            inputData[2 * inputHeight_ * inputWidth_ + i * inputWidth_ + j] = 
                img1f.at<cv::Vec3f>(i, j)[2] / 255.0f - mean_[2];

            // Image 2: channels 3, 4, 5
            inputData[3 * inputHeight_ * inputWidth_ + i * inputWidth_ + j] = 
                img2f.at<cv::Vec3f>(i, j)[0] / 255.0f - mean_[0];
            inputData[4 * inputHeight_ * inputWidth_ + i * inputWidth_ + j] = 
                img2f.at<cv::Vec3f>(i, j)[1] / 255.0f - mean_[1];
            inputData[5 * inputHeight_ * inputWidth_ + i * inputWidth_ + j] = 
                img2f.at<cv::Vec3f>(i, j)[2] / 255.0f - mean_[2];
        }
    }

    return inputData;
}

cv::Mat FlowNetsInferenceWrapper::visualizeOpticalFlow(const float* opticalFlow, int height, int width, 
                                                       int upscaleHeight, int upscaleWidth) {
    // Separate 2-channel data (flow_x and flow_y)
    cv::Mat flowX(height, width, CV_32F);
    cv::Mat flowY(height, width, CV_32F);

    for (int i = 0; i < height; ++i) {
        const float* xStartFlowPtr = opticalFlow;
        const float* yStartFlowPtr = opticalFlow + height * width;
        float* flowXPtr = flowX.ptr<float>(i);
        float* flowYPtr = flowY.ptr<float>(i);

        for (int j = 0; j < width; ++j) {
            flowXPtr[j] = xStartFlowPtr[i * width + j];
            flowYPtr[j] = yStartFlowPtr[i * width + j];
        }
    }

    // Resize to target dimensions
    cv::Mat flowXResized, flowYResized;
    cv::resize(flowX, flowXResized, cv::Size(upscaleWidth, upscaleHeight), 0, 0, cv::INTER_LINEAR);
    cv::resize(flowY, flowYResized, cv::Size(upscaleWidth, upscaleHeight), 0, 0, cv::INTER_LINEAR);

    // Calculate magnitude and angle
    cv::Mat magnitude, angle;
    cv::cartToPolar(flowXResized, flowYResized, magnitude, angle, true);

    // Normalize magnitude
    double magMin, magMax;
    cv::minMaxLoc(magnitude, &magMin, &magMax);
    magnitude -= magMin;
    magnitude.convertTo(magnitude, CV_8UC1, 255.0 / (magMax - magMin + 1e-5));

    // Create HSV image
    cv::Mat hsv(upscaleHeight, upscaleWidth, CV_8UC3);
    for (int i = 0; i < upscaleHeight; ++i) {
        uchar* hsvPtr = hsv.ptr<uchar>(i);
        const float* anglePtr = angle.ptr<float>(i);
        const uchar* magPtr = magnitude.ptr<uchar>(i);

        for (int j = 0; j < upscaleWidth; ++j) {
            hsvPtr[j * 3] = static_cast<uchar>(anglePtr[j] / 2); // Hue [0, 180]
            hsvPtr[j * 3 + 1] = 255;                             // Saturation
            hsvPtr[j * 3 + 2] = magPtr[j];                       // Value
        }
    }

    // Convert HSV to BGR
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    return bgr;
}

cv::Mat FlowNetsInferenceWrapper::visualizeDenseOpticalFlow(const cv::Mat& flow) {
    // Separate flow channels
    std::vector<cv::Mat> flowChannels(2);
    cv::split(flow, flowChannels); // flowChannels[0] = flow_x, flowChannels[1] = flow_y

    // Calculate magnitude and angle
    cv::Mat magnitude, angle;
    cv::cartToPolar(flowChannels[0], flowChannels[1], magnitude, angle, true);

    // Normalize magnitude to [0, 255] range
    double magMax;
    cv::minMaxLoc(magnitude, nullptr, &magMax);
    magnitude.convertTo(magnitude, CV_8UC1, 255.0 / (magMax + 1e-5));

    // Create HSV image
    cv::Mat hsv(flow.size(), CV_8UC3);
    for (int i = 0; i < flow.rows; ++i) {
        uchar* hsvPtr = hsv.ptr<uchar>(i);
        const float* anglePtr = angle.ptr<float>(i);
        const uchar* magPtr = magnitude.ptr<uchar>(i);

        for (int j = 0; j < flow.cols; ++j) {
            hsvPtr[j * 3] = static_cast<uchar>(anglePtr[j] / 2); // Hue: direction (0~180)
            hsvPtr[j * 3 + 1] = 255;                             // Saturation
            hsvPtr[j * 3 + 2] = magPtr[j];                       // Value: magnitude
        }
    }

    // Convert HSV to BGR
    cv::Mat flowImage;
    cv::cvtColor(hsv, flowImage, cv::COLOR_HSV2BGR);

    return flowImage;
} 