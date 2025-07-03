#pragma once

#include "inference_wrapper.h"
#include <opencv2/opencv.hpp>

/**
 * @brief FlowNets inference wrapper for optical flow estimation
 * 
 * This class provides a high-level interface for running FlowNets inference
 * on image pairs to estimate optical flow. It handles preprocessing, inference,
 * and postprocessing including visualization.
 */
class FlowNetsInferenceWrapper : public InferenceWrapper {
public:
    FlowNetsInferenceWrapper();
    ~FlowNetsInferenceWrapper() = default;

    /**
     * @brief Initialize the FlowNets model
     * @param modelPath Path to the ONNX model file
     * @param deviceId GPU device ID (0 for CPU)
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const std::string& modelPath, int deviceId = 0) override;

    /**
     * @brief Run inference on a pair of images
     * @param img1 First image (previous frame)
     * @param img2 Second image (current frame)
     * @return Optical flow visualization as cv::Mat
     */
    cv::Mat infer(const cv::Mat& img1, const cv::Mat& img2);

    /**
     * @brief Get input dimensions
     * @return Pair of (height, width)
     */
    std::pair<int, int> getInputDimensions() const;

    /**
     * @brief Get output dimensions
     * @return Pair of (height, width)
     */
    std::pair<int, int> getOutputDimensions() const;

    /**
     * @brief Visualize dense optical flow using OpenCV
     * @param flow Optical flow data
     * @return Visualized flow image
     */
    cv::Mat visualizeDenseOpticalFlow(const cv::Mat& flow);

    // Override for base class pure virtual
    cv::Mat infer(const cv::Mat& inputImage) override {
        throw std::runtime_error("Use infer(img1, img2) for FlowNetsInferenceWrapper");
    }

private:
    /**
     * @brief Preprocess input images for FlowNets
     * @param img1 First image
     * @param img2 Second image
     * @return Preprocessed tensor data
     */
    std::vector<float> preprocess(const cv::Mat& img1, const cv::Mat& img2);

    /**
     * @brief Visualize optical flow data
     * @param opticalFlow Raw optical flow data
     * @param height Original height
     * @param width Original width
     * @param upscaleHeight Target height for visualization
     * @param upscaleWidth Target width for visualization
     * @return Visualized optical flow as cv::Mat
     */
    cv::Mat visualizeOpticalFlow(const float* opticalFlow, int height, int width, 
                                 int upscaleHeight, int upscaleWidth);

private:
    int inputHeight_;
    int inputWidth_;
    int outputHeight_;
    int outputWidth_;
    
    // Normalization parameters for FlowNets
    const float mean_[3] = {0.45f, 0.432f, 0.411f};
}; 