#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "DeepLabV3InferenceWrapper.h"

const std::string modelDir = std::string(PROJECT_ROOT_DIR) + "/model";
const std::string dataDir = std::string(PROJECT_ROOT_DIR) + "/data";
const std::string onnxModelPath = modelDir + "/deeplabv3/deeplabv3_sim.onnx";
const std::string imageDataPath = dataDir + "/kitti_sample/%06d.png";

constexpr int32_t FRAME_NUM = 300;

// KITTI image dimensions: 1241 x 376
constexpr int32_t KITTI_WIDTH = 1241;
constexpr int32_t KITTI_HEIGHT = 376;

// Target dimensions for DeepLabV3 model
constexpr int32_t TARGET_WIDTH = 640;
constexpr int32_t TARGET_HEIGHT = 240;

// KITTI image preprocessing function
cv::Mat preprocessKittiImage(const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    
    // Crop to 1000x375 (center crop, same ratio)
    const int32_t cropWidth = 1000;
    const int32_t cropHeight = 375;
    const int32_t cropX = (inputImage.cols - cropWidth) / 2;
    const int32_t cropY = 0;
    
    cv::Mat croppedImage = inputImage(cv::Rect(cropX, cropY, cropWidth, cropHeight));
    
    // Resize to target dimensions (640x240) maintaining aspect ratio
    cv::Mat resizedImage;
    cv::resize(croppedImage, resizedImage, cv::Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, cv::INTER_LINEAR);
    
    return resizedImage;
}

int main() {
    // Create and initialize DeepLabV3 inference wrapper
    DeepLabV3InferenceWrapper deeplabWrapper;
    
    if (!deeplabWrapper.initialize(onnxModelPath, 0)) {
        std::cerr << "Failed to initialize DeepLabV3 model" << std::endl;
        std::cerr << "Model path: " << onnxModelPath << std::endl;
        return -1;
    }

    std::cout << "DeepLabV3 model initialized successfully" << std::endl;
    std::cout << "Model path: " << onnxModelPath << std::endl;
    
    // Get input dimensions
    auto [inputHeight, inputWidth] = deeplabWrapper.getInputDimensions();
    std::cout << "Input dimensions: " << inputHeight << "x" << inputWidth << std::endl;
    std::cout << "Target dimensions: " << TARGET_HEIGHT << "x" << TARGET_WIDTH << std::endl;

    for (int32_t idx = 0; idx < FRAME_NUM; idx++) {
        std::cout << "Processing " << idx << "th frame" << std::endl;

        char fileName[256];
        sprintf(fileName, imageDataPath.c_str(), idx);
        
        cv::Mat inputImage = cv::imread(fileName, cv::IMREAD_COLOR);
        
        if (inputImage.empty()) {
            std::cerr << "Failed to load image: " << fileName << std::endl;
            continue;
        }

        try {
            // Preprocess KITTI image (crop and resize)
            cv::Mat preprocessedImage = preprocessKittiImage(inputImage);
            
            // Display preprocessed image
            cv::imshow("Preprocessed Image (640x240)", preprocessedImage);
            
            // Perform inference
            cv::Mat segmentationResult = deeplabWrapper.infer(preprocessedImage);
            
            // Display result
            cv::imshow("Semantic Segmentation", segmentationResult);
            
            // Wait for key press (1ms delay)
            char key = cv::waitKey(1);
            if (key == 27) { // ESC key to exit
                break;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            continue;
        }
    }

    cv::destroyAllWindows();
    return 0;
}