#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "FlowNetsInferenceWrapper.h"

const std::string modelDir = std::string(PROJECT_ROOT_DIR) + "/model";
const std::string dataDir = std::string(PROJECT_ROOT_DIR) + "/data";
const std::string onnxModelPath = modelDir + "/flownets/flownets_sim.onnx";
const std::string imageDataPath = dataDir + "/kitti_sample/%06d.png";

constexpr int32_t FRAME_NUM = 300;

// KITTI image dimensions: 1241 x 376
constexpr int32_t KITTI_WIDTH = 1241;
constexpr int32_t KITTI_HEIGHT = 376;

// Target dimensions for FlowNets model
constexpr int32_t TARGET_WIDTH = 640;
constexpr int32_t TARGET_HEIGHT = 240;

// KITTI image preprocessing function for optical flow models
std::pair<cv::Mat, cv::Mat> preprocessKittiImagesForFlow(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty()) {
        throw std::runtime_error("Input images are empty");
    }
    
    // Crop to 1000x375 (center crop, same ratio as DeepLabV3)
    const int32_t cropWidth = 1000;
    const int32_t cropHeight = 375;
    const int32_t cropX = (img1.cols - cropWidth) / 2;
    const int32_t cropY = 0;
    
    cv::Mat croppedImg1 = img1(cv::Rect(cropX, cropY, cropWidth, cropHeight));
    cv::Mat croppedImg2 = img2(cv::Rect(cropX, cropY, cropWidth, cropHeight));

    cv::Mat resizedImg1;
    cv::resize(croppedImg1, resizedImg1, cv::Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, cv::INTER_LINEAR);
    cv::Mat resizedImg2;
    cv::resize(croppedImg2, resizedImg2, cv::Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, cv::INTER_LINEAR);
    
    return {resizedImg1, resizedImg2};
}

int main() {
    // Create and initialize FlowNets inference wrapper
    FlowNetsInferenceWrapper flownetsWrapper;
    
    if (!flownetsWrapper.initialize(onnxModelPath, 0)) {
        std::cerr << "Failed to initialize FlowNets model" << std::endl;
        std::cerr << "Model path: " << onnxModelPath << std::endl;
        return -1;
    }

    std::cout << "FlowNets model initialized successfully" << std::endl;
    std::cout << "Model path: " << onnxModelPath << std::endl;
    
    // Get input and output dimensions
    auto [inputHeight, inputWidth] = flownetsWrapper.getInputDimensions();
    std::cout << "Input dimensions: " << inputHeight << "x" << inputWidth << std::endl;

    for (int32_t idx = 0; idx < FRAME_NUM; idx++) {
        std::cout << "Processing " << idx << "th frame" << std::endl;

        // Load consecutive frames
        char fileName1[256], fileName2[256];
        sprintf(fileName1, imageDataPath.c_str(), idx);
        sprintf(fileName2, imageDataPath.c_str(), idx + 1);

        cv::Mat img1 = cv::imread(fileName1, cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(fileName2, cv::IMREAD_COLOR);

        if (img1.empty() || img2.empty()) {
            std::cerr << "Failed to load images: " << fileName1 << " or " << fileName2 << std::endl;
            continue;
        }

        try {
            // Preprocess KITTI images (crop)
            auto [croppedImg1, croppedImg2] = preprocessKittiImagesForFlow(img1, img2);
            
            // Display cropped images
            cv::Mat concatCroppedImg;
            cv::hconcat(croppedImg1, croppedImg2, concatCroppedImg);
            cv::imshow("Preprocessed Images", concatCroppedImg);
            
            // Run FlowNets inference (model will resize internally)
            cv::Mat flownetsFlow = flownetsWrapper.infer(croppedImg1, croppedImg2);
            cv::imshow("FlowNets Optical Flow", flownetsFlow);

            // Compare with OpenCV Dense Optical Flow on cropped images
            cv::Mat img1Gray, img2Gray;
            cv::cvtColor(croppedImg1, img1Gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(croppedImg2, img2Gray, cv::COLOR_BGR2GRAY);

            cv::Mat opencvFlow;
            cv::Ptr<cv::DenseOpticalFlow> dis = cv::DISOpticalFlow::create(0);
            dis->calc(img1Gray, img2Gray, opencvFlow);

            // Visualize OpenCV flow
            cv::Mat opencvFlowImage = flownetsWrapper.visualizeDenseOpticalFlow(opencvFlow);
            cv::imshow("OpenCV Dense Optical Flow", opencvFlowImage);

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
