#include "inference_wrapper/deeplabv3_inference_wrapper.h"

// Cityscapes train_id to color map (BGR for OpenCV)
const std::array<cv::Vec3b, 20> DeepLabV3InferenceWrapper::cityscapesTrainIdToColor_ = {{
    {128,  64,128}, // road          - dark magenta
    {232,  35,244}, // sidewalk      - light magenta
    { 70,  70, 70}, // building      - dark gray
    {156,102,102}, // wall          - slate blue
    {153,153,190}, // fence         - dusty rose
    {153,153,153}, // pole          - gray
    { 30,170,250}, // traffic light - orange
    {  0,220,220}, // traffic sign  - yellow
    { 35,142,107}, // vegetation    - olive green
    {152,251,152}, // terrain       - pale green
    {180,130, 70}, // sky           - steel blue
    { 60, 20,220}, // person        - crimson
    {  0,  0,255}, // rider         - bright red
    {142,  0,  0}, // car           - dark blue
    { 70,  0,  0}, // truck         - navy
    {100, 60,  0}, // bus           - dark cyan
    {100, 80,  0}, // train         - teal
    {230,  0,  0}, // motorcycle    - royal blue
    { 32, 11,119}, // bicycle       - maroon
    {  0,  0,  0}  // invalid       - black
}};

DeepLabV3InferenceWrapper::DeepLabV3InferenceWrapper() {
    // Base class constructor handles initialization
}

DeepLabV3InferenceWrapper::~DeepLabV3InferenceWrapper() {
    // Base class destructor handles cleanup
}

cv::Mat DeepLabV3InferenceWrapper::infer(const cv::Mat& inputImage) {
    if (!isInitialized()) {
        throw std::runtime_error("Model not initialized. Call initialize() first.");
    }

    // Preprocess input image using DeepLabV3 specific preprocessing
    cv::Mat preprocessedImage = preprocessImage(inputImage);

    // Create input tensor
    std::vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(
        OnnxHelper::CreateGPUTensor<float>(
            inputInfos_[0].shape, 
            reinterpret_cast<float*>(preprocessedImage.data), 
            inputHeight_ * inputWidth_ * 3
        )
    );

    // Execute inference using base class method
    auto outputTensors = executeInference(inputTensors);

    // Get output data using base class method
    float* semanticSegmentationTensor = getOutputTensorData(outputTensors, 0);

    // Postprocess output
    return postprocessOutput(semanticSegmentationTensor);
}

cv::Mat DeepLabV3InferenceWrapper::preprocessImage(const cv::Mat& inputImage) {
    // Convert to float and normalize to [0, 1]
    cv::Mat imgFloat;
    inputImage.convertTo(imgFloat, CV_32FC3);
    imgFloat = imgFloat / 255.0;

    // DeepLabV3 specific normalization with ImageNet mean and std
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    imgFloat = (imgFloat - mean) / std;

    // Convert to blob format (NCHW)
    cv::dnn::blobFromImage(imgFloat, imgFloat, 1.0, cv::Size(), cv::Scalar(), true, false, CV_32F);

    return imgFloat;
}

cv::Mat DeepLabV3InferenceWrapper::postprocessOutput(const float* logits) {
    cv::Mat labelMap = logitsToLabelMap(logits);
    return decodeCityscapesTarget(labelMap);
}

cv::Mat DeepLabV3InferenceWrapper::logitsToLabelMap(const float* logits) {
    int numClasses = 19;

    cv::Mat labelMap(inputHeight_, inputWidth_, CV_8UC1);

    for (int y = 0; y < inputHeight_; ++y) {
        for (int x = 0; x < inputWidth_; ++x) {
            float bestScore = logits[0 * inputHeight_ * inputWidth_ + y * inputWidth_ + x];
            int bestClass = 0;
            
            for (int c = 1; c < numClasses; ++c) {
                float score = logits[c * inputHeight_ * inputWidth_ + y * inputWidth_ + x];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }
            labelMap.at<uchar>(y, x) = static_cast<uchar>(bestClass);
        }
    }
    return labelMap;
}

cv::Mat DeepLabV3InferenceWrapper::decodeCityscapesTarget(const cv::Mat& labelMap) {
    CV_Assert(labelMap.type() == CV_8UC1);
    int height = labelMap.rows;
    int width = labelMap.cols;

    cv::Mat colorMap(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        const uchar* labelRow = labelMap.ptr<uchar>(y);
        cv::Vec3b* colorRow = colorMap.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            uchar label = labelRow[x];
            if (label >= 19) label = 19;  // Map 255 or invalid to black
            colorRow[x] = cityscapesTrainIdToColor_[label];
        }
    }
    return colorMap;
} 