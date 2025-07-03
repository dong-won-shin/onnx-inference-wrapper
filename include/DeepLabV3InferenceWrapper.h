#ifndef DEEPLABV3_INFERENCE_WRAPPER_H
#define DEEPLABV3_INFERENCE_WRAPPER_H

#include <array>
#include "InferenceWrapper.h"

class DeepLabV3InferenceWrapper : public InferenceWrapper {
public:
    DeepLabV3InferenceWrapper();
    ~DeepLabV3InferenceWrapper() override;

    // Override the pure virtual function from base class
    cv::Mat infer(const cv::Mat& inputImage) override;

private:
    // Override preprocessing for DeepLabV3 specific normalization
    cv::Mat preprocessImage(const cv::Mat& inputImage) override;
    
    // Postprocessing methods specific to DeepLabV3
    cv::Mat postprocessOutput(const float* logits);
    cv::Mat logitsToLabelMap(const float* logits);
    cv::Mat decodeCityscapesTarget(const cv::Mat& labelMap);
    
    // Cityscapes color mapping
    static const std::array<cv::Vec3b, 20> cityscapesTrainIdToColor_;
};

#endif // DEEPLABV3_INFERENCE_WRAPPER_H 