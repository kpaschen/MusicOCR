#ifndef training_hpp
#define training_hpp

#include <map>
#include <vector>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

namespace musicocr {

// Reads sample images and responses, can build training data
// and response matrices out of this.
class SampleData {
  public:
    cv::Mat makeSampleMatrix(const cv::Mat&) const;

    // Add one image and corresponding label.
    void addTrainingData(const cv::Mat&, int label);

    // If we have features and labels, train a classifier on them
    // and return it.
    bool trainClassifier(cv::Ptr<cv::ml::KNearest>);

    int runClassifier(cv::Ptr<cv::ml::KNearest>,
                      int neighbourCount,
                      cv::Mat& predictions,
                      cv::Mat& neighbours,
                      cv::Mat& dist);

  private:
    // All images will be resized to this width and height.
    static const int imageSize = 20;

    // accumulate features and labels in these internally.
    cv::Mat features, labels;
};

}  // namespace musicocr

#endif
