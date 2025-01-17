#ifndef training_hpp
#define training_hpp

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include "training_key.hpp"

namespace musicocr {

// Reads sample images and responses, can build training data
// and response matrices out of this.
class SampleData {
  public:
    cv::Mat makeSampleMatrix(const cv::Mat&, int xcoord, int ycoord) const;

    // Add one image and corresponding label.
    // Also adds base filename as metadata for debugging.
    void addTrainingData(const cv::Mat&, int label, int xcoord, int ycoord, const std::string& basename);

    void setPreprocessing(bool prep) { preprocess = prep; }

    bool isReadyToTrain() const;
    bool isReadyToRun() const;

    // If we have features and labels, train a classifier on them
    // and return it.
    bool trainClassifier(cv::Ptr<cv::ml::KNearest>);
    bool trainClassifier(cv::Ptr<cv::ml::SVM>);
    bool trainClassifier(cv::Ptr<cv::ml::DTrees>);

    int runClassifier(cv::Ptr<cv::ml::KNearest>,
                      int neighbourCount,
                      cv::Mat& predictions,
                      cv::Mat& neighbours,
                      cv::Mat& dist,
                      std::ostream& output) const;

    int runClassifier(cv::Ptr<cv::ml::SVM>,
                      cv::Mat& outcomes,
                      std::ostream& output) const;

    int runClassifier(cv::Ptr<cv::ml::DTrees>,
                      cv::Mat& outcomes,
                      std::ostream& output) const;

  private:
    // All images will be resized to this width and height.
    static const int imageSize = 20;

    // Preprocessing on/off
    bool preprocess;

    // accumulate features and labels in these internally.
    cv::Mat features, labels;
    std::vector<std::string> filenames;
};

}  // namespace musicocr

#endif
