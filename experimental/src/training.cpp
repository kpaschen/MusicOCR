#include "training.hpp"

namespace musicocr {

  using cv::Mat;
  using std::string;
  using std::vector;
  using std::map;

Mat SampleData::makeSampleMatrix(const Mat& smat) const {
  vector<float> sizeline(imageSize, 0.0);
  sizeline[0] = (float)smat.rows;
  sizeline[1] = (float)smat.cols;

  Mat ret;
  cv::resize(smat, ret, cv::Size(imageSize, imageSize),
             0, 0, cv::INTER_CUBIC);
  ret.convertTo(ret, CV_32F);
  ret.push_back(Mat(sizeline, true).t());
  return ret.reshape(1, 1);
}

void SampleData::addTrainingData(const cv::Mat& smat, int label) {
  features.push_back(makeSampleMatrix(smat));
  labels.push_back(label);
}

bool SampleData::trainClassifier(cv::Ptr<cv::ml::KNearest> knn) {
  if (features.rows == 0) {
    std::cerr << "Can't train a classifier on no data." << std::endl;
    return false;
  }
  if (features.rows != labels.rows) {
     std::cerr << "Need same number of rows as labels but have "
         << features.rows << " rows and "
         << labels.rows << " labels." << std::endl;
    return false;
  }
  labels.reshape(1, 1);
  labels.convertTo(labels, CV_32F);

  knn->train(features, cv::ml::ROW_SAMPLE, labels);
  return true;
}

int SampleData::runClassifier(cv::Ptr<cv::ml::KNearest> knn,
                              int neighbourCount,
                              cv::Mat& predictions,
                              cv::Mat& neighbours,
                              cv::Mat& dist) {
  knn->findNearest(features, neighbourCount, predictions, neighbours, dist); 
  std::cout << "I have " << labels.rows << " labels and "
            << features.rows << " samples." << std::endl;
  if (labels.rows == features.rows) {
    // predictions: one row, #samples columns
    // neighbours: neighbourCount rows, #samples columns
    // dist: neighbourCount rows, #samples columns
    size_t sameLabel = 0;
    for (size_t i = 0; i < features.rows; i++) {
      const float prediction = predictions.at<float>(i, 0);
      const int expected = labels.at<int>(0, i);
      std::cout << "predicted label for " << i << ": " 
                << prediction << std::endl;
      std::cout << "expected: " << expected << std::endl;
      if ((int)prediction == expected) {
        sameLabel++;
      }
      for (size_t j = 0; j < neighbourCount; j++) {
        // look at neighbours and dist
        float nb = neighbours.at<float>(i, j);
        float d = dist.at<float>(i, j);
        std::cout << "neighour " << j << ": " << nb
                  << ", dist: " << d << std::endl;
      }
    }
    return (int)((float)sameLabel * 100.0 / labels.rows);
  } else {
    std::cerr << "label count mismatch (bad testdata?)" << std::endl;
    return 0;
  }
  return 0;
}



}  // namespace musicocr
