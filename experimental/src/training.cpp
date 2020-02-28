#include "training.hpp"

namespace musicocr {

  using cv::Mat;
  using std::string;
  using std::vector;
  using std::map;


Mat SampleData::makeSampleMatrix(const Mat& smat, int xcoord, int ycoord) const {
  vector<float> sizeline(imageSize, 0.0);
  sizeline[0] = (float)smat.rows;
  sizeline[1] = (float)smat.cols;
  sizeline[2] = (float)xcoord;
  sizeline[3] = (float)ycoord;

  Mat ret;
  if (preprocess) {
    Mat horizontalStructure =  getStructuringElement(cv::MORPH_RECT, cv::Size(10, 1));
    Mat tmp;
    dilate(smat, tmp, horizontalStructure, cv::Point(-1, -1));
    erode(tmp, tmp, horizontalStructure, cv::Point(-1, -1));
    tmp = smat + ~tmp;
    threshold(tmp, tmp, 0.0f, 255, 12);
    tmp = ~tmp;
    cv::resize(tmp, ret, cv::Size(imageSize, imageSize), 0, 0, cv::INTER_CUBIC);
  } else {
    cv::resize(smat, ret, cv::Size(imageSize, imageSize), 0, 0, cv::INTER_CUBIC);
  }
  ret.convertTo(ret, CV_32F);
  ret.push_back(Mat(sizeline, true).t());
  return ret.reshape(1, 1);
}

void SampleData::addTrainingData(const cv::Mat& smat, int label,
                                 int xcoord, int ycoord,
				 const string& basename) {
  Mat bla = makeSampleMatrix(smat, xcoord, ycoord);
  // std::cout << "Matrix for " << basename << "(" << label << "):" << std::endl << bla << std::endl << std::endl;
  features.push_back(bla);
  labels.push_back(label);
  filenames.push_back(basename);
}

bool SampleData::isReadyToTrain() const {
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
  return true;
}

bool SampleData::trainClassifier(cv::Ptr<cv::ml::KNearest> model) {
  if (!isReadyToTrain()) { return false; }
  Mat trainingLabels = labels.clone();
  trainingLabels.reshape(1, 1);
  trainingLabels.convertTo(trainingLabels, CV_32F);

  // This takes model parameters and flags that are model type dependent.
  model->train(features, cv::ml::ROW_SAMPLE, trainingLabels);
  return true;
}

bool SampleData::trainClassifier(cv::Ptr<cv::ml::SVM> model) {
  if (!isReadyToTrain()) { return false; }
  Mat trainingLabels = labels.clone();
  trainingLabels.reshape(1, 1);
  // trainingLabels.convertTo(trainingLabels, CV_32F);
  // for svm, the labels have to be int not float.
  cv::Ptr<cv::ml::TrainData> trainingData = cv::ml::TrainData::create(
    features, cv::ml::ROW_SAMPLE, trainingLabels);

  // trainAuto crashes in various different ways.
  // model->trainAuto(trainingData, 4);
  model->train(trainingData);
  return true;
}

bool SampleData::trainClassifier(cv::Ptr<cv::ml::DTrees> model) {
  if (!isReadyToTrain()) { return false; }
  Mat trainingLabels = labels.clone();
  trainingLabels.reshape(1, 1);
  cv::Ptr<cv::ml::TrainData> trainingData = cv::ml::TrainData::create(
    features, cv::ml::ROW_SAMPLE, trainingLabels);

  std::cout << "training dtrees" << std::endl;
  model->train(trainingData);
  std::cout << "done training" << std::endl;
  return true;
}

int SampleData::runClassifier(cv::Ptr<cv::ml::DTrees> model,
                              cv::Mat& outcomes,
                              std::ostream& out) const {
  if (!isReadyToRun()) { return 0; }
  size_t sameLabel = 0;
  out << "index, expected, predicted, filename" << std::endl;
  for (size_t i = 0; i < features.rows; i++) {
    const Mat& row = features.row(i);
    float response = model->predict(row);
    int expected = labels.at<int>(0, i);
    out << i << ", " << expected << ", " << (int)response << ", " << filenames[i] << std::endl;
    outcomes.push_back(response);
    if ((int)response == expected) {
      sameLabel++;
    }
  }
  std::cout << "outcomes: " << outcomes.size() << std::endl;
  return (int)((float)sameLabel * 100.0 / features.rows);
}

int SampleData::runClassifier(cv::Ptr<cv::ml::SVM> svm,
                              cv::Mat& outcomes,
                              std::ostream& out) const {
  if (!isReadyToRun()) { return 0; }
  size_t sameLabel = 0;
  out << "index, expected, predicted" << std::endl;
  for (size_t i = 0; i < features.rows; i++) {
    const Mat& row = features.row(i);
    float response = svm->predict(row);
    int expected = labels.at<int>(0, i);
    out << i << ", " << expected << ", " << (int)response << std::endl;
    outcomes.push_back(response);
    if ((int)response == expected) {
      sameLabel++;
    }
  }
  std::cout << "outcomes: " << outcomes.size() << std::endl;
  return (int)((float)sameLabel * 100.0 / features.rows);
}

bool SampleData::isReadyToRun() const {
  std::cout << "I have " << labels.rows << " labels and "
            << features.rows << " samples." << std::endl;
  if (labels.rows != features.rows) {
    std::cerr << "label count mismatch (bad testdata?)" << std::endl;
    return false;
  }
  return true;
}

int SampleData::runClassifier(cv::Ptr<cv::ml::KNearest> knn,
                              int neighbourCount,
                              cv::Mat& predictions,
                              cv::Mat& neighbours,
                              cv::Mat& dist,
                              std::ostream& out) const {
  if (!isReadyToRun()) { return 0; }
  // findNearest only works for knn.
  knn->findNearest(features, neighbourCount, predictions, neighbours, dist); 
  // predictions: one row, #samples columns
  // neighbours: neighbourCount rows, #samples columns
  // dist: neighbourCount rows, #samples columns
  out << "index, expected, predicted, [neighbours]" << std::endl;
  size_t sameLabel = 0;
  for (size_t i = 0; i < features.rows; i++) {
    const float prediction = predictions.at<float>(i, 0);
    const int expected = labels.at<int>(0, i);
    out << i << ", " << expected << ", " << (float)prediction << ", ";
    if ((int)prediction == expected) {
      sameLabel++;
    }
    for (size_t j = 0; j < neighbourCount; j++) {
      // look at neighbours and dist
      float nb = neighbours.at<float>(i, j);
      float d = dist.at<float>(i, j);
      out << "neighour " << j << ": " << nb << " dist: " << d << ",";
    }
    out << std::endl;
  }
  return (int)((float)sameLabel * 100.0 / labels.rows);
}

}  // namespace musicocr
