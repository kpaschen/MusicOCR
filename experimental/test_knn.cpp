#include <opencv2/ml.hpp>

#include "training_fileutils.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "need a test data directory." << std::endl;
    return -1;
  }
  const std::string directory = argv[1];
  std::string modelfile("modelfile.yaml");
  if (argc > 2) {
    modelfile = argv[2];
  }
  std::string datasetname("");
  if (argc > 3) {
    datasetname = argv[3];
  }

  musicocr::SampleData* collector = new musicocr::SampleData();
  // Would be nicer to just pass the collector to readFiles.
  musicocr::SampleDataFiles files(collector);
  files.readFiles(directory, datasetname);

  cv::Ptr<cv::ml::KNearest> knn = cv::ml::StatModel::load<cv::ml::KNearest>(modelfile);
  cv::Mat predictions, neighbours, dist;
  int quality = collector->runClassifier(knn, 3, predictions, neighbours, dist);

  std::cout << "quality on data set " << directory << "/" << datasetname
            << ": " << quality << std::endl;

  return 0;
}
