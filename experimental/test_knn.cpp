#include <opencv2/ml.hpp>

#include "training_fileutils.hpp"
#include "training_key.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "TestKnn <test data directory> <model file to load> [filename pattern] " << std::endl;
    return -1;
  }
  const std::string directory = argv[1];
  std::string modelfile("");
  if (argc > 2) {
    modelfile = argv[2];
  }
  std::string fnamePattern("");
  if (argc > 3) {
    fnamePattern = argv[3];
  }
  std::string datasetname = musicocr::SampleDataFiles::datasetNameFromDirectoryName(
        directory);
  if (modelfile == "") {
    modelfile = "model." + datasetname;
  }

  musicocr::SampleData collector;
  musicocr::SampleDataFiles files;
  files.readFiles(directory, fnamePattern, musicocr::TrainingKey::statmodel);
  files.initCollector(directory, collector);

  {
  cv::Ptr<cv::ml::KNearest> knn =
     cv::ml::StatModel::load<cv::ml::KNearest>(
       musicocr::SampleDataFiles::modelFileName(modelfile, "knn"));
  cv::Mat predictions, neighbours, dist;
  std::ofstream out;
  out.open(musicocr::SampleDataFiles::makeModelOutputName(
    datasetname, "knn"));
  int quality = collector.runClassifier(knn, 3, predictions,
                                        neighbours, dist, out);

  std::cout << "knn quality on data set " << datasetname
            << ": " << quality << std::endl;
  }
  {
   cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(
       musicocr::SampleDataFiles::modelFileName(modelfile, "svm"));
   cv::Mat outcomes;
   std::ofstream out;
   out.open(musicocr::SampleDataFiles::makeModelOutputName(
     datasetname, "svm"));
   int quality = collector.runClassifier(svm, outcomes, out);
   std::cout << "svm quality on data set " << datasetname
             << ": " << quality << std::endl;
  }
  {
   cv::Ptr<cv::ml::DTrees> dtrees = cv::ml::StatModel::load<cv::ml::DTrees>(
       musicocr::SampleDataFiles::modelFileName(modelfile, "dtrees"));
   cv::Mat outcomes;
   std::ofstream out;
   out.open(musicocr::SampleDataFiles::makeModelOutputName(
     datasetname, "dtrees"));
   int quality = collector.runClassifier(dtrees, outcomes, out);
   std::cout << "dtree quality on data set " << datasetname
             << ": " << quality << std::endl;
  }
  return 0;
}
