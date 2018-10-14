#include <opencv2/ml.hpp>

#include "training_fileutils.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "TrainKnn <training data directory> [modelfile basename] [file name pattern]" << endl;
    return -1;
  }
  const string directory = argv[1];
  string modelfile("");
  if (argc > 2) {
    modelfile = argv[2];
  }
  string filenamepattern("");
  if (argc > 3) {
    filenamepattern = argv[3];
  }
  // use this for naming output files and models, but don't set
  // a file name pattern.
  string datasetname = musicocr::SampleDataFiles::datasetNameFromDirectoryName(
      directory);
  // This is for saving the outputs.
  if (modelfile == "") {
    modelfile = "model." + datasetname;  
  }

  musicocr::SampleData collector;
  musicocr::SampleDataFiles files;
  files.readFiles(directory, datasetname, filenamepattern, collector);

  cout << "read files, now starting training." << endl;

  { // KNN
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setIsClassifier(true);
  collector.trainClassifier(knn);

  // a really dumb evaluation of the model (test it on its own
  // training data).
  cv::Mat predictions, foo, bar;
  std::ofstream out;
  out.open(musicocr::SampleDataFiles::makeModelOutputName(
      datasetname, "KNN"));
  int quality = collector.runClassifier(knn, 3, predictions,
                                         foo, bar, out);

  std::cout << "knn quality on training set (" << directory
            << "/" << datasetname << "): " << quality << endl;

  // save model to file
  knn->save(musicocr::SampleDataFiles::modelFileName(
    modelfile, "knn"));
  std::cout << "model written to " << modelfile << ".knn.yaml" << std::endl;
  }
  { // SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);

    collector.trainClassifier(svm);
    cv::Mat outcomes;
    std::ofstream out;
    out.open(musicocr::SampleDataFiles::makeModelOutputName(
        datasetname, "SVM"));
    int quality = collector.runClassifier(svm, outcomes, out);
    std::cout << "svm quality on training set (" << directory
              << "/" << datasetname << "): " << quality << endl;
    svm->save(musicocr::SampleDataFiles::modelFileName(
      modelfile, "svm"));
    cout << "wrote svm model to " << modelfile << ".svm.yaml" << endl;
  }
  { // DTrees
    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
    dtree->setMaxCategories(20);
    dtree->setMaxDepth(10);
    dtree->setCVFolds(1);
    collector.trainClassifier(dtree);
    cv::Mat outcomes;
    std::ofstream out;
    out.open(musicocr::SampleDataFiles::makeModelOutputName(
        datasetname, "DTrees"));
    int quality = collector.runClassifier(dtree, outcomes, out);
    std::cout << "dtree quality on training set (" << directory
              << "/" << datasetname << "): " << quality << endl;
    dtree->save(musicocr::SampleDataFiles::modelFileName(
      modelfile, "dtrees"));
    cout << "wrote dtree model to " << modelfile << ".dtrees.yaml" << endl;
  }

  return 0;
}
