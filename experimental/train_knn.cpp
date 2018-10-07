#include <opencv2/ml.hpp>

#include "training_fileutils.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "need a training data directory." << endl;
    return -1;
  }
  string modelfile("modelfile.yaml");
  if (argc > 2) {
    modelfile = argv[2];
  } 
  string datasetname("");
  if (argc > 3) {
    datasetname = argv[3];
  }
  const string directory = argv[1];
  musicocr::SampleData collector;
  musicocr::SampleDataFiles files;
  files.readFiles(directory, datasetname, collector);

  cout << "read files, now starting training." << endl;

  { // KNN
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setIsClassifier(true);
  collector.trainClassifier(knn);

  // a really dumb evaluation of the model (test it on its own
  // training data).
  cv::Mat predictions, foo, bar;
  std::ofstream out;
  out.open("/tmp/modelout.KNN");
  int quality = collector.runClassifier(knn, 3, predictions,
                                         foo, bar, out);

  std::cout << "knn quality on training set (" << directory
            << "/" << datasetname << "): " << quality << endl;

  // save model to file
  knn->save(modelfile + ".knn");

  cout << "wrote model to " << modelfile << endl;
  }
  { // SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    collector.trainClassifier(svm);
    cv::Mat outcomes;
    std::ofstream out;
    out.open("/tmp/modelout.SVM");
    int quality = collector.runClassifier(svm, outcomes, out);
    std::cout << "svm quality on training set (" << directory
              << "/" << datasetname << "): " << quality << endl;
    svm->save(modelfile + ".svm");
    cout << "wrote model to " << modelfile << endl;
  }
  { // DTrees
    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
    dtree->setMaxCategories(20);
    dtree->setMaxDepth(10);
    dtree->setCVFolds(1);
    collector.trainClassifier(dtree);
    cv::Mat outcomes;
    std::ofstream out;
    out.open("/tmp/modelout.DTrees");
    int quality = collector.runClassifier(dtree, outcomes, out);
    std::cout << "dtree quality on training set (" << directory
              << "/" << datasetname << "): " << quality << endl;
    dtree->save(modelfile + ".dtree");
    cout << "wrote model to " << modelfile << endl;
  }

  return 0;
}
