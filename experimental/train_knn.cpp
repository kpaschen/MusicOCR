#include <opencv2/ml.hpp>

#include "training_fileutils.hpp"
#include "training_key.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "TrainKnn <training data directory> [modelfile basename] "
         << " [file name pattern]" << endl;
    return -1;
  }
  const string directory = argv[1];
  // This creates two modelfiles per model type, a 'normal' one and a 'fine' one.
  // The fine model uses the full range of categories for shapes. The other model
  // projects the categories onto a smaller set for training, so it can basically
  // only distinguish vertical/horizontal lines, "round" things, and "other".
  string modelfile("model.");
  string modelfile_fine("model.");
  // use this for naming output files and models, but don't set
  // a file name pattern.
  const string datasetname = musicocr::SampleDataFiles::datasetNameFromDirectoryName(directory);
  if (argc > 2) {
    // Modelfile names have to start with 'model'.
    modelfile.append(argv[2]);
    modelfile_fine.append(argv[2]).append("-fine");
  } else {
    modelfile.append(datasetname);
    modelfile_fine.append(datasetname + "-fine");
  }
  string filenamepattern("");
  if (argc > 3) {
    filenamepattern = argv[3];
  }

  musicocr::SampleData collector, collector_fine;
  collector_fine.setPreprocessing(true);
  musicocr::SampleDataFiles files, files_fine;
  files.readFiles(directory, filenamepattern, musicocr::TrainingKey::statmodel);
  files_fine.readFiles(directory, filenamepattern, musicocr::TrainingKey::basic);
  files.initCollector(directory, collector);
  files_fine.initCollector(directory, collector_fine);

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
  { // KNN - fine
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setIsClassifier(true);
  collector_fine.trainClassifier(knn);

  // a really dumb evaluation of the model (test it on its own
  // training data).
  cv::Mat predictions, foo, bar;
  std::ofstream out;
  out.open(musicocr::SampleDataFiles::makeModelOutputName(
      datasetname, "KNN"));
  int quality = collector.runClassifier(knn, 3, predictions,
                                         foo, bar, out);

  std::cout << "fine knn quality on training set (" << directory
            << "/" << datasetname << "): " << quality << endl;

  // save model to file
  knn->save(musicocr::SampleDataFiles::modelFileName(
    modelfile_fine, "knn"));
  std::cout << "fine model written to " << modelfile_fine << ".knn.yaml" << std::endl;
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
  { // SVM - fine
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);

    collector_fine.trainClassifier(svm);
    cv::Mat outcomes;
    std::ofstream out;
    out.open(musicocr::SampleDataFiles::makeModelOutputName(
        datasetname, "SVM"));
    int quality = collector.runClassifier(svm, outcomes, out);
    std::cout << "fine svm quality on training set (" << directory
              << "/" << datasetname << "): " << quality << endl;
    svm->save(musicocr::SampleDataFiles::modelFileName(
      modelfile_fine, "svm"));
    cout << "wrote svm model to " << modelfile_fine << ".svm.yaml" << endl;
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
  { // DTrees - fine
    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
    dtree->setMaxCategories(20);
    dtree->setMaxDepth(10);
    dtree->setCVFolds(1);
    collector_fine.trainClassifier(dtree);
    cv::Mat outcomes;
    std::ofstream out;
    out.open(musicocr::SampleDataFiles::makeModelOutputName(
        datasetname, "DTrees"));
    int quality = collector.runClassifier(dtree, outcomes, out);
    std::cout << "fine dtree quality on training set (" << directory
              << "/" << datasetname << "): " << quality << endl;
    dtree->save(musicocr::SampleDataFiles::modelFileName(
      modelfile_fine, "dtrees"));
    cout << "wrote dtree model to " << modelfile_fine << ".dtrees.yaml" << endl;
  }

  return 0;
}
