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
  musicocr::SampleData* collector = new musicocr::SampleData();
  // Would be nicer to just pass the collector to readFiles.
  musicocr::SampleDataFiles files(collector);
  files.readFiles(directory, datasetname);

  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setIsClassifier(true);
  collector->trainClassifier(knn);

  // save model to file
  knn->save(modelfile);

  cout << "wrote model to " << modelfile << endl;

  // then need another binary to test the model.
  return 0;
}
