#include <iostream>

#include "training_fileutils.hpp"
#include "training_key.hpp"

int main(int argc, char** argv) {

  if (argc < 2) {
    std::cerr << "DumpData <test data directory> <target file name>"
              << std::endl;
    return -1;
  }
  std::string directory = argv[1];
  std::string outfile("dump.csv");
  if (argc > 2) {
    outfile = argv[2];
  }

  musicocr::SampleData collector;
  musicocr::SampleDataFiles files;
  files.readFiles(directory, musicocr::TrainingKey::basic);

  std::ofstream out;
  out.open(outfile);
  files.dumpData(out, directory);

  std::cout << "Test data written to " << outfile << std::endl;

  return 0;
}
