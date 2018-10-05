#ifndef training_fileutils_hpp
#define training_fileutils_hpp

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <vector>

#include "training.hpp"

namespace musicocr {

// Reads sample images and responses from files.
class SampleDataFiles {
  public:
    SampleDataFiles() {
      data_collector.reset(new SampleData);
    }
    SampleDataFiles(SampleData* data) : data_collector(data) {}
    static bool parseFilename(char *filename,
                              char *imagesetname,
                              int *linenumber, int *index);

    // All files in dirname.
    void readFiles(const std::string& dirname);

    // Only one dataset.
    void readFiles(const std::string& dirname, const std::string& dataset);

  private:
    // dataset name -> [line number -> number of items]
    std::map<std::string, std::map<int, int>> datasets;
    // dataset name -> [line number -> list of responses]
    std::map<std::string, std::map<int, std::vector<int>>> responses;

    std::unique_ptr<SampleData> data_collector;
};

}  // namespace musicocr

#endif
