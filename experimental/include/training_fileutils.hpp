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
    static bool parseFilename(char *filename,
                              char *imagesetname,
                              int *linenumber, int *index,
                              int *xcoord, int *ycoord);

    // All files in dirname.
    void readFiles(const std::string& dirname, SampleData&);

    // Only one dataset.
    void readFiles(const std::string& dirname, const std::string& dataset,
                   SampleData&);

  private:
    // dataset name -> [line number -> list of x,y coords]
    std::map<std::string, std::map<int,
             std::vector<std::pair<int, int>>>> datasets;
    // dataset name -> [line number -> list of responses]
    std::map<std::string, std::map<int, std::vector<int>>> responses;
};

}  // namespace musicocr

#endif
