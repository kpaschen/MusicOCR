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

    static std::string datasetNameFromDirectoryName(const std::string&);
    static std::string makeModelOutputName(const std::string&, const std::string&);
    static std::string modelFileName(const std::string&, const std::string&);

    static int parseModelFileName(const std::string& model,
                                  char* trainingset,
                                  char* modeltype);

    // All files in dirname.
    void readFiles(const std::string& dirname, const std::string& dset, 
                   SampleData&);

    // Only one dataset.
    void readFiles(const std::string& dirname, const std::string& dataset,
                   const std::string& fname, SampleData&);

  private:
    // dataset name -> [line number -> list of x,y coords]
    std::map<std::string, std::map<int,
             std::map<int, std::pair<int, int>>>> datasets;
    // dataset name -> [line number -> list of responses]
    std::map<std::string, std::map<int, std::vector<int>>> responses;
};

}  // namespace musicocr

#endif
