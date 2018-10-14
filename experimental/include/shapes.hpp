#ifndef shapes_hpp
#define shapes_hpp

#include <fstream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "structured_page.hpp"

namespace musicocr {

using namespace cv;
using namespace std;

struct ContourConfig {
  int gaussianKernel = 3;
  double thresholdValue = 0.0;
  int thresholdType = 4 + THRESH_OTSU;
  int cannyMin = 80;
  int cannyMax = 90;
  int sobelKernel = 3;
  bool l2Gradient = false;
  int horizontalSizeFudge = 30;
  int horizontalHeight = 1;
};

class ShapeFinder {
 public:
   ShapeFinder(const ContourConfig& c) : config(c) {}

   void getTrainingDataForLine(const Mat& focused,
     const string& processedWindowName,
     const string& questionWindowName,
     const string& filename,
     ofstream& responsesFile
   );

   std::vector<cv::Rect> getContourBoxes(const Mat& focused, Mat&) const;

 private:
   ContourConfig config;

};

}  // namespace musicocr

#endif
