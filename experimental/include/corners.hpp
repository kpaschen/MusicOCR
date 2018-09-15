#ifndef corners_hpp
#define corners_hpp

#include <string>
#include <vector>
#include "opencv2/imgproc.hpp"

namespace musicocr {
using namespace cv;

  struct CornerConfig {
    int gaussianKernel = 1;  // 15 is also sometimes good
    double thresholdValue = 0.0;
    int thresholdType = 4 + THRESH_OTSU;
    int cannyMin = 80;
    int cannyMax = 90;
    int sobelKernel = 3;
    bool l2gradient = true;
    int houghResolution = 8;
    int houghResolutionRad = 118;
    int houghThreshold = 87;
    int houghMinLinLength = 50;
    int houghMaxLineGap = 29;
  };

  class CornerFinder {
   public:
    CornerFinder(const CornerConfig& c) : config(c) {}
    CornerFinder() {}
    std::vector<cv::Vec4i> find_lines(const cv::Mat& image) const;
    void adjustToCorners(const cv::Mat& image, cv::Mat& warp,
                         const std::vector<cv::Point>& corners);
    std::vector<cv::Point> find_corners(const std::vector<cv::Vec4i>& lines,
                                        int width, int height) const;
    static bool mostLinesAreHorizontal(const std::vector<cv::Vec4i>& lines);

    cv::Vec4i getOutline(std::vector<cv::Vec4i>& lines,
                         // 0: top, 1: left, 2: bottom, 3: right
                         int orientation,
                         int width, int height) const;
   private:
    CornerConfig config;
  };

}  // namespace musicocr

#endif
