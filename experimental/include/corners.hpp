#ifndef corners_hpp
#define corners_hpp

#include <string>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "utils.hpp"

namespace musicocr {
using namespace cv;

  struct CornerConfig {
    int gaussianKernel = 15;
    double thresholdValue = 0.0;
    int thresholdType = 4 + THRESH_OTSU;
    int cannyMin = 50;
    int cannyMax = 90;
    int sobelKernel = 3;
    bool l2gradient = false;
    double houghResolution = 1.0;
    double houghResolutionRad = 1.0;
    int houghThreshold = 50;
    int houghMinLinLength = 32;
    int houghMaxLineGap = 15;
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
    bool shouldRotate(const cv::Mat&) const;
    static bool mostLinesAreHorizontal(const std::vector<cv::Vec4i>& lines);
    void adjust(const cv::Mat& image, cv::Mat& target);

   private:
    cv::Vec4i getOutline(std::vector<cv::Vec4i>& lines,
                         // 0: top, 1: left, 2: bottom, 3: right
                         int orientation,
                         int width, int height) const;
    CornerConfig config;
  };

}  // namespace musicocr

#endif
