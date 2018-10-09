#include "corners.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace musicocr {
using namespace std;
using namespace cv;

std::vector<cv::Vec4i> CornerFinder::find_lines(const cv::Mat& image) const {
  // Input is expected to be a grayscale image.
  Mat tmp;
  GaussianBlur(image, tmp, Size(config.gaussianKernel, config.gaussianKernel),
               0, 0);
  threshold(tmp, tmp, config.thresholdValue, 255, config.thresholdType);
  Canny(tmp, tmp, config.cannyMin, config.cannyMax, config.sobelKernel,
        config.l2gradient);

  vector<Vec4i> lines;
  HoughLinesP(tmp, lines, 1, CV_PI/180.0, config.houghThreshold,
               config.houghMinLinLength, config.houghMaxLineGap);
  tmp.release();

  return lines;
}

bool CornerFinder::shouldRotate(const cv::Mat& image) const {
  Mat tmp;
  GaussianBlur(image, tmp, Size(config.gaussianKernel, config.gaussianKernel),
               0, 0);
  equalizeHist(tmp, tmp);
  Canny(tmp, tmp, config.cannyMin, config.cannyMax, config.sobelKernel,
        config.l2gradient);
  vector<Vec4i> lines;
  HoughLinesP(tmp, lines, 1, CV_PI/180.0, config.houghThreshold,
               config.houghMinLinLength, config.houghMaxLineGap);
  tmp.release();
  return !mostLinesAreHorizontal(lines);
}

cv::Point getIntersection(cv::Vec4i l1, cv::Vec4i l2) {
  cv::Point p(l1[0], l1[1]);
  cv::Point p2(l1[2], l1[3]);
  cv::Point q(l2[0], l2[1]);
  cv::Point q2(l2[2], l2[3]);
  cv::Point r(p2 - p);
  cv::Point s(q2 - q);
  if (r.cross(s) == 0) {
    throw std::logic_error("these lines should not be parallel.");
  }
  cv::Point x(q-p);
  double t = x.cross(s)/r.cross(s);
  return p + t * r;
}

// for efficiency, it's best to call this with 'lines' already preselected.
// don't call this repeatedly with the same set of lines for top, left etc.
cv::Vec4i CornerFinder::getOutline(std::vector<cv::Vec4i>& lines,
                     // 0: top, 1: left, 2: bottom, 3: right
                     int orientation,
                     int width, int height) const {
  cv::Vec4i outline;

  if (lines.size() == 0) {
    // This is fine, initialize outline to an edge of the image.
    switch(orientation) {
      case 0: return {0, 0, width, 0};  // top
      case 3: return {width, 0, width, height};
      case 2: return {0, height, width, height};
      case 1: return {0, 0, 0, height};
      default: throw std::logic_error("Illegal orientation, must be 0-3.");
    } 
  }
  if (lines.size() == 1) {
    const int dist = (orientation % 2 ? std::abs(lines[0][1] - lines[0][3])
                      : std::abs(lines[0][0] - lines[0][2]));
    return {lines[0][0], lines[0][1], lines[0][2], lines[0][3]};
  }
  switch(orientation) {
    case 0: std::sort(lines.begin(), lines.end(), moreTop); break;
    case 1: std::sort(lines.begin(), lines.end(), moreLeft); break;
    case 2: std::sort(lines.begin(), lines.end(), moreBottom); break;
    case 3: std::sort(lines.begin(), lines.end(), moreRight); break;
      default: throw std::logic_error("Illegal orientation, must be 0-3.");
  }
  // first point is lines[0][0],[1]. that segment ends at
  // lines[0][2],[3]
  int lastX = lines[0][2]; int lastY = lines[0][3];
  for (size_t i = 1; i < lines.size(); i++) {
    const cv::Vec4i line = lines[i];
    if (orientation % 2) {  // vertical lines
      // Is this roughly at the same width as lastX?
      if (std::abs(lastX - line[0]) < 6) {
         // And below lastY?
         if (line[1] >= lastY) {
           lastX = line[2]; lastY = line[3];
           continue;
         }
      }
      break;
    } else {  // horizontal lines
      // Is this roughly at the same height as lastY?
      if (std::abs(lastY - line[1]) < 6) {
         // And to the right of lastX?
         if (line[0] >= lastX) {
           lastX = line[2]; lastY = line[3];
           continue;
         }
      }
      break;
    }
  }
  // This defaults to just the first line segment if no extension was
  // possible.
  return {lines[0][0], lines[0][1], lastX, lastY};
}

std::vector<cv::Point> CornerFinder::find_corners(
    const std::vector<cv::Vec4i>& lines, int width, int height) const {
  std::vector<cv::Vec4i> bottomLines, topLines,
                         leftLines, rightLines;
  std::vector<cv::Vec4i> verticalLines;

  const float thirdHeight = (float)height / 3.0;
  const float bottomThird = (float)height - thirdHeight;
  const float thirdWidth = (float)width / 3.0;
  const float rightThird = (float)width - thirdWidth;

  for (const auto& line: lines) {
    const short h = lineIsHorizontal(line);
    if (h == 1) {
      if (line[1] < thirdHeight) {
        topLines.push_back(line);
      } else if (line[1] > bottomThird) {
        bottomLines.push_back(line);
      }
    } else if (h == 0) {
      if (line[0] < thirdWidth) {
        leftLines.push_back(line);
      } else if(line[0] > rightThird) {
        rightLines.push_back(line);
      }
    }
  }

  cv::Vec4i topLine, bottomLine, leftLine, rightLine;
  topLine = getOutline(topLines, 0, width, height); 
  bottomLine = getOutline(bottomLines, 2, width, height);
  leftLine = getOutline(leftLines, 1, width, height);
  rightLine = getOutline(rightLines, 3, width, height);

  const int topLineWidth = std::abs(topLine[0] - topLine[2]);
  if (width / topLineWidth > 18) {
    cout << "top line too short, snapping to edge of paper." << endl;
    topLine = {0, 0, width, 0};
  }
  const int bottomLineWidth = std::abs(bottomLine[0] - bottomLine[2]);
  if (width / bottomLineWidth > 18) {
    cout << "bottom line too short, snapping to edge of paper." << endl;
    bottomLine = {0, height, width, height};
  }
  const int leftLineHeight = std::abs(leftLine[1] - leftLine[3]);
  if (height / leftLineHeight > 18) {
    cout << "left line too short, snapping to edge of paper." << endl;
    leftLine = {0, 0, 0, height};
  }
  const int rightLineHeight = std::abs(rightLine[1] - rightLine[3]);
  if (height / rightLineHeight > 18) {
    cout << "right line too short, snapping to edge of paper." << endl;
    rightLine = {width, 0, width, height};
  }

  // Extend lines to edge of paper
  topLine[0] = 0; topLine[2] = width;
  bottomLine[0] = 0; bottomLine[2] = width;
  leftLine[1] = 0; leftLine[3] = height;
  rightLine[1] = 0; rightLine[3] = height;

  // intersection of topLine and leftLine == top left corner
  Point topLeft = getIntersection(topLine, leftLine);
  Point topRight = getIntersection(topLine, rightLine);
  Point bottomLeft = getIntersection(bottomLine, leftLine);
  Point bottomRight = getIntersection(bottomLine, rightLine);

  return {topLeft, topRight, bottomRight, bottomLeft};
}


// Decide if most (actually, at least half) of the lines are horizontal.
// This is intended to be used on the output of a hough lines detection that
// should have found (pieces of) the outer edges of the paper as well as
// (usually) some of the printed lines. Since the printed lines are always
// horizontal when you read the notes, it makes sense to orient the paper
// so the printed lines are actually horizontal. This can still leave us with
// an upside down image, which we may try to correct later.
bool CornerFinder::mostLinesAreHorizontal(const std::vector<cv::Vec4i>& lines) {
  int horizontalCount = 0;
  int verticalCount = 0;
  for (const auto& vec: lines) {
    short h = lineIsHorizontal(vec);
    if (h == 1) { 
      horizontalCount++;
    } else if (h == 0) {
      verticalCount++;
    }
  }
  return horizontalCount >= verticalCount;
}

void CornerFinder::adjustToCorners(const cv::Mat& image, cv::Mat& warp,
                     const std::vector<cv::Point>& corners) {
  Point2f source[4] = {Point2f(corners[0]), Point2f(corners[1]), Point2f(corners[2]),
                    Point2f(corners[3])};
  Point2f target[4] = {Point2f(0.0, 0.0), Point2f(image.cols, 0.0),
                    Point2f(image.cols, image.rows), Point2f(0.0, image.rows)};

  Mat lambda = getPerspectiveTransform(source, target);
  warpPerspective(image, warp, lambda, warp.size());
}

void CornerFinder::adjust(const cv::Mat& image, cv::Mat& target) {
  std::vector<cv::Vec4i> lines = find_lines(image);
  std::vector<cv::Point> corners = find_corners(lines, image.cols, image.rows); 
  adjustToCorners(image, target, corners);
  if (shouldRotate(target)) {
    cv::rotate(target, target, cv::ROTATE_90_COUNTERCLOCKWISE);
  }
}

} // namespace musicocr
