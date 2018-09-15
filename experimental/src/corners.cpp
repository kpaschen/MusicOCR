#include "corners.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace musicocr {
using namespace std;
using namespace cv;

// Returns 1 for horizontal, 0 for vertical, -1 for neither.
short lineIsHorizontal(const cv::Vec4i& vec) {
    // vec is of type Vec4i, which is a quadruple of integers.
    // vec[0], vec[1] is the starting point, vec[2], vec[3] the end point
    // of a line. 
    const int dx = std::abs(vec[0] - vec[2]);
    const int dy = std::abs(vec[1] - vec[3]);

    if (dx == 0 && dy == 0) {
       // This is not a line.
       throw std::logic_error("Illegal line of length 0.");
    }
    if (dy == 0) { return 1; }
    if (dx == 0) { return 0; }
    const float dd = (float)dy / dx;
    if (dd < 0.8) {
      return 1;
    }
    if (dd > 1.2) {
      return 0;
    }
    return -1;  // neither
}

std::vector<cv::Vec4i> CornerFinder::find_lines(const cv::Mat& image) const {
  // Input is expected to be a grayscale image.
  Mat tmp;
  GaussianBlur(image, tmp, Size(config.gaussianKernel, config.gaussianKernel),
               0, 0);
  namedWindow("Blurred", WINDOW_AUTOSIZE);
  imshow("Blurred", tmp);
  threshold(tmp, tmp, config.thresholdValue, 255, config.thresholdType);
  namedWindow("Threshold", WINDOW_AUTOSIZE);
  imshow("Threshold", tmp);
  Canny(tmp, tmp, config.cannyMin, config.cannyMax, config.sobelKernel,
        config.l2gradient);
  namedWindow("Canny", WINDOW_AUTOSIZE);
  imshow("Canny", tmp);

  // Somehow this finds all the small horizontal lines
  vector<Vec4i> lines;
  HoughLinesP(tmp, lines, config.houghResolution, config.houghResolutionRad,
              config.houghThreshold, config.houghMinLinLength,
              config.houghMaxLineGap);
  return lines;
}

// Order by y coordinate, using x to break ties.
bool moreTop (const cv::Vec4i& line1, const cv::Vec4i& line2) {
  if (line1[1] > line2[1]) return false;
  if (line1[1] < line2[1]) return true;
  return line1[0] < line2[0];
}

bool moreBottom (const cv::Vec4i& line1, const cv::Vec4i& line2) {
  if (line1[1] > line2[1]) return true;
  if (line1[1] < line2[1]) return false;
  return line1[0] < line2[0];
}

// Order by x coordinate, using y to break ties.
bool moreRight (const cv::Vec4i& line1, const cv::Vec4i& line2) {
  if (line1[0] > line2[0]) return true;
  if (line1[0] < line2[0]) return false;
  return line1[1] < line2[1];
}

bool moreLeft (const cv::Vec4i& line1, const cv::Vec4i& line2) {
  if (line1[0] > line2[0]) return false;
  if (line1[0] < line2[0]) return true;
  return line1[1] < line2[1];
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
      case 0: return {0, 0, width, 0};
      case 1: return {width, 0, width, height};
      case 2: return {0, height, width, height};
      case 3: return {0, 0, 0, height};
      default: throw std::logic_error("Illegal orientation, must be 0-3.");
    } 
  }
  if (lines.size() == 1) {
    // Should we have a minimal length for these?
    const int dist = (orientation % 2 ? std::abs(lines[0][0] - lines[0][2])
                      : std::abs(lines[0][1] - lines[0][3]));
    cout << "only one line, of length " << dist << endl;
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
      // Is this roughtly at the same height as lastY?
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
  const float topThird = (float)height - thirdHeight;
  const float thirdWidth = (float)width / 3.0;
  const float rightThird = (float)width - thirdWidth;

  for (const auto& line: lines) {
    const short h = lineIsHorizontal(line);
    if (h == 1) {
      if (line[1] < thirdHeight) {
        topLines.push_back(line);
      } else if (line[1] > topThird) {
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

} // namespace musicocr
