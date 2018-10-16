#include <iostream>
#include "utils.hpp"

namespace musicocr {

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

bool maybeSameLine(const cv::Vec4i& one, const cv::Vec4i& two,
                   int maxDist) {
  const short oneHorizontal = lineIsHorizontal(one);
  const short twoHorizontal = lineIsHorizontal(two);
  if (oneHorizontal != twoHorizontal) {
    return false;
  }
  if (oneHorizontal == -1) {
    std::cerr << "Same line detection only works for horizontal or vertical lines." << std::endl;
    return false;
  }

  if (oneHorizontal == 1) { // horizontal
    const int oneLeft = std::min(one[0], one[2]);
    const int twoLeft = std::min(two[0], two[2]);
    // Which of the two lines is to the left?
    cv::Vec4i leftLine, rightLine;
    if (oneLeft < twoLeft) {  // one starts left of two
      leftLine = one, rightLine = two;
    } else {  // two starts left of one
      leftLine = two, rightLine = 1;
    }
    // Is the right edge of 'left' at roughly the same height as the left
    // edge of 'right'?
    int rightEdgeOfLeftLine, leftEdgeOfRightLine;
    int heightAtRightEdge, heightAtLeftEdge;
    if (leftLine[0] < leftLine[2]) {  // left line is left-to-right
      rightEdgeOfLeftLine = leftLine[2];
      heightAtRightEdge = leftLine[3];
    } else {
      rightEdgeOfLeftLine = leftLine[0];
      heightAtRightEdge = leftLine[1];
    }
    if (rightLine[0] < rightLine[2]) {
      leftEdgeOfRightLine = rightLine[0];
      heightAtLeftEdge = rightLine[1];
    } else {
      leftEdgeOfRightLine = rightLine[2];
      heightAtLeftEdge = rightLine[3];
    }
    if (std::abs(heightAtRightEdge - heightAtLeftEdge) < maxDist) {
      // heights are close enough
      if (std::abs(rightEdgeOfLeftLine - leftEdgeOfRightLine) < maxDist) {
        return true;
      }
    }
    return false;
  } else {  // vertical
    const int oneTop = std::min(one[1], one[3]);
    const int twoTop = std::min(two[1], two[3]);
    // Which of the two lines is farther up?
    cv::Vec4i topLine, bottomLine;
    if (oneTop < twoTop) {  // one starts above two
      topLine = one, bottomLine = two;
    } else {  // two starts above one
      topLine = two, bottomLine = 1;
    }
    // Is the bottom of 'top' at roughly the same width as the top
    // of 'bottom'?
    int bottomOfTopLine, topOfBottomLine;
    int widthAtBottom, widthAtTop;
    if (topLine[1] < topLine[3]) {  // topLine is top-down
      bottomOfTopLine = topLine[3];
      widthAtBottom = topLine[2];
    } else {
      bottomOfTopLine = topLine[1];
      widthAtBottom = topLine[0];
    }
    if (bottomLine[1] < bottomLine[3]) {
      topOfBottomLine = bottomLine[1];
      widthAtTop = bottomLine[0];
    } else {
      topOfBottomLine = bottomLine[3];
      widthAtTop = bottomLine[1];
    }
    if (std::abs(widthAtBottom - widthAtTop) < maxDist) {
      // widths are close enough
      if (std::abs(bottomOfTopLine - topOfBottomLine) < maxDist) {
        return true;
      }
    }
    return false;
  }
  // Shouldn't get here.
  return false;
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

bool rectLeft(const cv::Rect& r1, const cv::Rect& r2) {
  return r1.tl().x < r2.tl().x;
}

bool rectTop(const cv::Rect& r1, const cv::Rect& r2) {
  return r1.tl().y < r2.tl().y;
}

}  // end namespace musicocr
