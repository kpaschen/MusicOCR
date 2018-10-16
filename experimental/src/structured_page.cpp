#include <iostream>
#include <opencv2/highgui.hpp>
#include <unordered_map>

#include "structured_page.hpp"
#include "utils.hpp"

namespace musicocr {

using namespace std;
using namespace cv;

// Find horizontal lines
vector<Vec4i> Sheet::find_lines(const Mat& processed) const {
  Mat tmp;
  equalizeHist(processed, tmp);
  Mat horizontalStructure = getStructuringElement(MORPH_RECT,
      Size(processed.cols/30, 1));
  dilate(tmp, tmp, horizontalStructure, Point(-1, -1));
  erode(tmp, tmp, horizontalStructure, Point(-1, -1));
  GaussianBlur(tmp, tmp, Size(config.gaussianKernel, config.gaussianKernel), 0, 0);
  Canny(tmp, tmp, config.cannyMin, config.cannyMax, config.sobel,
        config.l2Gradient);

  vector<Vec4i> lines;
  HoughLinesP(tmp, lines, 1, CV_PI/180.0, config.houghThreshold,
              config.houghMinLineLength, config.houghMaxLineGap);
  cout << "findLines: " << lines.size() << " found." << endl;
  tmp.release();

  return lines;
}

vector<Vec4i> Sheet::findVerticalLines(const Mat& processed) const {
  Mat tmp;
  Mat horizontalStructure = getStructuringElement(MORPH_RECT,
      Size(processed.cols/30, 1));
  dilate(processed, tmp, horizontalStructure, Point(-1, -1));
  erode(tmp, tmp, horizontalStructure, Point(-1, -1));
  tmp = processed + ~tmp;
/*
  Mat verticalStructure = getStructuringElement(MORPH_RECT,
     Size(1, processed.rows/30));
  erode(tmp, tmp, verticalStructure, Point(-1, -1));
  dilate(tmp, tmp, verticalStructure, Point(-1, -1));
*/
/*
  GaussianBlur(tmp, tmp, Size(7, 7), 0, 0);
  threshold(tmp, tmp, config.thresholdValue, 255, config.thresholdType);
*/

  Canny(tmp, tmp, config.cannyMin, config.cannyMax, config.sobel,
        config.l2Gradient);

  vector<Vec4i> lines;
  HoughLinesP(tmp, lines, 1, CV_PI/180.0, 100, 50, 15);
  // HoughLinesP(tmp, lines, 1, CV_PI/180.0, 50, 23, 15);
  cout << "findVerticalLines: " << lines.size() << " found." << endl;
  tmp.release();

  return lines;
}

vector<int> Sheet::getSheetInfo() const {
  vector<int> ret;
  for (const auto& g : lineGroups) {
    ret.push_back(g->size());
  }
  return ret;
}

void Sheet::printSheetInfo() const {
  cout << "Sheet has " << size() << " line groups." << endl;
  if (config.voices > 0) {
    cout << "set number of voices: " << config.voices << endl;
  } else {
    vector<int> voices = getSheetInfo();
    for (int v : voices) {
      cout << "line group with " << v << " voices." << endl;
    }
  }
}

void Sheet::analyseLines(vector<Vec4i>& horizontal,
     vector<Vec4i>& vertical, const Mat& clines) {
  vector<SheetLine> sheetLines;
  std::sort(horizontal.begin(), horizontal.end(), moreTop);
  SheetLine::collectSheetLines(horizontal, &sheetLines, clines);
  std::pair<int, int> leftRight = overallLeftRight(sheetLines, clines.cols);
  if (leftRight.first < 0) leftRight.first = 0;
  for (auto& sl : sheetLines) {
    sl.updateBoundingBox(leftRight, clines);
  }

  // Go over vertical lines, skipping those outside sheet margins.
  std::vector<cv::Vec4i> sortedVerticalLines;
  for (const auto& v : vertical) {
    if ((v[0] < leftRight.first && v[2] < leftRight.first) ||
        (v[0] > leftRight.second && v[2] > leftRight.second)) {
      continue;  // skip, it's outside the margins
    }
    // could probably use std::set and moreLeft as the comparator
    // to avoid the extra sort.
    sortedVerticalLines.push_back(v);
  }
  std::sort(sortedVerticalLines.begin(), sortedVerticalLines.end(),
            musicocr::moreLeft);

  initLineGroups(sortedVerticalLines, sheetLines, clines);
}

void Sheet::initLineGroups(const vector<Vec4i>& verticalLines,
                           const vector<SheetLine>& sheetLines,
                           const Mat& clines) {
  LineGroup *group = new LineGroup();
  addLineGroup(group);

  // This is probably wrong because sometimes there are lines that
  // are in between voice groups, or that have a title in them.
  if (config.voices > 0) {
    for (const auto& l : sheetLines) {
      if (group->size() >= config.voices) {
        group = new LineGroup();
        addLineGroup(group);
      }
      group->addSheetLine(l);
    }
    return;
  }

  vector<Vec4i> crossings;
  for (const auto& l : sheetLines) {
    bool startNewGroup = false;
    if (group->size() == 4) {
       startNewGroup = true;
    }
    else if (group->size() > 0 && crossings.size() == 0) {
      startNewGroup = true;
    }
    // Compute newCrossings and intersection count in any case.
    std::vector<Vec4i> newCrossings;
    int intersectSize = 0;
    for (const auto& v : verticalLines) {
      line(clines, Point(v[0], v[1]), Point(v[2], v[3]),
           Scalar(0, 0, 255), 1);
      if (l.crossedBy(v)) {
        newCrossings.push_back(v);
        for (const auto& c : crossings) {
          if (v[0] == c[0] && v[1] == c[1] && v[2] == c[2] && v[3] == c[3]) {
            intersectSize++;
            break;
          }
        }
      }
    }
    // now we have newCrossings and intersectSize
    // There's already a line, and there are crossings.
    if (group->size() > 0 && crossings.size() > 0) {
      // 2 may be too low a threshold.
      if (intersectSize < 2) { startNewGroup = true; }
    }
    if (startNewGroup) {
      group = new LineGroup();
      addLineGroup(group);
      group->addSheetLine(l);
    } else {
      group->addSheetLine(l);
    }
    // update crossings.
    crossings.clear();
    for (const auto& n : newCrossings) {
      crossings.push_back(n);
    }
  }
}

size_t Sheet::getLineCount() const {
 size_t s = 0;
 for (const auto& lg : lineGroups) {
   s += lg->size();
 }
 return s;
}

const SheetLine& Sheet::getNthLine(size_t i) const {
 size_t cur = 0;
 for (const auto& lg : lineGroups) {
   if (cur + lg->size() > i) {
     return lg->getNthVoice(i-cur);
   }
   cur += lg->size();
 }
 throw(std::runtime_error("Bad line index."));
}


pair<int, int> Sheet::overallLeftRight(const vector<SheetLine>& lines,
                                       int maxWidth) {
  // histograms of left and right borders
  unordered_map<int, int> leftBorders;
  unordered_map<int, int> rightBorders;

  static const int fudge = 5;

  // Divide left and right edge values by five to accumulate
  // more points.
  for (const auto& line: lines) {
   const int l = line.getLeftEdge() / fudge;
   const int r = line.getRightEdge() / fudge;
   int count = leftBorders[l];
   leftBorders[l] = count + 1;
   count = rightBorders[r];
   rightBorders[r] = count + 1;
  }

  // Max values of left and right edge.
  pair<int, int> maxLeft(0, 0);
  pair<int, int> maxRight(0, 0);
  for (const pair<int, int>& left : leftBorders) {
    if (left.second > maxLeft.second) {
      maxLeft = left;
    }
  }
  for (const pair<int, int>& right : rightBorders) {
    if (right.second > maxRight.second) {
      maxRight = right;
    }
  }
  return pair<int, int>(
    std::max(0, fudge * maxLeft.first - fudge),
    std::min(maxWidth, fudge * maxRight.first + fudge));;
}

SheetLine::SheetLine(const vector<Vec4i>& l, const Mat& wholePage) {
  for (const auto x : l) {
    lines.push_back(x);
  }
  Vec4i topLine = TopLine(lines); 
  line(wholePage, Point(topLine[0], topLine[1]),
                  Point(topLine[2], topLine[3]),
                  Scalar(255, 255, 255), 5);

  boundingBox = BoundingBox(lines, wholePage.rows, wholePage.cols);
  rectangle(wholePage, boundingBox, Scalar(255, 255, 255), 5);
}

void SheetLine::updateBoundingBox(const pair<int, int>& newLeftRight, const Mat& wholePage) {
  const int bottom = boundingBox.br().y;
  const int top = boundingBox.tl().y;
  boundingBox = Rect(Point(newLeftRight.first, top),
                     Point(newLeftRight.second, bottom));
  rectangle(wholePage, boundingBox, Scalar(255, 0, 0), 5);
}

bool SheetLine::crossedBy(const cv::Vec4i& vertical) const {
 const Point topLeft = boundingBox.tl();
 const Point bottomRight = boundingBox.br();

 // Discard lines that are to the left or right of the bounding box.
 // This could be done one level above for speed if necessary.
 if (vertical[0] < topLeft.x  || vertical[0] > bottomRight.x) {
   return false;
 }

 const int top = std::min(vertical[1], vertical[3]);
 const int bottom = std::max(vertical[1], vertical[3]);

 return (top <= (bottomRight.y - verticalPaddingPx/2)
         && bottom >= (topLeft.y + verticalPaddingPx/2));
}

Vec4i SheetLine::TopLine(const vector<Vec4i>& l) {
  // l is assumed to be ordered top to bottom.
  Vec4i topLine = l[0];
  const int topHeight = topLine[1];
  const int leftEdge = topLine[0];
  for (size_t i = 1; i < l.size(); i++) {
    const Vec4i& curr = l[i];
    // Is either edge of curr to the left of topLine[2] ?
    if (curr[0] < topLine[2] || curr[2] < topLine[2]) {
      break;
    }
    // Else, extend topLine
    topLine[2] = curr[2];
    topLine[3] = curr[3];
  }
  return topLine;
}

Rect SheetLine::BoundingBox(const vector<Vec4i>& l, int rows,
                            int cols) {
  // This is per-sheetline, might want to determine left/right
  // boundaries at sheet level though. Top and Bottom are
  // relatively straightforward.

  // Make sure to snap these to the actual edges of the sheet if
  // necessary.
  const int top = std::max(0, l[0][1] - verticalPaddingPx);
  const int bottom = std::min(rows, l.back()[1] + verticalPaddingPx);

  // Keep track of the leftmost and rightmost three points.
  // This is so we can discard outliers due to artifacts.
  int left1 = -1, left2 = -1, left3 = -1;
  int right1 = -1, right2 = -1, right3 = -1;
  for (const auto& x : l) {
    int l, r;
    if (x[0] < x[2]) { 
      l = x[0]; r = x[2];
    } else {
      l = x[2]; r = x[0];
    }
    if (left1 == -1) left1 = l;
    else if (left2 == -1) left2 = l;
    else if (left3 == -1) left3 = l;
    else if (l < left1) left1 = l;
    else if (l < left2) left2 = l;
    else if (l < left3) left3 = l;

    if (right1 == -1) right1 = r;
    else if (right2 == -1) right2 = r;
    else if (right3 == -1) right3 = r;
    else if (r > right1) right1 = r;
    else if (r > right2) right2 = r;
    else if (r > right3) right3 = r;
  }
  int left = std::max(0, (left1 < left2 ? (left2 < left3 ? left3 : left2)
                                : left1) - horizontalPaddingPx);
  if (left < 0) left = 0;
  int right = std::min(cols, (right1 > right2 ? (right2 > right3 ? right3
                                    : right2) : right1) + horizontalPaddingPx);
  if (right >= cols) right = cols = 1;
   
  return Rect(Point(left, top), Point(right, bottom));
}

void SheetLine::collectSheetLines(const vector<Vec4i>& horizontalLines,
                                  vector<SheetLine> *sheetLines,
                                  const Mat& clines) {
  // lastLineHeight is where the previous sheetline ended. Initialize
  // to top of page.
  int lastLineHeight = 0;
  vector<Vec4i> currentGroup;
  currentGroup.push_back(horizontalLines[0]);
  const Scalar sheetLineColour(120, 0, 0);
  const Scalar smallSheetLineColour(0, 120, 0);
  const Scalar betweenLinesColour(0, 0, 120);
  for (size_t i = 1; i < horizontalLines.size(); i++) {
    const Vec4i curr = horizontalLines[i];
    // gap == vertical distance between this line and the previous one.
    const int gap = std::abs(curr[1] - currentGroup.back()[1]);
    if (gap < 7) {
      currentGroup.push_back(curr);
      lastLineHeight = curr[1];
       line(clines, Point(curr[0], curr[1]), Point(curr[2], curr[3]),
            sheetLineColour, 2);
      continue;
    }
    if (gap >= 20) {
      // How high is the current sheet line?
      const int height = currentGroup.back()[1] - currentGroup[0][1];
      // cout << "Current height: " << height << endl;
      if (height >= 20) {
        // Finalize the current set
        sheetLines->push_back(SheetLine(currentGroup, clines));
        lastLineHeight = currentGroup.back()[1];
        // Begin a new sheet line.
        currentGroup.clear();
        currentGroup.push_back(curr);
        line(clines, Point(curr[0], curr[1]), Point(curr[2], curr[3]),
             sheetLineColour, 1);
      } else {
        // cout << "Patchy sheet line?" << endl;
        // Draw the line just below the patchy line. 
        // we'll add this to the current group anyway.
        currentGroup.push_back(curr);
        line(clines, Point(curr[0], curr[1]), Point(curr[2], curr[3]),
             smallSheetLineColour, 1);
        continue;
      }
    }  // gap >= 7 and gap < 20
    // cout << "Some stuff between lines? " << endl;
    line(clines, Point(curr[0], curr[1]), Point(curr[2], curr[3]),
         betweenLinesColour, 1);
  }
  sheetLines->push_back(SheetLine(currentGroup, clines));
}

}  // namespace musicocr
