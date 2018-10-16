#include <iostream>
#include <opencv2/highgui.hpp>
#include <unordered_map>

#include "structured_page.hpp"
#include "utils.hpp"

namespace musicocr {

using namespace std;
using namespace cv;

vector<cv::Rect> Sheet::find_lines_outlines(const Mat& processed) const {
  Mat tmp;
  Mat horizontalStructure = getStructuringElement(MORPH_RECT,
      Size(processed.cols/30, 1));
  dilate(processed, tmp, horizontalStructure, Point(-1, -1));
  erode(tmp, tmp, horizontalStructure, Point(-1, -1));
  adaptiveThreshold(tmp, tmp, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY, 15, -2);
  GaussianBlur(tmp, tmp, Size(9, 9), 0, 0);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(tmp, contours, hierarchy, RETR_TREE,
        CHAIN_APPROX_SIMPLE, Point(0, 0));
  vector<Rect> rectangles;
  for (int i = 0; i < contours.size(); i++) {
    vector<Point> hull;
    convexHull(Mat(contours[i]), hull, false);
    rectangles.emplace_back(boundingRect(Mat(hull)));
  }
  return rectangles;
}

vector<Vec4i> Sheet::findVerticalLines(const Mat& processed) const {
  Mat tmp;
  Mat horizontalStructure = getStructuringElement(MORPH_RECT,
      Size(processed.cols/30, 1));
  dilate(processed, tmp, horizontalStructure, Point(-1, -1));
  erode(tmp, tmp, horizontalStructure, Point(-1, -1));
  tmp = processed + ~tmp;

  threshold(tmp, tmp, config.thresholdValue, 255, 12);
  GaussianBlur(tmp, tmp, Size(3, 3), 0, 0);

  vector<Vec4i> lines;
  HoughLinesP(tmp, lines, 1, CV_PI/180.0, 100, 50, 15);
  // cout << "findVerticalLines: " << lines.size() << " found." << endl;
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
  for (const auto& g : lineGroups) {
    for (size_t i = 0; i < g->size(); i++) {
      const SheetLine& v = g->getNthVoice(i);
      cout << "v inner box: " << v.getInnerBox() 
           << ": height " << v.getInnerBox().height
           << ", width " << v.getInnerBox().width
           << endl;
    }
  }
}

void Sheet::analyseLines(const vector<Rect>& horizontalOutlines,
                         const vector<Vec4i>& vertical,
                         const Mat& clines) {
  vector<Rect> horizontal;
  for (const auto& r : horizontalOutlines) {
    const int area = r.area();
    // experimentally determined values. These work ok for A4
    // paper with what I'd consider "standard" lining.
    if (area < 20000 || area > 80000 || r.height < 40) {
      // Skip this, it's most likely not a sheet line.
      continue;
    }
    rectangle(clines, r, Scalar(0, 0, 255), 2);
    horizontal.push_back(r);
  }
  std::sort(horizontal.begin(), horizontal.end(), musicocr::rectTop);
  vector<SheetLine> sheetLines;
  for (const auto& h : horizontal) {
    sheetLines.emplace_back(h, clines);
  }
  std::pair<int, int> leftRight = overallLeftRight(sheetLines, clines.cols);
  for (auto& sl : sheetLines) {
    sl.updateBoundingBox(leftRight, clines);
  }
  // Go over vertical lines, skipping those outside sheet margins.
  std::vector<cv::Vec4i> sortedVerticalLines;
  for (const auto& v : vertical) {
    if (musicocr::lineIsHorizontal(v)) continue;
    if ((v[0] < leftRight.first && v[2] < leftRight.first) ||
        (v[0] > leftRight.second && v[2] > leftRight.second)) {
      continue;  // skip, it's outside the margins
    }
    // could probably use std::set and moreLeft as the comparator
    // to avoid the extra sort.
    sortedVerticalLines.push_back(v);
    // line(clines, Point(v[0], v[1]), Point(v[2], v[3]), Scalar(0, 255, 0), 1);

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
      //line(clines, Point(v[0], v[1]), Point(v[2], v[3]),
      //     Scalar(0, 0, 255), 1);
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

SheetLine& Sheet::getNthLine(size_t i) const {
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
    std::min(maxWidth, fudge * maxRight.first + fudge));
}

SheetLine::SheetLine(const Rect& r, const Mat& page) {
  innerBox = r;
  boundingBox = BoundingBox(r, page.rows, page.cols);
  cout << "setting inner box to " << innerBox << endl;
  cout << " and outer box to " << boundingBox << endl;
  rectangle(page, innerBox, Scalar(127, 0, 0), 1);
}

void SheetLine::updateBoundingBox(const pair<int, int>& newLeftRight, const Mat& wholePage) {
  const int bottom = boundingBox.br().y;
  const int top = boundingBox.tl().y;
  // Don't cross the inner box.
  const int left = std::min(newLeftRight.first, innerBox.tl().x);
  const int right = std::max(newLeftRight.second, innerBox.br().x);
  boundingBox = Rect(Point(left, top),
                     Point(right, bottom));
  rectangle(wholePage, boundingBox, Scalar(255, 255, 255), 1);
}

bool SheetLine::crossedBy(const cv::Vec4i& vertical) const {
 const Point topLeft = innerBox.tl();
 const Point bottomRight = innerBox.br();

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

Rect SheetLine::BoundingBox(const Rect& r, int rows, int cols) {
  const int top = std::max(0, r.tl().y - verticalPaddingPx);
  const int bottom = std::min(rows, r.br().y + verticalPaddingPx);
  const int left = std::max(0, r.tl().x - verticalPaddingPx);
  const int right = std::min(cols, r.br().x + verticalPaddingPx);
  return Rect(Point(left, top), Point(right, bottom));
}

}  // namespace musicocr
