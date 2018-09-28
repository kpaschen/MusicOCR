#include <iostream>
#include <opencv2/highgui.hpp>
#include <unordered_map>

#include "structured_page.hpp"
#include "utils.hpp"

namespace musicocr {

using namespace std;
using namespace cv;

vector<Vec4i> Sheet::find_lines(const Mat& warped) const {
  Mat tmp;
  GaussianBlur(warped, tmp, Size(config.gaussianKernel,
               config.gaussianKernel), 0, 0);
  threshold(tmp, tmp, config.thresholdValue, 255, config.thresholdType);
  Canny(tmp, tmp, config.cannyMin, config.cannyMax, config.sobel,
        config.l2Gradient);

  vector<Vec4i> lines;
  HoughLinesP(tmp, lines, 1, CV_PI/180.0, config.houghThreshold,
              config.houghMinLineLength, config.houghMaxLineGap);

  return lines;
}


void Sheet::printSheetInfo() const {
  cout << "Sheet has " << size() << " line groups." << endl;
  if (config.voices > 0) {
    cout << "set number of voices: " << config.voices << endl;
  } else {
    for (const auto& g : lineGroups) {
      cout << "line group with " << g->size() << " voices." << endl;
    }
  }
}

void Sheet::analyseLines(const vector<Vec4i>& lines, const Mat& clines) {
  vector<Vec4i> horizontal;
  vector<Vec4i> vertical;
  for (const auto& l : lines) {
    const short h = lineIsHorizontal(l);
    if (h == 1) horizontal.push_back(l);
    else if (h == 0) vertical.push_back(l);
  }

  vector<SheetLine> sheetLines;
  std::sort(horizontal.begin(), horizontal.end(), moreTop);
  SheetLine::collectSheetLines(horizontal, &sheetLines, clines);
  std::pair<int, int> leftRight = overallLeftRight(sheetLines);
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


pair<int, int> Sheet::overallLeftRight(const vector<SheetLine>& lines) {
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
  return pair<int, int>(fudge * maxLeft.first, fudge * maxRight.first + fudge);
}

SheetLine::SheetLine(const vector<Vec4i>& l, const Mat& wholePage) {
  for (const auto x : l) {
    lines.push_back(x);
  }
  Vec4i topLine = TopLine(lines); 

  const int heightDiff = topLine[1] - topLine[3];

  line(wholePage, Point(topLine[0], topLine[1]), Point(topLine[2], topLine[3]),
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
  const int left = std::max(0, (left1 < left2 ? (left2 < left3 ? left3 : left2)
                                : left1) - horizontalPaddingPx);
  const int right = std::min(cols, (right1 > right2 ? (right2 > right3 ? right3
                                    : right2) : right1) + horizontalPaddingPx);
   
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
      // line(clines, Point(curr[0], curr[1]), Point(curr[2], curr[3]),
      //      sheetLineColour, 2);
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



// This isn't adding much, just keeping the code for now.
void maybeCombineVerticalLines(
   const std::vector<cv::Vec4i>& sortedVerticalLines,
   std::vector<cv::Vec4i> *combinedVerticalLines) {
  int advance = 1;
  for (size_t i = 0; i < sortedVerticalLines.size(); i += advance) {
    const cv::Vec4i& tmp = sortedVerticalLines[i];
    cv::Vec4i currentLine;
    // make sure currentLine is top-to-bottom.
    if (tmp[1] > tmp[3]) {
      currentLine = {tmp[2], tmp[3], tmp[0], tmp[1]};
    } else {
      currentLine = {tmp[0], tmp[1], tmp[2], tmp[3]};
    }
    advance = 1;
    for (size_t j = i+1; j < sortedVerticalLines.size(); j++) {
      const cv::Vec4i& candidate = sortedVerticalLines[j];
      if (musicocr::maybeSameLine(currentLine, candidate, 6)) {
        int newBottom, newWidth;
        if (candidate[1] < candidate[3]) {
          newBottom = candidate[3];
          newWidth = candidate[2];
        } else {
          newBottom = candidate[1];
          newWidth = candidate[0];
        }
        advance++;
        currentLine[2] = newWidth;
        currentLine[3] = newBottom;
      }
    }
    combinedVerticalLines->push_back(currentLine);
    advance = 1;
  }
}


}  // namespace musicocr
