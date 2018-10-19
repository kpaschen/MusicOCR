#include <iostream>
#include <map>
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
                         const Mat& focused) {
  vector<Rect> horizontal;
  for (const auto& r : horizontalOutlines) {
    const int area = r.area();
    // experimentally determined values. These work ok for A4
    // paper with what I'd consider "standard" lining.
    if (area < 20000 || area > 80000 || r.height < 40) {
      // Skip this, it's most likely not a sheet line.
      continue;
    }
    horizontal.push_back(r);
  }
  std::sort(horizontal.begin(), horizontal.end(), musicocr::rectTop);
  vector<SheetLine> sheetLines;
  for (const auto& h : horizontal) {
    sheetLines.emplace_back(h, focused);
  }
  std::pair<int, int> leftRight = overallLeftRight(
      sheetLines, focused.cols);

  // Go over vertical lines, skipping those outside sheet margins.
  // could do this later and avoid having to compute overall left right.
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

  }
  std::sort(sortedVerticalLines.begin(), sortedVerticalLines.end(),
            musicocr::moreLeft);
  initLineGroups(sortedVerticalLines, sheetLines);
}

void Sheet::initLineGroups(const vector<Vec4i>& verticalLines,
                           const vector<SheetLine>& sheetLines) {
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
  viewPort = page(boundingBox).clone();
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
  const int left = std::max(0, r.tl().x - horizontalPaddingPx);
  const int right = std::min(cols, r.br().x + horizontalPaddingPx);
  return Rect(Point(left, top), Point(right, bottom));
}

vector<Vec4i> SheetLine::obtainGridlines() const {
  const Rect relative = innerBox - boundingBox.tl();
  Mat tmp = viewPort.clone();
  const int edHorizontalWidth = tmp.cols / 30;
  const Mat horizontalStructure = getStructuringElement(MORPH_RECT,
    Size(edHorizontalWidth, 1));
  dilate(tmp, tmp, horizontalStructure, Point(-1, -1));
  erode(tmp, tmp, horizontalStructure, Point(-1, -1));
  GaussianBlur(tmp, tmp, Size(3, 3), 0, 0); 

  // Might be better off with histogram equalization here for
  // some inputs.
  threshold(tmp, tmp, 0.0, 255, 12);

  // Without Canny, you get a lot of noise in the lines.
  Canny(tmp, tmp, 80, 90, 3, false);
  vector<Vec4i> lines;
  HoughLinesP(tmp, lines, 1, 2.0 * CV_PI/180.0, 70, 23, 32);
  tmp.release();
  return lines;
}

void SheetLine::accumulateHorizontalLines(const vector<Vec4i>& lines) {
  vector<Vec4i> innerLines;

  // Make sure we don't already have horizontal lines.
  horizontalLines.clear();
  // The lines that are coming in are relative to the bounding
  // box of this sheet line.
  const Rect& relative = innerBox - boundingBox.tl();
  for (const auto& l : lines) {
    // Skip lines that are outside the inner box.
    if (!relative.contains(Point(l[0], l[1])) ||
        !relative.contains(Point(l[2], l[3]))) {
      continue;
    }
    innerLines.push_back(l);
  }

  // Too few inner lines.
  if (innerLines.size() < minHorizontalLines) {
    cout << "only " << innerLines.size() << " lines inside inner box" << endl;
    realMusicLine = false;
    return;
  }

  std::sort(innerLines.begin(), innerLines.end(), musicocr::moreRight);
  std::sort(innerLines.begin(), innerLines.end(), musicocr::moreTop);
  vector<Vec4i> horizontals;
  for (size_t i = 0; i < innerLines.size(); i++) {
    const auto& l = innerLines[i];
    int currentHeight = l[1];
    Point lp(l[0], l[1]); Point rp(l[2], l[3]);
    int top = std::min(l[1], l[3]), bottom = std::max(l[1], l[3]);
    // Try to extend l rightwards.
    for (size_t j = i+1; j < innerLines.size(); j++) {
      const auto& k = innerLines[j];
      if (k[1] - bottom <= 2) {
        if (k[2] <= lp.x || k[0] >= rp.x) {
          if (k[2] <= lp.x) {
            lp = Point(k[0], k[1]);
          }
          if (k[0] >= rp.x) {
            rp = Point(k[2], k[3]);
          }
          top = std::min(top, std::min(k[1], k[3]));
          bottom = std::max(bottom, std::max(k[1], k[3]));
        } else {
          i = j + 1;
          break;
        }
      } else {
        i = j + 1;
        break;
      }
    }
    horizontals.emplace_back(Vec4i(lp.x, lp.y, rp.x, rp.y));
  }

  if (horizontals.size() < minHorizontalLines) {
    cout << "only " << horizontals.size() << " lines after merging" << endl;
    realMusicLine = false;
    return;
  }

  vector<Vec4i> goodLines;
  int lastHeight = 0;
  int bestLeft = 500, bestRight = 0;
  for (const auto& candidate : horizontals) {
    if (!goodLines.empty() && candidate[1] - lastHeight >= 12
         && lastHeight < 50) {
      cout << "resetting at height " << lastHeight << endl;
      bestLeft = 500, bestRight = 0;
      goodLines.clear();
    }
    goodLines.push_back(candidate);
    lastHeight = candidate[1];
    if (candidate[0] < bestLeft) bestLeft = candidate[0];
    if (candidate[2] > bestRight) bestRight = candidate[2];
  }
  if (goodLines.size() < minHorizontalLines) {
    cout << "too few lines after grouping" << endl;
    realMusicLine = false;
    return;
  }
  if (std::abs(goodLines[0][1] - goodLines.back()[1]) < 25) {
    cout << "total height not enough" << endl;
    realMusicLine = false;
    return;
  }
  cout << "got " << goodLines.size() << " horizontal lines"
       << " and total height "
       << goodLines[0][1] << " to " << goodLines.back()[1]
       << endl;
  for (const auto& l : goodLines) {
    horizontalLines.push_back(l);
  }
}

float SheetLine::getSlope() const {
  float totalSlope = 0.0;
  int totalWidth = 0;
  for (const auto& l : horizontalLines) {
    const float slope = (float)(l[1]-l[3])/std::abs(l[0] - l[2]);
    const int xwidth = std::abs(l[2] - l[0]);
    totalWidth += xwidth;
    totalSlope += xwidth * slope;
    cout << "line with xwdith " << xwidth
         << " and  slope " << slope << endl;
  }
  float averageSlope = 0.0;
  if (totalWidth > 10) {
    averageSlope = totalSlope / totalWidth;
  }
  cout << "average slope: " << averageSlope << endl;
  return averageSlope;
}

void SheetLine::coordinates(Mat& show) const {
  if (!realMusicLine) {
    cout << "not a real music line." << endl;
    return;
  }
  // go from innerbox left to right edge in horizontal segments
  // of about 50px. for each segment, determine top and bottom
  // horizontal line.
  const Rect relative = innerBox - boundingBox.tl();
  const int left = relative.tl().x;
  const int right = relative.br().x;
  vector<Vec4i> toplines;
  vector<Vec4i> bottomlines;
  const int segment = 20;
  for (int xcoord = left; xcoord < right; xcoord += segment) {
    int top = relative.br().y;
    int bottom = relative.tl().y;
    for (const auto& l : horizontalLines) {
      const int lleft = std::min(l[0], l[2]);
      const int lright = std::max(l[0], l[2]);
      const int ltop = std::min(l[1], l[3]);
      const int lbottom = std::max(l[1], l[3]);
      // Does line l overlap the current segment?
      if (lleft <= xcoord + segment && lright >= xcoord) {
        if (ltop < top) top = ltop;
        if (lbottom > bottom) bottom = lbottom;
      }
    }
    if (top == relative.br().y) top = relative.tl().y;
    if (bottom == relative.tl().y) bottom = relative.br().y;
    toplines.emplace_back(Vec4i({xcoord, top, std::min(xcoord+segment, right), top}));
    bottomlines.emplace_back(Vec4i({xcoord, bottom, std::min(xcoord+segment, right), bottom}));
  }
  map<int, int> topcounts; map<int, int> bottomcounts;
  for (const auto& topl : toplines) {
    topcounts[topl[1]] += 1;
    line(show, Point(topl[0], topl[1]), Point(topl[2], topl[3]),
         Scalar(0, 255, 0), 1);
  }
  for (const auto& bottoml : bottomlines) {
    bottomcounts[bottoml[1]] += 1;
    line(show, Point(bottoml[0], bottoml[1]), Point(bottoml[2], bottoml[3]),
         Scalar(0, 0, 255), 1);
  }
  int maxtop = 0; int maxcount = 0;
  for (const auto& tl : topcounts) {
    if (tl.second > maxcount) {
      maxtop = tl.first;
      maxcount = tl.second;
    }
  }
  cout << maxcount << " segments have top " << maxtop << endl;
  int maxbottom = 0;
  maxcount = 0;
  for (const auto& tl : bottomcounts) {
    if (tl.second > maxcount) {
      maxbottom = tl.first;
      maxcount = tl.second;
    }
  }
  cout << maxcount << " segments have bottom " << maxbottom << endl;
  line(show, Point(left, maxtop), Point(right, maxtop),
       Scalar(255, 0, 0), 1);
  line(show, Point(left, maxbottom), Point(right, maxbottom),
       Scalar(255, 0, 0), 1);
}

void SheetLine::rotateViewPort(float slope) {
  const Rect relative = innerBox - boundingBox.tl();
  rotationSlope = slope;
  const Point2f ctr((float)relative.tl().x, (float)relative.tl().y/2.0);
  Mat r = getRotationMatrix2D(ctr, (-1.0) * slope * 45.0, 1.0);
  warpAffine(viewPort, viewPort, r,
             Size(viewPort.cols, viewPort.rows),
             cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
}


}  // namespace musicocr
