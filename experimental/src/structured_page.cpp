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

vector<vector<int>> Sheet::getSheetInfo() const {
  vector<vector<int>> ret;
  for (const auto& g : lineGroups) {
    ret.push_back(g->lines);
  }
  return ret;
}

void Sheet::printSheetInfo() const {
  cout << "Sheet has " << size() << " line groups." << endl;
  if (config.voices > 0) {
    cout << "set number of voices: " << config.voices << endl;
  } else {
    vector<vector<int>> voices = getSheetInfo();
    for (vector<int> v : voices) {
      cout << "line group with " << v.size() << " voices:";
      for (int i : v) {
        cout << i << ", ";
      }
      cout << endl;
    }
  }
  for (const auto& l : sheetLines) {
    cout << "inner box: " << l.getInnerBox() 
         << ": height " << l.getInnerBox().height
         << ", width " << l.getInnerBox().width
         << endl;
  }
}

void Sheet::createSheetLines(const vector<Rect>& outlines, const Mat& focused) {
  vector<Rect> horizontal;
  for (const auto& r : outlines) {
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
  for (const auto& h : horizontal) {
    sheetLines.emplace_back(h, focused);
  }
  for (auto& sl : sheetLines) {
    vector<Vec4i> lines = sl.obtainGridlines();
    sl.accumulateHorizontalLines(lines);
    // xxx not sure if this is pulling its weight.
    const float slope = sl.getSlope();
    if (std::abs(slope) >= 0.025) {
      cout << "rotate by " << (std::abs(slope) * 45.0)
           << " degrees " << (slope < 0 ? "counterclockwise"
                              : "clockwise") << endl;
      sl.rotateViewPort(slope);
      cout << "line finding post-rotation" << endl;
      lines = sl.obtainGridlines();
      sl.accumulateHorizontalLines(lines);
    }
  }
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

void SheetLine::printInfo(Mat& draw) const {
  const Vec4i& l = horizontalLines[0];
  line(draw, boundingBox.tl() + Point(l[0], l[1]),
             boundingBox.tl() + Point(l[2], l[3]), Scalar(0, 255, 0), 1);
  const Vec4i& b = horizontalLines.back();
  line(draw, boundingBox.tl() + Point(b[0], b[1]),
             boundingBox.tl() + Point(b[2], b[3]), Scalar(0, 255, 0), 1);

  //cout << "inner box: " << innerBox
  //     << ", horizontal lines go from "
  //     << horizontalLines[0][1] << " to "
  //     << horizontalLines.back()[1] << endl;
}

float SheetLine::getSlope() const {
  float totalSlope = 0.0;
  int totalWidth = 0;
  for (const auto& l : horizontalLines) {
    const float slope = (float)(l[1]-l[3])/std::abs(l[0] - l[2]);
    const int xwidth = std::abs(l[2] - l[0]);
    totalWidth += xwidth;
    totalSlope += xwidth * slope;
  }
  float averageSlope = 0.0;
  if (totalWidth > 10) {
    averageSlope = totalSlope / totalWidth;
  }
  cout << "average slope: " << averageSlope << endl;
  return averageSlope;
}

// TODO: this does everything in local-relative coordinates
// TODO: not sure if this is good enough
std::pair<int, int> SheetLine::coordinates() {
  if (!realMusicLine) {
    cout << "not a real music line." << endl;
    return std::make_pair(0, 0);
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
  }
  for (const auto& bottoml : bottomlines) {
    bottomcounts[bottoml[1]] += 1;
  }
  int maxtop = 0; int maxcount = 0;
  for (const auto& tl : topcounts) {
    if (tl.second > maxcount) {
      maxtop = tl.first;
      maxcount = tl.second;
    }
  }
  //cout << maxcount << " segments have top " << maxtop << endl;
  int maxbottom = 0;
  maxcount = 0;
  for (const auto& tl : bottomcounts) {
    if (tl.second > maxcount) {
      maxbottom = tl.first;
      maxcount = tl.second;
    }
  }
  //cout << maxcount << " segments have bottom " << maxbottom << endl;
  return std::make_pair(maxtop, maxbottom);
}

std::pair<int, int> SheetLine::getCoordinates() const {
  return std::make_pair(horizontalLines[0][1], horizontalLines.back()[1]);
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
