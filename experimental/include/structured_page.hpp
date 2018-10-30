#ifndef structured_page_hpp
#define structured_page_hpp

#include <memory>
#include <opencv2/imgproc.hpp>
#include <vector>


namespace musicocr {

class SheetLine;
class LineGroup;

struct SheetConfig {
  int voices = 0; // 0: determine algorithmically
  int gaussianKernel = 3;
  double thresholdValue = 0.0; 
  int thresholdType = 3;  // no OTSU
  int cannyMin = 80;
  int cannyMax = 90;
  int sobel = 3;
  bool l2Gradient = false;
  int houghThreshold = 100;
  int houghMinLineLength = 50;
  int houghMaxLineGap = 15;
};

class Sheet {
 public:
   Sheet(const SheetConfig& c) : config(c) {}
   Sheet() {}

   // This finds contours of lines.
   std::vector<cv::Rect> find_lines_outlines(const cv::Mat&) const;

   // x,y,s,t,b,l
   std::vector<cv::Vec4i> findVerticalLines(const cv::Mat&) const;

   // Take contours (bounding boxes) as found by find_lines_outlines
   // and create sheet lines for them.
   // Also performs corrective local rotations and initialised
   // per-sheetline horizontal lines (for coordinate finding).
   void createSheetLines(const std::vector<cv::Rect>&, const cv::Mat&);

   void analyseLines(const std::vector<cv::Rect>&,
                     const std::vector<cv::Vec4i>&, const cv::Mat&);

   // Determine general left/right margins.
   static std::pair<int, int> overallLeftRight(
     const std::vector<SheetLine>&, int maxWidth);

   size_t size() const { return lineGroups.size(); }
   size_t getLineCount() const { return sheetLines.size(); }

   const LineGroup& getNthLineGroup(size_t i) const {
     return *lineGroups[i];
   }
   SheetLine& getNthLine(size_t i) { return sheetLines[i]; }

   void printSheetInfo() const; 
   std::vector<std::vector<int>> getSheetInfo() const;

   int medianLineHeight;
   int medianLineDistance;

 private:
   void addLineGroup(LineGroup* group) {
     lineGroups.push_back(std::unique_ptr<LineGroup>(group));
   }

   void initLineGroups(const std::vector<cv::Vec4i>& verticalLines);

   Sheet(const Sheet&) = delete;
   std::vector<std::unique_ptr<LineGroup>> lineGroups;
   std::vector<SheetLine> sheetLines;
   SheetConfig config;
};

class LineGroup {
 public:
   LineGroup() {}
   size_t size() const { return lines.size(); }
   void addSheetLine(int i) {
     lines.push_back(i);
   }
   int getNthVoice(size_t i) const { return lines[i]; }
   std::vector<int> lines;
};

class SheetLine {
 public:

   // Initialize a sheet line. Rect is the inner
   // bounding box, coordinates relative to Mat.
   // Mat is a greyscale, corner-adjusted page.
   // SheetLine will initialize its local viewport
   // to Rect(Mat).
   SheetLine(const cv::Rect&, const cv::Mat&);
   bool crossedBy(const cv::Vec4i&) const;

   int getLeftEdge() const { return boundingBox.tl().x; }
   int getRightEdge() const { return boundingBox.br().x; }

   const cv::Rect& getBoundingBox() const { return boundingBox; }
   const cv::Rect& getInnerBox() const { return innerBox; }

   // Transform a clone of viewport and obtain horizontal lines.
   std::vector<cv::Vec4i> obtainGridlines() const;

   // Limit /lines/ to those inside the inner box. Attempt
   // to join pieces of the same horizontal line.
   // Will return early and set realMusicLine to false if too few
   // horizontal lines are found.
   void accumulateHorizontalLines(const std::vector<cv::Vec4i>& lines);

   float getSlope() const;

   bool isRealMusicLine() const { return realMusicLine; }

   // How much has this been rotated relative to the whole page.
   float getRotationSlope() const { return rotationSlope; }
   void rotateViewPort(float angle);

   // Try to improve coordinate finding done in accumulateHorizontalLines.
   std::pair<int, int> coordinates();

   // Return previously found coordinates (top, bottom line).
   std::pair<int, int> getCoordinates() const;
   const cv::Mat& getViewPort() const { return viewPort; }

   void printInfo(cv::Mat& draw) const;

 private:
   static const int verticalPaddingPx = 20;
   static const int horizontalPaddingPx = 10;
   static const int minHorizontalLines = 4;

   static cv::Rect BoundingBox(const cv::Rect&, int rows, int cols);

   cv::Mat viewPort;
   cv::Rect boundingBox, innerBox;
   std::vector<cv::Vec4i> horizontalLines;

   float rotationSlope = 0.0;

   // flip this to false when it turns out this line doesn't contain
   // music notes.
   bool realMusicLine = true;
};

}  // namespace musicocr

#endif  // structured_page_hpp
