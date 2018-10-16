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
  int gaussianKernel = 3; // 7 or 9?
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

   // This finds the horizontal lines.
   std::vector<cv::Vec4i> find_lines(const cv::Mat& warped) const;

   std::vector<cv::Vec4i> findVerticalLines(const cv::Mat&) const;

   // Determine general left/right margins.
   static std::pair<int, int> overallLeftRight(
     const std::vector<SheetLine>&, int maxWidth);

   void addLineGroup(LineGroup* group) {
     lineGroups.push_back(std::unique_ptr<LineGroup>(group));
   }

   size_t size() const {
     return lineGroups.size();
   }
   size_t getLineCount() const;

   const LineGroup& getNthLineGroup(size_t i) const {
     return *lineGroups[i];
   }
   const SheetLine& getNthLine(size_t i) const;

   void analyseLines(std::vector<cv::Vec4i>&,
                     std::vector<cv::Vec4i>&, const cv::Mat&);
   void printSheetInfo() const; 
   std::vector<int> getSheetInfo() const;

 private:
   void initLineGroups(const std::vector<cv::Vec4i>& verticalLines,
                       const std::vector<SheetLine>& sheetLines,
                       const cv::Mat& clines);

   Sheet(const Sheet&) = delete;
   std::vector<std::unique_ptr<LineGroup>> lineGroups;
   SheetConfig config;
};

class LineGroup {
 public:
   LineGroup() {}
   size_t size() const { return lines.size(); }
   void addSheetLine(const SheetLine& line) {
     lines.emplace_back(line);
   }

   const SheetLine& getNthVoice(size_t i) const { return lines[i]; }

 private:
   std::vector<SheetLine> lines;
};

class SheetLine {
 public:
   SheetLine(const std::vector<cv::Vec4i>&, const cv::Mat&);
   bool crossedBy(const cv::Vec4i&) const;

   int getLeftEdge() const { return boundingBox.tl().x; }
   int getRightEdge() const { return boundingBox.br().x; }
   void updateBoundingBox(const std::pair<int, int>&, const cv::Mat&);

   const cv::Rect& getBoundingBox() const { return boundingBox; }

   static const int verticalPaddingPx = 20;
   static const int horizontalPaddingPx = 10;

   static void collectSheetLines(const std::vector<cv::Vec4i>&,
                                 std::vector<SheetLine>*,
                                 const cv::Mat&);

 private:
   static cv::Vec4i TopLine(const std::vector<cv::Vec4i>&);
   static cv::Rect BoundingBox(const std::vector<cv::Vec4i>&,
                               int rows, int cols);

   // Lines found by houghp.
   std::vector<cv::Vec4i> lines;
   cv::Rect boundingBox;
};


}  // namespace musicocr


#endif  // structured_page_hpp
