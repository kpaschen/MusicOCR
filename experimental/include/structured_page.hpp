#ifndef structured_page_hpp
#define structured_page_hpp

#include <memory>
#include <opencv2/imgproc.hpp>
#include <vector>


namespace musicocr {

class SheetLine;
class LineGroup;

class Sheet {
 public:
   Sheet() : voices_(0) {}
   Sheet(int voices) : voices_(voices) {}

   // Determine general left/right margins.
   static std::pair<int, int> overallLeftRight(const std::vector<SheetLine>&);

   void addLineGroup(LineGroup* group) {
     lineGroups.push_back(std::unique_ptr<LineGroup>(group));
   }

   size_t size() const {
     return lineGroups.size();
   }

   void initLineGroups(const std::vector<cv::Vec4i>& verticalLines,
                       const std::vector<SheetLine>& sheetLines,
                       const cv::Mat& clines);

   void printSheetInfo() const; 

 private:
   Sheet(const Sheet&) = delete;
   std::vector<std::unique_ptr<LineGroup>> lineGroups;
   int voices_;  // 0: determine based on layout.
};

class LineGroup {
 public:
   LineGroup() {}
   size_t size() const { return lines.size(); }
   void addSheetLine(const SheetLine& line) {
     lines.emplace_back(line);
   }

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

   static const int verticalPaddingPx = 20;
   static const int horizontalPaddingPx = 10;

   static void collectSheetLines(const std::vector<cv::Vec4i>&,
                                 std::vector<SheetLine>*,
                                 const cv::Mat&);

 private:
   static cv::Vec4i TopLine(const std::vector<cv::Vec4i>&);
   static cv::Rect BoundingBox(const std::vector<cv::Vec4i>&);

   // Lines found by houghp.
   std::vector<cv::Vec4i> lines;
   cv::Rect boundingBox;
};


}  // namespace musicocr


#endif  // structured_page_hpp
