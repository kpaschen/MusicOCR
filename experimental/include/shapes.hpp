#ifndef shapes_hpp
#define shapes_hpp

#include <fstream>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include "recognition.hpp"
#include "structured_page.hpp"
#include "training_key.hpp"

namespace musicocr {

using namespace cv;
using namespace std;

struct ContourConfig {
  int gaussianKernel = 3;
  double thresholdValue = 0.0;
  int thresholdType = 4 + THRESH_OTSU;
  int cannyMin = 80;
  int cannyMax = 90;
  int sobelKernel = 3;
  bool l2Gradient = false;
  int horizontalSizeFudge = 30;
  int horizontalHeight = 1;
};

class Shape;

class ShapeFinder {
 public:
   ShapeFinder(const ContourConfig& c) : config(c) {}

   void getTrainingDataForLine(const Mat& focused,
     const string& processedWindowName,
     const string& questionWindowName,
     const string& filename,
     ofstream& responsesFile
   );

   const std::vector<cv::Rect>& getContourBoxes(const Mat& focused);

   void initLineScan(const musicocr::SheetLine& sheetLine,
                     const cv::Ptr<cv::ml::StatModel>& statModel);

   // Go over the image enclosed by sheetLine, detect contours and feed
   // them to statModel and ocr. Decide on their categorization.
   void scanLine(const musicocr::SheetLine& sheetLine,
                 const cv::Ptr<cv::ml::StatModel>& statModel,
                 const Scanner& ocr,
                 const string& processedWindowName,
                 const string& questionWindowName);

   // 0: no voice connectors found.
   // -1: not determined yet.
   // 1: top of several voices
   // 2: middle of several voices
   // 3: bottom of several voices
   int getVoicePosition() const { return voicePosition; }

   const std::vector<int> getBarPositions() const;
   const Shape* getBarAt(int x) const;

 private:
   ContourConfig config;

   std::vector<cv::Rect> contourBoxes;

   // This maps horizontal positions to vectors of shapes who have
   // this horizontal value as their tl().x.
   std::map<int, std::vector<std::unique_ptr<Shape>>> shapes;

   // bar lines by x coordinate.
   std::map<int, Shape*> barLines;

   cv::Mat preprocess(const Mat& img);

   int voicePosition = -1;

   // Initialise shapes based on rectangles:
   // create shapes with top-level categories and discover
   // neighbourhood relations.
   void firstPass(const std::vector<cv::Rect>& rectangles,
                  const cv::Mat& viewPort,
                  const cv::Ptr<cv::ml::StatModel>& statModel);

   void scanForBarLines(const cv::Mat& viewPort,
                        const cv::Rect& relativeInnerBox,
                        const std::pair<int, int>& slCoords); 
};

class Shape {
 public:
   Shape(const cv::Rect& rect);

   void print() const;

   const Rect& getRectangle() const { return rectangle; }

   void updateBelief(TrainingKey::Category, int diff);

   // Get the category with the strongest belief. Ties
   // resolved arbitrarily.
   TrainingKey::Category getMostLikelyCategory() const;

   // category confidence / sum of all confidences
   int categoryConfidence(TrainingKey::Category category) const;

   void setTopLevelCategory(TrainingKey::TopLevelCategory cat) {
     topLevelCategory = cat;
   }

   TrainingKey::TopLevelCategory getTopLevelCategory() const {
     return topLevelCategory;
   }

   // compass directions plus inside/outside
   enum Neighbourhood {
     UNKNOWN, N, NE, E, SE, S, SW, W, NW, IN, AROUND, INTERSECT
   };

   // Decide if shape is adjacent to this and add it if so.
   void maybeAddNeighbour(Shape* shape);

   size_t getNumberOfNeighbours() const;

   const std::map<Neighbourhood, std::vector<Shape*>>& getNeighbours() const {
     return neighboursByDirection;
   }

 private:
   // Distances (in pixels) up to this are considered 'adjacent'.
   static const int smallDistance = 2;

   // Location of this shape relative to the sheet line.
   // Also, size.
   cv::Rect rectangle;

   // Map TrainingKey categories to likelihoods between 0 and 100.
   std::map<TrainingKey::Category, int> beliefs;

   // This is the category returned by the stat model.
   TrainingKey::TopLevelCategory topLevelCategory;

   // Does not take ownership of shapes.
   std::map<Neighbourhood, std::vector<Shape*>> neighboursByDirection;

   void addNeighbour(Neighbourhood where, Shape *shape);
};

}  // namespace musicocr

#endif
