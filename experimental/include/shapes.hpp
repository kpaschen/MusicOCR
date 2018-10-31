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

class CompositeShape {
  public:
    // Unknown: default initial state
    // Note: note head plus optional head, dot, accidentals, expressive marks
    //   -- expressive marks and connectors can link multiple notes
    //   -- together into a NOTEGROUP.
    //   -- chords are notegroups
    //   -- breaks are notes
    // Linestart: combination of bar line, clef, time, accidentals usually
    //     found at start of piece (but may also occur in middle).
    // Barline: single or double line, long or short, possibly including
    //   repeat marks.
    // Other: unidentified item within the inner box (could be e.g. 'A7' or similar, 
    //    or just an item that wasn't parsed properly).
    // Outofline: outside the inner box and unidentified (not clearly a note head
    //   or vertical line, possibly writing or piece of adjacent line).
    
    // Not sure if maybe note and notegroup should be one type?
    enum CompositeType {
      UNKNOWN, NOTE, NOTEGROUP, LINESTART, BARLINE, OTHER, OUTOFLINE
    };

    // Does not take ownership of Shape.
    CompositeShape(CompositeType, Shape*);

    // Does not take ownership of Shape.
    void addShape(Shape*);

    const cv::Rect& getRectangle() const { return boundingBox; }
    CompositeType getType() const { return type; }

  private:
    // Each shape has its position (rectangle) and neighbours list.
    std::vector<Shape *> shapes;

    // Box containing all the shapes.
    cv::Rect boundingBox;
    CompositeType type;
};

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
                 const cv::Ptr<cv::ml::StatModel>& fineStatModel,
                 const Scanner& ocr,
                 const string& processedWindowName,
                 const string& questionWindowName);

   // 0: no voice connectors found.
   // -1: not determined yet.
   // 1: top of several voices
   // 2: middle of several voices
   // 3: bottom of several voices
   int getVoicePosition() const { return voicePosition; }

   std::unique_ptr<CompositeShape>&
     addCompositeShape(CompositeShape::CompositeType, Shape*);

   const std::vector<std::unique_ptr<CompositeShape>>& getComposites() const {
     return compositeShapes;
   }

 private:
   ContourConfig config;

   std::vector<cv::Rect> contourBoxes;

   // This maps horizontal positions to vectors of shapes who have
   // this horizontal value as their tl().x.
   std::map<int, std::vector<std::unique_ptr<Shape>>> shapes;

   // All composite shapes (including bar lines)
   std::vector<std::unique_ptr<CompositeShape>> compositeShapes;

   cv::Mat preprocess(const Mat& img);

   int voicePosition = -1;

   // Returns true if s is part of a composite, false otherwise.
   bool isShapeInComposite(const Shape& s) const;

   // Initialise shapes based on rectangles:
   // create shapes with top-level categories and discover
   // neighbourhood relations.
   void firstPass(const std::vector<cv::Rect>& rectangles,
                  const cv::Mat& viewPort,
                  const cv::Ptr<cv::ml::StatModel>& statModel);

   void scanForBarLines(const cv::Mat& viewPort,
                        const cv::Rect& relativeInnerBox,
                        const std::pair<int, int>& slCoords); 

   // look for items that are probably writing between lines or that belong
   // to another sheetline.
   void scanForDiscards(const cv::Rect& relativeInnerBox);
   void scanForNotes(const cv::Rect& relativeInnerBox);
   void scanStartOfLine(const cv::Rect& relativeInnerBox);
};

class Shape {
 public:
   Shape(const cv::Rect& rect);

   void print() const;

   // This is the rectangle where the shape is located relative to
   // the enclosing sheet line's bounding box.
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
