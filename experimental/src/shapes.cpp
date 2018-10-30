#include "shapes.hpp"
#include "training.hpp"
#include "utils.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

namespace musicocr {

const std::vector<cv::Rect>& ShapeFinder::getContourBoxes(const Mat& focused) {
  if (contourBoxes.size() > 0) { return contourBoxes; }
  Mat processed;
  focused.copyTo(processed);
  // erode, dilate horizontally to get just the horizontal lines.
  const int edHorizontalWidth = processed.cols / config.horizontalSizeFudge;

  // Could just create this once when config gets read.
  Mat horizontalStructure = getStructuringElement(MORPH_RECT,
    Size(edHorizontalWidth, config.horizontalHeight));

  dilate(processed, processed, horizontalStructure, Point(-1, -1));
  erode(processed, processed, horizontalStructure, Point(-1, -1));

  // Now 'subtract' the horizontal lines from 'focused'.
  processed = focused + ~processed;
  // threshold, blur, canny
  threshold(processed, processed, config.thresholdValue, 255, config.thresholdType);
  Mat tmp = Mat::zeros(processed.rows, processed.cols, processed.type());
  GaussianBlur(processed, tmp, Size(config.gaussianKernel, config.gaussianKernel),
               0, 0);
  tmp.copyTo(processed);

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(processed, contours, hierarchy, RETR_TREE,
               CHAIN_APPROX_SIMPLE, Point(0, 0));

  contourBoxes.resize(contours.size());
  vector<vector<Point>> hull(contours.size());
  for (int i = 0; i < contours.size(); i++) {
    //drawContours(cont, contours, i, Scalar(255, 0, 0), 1, 8,
    //  hierarchy, 0, Point(0, 0));
    convexHull(Mat(contours[i]), hull[i], false);
    contourBoxes[i] = boundingRect(Mat(hull[i]));
  }
  std::sort(contourBoxes.begin(), contourBoxes.end(), musicocr::rectLeft);
  return contourBoxes;
}

void ShapeFinder::firstPass(const std::vector<cv::Rect>& rectangles,
                            const cv::Mat& viewPort,
                            const cv::Ptr<cv::ml::StatModel>& statModel) {
  SampleData sd;
  TrainingKey key;
  for (const auto& rect : rectangles) {
    Mat partial = Mat(viewPort, rect);
    // what does the system think this is.
    cv::Mat sample = sd.makeSampleMatrix(partial, rect.tl().x, rect.tl().y);
    float prediction = statModel->predict(sample);
    const TrainingKey::TopLevelCategory cat =
        static_cast<TrainingKey::TopLevelCategory>((int)prediction);
    Shape *shape = new Shape(rect);
    shape->setTopLevelCategory(cat);
    // Go over known shapes and add this one to their neighbour lists.
    // There is nothing there yet to the right of this rectangle, so only
    // need to look at whether things' left or right edge is near this
    // one's left edge.
    for (auto& coordshapes : shapes) {
      vector<unique_ptr<Shape>>& theseshapes = coordshapes.second;
      for (auto& s : theseshapes) {
        s->maybeAddNeighbour(shape);
      }
    }
    // Insert into shapes map at tl corner horizontal coordinate.
    const int xcoord = rect.tl().x;
    auto rects = shapes.find(xcoord);
    if (rects == shapes.end()) {
      std::vector<std::unique_ptr<Shape>> s;
      shapes.emplace(xcoord, std::vector<std::unique_ptr<Shape>>());
      rects = shapes.find(xcoord);
    }
    std::vector<std::unique_ptr<Shape>>& s = rects->second;
    s.emplace_back(shape);
  }
}

void ShapeFinder::scanForBarLines(const cv::Mat& viewPort,
                                  const cv::Rect& relativeInnerBox,
                                  const std::pair<int, int>& slCoords) {
  const int slHeight = slCoords.second - slCoords.first;
  const int leftEdge = relativeInnerBox.tl().x;
  const int rightEdge = relativeInnerBox.br().x;
  const int topEdge = relativeInnerBox.tl().y;
  const int bottomEdge = relativeInnerBox.br().y;

  // TODO: put this in a dedicated beginning-of-line scanner.
  // First, look for the left-most bar line, if any.
  // A line can start in several different ways:
  // - bar line followed by notes
  // - bar line followed by clef, then notes 
  // - just a clef followed by notes
  // 'notes' can include time (e.g. "4/4", accidentals) 
  // if there is no bar line at the start, then either there is
  // an implied bar line at the beginning, or there will be one
  // (possibly a double line with repeat signs, etc) just before the
  // actual music notes begin.

  // First, look for voice connectors (long bar lines).
  map<int, int> positions;
  bool haveFirstVline = false;
  for (const auto& siter : shapes) {
    const int xcoord = siter.first;
    const vector<std::unique_ptr<Shape>>& list = siter.second;
    for (size_t i = 0; i < list.size(); i++) {
      const Rect& r = list[i]->getRectangle();
      const auto cat = list[i]->getTopLevelCategory();
      const int height = r.br().y - r.tl().y;
      bool processThis = false;
      if (cat == TrainingKey::TopLevelCategory::composite) {
        const int width = r.br().x - r.tl().x;
        if (height / width >= 3) {
          // probably a vertical line. other vertical lines can be included
          // in larger composites, but won't be identified here.
          processThis = true;
        } else if (!haveFirstVline && xcoord < 30) {
          processThis = true;
        }
        haveFirstVline = true;
      }
      else if (cat == TrainingKey::TopLevelCategory::vline) {
        processThis = true;
        haveFirstVline = true;
      }
      if (!processThis) continue;
      bool aboveTop = r.tl().y < topEdge;
      bool belowBottom = r.br().y > bottomEdge;
      int position = -1;
      if (height > slHeight + 20) {
        if (!aboveTop && belowBottom) { position = 1; }  // top voice
        else if (aboveTop && belowBottom) { position = 2; } // middle voice
        else if (aboveTop && !belowBottom) { position = 3; }  // bottom voice
        positions.emplace(xcoord, position);
      }
    }
  }
  map<int, int> countByPosition;
  for (const auto& p: positions) {
    auto x = countByPosition.find(p.second);
    if (x == countByPosition.end()) {
      countByPosition.emplace(p.second, 1);
    } else {
      x->second++;
    }
  }
  int maxPos = 0, maxCount = 0;
  for (const auto& c : countByPosition) {
    if (c.second > maxCount) {
      maxCount = c.second;
      maxPos = c.first;
    }
  }
  voicePosition = maxPos;

  if (voicePosition > 0) {
    for (auto& siter : shapes) {
      const int xcoord = siter.first;
      const auto& pos = positions.find(xcoord);
      if (pos == positions.end()) continue;
      vector<std::unique_ptr<Shape>>& list = siter.second;
      for (size_t i = 0; i < list.size(); i++) {
        const auto cat = list[i]->getTopLevelCategory();
        if (cat == TrainingKey::TopLevelCategory::vline ||
            cat == TrainingKey::TopLevelCategory::composite) {
          cout << "adding bar line at " << xcoord << endl;
          barLines.emplace(xcoord, list[i].get());
          list[i]->updateBelief(TrainingKey::Category::bar, 10);
        }
      }
    }
    return;
  }

  // Is this a single voice line? Then we don't have any bar lines yet.
  haveFirstVline = false;

  // Look for the rightmost x coordinate of a vertical line or composite.
  int lastXCoord = -1;
  for (auto cr = shapes.crbegin(); cr != shapes.crend(); ++cr) {
    const vector<std::unique_ptr<Shape>>& list = cr->second;
    for (size_t i = 0; i < list.size(); i++) {
      const auto cat = list[i]->getTopLevelCategory();
      if (cat == TrainingKey::TopLevelCategory::vline || 
          cat == TrainingKey::TopLevelCategory::composite) {
        lastXCoord = cr->first;
        break;
      }
    }
    if (lastXCoord != -1) break;
  }
  for (const auto& siter : shapes) {
    const int xcoord = siter.first;
    if (xcoord < leftEdge) continue;
    if (xcoord > rightEdge) break;
    const vector<std::unique_ptr<Shape>>& list = siter.second;
    for (size_t i = 0; i < list.size(); i++) {
      const auto cat = list[i]->getTopLevelCategory();
      if (cat != TrainingKey::TopLevelCategory::vline &&
          cat != TrainingKey::TopLevelCategory::composite) continue;
      const Rect& r = list[i]->getRectangle();
      const Rect insideInnerRect = r & relativeInnerBox;
   
      if (insideInnerRect.area() == 0) continue;

      // There might be something else there besides the bar line
      // but there's certainly the end of a bar.
      if (xcoord == lastXCoord) {
//        cout << "end of line barline at " << xcoord << endl;
        barLines.emplace(xcoord, list[i].get());
        list[i]->updateBelief(TrainingKey::Category::bar, 10);
        break;
      }

      if (cat == TrainingKey::TopLevelCategory::composite) {
        const int width = r.br().x - r.tl().x;
        const int height = r.br().y - r.tl().y;
        if ((float)height / width < 3.5f) {
          continue;
        }
      }

      if (!haveFirstVline && xcoord < 30) {
        bool isFirstVline = true;
        for (const auto& s : shapes) {
          if (s.first > xcoord) { break; }
          for (const auto& sh : s.second) {
            const auto& otherRect = sh->getRectangle();
            if (otherRect.tl() == r.tl()) { break; }
            const int area = (otherRect & relativeInnerBox).area();
            if (area == 0) { continue; }
            const auto cat2 = sh->getTopLevelCategory();
            if (cat2 == TrainingKey::TopLevelCategory::vline ||
                  cat2 == TrainingKey::TopLevelCategory::composite) {
              isFirstVline = false;
              break;
            }
            if (!isFirstVline) break;
          }
          if (!isFirstVline) break;
        }
        if (isFirstVline) {
          //cout << "beginning of line barline: " << xcoord << endl;
          barLines.emplace(xcoord, list[i].get());
          list[i]->updateBelief(TrainingKey::Category::bar, 10);
          haveFirstVline = true;
          continue;
        }
      }
      const int height = r.br().y - r.tl().y;
      // right height?
      if (height < slHeight - 6) {
        continue;
      }
      // are the ends near the upper/lower horizontal lines?
      if (std::abs(r.tl().y - slCoords.first) > 5 ||
          std::abs(r.br().y - slCoords.second) > 5) {
        continue; 
      }

      const auto& neighbours = list[i]->getNeighbours();
      bool noteNeck = false;
      for (const auto& nb : neighbours) {
        if (nb.first == Shape::IN || nb.first == Shape::AROUND) { continue; }
        if (nb.first == Shape::E || nb.first == Shape::W) { continue; }
        for (const auto& sp : nb.second) {
          const auto cat = sp->getTopLevelCategory();
          if (cat == TrainingKey::TopLevelCategory::round ||
              cat == TrainingKey::TopLevelCategory::hline) {
            noteNeck = true;
            break;
          }
        }
        if (noteNeck) break;
      }
      if (noteNeck) continue;
      // Still here? Then it's probably a bar line.
      //cout << "conditions met for barline: " << xcoord << endl;
      barLines.emplace(xcoord, list[i].get());
      list[i]->updateBelief(TrainingKey::Category::bar, 10);
    }
  }
  // Thin out the bar lines. Assume the first bar line is correct
  // and that bar lines are at least 40 and at most 150 px apart.
  int previousBarX = barLines.cbegin()->first;
  int beforePrevious = -1;
  vector<int> droplist;
  for (auto& bl : barLines) {
    if (bl.first == previousBarX) continue;
    const int distance = bl.first - previousBarX;
    cout << "bar line distance: " << distance << endl; 
    if (distance < 40) {
      // Should one of these be dropped?
      cout << "triplet: " << beforePrevious << ", " << previousBarX
           << ", " << bl.first << endl;
      if (beforePrevious == -1) {
        // previousBarX is probably the beginning of the line, so drop
        // bl.first from bar lines list.
        droplist.push_back(bl.first);
      } else {
        // drop the middle item.
        droplist.push_back(previousBarX);
      }
      continue;  // this avoids updating beforePrevious and previousBarX.
    }
    beforePrevious = previousBarX;
    previousBarX = bl.first;
  }
  cout << "dropping " << droplist.size() << " bar lines." << endl;
  for (int i : droplist) {
    cout << "dropping entry at " << i << endl; 
    Shape* s = barLines.find(i)->second;
    s->updateBelief(TrainingKey::Category::bar, -10);
    s->print();
    barLines.erase(i);
  }
}

void ShapeFinder::initLineScan(const SheetLine& sheetLine,
                               const cv::Ptr<cv::ml::StatModel>& statModel) {
  Mat viewPort = sheetLine.getViewPort().clone();
  const vector<Rect>& rectangles = getContourBoxes(viewPort); 
  firstPass(rectangles, viewPort, statModel);
  const std::pair<int, int> tb = sheetLine.getCoordinates();
  const Rect relative = sheetLine.getInnerBox() - sheetLine.getBoundingBox().tl();
  scanForBarLines(viewPort, relative, tb);
  cout << "voice position: " << voicePosition << endl;
}

const std::vector<int> ShapeFinder::getBarPositions() const {
  std::vector<int> ret;
  for (const auto& bl : barLines) {
    ret.emplace_back(bl.first);
  }
  return ret;
}

const Shape* ShapeFinder::getBarAt(int x) const {
  const auto& b = barLines.find(x);
  if (b == barLines.end()) {
    cerr << "no bar line exists at position " << x << endl;
    return NULL;
  }
  return b->second; 
}

void ShapeFinder::scanLine(const SheetLine& sheetLine,
                           const cv::Ptr<cv::ml::StatModel>& statModel,
                           const Scanner& ocr,
                           const string& processedWindowName,
                           const string& questionWindowName) {

  initLineScan(sheetLine, statModel);
  cout << "voice position: " << voicePosition << endl;
  Mat viewPort = sheetLine.getViewPort().clone();

  Mat cont;
  cvtColor(viewPort, cont, COLOR_GRAY2BGR);

  const Rect relative = sheetLine.getInnerBox() - sheetLine.getBoundingBox().tl();
  rectangle(cont, relative, Scalar(127, 0, 0), 1);

  const std::pair<int, int> tb = sheetLine.getCoordinates();
  line(cont, Point(0, tb.first), Point(cont.cols, tb.first),
       Scalar(0, 127, 0), 1);
  line(cont, Point(0, tb.second), Point(cont.cols, tb.second),
       Scalar(0, 127, 0), 1);
  TrainingKey key;

  Scalar colour(0, 0, 127);
  colour = Scalar(127, 0, 0);
  for (const auto& bl: barLines) {
    const Shape *s = bl.second;
    const Rect& r = s->getRectangle();
    rectangle(cont, r, colour, 2);
  }

  // TODO: scan for notes and note heads, see if they can be
  // put together with complexes that could be accidentals,
  // dots, and connector pieces.

  const int slHeight = tb.second - tb.first;
  for (auto& shapeAtX : shapes) {
    const int xcoord = shapeAtX.first;
    vector<std::unique_ptr<Shape>>& list = shapeAtX.second;
    if (list.size() > 1) {
      cout << "There are " << list.size() << " shapes at " << xcoord << endl;
    }

    for (size_t i = 0; i < list.size(); i++) {
      colour = Scalar(0, 0, 127);
      const Rect& r = list[i]->getRectangle();
      Rect insideInnerRect = r & relative;
      size_t nbcount = list[i]->getNumberOfNeighbours();
      list[i]->print();

      const auto cat = list[i]->getTopLevelCategory();

      switch (cat) {
        case TrainingKey::TopLevelCategory::round:
          {
            if (r.area() > 50) {
              // note heads are generally at least this size.
              // they can be outside the core area.
              cout << "size says note head" << endl;
              list[i]->updateBelief(TrainingKey::Category::notehead, 1);
            }
            if (insideInnerRect.area() == 0) {
                if (nbcount == 0 && r.area() <= 50) {
                // outside the core area and small and isolated -> probably a speck
                list[i]->updateBelief(TrainingKey::Category::speck, 2);  
              } else {
                // Does have neighbours, don't skip it yet.
                list[i]->updateBelief(TrainingKey::Category::piece, 1);  
              }
            }   
            // increase piece likelihood if this is inside another item.
            const auto& neighbours = list[i]->getNeighbours();
            if (neighbours.find(Shape::Neighbourhood::AROUND)
                  != neighbours.end()) {
              list[i]->updateBelief(TrainingKey::Category::piece, 1);  
              cout << "inside another item, probably a piece" << endl;
            }
            auto x = list[i]->getMostLikelyCategory();
            if (x == TrainingKey::Category::speck) colour = Scalar(127, 127, 127);
          }
          break;
        case TrainingKey::TopLevelCategory::vline:
        {
          if (insideInnerRect.area() == 0) {
              if (nbcount == 0 && r.area() <= 50) {
              // outside the core area and small and isolated -> probably a speck
              list[i]->updateBelief(TrainingKey::Category::speck, 2);  
            } else {
              // Does have neighbours, don't skip it yet.
              list[i]->updateBelief(TrainingKey::Category::piece, 1);  
            }
          }   
        }
        break;

        default: break;
      }

      // Decide whether to display this or skip it.
      if (list[i]->getMostLikelyCategory() == TrainingKey::Category::bar) {
        if (list[i]->getTopLevelCategory() == TrainingKey::TopLevelCategory::vline) {
          // straightforward bar line, skip it.
          continue;
        }
      }
      if (list[i]->getMostLikelyCategory() == TrainingKey::Category::speck) {
        continue;
      }

      rectangle(cont, r, colour, 2);
      imshow(processedWindowName, cont);
      Mat partial = Mat(viewPort, r);

      Mat scaleup;
      resize(partial, scaleup, Size(), 2.0, 2.0, INTER_CUBIC);
      imshow(questionWindowName, scaleup);

      // TODO: send prep into stat model?
      waitKey(0);
      Mat prep = preprocess(partial);
      resize(prep, scaleup, Size(), 2.0, 2.0, INTER_CUBIC);
      imshow(questionWindowName, scaleup);
      ocr.process(scaleup);
      waitKey(0);
    }
  }
}

Mat ShapeFinder::preprocess(const Mat& img) {
  // Run horizontal dilate/erode and subtract to get rid of the 
  // horizontal lines. Then use threshold.
  // This might be useful for recognition/training in general?
  Mat horizontalStructure = getStructuringElement(MORPH_RECT,
    Size(10, 1));
  Mat tmp;
  dilate(img, tmp, horizontalStructure, Point(-1, -1));
  erode(tmp, tmp, horizontalStructure, Point(-1, -1));
  tmp = img + ~tmp;
  threshold(tmp, tmp, config.thresholdValue, 255, config.thresholdType);
  tmp = ~tmp;
  return tmp;
}


void ShapeFinder::getTrainingDataForLine(const Mat& focused, 
  const string& processedWindowName,
  const string& questionWindowName,
  const string& filename,
  ofstream& responsesFile) {

  // This is just for showing the contours.
  Mat cont;
  cvtColor(focused, cont, COLOR_GRAY2BGR);

  vector<Rect> rectangles = getContourBoxes(focused); 

  Mat partial, scaleup;
  char fname[200];
  for (int i = 0; i < rectangles.size(); i++) {
    rectangle(cont, rectangles[i], Scalar(0, 0, 127), 2);
    imshow(processedWindowName, cont);

    partial = Mat(focused, rectangles[i]);

    cout << "showing contour with area " << rectangles[i].area()
         << " at coordinates " << rectangles[i].tl()
         << " to " << rectangles[i].br() << endl;

    // TODO: sharpen contours in the image some more.
    // horizontal dilate/erode, subtract, threshold, invert seems to be ok-ish?

    // Scaling up more just makes the image too blurry.
    resize(partial, scaleup, Size(), 2.0, 2.0, INTER_CUBIC);

    imshow(questionWindowName, scaleup);

    int cat = waitKeyEx(0);
    cout << "category: " << cat << endl;
    if (cat == 'q') return;

    // This is awkward, because I'm encoding the coordinates in
    // the filename, but seems like the least bad option.
    sprintf(fname, "%s.%d.%d.%d.png", filename.c_str(), i,
            rectangles[i].tl().x, rectangles[i].tl().y);

    // Partial is an 8-bit grayscale image.
    if (imwrite(fname, partial)) {
      responsesFile << i << ": " << cat << endl;
    } else {
      cerr << "Failed to save image to file " << fname << endl;
    }
  }
  cout << "Done with this line." << endl;
}

Shape::Shape(const cv::Rect& rect) {
  rectangle = rect;
}

void Shape::updateBelief(TrainingKey::Category key, int diff) {
  const auto& b = beliefs.find(key);
  if (b == beliefs.end()) {
    beliefs.emplace(key, diff);
  } else {
    beliefs[key] += diff;
  }
}

TrainingKey::Category Shape::getMostLikelyCategory() const {
  TrainingKey::Category maxCat = TrainingKey::Category::undefined;
  int maxBelief = 0;
  for (const auto& b : beliefs) {
    if (b.second > maxBelief) {
      maxCat = b.first;
      maxBelief = b.second;
    }
  } 
  return maxCat;
}

int Shape::categoryConfidence(TrainingKey::Category category) const {
  const auto& b = beliefs.find(category);
  if (b == beliefs.end()) {
    return 0;
  }
  return b->second;
}

size_t Shape::getNumberOfNeighbours() const {
  size_t ret = 0;
  for (const auto& nb : neighboursByDirection) {
    ret += nb.second.size();
  }
  return ret;
}

void Shape::addNeighbour(Neighbourhood where, Shape *shape) {
  auto n = neighboursByDirection.find(where);
  if (n == neighboursByDirection.end()) {
    std::vector<Shape*> neighbours({ shape });
    neighboursByDirection.emplace(where, neighbours);
  } else {
    std::vector<Shape*>& neighbours = n->second;
    neighbours.push_back(shape);
  }
}

void Shape::print() const {
  cout << "rectangle: " << rectangle << endl;
  TrainingKey key;
  cout << "tl category: " << key.getCategoryName(getTopLevelCategory()) << endl;
  for (const auto& b : beliefs) {
    cout << "is it a " << key.getCategoryName(b.first) << "? " << b.second << endl;
  }
}

void Shape::maybeAddNeighbour(Shape *shape) {
  // First check containment. Line scanning always finds the enclosing
  // shape before the contained shapes, so the check only goes one way.
  const Rect& shapeRect = shape->getRectangle();

  // Containment
  if (shapeRect.tl().x >= rectangle.tl().x &&
      shapeRect.tl().y >= rectangle.tl().y &&
      shapeRect.br().x <= rectangle.br().x &&
      shapeRect.br().y <= rectangle.br().y) {
    // 'this' is around shape
    shape->addNeighbour(Neighbourhood::AROUND, this);
    // shape is inside 'this'
    this->addNeighbour(Neighbourhood::IN, shape);
    return; 
  } 

  {
  Rect tmp = shapeRect & rectangle;
  if (tmp.area() > 0) {
    shape->addNeighbour(Neighbourhood::INTERSECT, this);
    this->addNeighbour(Neighbourhood::INTERSECT, shape);
    return;
  }
  }

  // Because of the way line scanning works, shape will not be to the left
  // of this.
  // Nearby horizontally?
  if (std::abs(shapeRect.tl().x - rectangle.br().x) <= smallDistance) {
    // NE means the top of shapeRect is above the top of this and the
    // bottom of shapeRect is either at most smallDistance above the
    // top of this or below the top of this and above the bottom of this.
    if (shapeRect.tl().y <= rectangle.tl().y &&
         (std::abs(shapeRect.br().y - rectangle.tl().y) <= smallDistance ||
          shapeRect.br().y > rectangle.tl().y &&
          shapeRect.br().y <= rectangle.br().y)) {
      this->addNeighbour(Neighbourhood::NE, shape);
      shape->addNeighbour(Neighbourhood::SW, this);
      return; 
    }
    // SE means the bottom of shapeRect is below the bottom of this and
    // the top of shapeRect is either at most smallDistance below the
    // bottom of this or between the top and bottom of this.
    if (shapeRect.br().y >= rectangle.br().y &&
       (std::abs(shapeRect.tl().y - rectangle.br().y) <= smallDistance ||
        shapeRect.tl().y > rectangle.tl().y &&
        shapeRect.tl().y <= rectangle.br().y)) {
      this->addNeighbour(Neighbourhood::SE, shape);
      shape->addNeighbour(Neighbourhood::NW, this);
      return; 
    }
    // E means either both the top and bottom of shapeRect are between
    // the top and bottom of this, or the top is above and the bottom is below.
    if (shapeRect.tl().y >= rectangle.tl().y &&
        shapeRect.br().y <= rectangle.br().y) {
        this->addNeighbour(Neighbourhood::E, shape);
        shape->addNeighbour(Neighbourhood::W, this);
        return;
    }
    if (shapeRect.tl().y <= rectangle.tl().y &&
        shapeRect.br().y >= rectangle.br().y) {
        this->addNeighbour(Neighbourhood::E, shape);
        shape->addNeighbour(Neighbourhood::W, this);
        return;
    }
  }
  if (std::abs(rectangle.tl().y - shapeRect.br().y) <= smallDistance) {
    // This is N if shaperect overlaps horizontally with this.
      if (shapeRect.tl().x <= rectangle.br().x) {
        this->addNeighbour(Neighbourhood::N, shape);
        shape->addNeighbour(Neighbourhood::S, this);
        return;
      }
    return;
  }
  if (std::abs(shapeRect.tl().y - rectangle.br().y) <= smallDistance) {
    // This is S if shaperect is within the horizontal size of this.
      if (shapeRect.tl().x <= rectangle.br().x) {
        this->addNeighbour(Neighbourhood::S, shape);
        shape->addNeighbour(Neighbourhood::N, this);
        return;
      }
  }
  // Not a neighbour. It might be possible to return bool from this method
  // to stop scanning for performance reasons. 
}

}  // namespace
