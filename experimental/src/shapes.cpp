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
                            const cv::Ptr<cv::ml::StatModel>& statModel,
                            const cv::Ptr<cv::ml::StatModel>& fineStatModel) {
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
    if (fineStatModel && fineStatModel->isTrained()) {
      float prediction2 = fineStatModel->predict(sample);
      const TrainingKey::Category cat2 =
          static_cast<TrainingKey::Category>((int)prediction2);
      shape->setCategory(cat2);
    } 
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

  // First, look for voice connectors (long bar lines).
  map<int, int> positions;
  bool haveFirstVline = false;
  for (const auto& siter : shapes) {
    const int xcoord = siter.first;
    const vector<std::unique_ptr<Shape>>& list = siter.second;
    for (size_t i = 0; i < list.size(); i++) {
      const Rect& r = list[i]->getRectangle();
      const auto cat = list[i]->getTopLevelCategory();
      const auto fcat = list[i]->getCategory();
      const int height = r.br().y - r.tl().y;
      bool processThis = false;
      if (fcat == TrainingKey::Category::vertical) {
        processThis = true;
        haveFirstVline = true;
      } else if (cat == TrainingKey::TopLevelCategory::composite) {
        const int width = r.br().x - r.tl().x;
        if (height / width >= 3) {
          // probably a vertical line. other vertical lines can be included
          // in larger composites, but won't be identified here.
          processThis = true;
        } else if (!haveFirstVline && xcoord < 30) {
          processThis = true;
        }
        haveFirstVline = true;
      } else if (cat == TrainingKey::TopLevelCategory::vline) {
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

  std::map<int, Shape*> barLines;

  if (voicePosition > 0) {
    for (auto& siter : shapes) {
      const int xcoord = siter.first;
      const auto& pos = positions.find(xcoord);
      if (pos == positions.end()) continue;
      vector<std::unique_ptr<Shape>>& list = siter.second;
      for (size_t i = 0; i < list.size(); i++) {
        const auto cat = list[i]->getTopLevelCategory();
        if (cat == TrainingKey::TopLevelCategory::vline ||
            cat == TrainingKey::TopLevelCategory::composite ||
            list[i]->getCategory() == TrainingKey::Category::vertical) {
          cout << "adding bar line at " << xcoord << endl;
          barLines.emplace(xcoord, list[i].get());
        }
      }
    }
  } else {

  // Is this a single voice line? Then we don't have any bar lines yet.
  haveFirstVline = false;

  // Look for the rightmost x coordinate of a vertical line or composite.
  int lastXCoord = -1;
  for (auto cr = shapes.crbegin(); cr != shapes.crend(); ++cr) {
    const vector<std::unique_ptr<Shape>>& list = cr->second;
    for (size_t i = 0; i < list.size(); i++) {
      const auto cat = list[i]->getTopLevelCategory();
      if (cat == TrainingKey::TopLevelCategory::vline || 
          cat == TrainingKey::TopLevelCategory::composite ||
          list[i]->getCategory() == TrainingKey::Category::vertical) {
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
          cat != TrainingKey::TopLevelCategory::composite && 
          list[i]->getCategory() != TrainingKey::Category::vertical) continue;
      const Rect& r = list[i]->getRectangle();
      const Rect insideInnerRect = r & relativeInnerBox;
   
      if (insideInnerRect.area() == 0) continue;

      // There might be something else there besides the bar line
      // but there's certainly the end of a bar.
      if (xcoord == lastXCoord) {
//        cout << "end of line barline at " << xcoord << endl;
        barLines.emplace(xcoord, list[i].get());
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
                cat2 == TrainingKey::TopLevelCategory::composite ||
                sh->getCategory() == TrainingKey::Category::vertical) {
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
    s->print();
    barLines.erase(i);
  }
  }  // end of the 'else' case. below code gets executed for both
     // long and short bar lines.
  for (const auto& bl : barLines) {
    unique_ptr<CompositeShape>& composite = addCompositeShape(
        CompositeShape::CompositeType::BARLINE, bl.second);
  }
}

void ShapeFinder::initLineScan(const SheetLine& sheetLine,
                               const cv::Ptr<cv::ml::StatModel>& statModel,
                               const cv::Ptr<cv::ml::StatModel>& fineStatModel) {
  Mat viewPort = sheetLine.getViewPort().clone();
  const vector<Rect>& rectangles = getContourBoxes(viewPort); 
  firstPass(rectangles, viewPort, statModel, fineStatModel);
  const std::pair<int, int> tb = sheetLine.getCoordinates();
  const Rect relative = sheetLine.getInnerBox() - sheetLine.getBoundingBox().tl();
  scanForBarLines(viewPort, relative, tb);
}

void ShapeFinder::scanForNotes(const Rect& relativeInnerBox) {

  // the fine model (dtrees) is pretty good at detecting note heads.
  // when it says 'quarter break', there's a good chance it's actually
  // a note neck.
  // when it says 'note' it could be a note neck or a note. 
  // when it says 'sharp', it's probably correct
  // when it says 'bass clef', it's probably wrong
  // when it says 'flat', it could be a sharp or a note head instead.

  for (auto& siter : shapes) {
    if (siter.first < (relativeInnerBox.tl().x - 2)) continue;
    vector<std::unique_ptr<Shape>>& list = siter.second;
    for (size_t i = 0; i < list.size(); i++) {
      if (isShapeInComposite(*list[i])) continue;
      // Is this item inside another one?
      // xxx 
      const auto cat = list[i]->getCategory();
      if (cat == TrainingKey::Category::notehead ||
          cat == TrainingKey::Category::note) {
        // This will add potential neighbours, but there can be
        // relevant items that aren't caught by neighbourhood relations.
        // TODO look a little further.
        // look for dots to the right, note necks above/below
        // accidentals to the left
        // possibly connector lines, expressive marks above/below
        std::unique_ptr<CompositeShape>& comp = addCompositeShape(
          CompositeShape::CompositeType::NOTE, list[i].get()); 
      }
    }
  }
}

void ShapeFinder::scanForDiscards(const Rect& relativeInnerBox) {
  for (auto& siter : shapes) {
    vector<std::unique_ptr<Shape>>& list = siter.second;
    for (size_t i = 0; i < list.size(); i++) {
      if (isShapeInComposite(*list[i])) continue;
      // Is it outside the inner box?
      if ((list[i]->getRectangle() & relativeInnerBox).area() != 0) continue;
      // Are any of its neighbours inside the inner box?
      bool foundInsideNeighbour = false;
      for (const auto& nb : list[i]->getNeighbours()) {
        const vector<Shape*>& sp = nb.second;
        for (size_t j = 0; j < sp.size(); j++) {
          if ((sp[j]->getRectangle() & relativeInnerBox).area() != 0) {
            foundInsideNeighbour = true;
            break;
          }
        }
        if (foundInsideNeighbour) break;
      }
      if (foundInsideNeighbour) continue;
      std::unique_ptr<CompositeShape>& comp = addCompositeShape(
        CompositeShape::CompositeType::OUTOFLINE, list[i].get()); 
    }
  }
}

void ShapeFinder::scanStartOfLine(const Rect& relativeInnerBox) {
  // A line can start in several different ways:
  // - optional bar line
  // - optional clef, optionally followed by time and accidentals
  // if there is no bar line at the start, then either there is
  // an implied bar line at the beginning, or there will be one
  // (possibly a double line with repeat signs, etc) just before the
  // actual music notes begin.

  // This method should be called after bar lines have been identified.

  // 'state machine': 
  // -1: starting state
  // 1: have found bar line
  // 2: have found potential clef
  // 3: have found clef followed by complex that could be time
  // 4: have found things followed by complex that could be accidentals
  // 5: have found things followed by complex that could be wide bar line
  int state = -1;
  for (const auto& siter : shapes) {
    // skip items before left edge of bounding box
    if (siter.first < (relativeInnerBox.tl().x - 2)) continue;
    // Arbitrary: 'start' of line does not go farther than this.
    // TODO: this needs to be fixed for in-between changes in clef etc.
    if (siter.first >= 100) return;
    // Finding a note head signals we're past the start of line.
    const vector<std::unique_ptr<Shape>>& list = siter.second;
    for (size_t i = 0; i < list.size(); i++) {
      const auto cat = list[i]->getTopLevelCategory();
      if (cat == TrainingKey::TopLevelCategory::round &&
          list[i]->getRectangle().area() >= 50) {
        return;
      }
      // TODO: need to train ocr or cascade to recognise clefs
      // For now, look for things that are 'complex' and the right size.
      if (cat == TrainingKey::TopLevelCategory::composite) {
        cout << "composite found at " << siter.first
             << "with size " << list[i]->getRectangle() << endl;
      }
    }
  }
}

bool ShapeFinder::isShapeInComposite(const Shape& s) const {
  const Rect& r = s.getRectangle();
  for (const auto& comp : compositeShapes) {
    const Rect& r2 = comp->getRectangle();
    if ((r & r2) == r) return true;
  }
  return false;
}

void ShapeFinder::scanLine(const SheetLine& sheetLine,
                           const cv::Ptr<cv::ml::StatModel>& statModel,
                           const cv::Ptr<cv::ml::StatModel>& fineStatModel,
                           const Scanner& ocr,
                           const string& processedWindowName,
                           const string& questionWindowName) {

  if (voicePosition == -1) {
    initLineScan(sheetLine, statModel, fineStatModel);
  }
  cout << "voice position: " << voicePosition << endl;
  Mat viewPort = sheetLine.getViewPort().clone();

  Mat cont;
  cvtColor(viewPort, cont, COLOR_GRAY2BGR);

  const Rect relative = sheetLine.getInnerBox() - sheetLine.getBoundingBox().tl();
  rectangle(cont, relative, Scalar(127, 0, 0), 1);
  scanStartOfLine(relative);
  scanForDiscards(relative);
  scanForNotes(relative);

  const std::pair<int, int> tb = sheetLine.getCoordinates();
  line(cont, Point(0, tb.first), Point(cont.cols, tb.first),
       Scalar(0, 127, 0), 1);
  line(cont, Point(0, tb.second), Point(cont.cols, tb.second),
       Scalar(0, 127, 0), 1);
  TrainingKey key;

  Scalar colour(0, 0, 127);
  for (const auto& cm : compositeShapes) {
    const Rect& r = cm->getRectangle();
    switch(cm->getType()) {
      case CompositeShape::CompositeType::BARLINE:
        colour = Scalar(127, 0, 0); 
        break;
      case CompositeShape::CompositeType::NOTE:
        colour = Scalar(0, 127, 0); 
        break;
      case CompositeShape::CompositeType::OUTOFLINE:
        colour = Scalar(0, 0, 127); 
        break;
      default: 
        break;
    }
    rectangle(cont, r, colour, 2);
  }

  SampleData sd;
  const int slHeight = tb.second - tb.first;
  for (auto& shapeAtX : shapes) {
    vector<std::unique_ptr<Shape>>& list = shapeAtX.second;
    for (size_t i = 0; i < list.size(); i++) {
      list[i]->print();

      // Decide whether to display this or skip it.
      //if (isShapeInComposite(*list[i])) continue;
      //if (list[i]->getMostLikelyCategory() == TrainingKey::Category::speck) {
      //  continue;
      // }

      rectangle(cont, list[i]->getRectangle(), Scalar(0, 0, 0), 2);
      imshow(processedWindowName, cont);
      Mat partial = Mat(viewPort, list[i]->getRectangle());

      Mat scaleup;

      Mat prep = preprocess(partial);
      resize(prep, scaleup, Size(), 2.0, 2.0, INTER_CUBIC);
      imshow(questionWindowName, scaleup);
      ocr.process(scaleup);
      int input = waitKeyEx(0);
      if (input == 'q') {
        return;
      }
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

std::unique_ptr<CompositeShape>&
   ShapeFinder::addCompositeShape(
     CompositeShape::CompositeType type, Shape* shape) {
  CompositeShape* composite = new CompositeShape(type, shape);
  // Add some neighbours of shape to composite.
  const map<Shape::Neighbourhood, vector<Shape*>>& nb = shape->getNeighbours();
  for (const auto& nbiter : nb) {
    if (type == CompositeShape::CompositeType::BARLINE) {
       // Skip 'E' and 'W' neighbours for bar line types.
       if (nbiter.first == Shape::Neighbourhood::E ||
           nbiter.first == Shape::Neighbourhood::W) continue;
       for (const auto& s : nbiter.second) {
         const auto cat = s->getTopLevelCategory();
         if (cat != TrainingKey::TopLevelCategory::round &&
             cat != TrainingKey::TopLevelCategory::vline) continue;
         composite->addShape(s);
       }
    } else if (type == CompositeShape::CompositeType::NOTE) {
      // add 'complex' if it's to the left (accidental)
      // add 'dot' if it's to the right
      // add connector lines and verticals if they're above/below
      for (const auto& s : nbiter.second) {
        const auto cat = s->getTopLevelCategory();
        switch(cat) {
          case TrainingKey::TopLevelCategory::round:
            cout << "neighbour of size " << s->getRectangle().area()
                 << " in direction " << nbiter.first << endl;
            if (s->getRectangle().area() < 12 &&
                (nbiter.first == Shape::Neighbourhood::E ||
                 nbiter.first == Shape::Neighbourhood::SE)) {
              composite->addShape(s);
            }
          break;
          case TrainingKey::TopLevelCategory::composite:
            cout << "composite neighbour of size " << s->getRectangle().area()
                 << " in direction " << nbiter.first << endl;
            if (nbiter.first == Shape::Neighbourhood::W ||
                nbiter.first == Shape::Neighbourhood::SW ||
                nbiter.first == Shape::Neighbourhood::NW) {
              composite->addShape(s);
            }
            break;
          case TrainingKey::TopLevelCategory::hline:
            cout << "connector neighbour of size " << s->getRectangle().area()
                 << " in direction " << nbiter.first << endl;
            if (nbiter.first == Shape::Neighbourhood::N ||
                nbiter.first == Shape::Neighbourhood::S ||
                nbiter.first == Shape::Neighbourhood::SE ||
                nbiter.first == Shape::Neighbourhood::SW ||
                nbiter.first == Shape::Neighbourhood::NW ||
                nbiter.first == Shape::Neighbourhood::NE) {
              composite->addShape(s);
            }
            break;
           default: break;
        }
      }
    } else {  // default: add all neighbours.
       for (const auto& s : nbiter.second) {
         composite->addShape(s);
       }
    }
  }
  compositeShapes.emplace_back(composite);
  return compositeShapes.back();
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

CompositeShape::CompositeShape(CompositeShape::CompositeType type,
                               Shape* shape) : type(type) {
  shapes.push_back(shape); 
  boundingBox = shape->getRectangle();
}

void CompositeShape::addShape(Shape* shape) {
  shapes.push_back(shape);
  boundingBox |= shape->getRectangle();
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
  cout << "category: " << key.getCategoryName(getCategory()) << endl;
  for (const auto& nb : neighboursByDirection) {
    const std::vector<Shape*>& n = nb.second;
    cout << n.size() << " neighbours in direction " << nb.first
         << ": ";
    for (int i = 0; i < n.size(); i++) {
      cout << n[i]->getCategory() << ", ";
    }
    cout << endl;
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
