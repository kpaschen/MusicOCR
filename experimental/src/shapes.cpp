#include "shapes.hpp"
#include "utils.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

namespace musicocr {

void ShapeFinder::getTrainingDataForLine(const Mat& focused, 
  const string& processedWindowName,
  const string& questionWindowName,
  const string& filename,
  ofstream& responsesFile) {
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
#if 0
  Canny(tmp, processed, config.cannyMin, config.cannyMax,
        config.sobelKernel, config.l2Gradient);
#endif

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(processed, contours, hierarchy, RETR_TREE,
               CHAIN_APPROX_SIMPLE, Point(0, 0));

  // This is just for showing the contours.
  Mat cont;
  cvtColor(processed, cont, COLOR_GRAY2BGR);
  vector<Rect> rectangles(contours.size());
  vector<vector<Point>> hull(contours.size());
  for (int i = 0; i < contours.size(); i++) {
    auto colour = Scalar(0, 0, 255);
    if (hierarchy[i][3] == -1) {
      colour = Scalar(255, 0, 0);
    } else if (hierarchy[i][2] == -1) {
      colour = Scalar(0, 255, 0);
    }
    drawContours(cont, contours, i, colour, 1, 8, hierarchy, 0, Point(0, 0));
    convexHull(Mat(contours[i]), hull[i], false);
    rectangles[i] = boundingRect(Mat(hull[i]));
  }
  std::sort(rectangles.begin(), rectangles.end(), musicocr::rectLeft);
  Mat partial, scaleup;
  char fname[200];
  for (int i = 0; i < rectangles.size(); i++) {
    rectangle(cont, rectangles[i], Scalar(0, 0, 127), 2);
    imshow(processedWindowName, cont);

    partial = Mat(focused, rectangles[i]);

    cout << "showing contour with area " << rectangles[i].area()
         << " at coordinates " << rectangles[i].tl()
         << " to " << rectangles[i].br() << endl;

    // Scaling up more just makes the image too blurry.
    resize(partial, scaleup, Size(), 2.0, 2.0, INTER_CUBIC);

    imshow(questionWindowName, scaleup);

    int cat = waitKeyEx(0);
    cout << "category: " << cat << endl;
    if (cat == 'q') return;

    // May want to save the coordinates too.
    sprintf(fname, "%s.%d.png", filename.c_str(), i);

    // Partial is an 8-bit grayscale image.
    if (imwrite(fname, partial)) {
      responsesFile << i << ": " << cat << endl;
    } else {
      cerr << "Failed to save image to file " << fname << endl;
    }
  }
}

}  // namespace
