#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include "corners.hpp"
#include "structured_page.hpp"
#include "shapes.hpp"
#include "training.hpp"

using namespace cv;
using namespace std;

// gray is the original image in grayscale.
// focused is the matrix being worked on (all of gray or
// a portion of it). processed is the result of the current
// work. cdst is a colour version of processed, can have
// extra markings on it in colour.
Mat gray, focused, processed, cdst;

int gaussianKernel = 3;
int thresholdValue = 0;
int thresholdType = 4;
int useOtsu = 1;  // only true or false
int cannyMin = 80;
int cannyMax = 90;
int sobelKernel = 3;
int l2gradient = 0;  // only true or false
int houghResolution = 1; // resolution in pixels.
int houghResolutionRad = 1; // resolution in degree radians.
int houghThreshold = 100;
int houghMinLinLength = 50;
int houghMaxLineGap = 15;

// threshold algorithm.
// int edthreshold = ADAPTIVE_THRESH_MEAN_C;
int athreshold = ADAPTIVE_THRESH_GAUSSIAN_C;
// threshold type is always THRESH_BINARY
int aBlockSize = 15; // must be odd
int aSubtractMe = 2;
int aAddMe = 0;

// the width of the horizontal structuring element will be
// inputmatrix.cols / edHorizontalSizeFudge
int edHorizontalSizeFudge = 30;
int edHorizontalHeight = 1;
int edVerticalSizeFudge = 30;  // 7 is good for finding mainly the bars
int edVerticalWidth = 1;

void correctConfig() {
  if (gaussianKernel % 2 == 0) {
    gaussianKernel += 1;
  }
  if (sobelKernel < 3) {
    sobelKernel = 3;
  }
  if (sobelKernel % 2 == 0) {
    sobelKernel += 1;
  } 
  if (sobelKernel > 7) {
    sobelKernel = 7;
  }
  if (houghResolution < 1.0) { houghResolution = 1.0; }
  if (aBlockSize % 2 == 0) {
    aBlockSize++;
  }
  if (aBlockSize <= 1) {
    aBlockSize = 3;
  }
}

void makeConfig(musicocr::CornerConfig *config) {
  correctConfig();
  config->gaussianKernel = gaussianKernel;
  config->thresholdValue = (double)thresholdValue/255.0;
  config->thresholdType = thresholdType + (useOtsu ? THRESH_OTSU : 0);
  config->cannyMin = cannyMin;
  config->cannyMax = cannyMax;
  config->sobelKernel = sobelKernel;
  config->l2gradient = (l2gradient != 0);
  config->houghResolution = houghResolution;
  config->houghResolutionRad = (double)houghResolutionRad * CV_PI / 180.0;
  if (config->houghResolutionRad < 1.0) { config->houghResolutionRad = 1.0; }
  config->houghThreshold = houghThreshold;
  config->houghMinLinLength = houghMinLinLength;
  config->houghMaxLineGap = houghMaxLineGap;
}

void makeSheetConfig(musicocr::SheetConfig *config) {
  correctConfig();
  config->gaussianKernel = gaussianKernel;
  config->thresholdValue = (double)thresholdValue/255.0;
  config->thresholdType = thresholdType + (useOtsu ? THRESH_OTSU : 0);
  config->cannyMin = cannyMin;
  config->cannyMax = cannyMax;
  config->sobel = sobelKernel;
  config->l2Gradient = (l2gradient != 0);
//  config->houghResolution = houghResolution;
//  config->houghResolutionRad = (double)houghResolutionRad * CV_PI / 180.0;
//  if (config->houghResolutionRad < 1.0) { config->houghResolutionRad = 1.0; }
  config->houghThreshold = houghThreshold;
  config->houghMinLineLength = houghMinLinLength;
  config->houghMaxLineGap = houghMaxLineGap;
}

void makeContourConfig(musicocr::ContourConfig *config) {
  correctConfig();
  config->gaussianKernel = gaussianKernel;
  config->thresholdValue = (double)thresholdValue/255.0;
  config->thresholdType = thresholdType + (useOtsu ? THRESH_OTSU : 0);
  config->cannyMin = cannyMin;
  config->cannyMax = cannyMax;
  config->sobelKernel = sobelKernel;
  config->l2Gradient = (l2gradient != 0);

  config->horizontalSizeFudge = edHorizontalSizeFudge;
  config->horizontalHeight = 1;
}

void onTrackbar(int, void *) { }

void setupTrackbars(const string& windowName) {
  // Slider for the Gaussian Blur
  createTrackbar("Gaussian Kernel Size", windowName, &gaussianKernel,
                 15, onTrackbar);

  // Slider values for thresholding.
  createTrackbar("Threshold value", windowName, &thresholdValue, 255,
                 onTrackbar);
  createTrackbar("Threshold type", windowName, &thresholdType,
                 THRESH_TOZERO_INV, onTrackbar);
  createTrackbar("Use Otsu", windowName, &useOtsu, 1, onTrackbar);

  // Sliders for canny.
  createTrackbar("Canny Min", windowName, &cannyMin, 255, onTrackbar);
  createTrackbar("Canny Max", windowName, &cannyMax, 255, onTrackbar);
  createTrackbar("Sobel", windowName, &sobelKernel, 7, onTrackbar);
  createTrackbar("l2gradient", windowName, &l2gradient, 1, onTrackbar);

  // Sliders for hough.
//  createTrackbar("Hough Resolution Pixels", windowName, &houghResolution, 10,
//                 onTrackbar);
//  createTrackbar("Hough Resolution Radians", windowName, &houghResolutionRad,
//                 180, onTrackbar);
  createTrackbar("Hough Threshold", windowName, &houghThreshold,
                 255, onTrackbar);
  createTrackbar("Hough Min Line Length (P)", windowName, &houghMinLinLength,
                 100, onTrackbar);
  createTrackbar("Hough Max Line Gap (P)", windowName, &houghMaxLineGap,
                 100, onTrackbar);

  // adaptive threshold
  createTrackbar("ad. threshold", windowName,
                 &athreshold, 1, onTrackbar);
  createTrackbar("a block size", windowName,
                 &aBlockSize, 15, onTrackbar);
//  createTrackbar("e-d subtract const", windowName,
//                 &aSubtractMe, 10, onTrackbar);
//  createTrackbar("a add const", windowName,
//                 &aAddMe, 10, onTrackbar);

  // erode-dilate
  createTrackbar("e-d horizontal size fudge", windowName,
                 &edHorizontalSizeFudge, 30, onTrackbar);
//  createTrackbar("e-d horizontal height", windowName,
//                 &edHorizontalHeight, 10, onTrackbar);
  createTrackbar("e-d vertical size fudge",
                 windowName, &edVerticalSizeFudge, 50, onTrackbar);
//  createTrackbar("e-d vertical width", windowName,
//                &edVerticalWidth, 10, onTrackbar);
}

void findCorners();
void findLines();
void processImage();

int main(int argc, char** argv) {
  if (argc < 2) {
   cerr << "DisplayImage <Path to Image>";
   return -1; 
  }
  string filename = argv[1];
  Mat image;
  image = imread(filename, 1);
  if (!image.data) {
    cerr << "No image data.";
    return -1;
  }
  {
    filename = filename.substr(filename.find_last_of("/") + 1);
    size_t x = filename.find_last_of('.');
    if (x > 0) filename = filename.substr(0, x);
  }
  cout << "base name of file: " << filename << endl;

  resize(image, image, Size(), 0.2, 0.2, INTER_AREA);
  namedWindow("Original Image", WINDOW_AUTOSIZE);
  namedWindow("Processed", WINDOW_AUTOSIZE);

  // everything starts with 'gray'.
  cvtColor(image, gray, COLOR_BGR2GRAY);
  imshow("Original Image", gray);

  namedWindow("Controls", WINDOW_AUTOSIZE);
  setupTrackbars("Controls");

  focused = gray.clone();
  focused.copyTo(processed);

  int input;
  bool quit = false;
  while(!quit) {
    cout << "top level menu. "
         << endl << "'r': reset processed to original. "
         << endl << "'c': find corners and go to grid finding. "
         << endl << "'g': find grid and go to navigation. "
         << endl << "'p': enter processing loop."
         << endl << "'x': rotate image."
         << endl;
    input = waitKeyEx(0);
    switch(input) {
      case 'r':
        focused.copyTo(processed);
        imshow("Processed", processed);
        break;
      case 'c':
        findCorners();
        break;
      case 'g':
        findLines();
        break;
      case 'p':
        processImage();
        break;
      case 'x':
        cv::rotate(processed, processed, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::rotate(focused, focused, cv::ROTATE_90_COUNTERCLOCKWISE);
        imshow("Processed", processed);
        break;
      case 'q':
        cout << "Bye." << endl;
        quit = true;
        break;
      default:
        break;
    }
  }
  return 0;
}

void findCorners() {
  // This uses the defaults in the corner finder.
  musicocr::CornerFinder cornerFinder;
  cornerFinder.adjust(gray, focused);
#if 0
  vector<Vec4i> lines = cornerFinder.find_lines(gray);
  vector<Point> corners = cornerFinder.find_corners(lines, gray.cols, gray.rows);
  cout << "points: " << corners[0] << ", " << corners[1] << ", "
       << corners[2] << ", " << corners[3] << endl;

  // Crop to corners.
  Rect c(corners[0], corners[2]);
  cout << "cropping to " << c << endl;

  cornerFinder.adjustToCorners(gray, focused, corners);
  bool rotate = cornerFinder.shouldRotate(focused);
  if (rotate) {
    cv::rotate(focused, focused, cv::ROTATE_90_COUNTERCLOCKWISE);
  }
  //focused = Mat(gray, c);
#endif
  focused.copyTo(processed);
  imshow("Processed", processed);
}


void findLines() {
  musicocr::Sheet sheet;
  vector<Vec4i> lines = sheet.find_lines(processed);
  vector<Vec4i> verticalLines = sheet.findVerticalLines(processed);
  cvtColor(processed, cdst, COLOR_GRAY2BGR);

  sheet.analyseLines(lines, verticalLines, cdst);
  sheet.printSheetInfo();
  imshow("Processed", cdst);
}

void processImage() {
  Mat tmp;
  Mat last;
  int input;
  while (true) {
    cout << "processing image, select operation." << endl;
    input = waitKeyEx(0);
    if (input == 'q') break;
    last = processed.clone();
    correctConfig();
    switch(input) {
    case 'a' : // adaptive threshold
    {
      const int additionConstant = (aSubtractMe > 0)
        ? (0 - aSubtractMe) : aAddMe;
      tmp = Mat::zeros(processed.rows, processed.cols, processed.type());
      cout << "adaptiveThreshold 255, " << athreshold
           << ", THRESH_BINARY, " << aBlockSize << ", "
           << additionConstant << endl; 
      adaptiveThreshold(processed, tmp, 255, athreshold,
                        THRESH_BINARY, aBlockSize, additionConstant);
      tmp.copyTo(processed);
      imshow("Processed", processed);
    }
    break;
    case 'b' : // gaussian blur
    {
      tmp = Mat::zeros(processed.rows, processed.cols, processed.type());
      cout << "gaussian blur " << gaussianKernel
           << ", 0, 0" << endl;
      GaussianBlur(processed, tmp, Size(gaussianKernel, gaussianKernel), 0, 0);
      tmp.copyTo(processed); 
      imshow("Processed", processed);
    }
    break;
    case 'c' : // canny
    {
      cout << "canny " << cannyMin << ", "
           << cannyMax << ", " << sobelKernel
           << ", " << l2gradient << endl;
      Canny(processed, processed, cannyMin, cannyMax,
            sobelKernel, (l2gradient > 0));
      imshow("Processed", processed);
    }
    break;
    case 'd': // dilate vertically
    {
      tmp = processed.clone();
      const int edVerticalHeight = tmp.rows / edVerticalSizeFudge;
      cout << "dilate vertically with "
           << edVerticalWidth << ", " 
           << edVerticalHeight << endl;
      Mat verticalStructure = getStructuringElement(MORPH_RECT,
        Size(edVerticalWidth, edVerticalHeight));
      dilate(tmp, tmp, verticalStructure, Point(-1, -1));
      tmp.copyTo(processed);
      imshow("Processed", processed);
    }
    break;
    case 'e': // e, erode vertically
    {
      tmp = processed.clone();
      const int edVerticalHeight = tmp.rows / edVerticalSizeFudge;
      cout << "erode vertically with "
           << edVerticalWidth << ", "
           << edVerticalHeight << endl;
      Mat verticalStructure = getStructuringElement(MORPH_RECT,
        Size(edVerticalWidth, tmp.rows / edVerticalSizeFudge));
      erode(tmp, tmp, verticalStructure, Point(-1, -1));
      tmp.copyTo(processed);
      imshow("Processed", processed);
    }
    break;
    case 'x': // dilate horizontally
    // (x,y,t,s,b,f)
    {
      tmp = processed.clone();
      const int edHorizontalWidth = tmp.cols / edHorizontalSizeFudge;
      cout << "dilate horizontally with "
           << edHorizontalWidth << ", "
           << edHorizontalHeight << endl;
      Mat horizontalStructure = getStructuringElement(MORPH_RECT,
        Size(edHorizontalWidth, edHorizontalHeight));
      dilate(tmp, tmp, horizontalStructure, Point(-1, -1));
      tmp.copyTo(processed);
      imshow("Processed", processed);
    }
    break;
    case 'y': // erode horizontally
    {
      tmp = processed.clone();
      const int edHorizontalWidth = tmp.cols / edHorizontalSizeFudge;
      cout << "erode horizontally with "
           << edHorizontalWidth << ", "
           << edHorizontalHeight << endl;
      Mat horizontalStructure = getStructuringElement(MORPH_RECT,
        Size(edHorizontalWidth, edHorizontalHeight));
      erode(tmp, tmp, horizontalStructure, Point(-1, -1));
      tmp.copyTo(processed);
      imshow("Processed", processed);
    }
    break;
    case 'f': // f, find contours
    // first: x, y, s. then t, b (gaussian 3 or more).
    // gaussian kernel 7 the bounding boxes contain too much
    // but might be good for text. 5 is also not so good.
    // 3 has ok segmentation.
    {
      vector<vector<Point>> contours;
      vector<Vec4i> hierarchy;
      // maybe just the outer contours are enough.
      findContours(processed, contours, hierarchy, RETR_TREE,
        //CHAIN_APPROX_NONE, Point(0, 0));
        CHAIN_APPROX_SIMPLE, Point(0, 0));
      cout << "found " << contours.size() << " contours." << endl;
      cvtColor(processed, cdst, COLOR_GRAY2BGR);
      vector<Rect> rectangles(contours.size());
      vector<vector<Point>> hull(contours.size());
      for (int i = 0; i < contours.size(); i++) {
        auto colour = Scalar(0, 0, 255);
        if (hierarchy[i][3] == -1) {
          colour = Scalar(255, 0, 0);
        } else if (hierarchy[i][2] == -1) {
          colour = Scalar(0, 255, 0);
        }
        drawContours(cdst, contours, i, colour, 1, 8,
                     hierarchy, 0, Point(0, 0));
        convexHull(Mat(contours[i]), hull[i], false);
        rectangles[i] = boundingRect(Mat(hull[i]));
      }
      imshow("Processed", cdst);
    }
    break;
    case 'h': // h, equalizehist
      cout << "equalize histograms." << endl;
      equalizeHist(processed, processed);
      imshow("Processed", processed);
    break;
    case 'i': // i, invert
      cout << "inverting " << endl;
      tmp = ~processed;
      tmp.copyTo(processed);
      imshow("Processed", processed);
     break;
     case 'l': // hough line finding
     {
       cout << "houghlinesp " << houghResolution << ", "
            << (float)houghResolutionRad * CV_PI/180.0 << ", "
            << houghThreshold << ", "
            << houghMinLinLength << ", "
            << houghMaxLineGap << endl;
       vector<Vec4i> lines;
       HoughLinesP(processed, lines, houghResolution, 
                   (float)houghResolutionRad * CV_PI/180.0,
                    houghThreshold,
                    houghMinLinLength, houghMaxLineGap);
       cvtColor(processed, cdst, COLOR_GRAY2BGR);
       for (const auto& l: lines) {
         line(cdst, Point(l[0], l[1]), Point(l[2], l[3]),
              Scalar(255, 0, 0), 1);
       }
       imshow("Processed", cdst);
     }
     break;
     case 'r': // reset
       cout << "reset to previous state" << endl;
       processed = last;
       imshow("Processed", processed);
       break;
     case 's': // processed := focused - proceessed
       cout << "subtract" << endl;
       cout << "currently focused has type " << focused.type()
            << " and size " << focused.size() << endl;
       cout << "processed: " << processed.type() << ", "
            << processed.size() << endl;
       processed = focused + ~processed;
       imshow("Processed", processed);
     break;
     case 't' : // non-adaptive threshold
     {
       cout << "threshold " << (float)thresholdValue/255.0
            << ", " << thresholdType << ", "
            << useOtsu << endl;
       threshold(processed, processed, (float)thresholdValue/255.0,
            255, (thresholdType + (useOtsu ? THRESH_OTSU : 0)));
       imshow("Processed", processed);
     }
     break;
     case 'z' : // try grid line finding.
       {
       musicocr::SheetConfig config;
       makeSheetConfig(&config);
       musicocr::Sheet sheet(config);
       vector<Vec4i> lines = sheet.find_lines(processed);
       vector<Vec4i> v = sheet.findVerticalLines(processed);
       cout << "Found " << lines.size() << " lines." <<
               " and " << v.size() << " vertical lines." << endl;
       cvtColor(processed, cdst, COLOR_GRAY2BGR);
       for (const auto& l: lines) {
         line(cdst, Point(l[0], l[1]), Point(l[2], l[3]),
              Scalar(255, 0, 0), 1);
       }
       sheet.analyseLines(lines, v, cdst);
       sheet.printSheetInfo();
       imshow("Processed", cdst);
       }
     break;
     default:
       cout << "unknown operation " << input << endl;
       break;
    }
  }
}