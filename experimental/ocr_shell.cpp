#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include "corners.hpp"
#include "recognition.hpp"
#include "structured_page.hpp"
#include "shapes.hpp"
#include "training.hpp"
#include "training_fileutils.hpp"
#include "training_key.hpp"

using namespace cv;
using namespace std;

// gray is the original image in grayscale.
// focused is the matrix being worked on (all of gray or
// a portion of it). processed is the result of the current
// work. cdst is a colour version of processed, can have
// extra markings on it in colour.
Mat gray, focused, processed, cdst;
musicocr::Sheet sheet;
cv::Ptr<cv::ml::StatModel> statModel;

string filename;

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
void navigateSheet();
void processImage();
void scanImage();

int main(int argc, char** argv) {
  if (argc < 2) {
   cerr << "OcrShell <Path to Image> [<model file name>]";
   return -1; 
  }
  filename = argv[1];
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

  if (argc > 2) {
    const string& modelfile = argv[2];
    // We need the type of the model because statmodel::load
    // is templatized.
    char modeltype[20];
    char trainingset[100];
    int x = musicocr::SampleDataFiles::parseModelFileName(
      modelfile, trainingset, modeltype);
    if (x != 2) {
      cerr << "Unrecognised model type in file " << modelfile
           << ", not loading a model." << endl;
    } else {
      cout << "model type: " << modeltype << endl;
      if (strcmp(modeltype, "knn") == 0) {
        statModel = cv::ml::StatModel::load<cv::ml::KNearest>(modelfile);
        cout << "loaded knn model" << endl;
      } else if (strcmp(modeltype, "svm") == 0) {
        statModel = cv::ml::StatModel::load<cv::ml::SVM>(modelfile);
        cout << "loaded svm model" << endl;
      } else if (strcmp(modeltype, "dtrees") == 0){
        statModel = cv::ml::StatModel::load<cv::ml::DTrees>(modelfile);
        cout << "loaded dtree model" << endl;
      }
      else {
        cerr << "Unrecognised model type in file " << modelfile
             << ", not loading a model." << endl;
      }
    }
  }

  resize(image, image, Size(), 0.2, 0.2, INTER_AREA);
  namedWindow("Original Image", WINDOW_AUTOSIZE);
  namedWindow("Processed", WINDOW_AUTOSIZE);
  namedWindow("What is this?", WINDOW_AUTOSIZE);

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
         << endl << "'c': find corners. "
         << endl << "'g': find grid. "
         << endl << "'n': start navigating line by line."
         << endl << "'p': enter processing loop."
         << endl << "'s': scan."
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
      case 'n':
        navigateSheet();
        break;
      case 'p':
        processImage();
        break;
      case 'q':
        cout << "Bye." << endl;
        quit = true;
        break;
      case 's':
        scanImage();
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
  focused.copyTo(processed);
  imshow("Processed", processed);
}

void findLines() {
  vector<Rect> lineContours = sheet.find_lines_outlines(focused);

  cout << "have line countours, drawing." << endl;

  cvtColor(focused, cdst, COLOR_GRAY2BGR);
  for (const auto& r : lineContours) {
    rectangle(cdst, r, Scalar(255, 0, 0), 1);
  }
  imshow("Processed", cdst);

  cout << "creating sheet lines." << endl;

  sheet.createSheetLines(lineContours, focused);

//  sheet.printSheetInfo();
}

void scanImage() {
  if (!sheet.getLineCount()) {
    cerr << "Need to use 'g' first to set up a sheet." << endl;
    return;
  }
  if (!statModel || !statModel->isTrained()) {
    cerr << "Missing a trained model." << endl;
    return;
  }
  musicocr::ContourConfig config;
  makeContourConfig(&config);
  int previousVoicePosition = 0;
  for (size_t i = 0; i < sheet.getLineCount(); i++) {
    auto& sl = sheet.getNthLine(i);
    if (!sl.isRealMusicLine()) continue;

    musicocr::ShapeFinder* sf = new musicocr::ShapeFinder(config);
    sl.setShapeFinder(sf);
    sf->initLineScan(sl, statModel);  // TODO: move this to setShapeFinder?

    int voicePosition = sf->getVoicePosition();
    cout << "line " << i << " has voice position " << voicePosition << endl;

    // xxx: compare with previousVoicePosition to decide if this is plausible.

    // determine theoretical top/bottom of bar lines based on voice position
    const cv::Rect& bb = sl.getBoundingBox();
    const int top = bb.tl().y;
    const int bottom = bb.br().y;

    // These are relative to bb.
    const std::pair<int, int> coords = sl.getCoordinates();
    int bltop, blbottom;
    switch(voicePosition) {
      case 0: // just individual lines
        bltop = coords.first + top;
        blbottom = coords.second + top;
        break;
      case 1: // top voice
        bltop = coords.first + top; 
        blbottom = bottom;
        break;
      case 2: // middle voice
        bltop = top;
        blbottom = bottom;
        break;
      case 3: // bottom voice
        bltop = top;
        blbottom = coords.second + top;
        break;
    }
    previousVoicePosition = voicePosition;

    const std::vector<int> bp = sf->getBarPositions();
    for (size_t i = 0; i < bp.size(); i++) {
      const musicocr::Shape* s = sf->getBarAt(bp[i]);
      const cv::Rect& sr = s->getRectangle() + bb.tl();
      const int middleX = sr.tl().x + (sr.br().x - sr.tl().x) / 2;
      line(cdst, cv::Point(middleX, bltop), cv::Point(middleX, blbottom),
           Scalar(0, 0, 127), 1);
      rectangle(cdst, sr, Scalar(0, 200, 0), 1);
    }
  }
  imshow("Processed", cdst);
}

void navigateSheet() {
  if (!sheet.getLineCount()) {
    cerr << "Need to use 'g' first to set up a sheet." << endl;
    return;
  }
  int lineIndex = 0;
  int input;
  
  focused = sheet.getNthLine(lineIndex).getViewPort().clone();
  focused.copyTo(processed);
  imshow("Processed", processed);

  musicocr::ContourConfig config;
  makeContourConfig(&config);

  size_t sheetSize = sheet.getLineCount();
  bool quit = false;

  while(!quit) {
    cout << "Per-line menu. Cursor keys to navigate, 'p' to fiddle, "
         << "'t' to train, 'd' to run a model or 'q' to quit. "
         << "'d' needs a model loaded." << endl;
    cout << "Current Line is " << (sheet.getNthLine(lineIndex).isRealMusicLine() ? "" : "not ")
         << "a real music line." << endl;
    input = waitKeyEx(0);
    switch(input) {
      case 65364: // cursor down
        {
        lineIndex++;
        if (lineIndex >= sheetSize) lineIndex -= sheetSize; 
        focused = sheet.getNthLine(lineIndex).getViewPort().clone();
        focused.copyTo(processed);
        imshow("Processed", processed);
        }
        break;
      case 65362: // cursor up
        {
        lineIndex--;
        if (lineIndex < 0) lineIndex += sheetSize;
        focused = sheet.getNthLine(lineIndex).getViewPort().clone();
        focused.copyTo(processed);
        imshow("Processed", processed);
        }
        break;
      case 'p': 
        processImage();
        break;
      case 't':
        // step through current line providing classification; q exits.
        // There is no need to specify a model type here because the
        // training data collected works for all stat model types.
        {
        ofstream responseStream;
        char filenameBase[200];
        sprintf(filenameBase, "training/data/%s.%d", filename.c_str(), lineIndex);
        char responseFileName[250];
        sprintf(responseFileName, "training/data/responses.%s.%d",
                filename.c_str(), lineIndex);
        // this will overwrite the file if it exists.
        responseStream.open(responseFileName);
        musicocr::ShapeFinder shapeFinder(config);
        shapeFinder.getTrainingDataForLine(
          processed, "Processed", "What is this?", filenameBase,
          responseStream);
        responseStream.close();
        }
        break;
      case 'd':
        {
          if (!statModel || !statModel->isTrained()) {
            cerr << "Missing a trained model." << endl;
            break;
          }
          auto& sheetLine = sheet.getNthLine(lineIndex);
          if (!sheetLine.hasShapeFinder()) {
            musicocr::ContourConfig config;
            makeContourConfig(&config);
            musicocr::ShapeFinder* sf = new musicocr::ShapeFinder(config);
            sheetLine.setShapeFinder(sf);
            sf->initLineScan(sheetLine, statModel);
          }
 
          musicocr::Scanner scanner;
          sheetLine.getShapeFinder().scanLine(sheetLine, statModel, scanner, "Processed",
            "What is this?");
        }
      break;
      case 'q': 
        quit = true;
        break;
        
      default: break;
    }
  }
}

void processImage() {
  Mat tmp;
  int input;
  while (true) {
    cout << "processing image, select operation." << endl;
    input = waitKeyEx(0);
    if (input == 'q') break;
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
        cout << "rectangle with size " << rectangles[i].area() << endl;
        rectangle(cdst, rectangles[i], Scalar(255, 255, 255), 1);
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
       focused.copyTo(processed);
       imshow("Processed", processed);
       break;
     case 's': // processed := focused - proceessed
       cout << "subtract" << endl;
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
     default:
       cout << "unknown operation " << input << endl;
       break;
    }
  }
}
