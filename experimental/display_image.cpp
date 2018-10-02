#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "corners.hpp"
#include "structured_page.hpp"

using namespace cv;
using namespace std;

Mat gray, warped;

bool tuningCornerDetection = false;
bool tuningHarrisCorners = false;
bool tuningGridLineDetection = false;
bool tuningContoursInLine = false;

// Only needed if tuning corner detection
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

// Only needed if tuning harris corner detection
int blockSize = 2;
int apertureSize = 3;
int k = 4;
int harrisThreshold = 200;

// Only needed if tuning grid line detection
int gridLineGaussianKernel = 9;  // 7 or 9 ...
int gridLineThresholdValue = 0;
int gridLineThresholdType = 3;
int gridLineUseOtsu = 0;
int gridLineCannyMin = 80;
int gridLineCannyMax = 121;
int gridLineSobel = 5;
int gridLineL2Gradient = 0;
int gridLineHoughThreshold = 82;
int gridLineHoughMinLinLength = 23;
int gridLineHoughMaxLineGap = 15;

// Tuning contour finding
int contourGaussianKernel = 1;
int contourThresholdValue = 0;
int contourThresholdType = 4;
int contourUseOtsu = 1;  // only true or false
int contourCannyMin = 80;
int contourCannyMax = 90;
int contourSobelKernel = 3;
int contourL2Gradient = 0;  // only true or false
// These are peculiarly bad. shouldn't it do better on a very
// clear canny view?
int contourHoughThreshold = 70;
int contourHoughMLL = 23;
int contourHoughMLG = 32;
int contourHoughResolution = 1; // resolution in pixels.
int contourHoughResolutionRad = 1; // resolution in degree radians.
// threshold algorithm.
// int edthreshold = ADAPTIVE_THRESH_MEAN_C;
int edthreshold = ADAPTIVE_THRESH_GAUSSIAN_C;
// threshold type is always THRESH_BINARY
int edBlockSize = 15; // must be odd
int edSubtractMe = 2;
int edAddMe = 0;

// the width of the horizontal structuring element will be
// inputmatrix.cols / edHorizontalSizeFudge
int edHorizontalSizeFudge = 30;
int edHorizontalHeight = 1;
int edVerticalSizeFudge = 7;  // 7 is good for finding mainly the bars
int edVerticalWidth = 1;


// Only needed if tuning corner detection
void makeCornerConfig(musicocr::CornerConfig *config) {
  if (gaussianKernel % 2 == 0) {
    gaussianKernel += 1;
  }
  config->gaussianKernel = gaussianKernel;
  config->thresholdValue = (double)thresholdValue/255.0;
  config->thresholdType = thresholdType + (useOtsu ? THRESH_OTSU : 0);
  config->cannyMin = cannyMin;
  config->cannyMax = cannyMax;
  if (sobelKernel < 3) {
    sobelKernel = 3;
  }
  if (sobelKernel % 2 == 0) {
    sobelKernel += 1;
  } 
  if (sobelKernel > 7) {
    sobelKernel = 7;
  }
  config->sobelKernel = sobelKernel;
  config->l2gradient = (l2gradient != 0);
  if (houghResolution < 1.0) { houghResolution = 1.0; }
  config->houghResolution = houghResolution;
  config->houghResolutionRad = (double)houghResolutionRad * CV_PI / 180.0;
  if (config->houghResolutionRad < 1.0) { config->houghResolutionRad = 1.0; }
  config->houghThreshold = houghThreshold;
  config->houghMinLinLength = houghMinLinLength;
  config->houghMaxLineGap = houghMaxLineGap;
}

void makeSheetConfig(musicocr::SheetConfig *config) {
  if (gridLineGaussianKernel % 2 == 0) {
    gridLineGaussianKernel += 1;
  }  
  config->gaussianKernel = gaussianKernel;
  config->thresholdValue = (double)gridLineThresholdValue/255.0;
  config->thresholdType = gridLineThresholdType + (gridLineUseOtsu ? THRESH_OTSU : 0);
  config->cannyMin = gridLineCannyMin;
  config->cannyMax = gridLineCannyMax;
  if (gridLineSobel % 2 == 0) {
    gridLineSobel++;
  }
  if (gridLineSobel < 3) { gridLineSobel = 3; }
  if (gridLineSobel > 7) { gridLineSobel = 7; }
  config->sobel = gridLineSobel;
  config->l2Gradient = (gridLineL2Gradient != 0);
  config->houghThreshold = gridLineHoughThreshold;
  config->houghMinLineLength = gridLineHoughMinLinLength;
  config->houghMaxLineGap = gridLineHoughMaxLineGap;
}

void onTrackbar(int, void *) {
  if (tuningCornerDetection) {
    musicocr::CornerConfig cornerConfig;
    makeCornerConfig(&cornerConfig);
    musicocr::CornerFinder cornerFinder(cornerConfig);
  
    vector<Vec4i> lines = cornerFinder.find_lines(gray);
    Mat cdst;
    cvtColor(gray, cdst, COLOR_GRAY2BGR);

    vector<Point> corners = cornerFinder.find_corners(lines, gray.cols, gray.rows);
    line(cdst, corners[0], corners[1], Scalar(0, 126, 0), 3);   // top
    line(cdst, corners[1], corners[2], Scalar(0, 255, 0), 3);   // right
    line(cdst, corners[2], corners[3], Scalar(0, 0, 126), 3);   // bottom
    line(cdst, corners[3], corners[0], Scalar(0, 0, 255), 3);   // left

  // Crop to corners.
  Rect c(corners[0], corners[2]);
  rectangle(cdst, corners[0], corners[2], Scalar(255, 255, 255), 6);

  warped = Mat(gray, c);
  // somehow just cropping is better than warping most of the time.
  //warped = Mat::zeros(cropped.rows, cropped.cols, cropped.type());
  //cornerFinder.adjustToCorners(cropped, warped, corners);

  imshow("Warped", warped);
  }

  if (tuningHarrisCorners) {
    Mat dst = Mat::zeros(gray.size(), CV_32FC1);
    if (apertureSize % 2 == 0) {
      apertureSize += 1;
    }
    if (apertureSize > 31) {
      apertureSize = 31;
    }  
    cornerHarris(gray, dst, blockSize, apertureSize, (double)k/100.0, BORDER_DEFAULT);
    // does this need to be normalised?
    for (int j = 0; j < dst.rows; j++) {
      for (int i = 0; i < dst.cols; i++) {
        if ((int)dst.at<float>(j, i) > harrisThreshold) {
          circle(dst, Point(i,j), 5, Scalar(0, 0, 0), 2, 8, 0); 
        }
      }
    }
    namedWindow("bla", WINDOW_AUTOSIZE);
    imshow("bla", dst);
  }
  if (tuningGridLineDetection) {
    musicocr::SheetConfig config;
    makeSheetConfig(&config);

    musicocr::Sheet sheet(config);

    vector<Vec4i> lines = sheet.find_lines(warped);
    {
      bool horizontalp = musicocr::CornerFinder::mostLinesAreHorizontal(lines);

      if (!horizontalp) {
        cout << "should rotate this image" << endl;
        cv::rotate(warped, warped, cv::ROTATE_90_COUNTERCLOCKWISE);
        imshow("Warped", warped);
        // Could swap coordinates on lines if that's more efficient.
        // Alternatively, move the decision to rotate into the sheet class
        // and have sheet rotate warped.
        lines = sheet.find_lines(warped);
      }
    }
    Mat cdst;
    cvtColor(warped, cdst, COLOR_GRAY2BGR);
    sheet.analyseLines(lines, cdst);

    sheet.printSheetInfo();

    imshow("Grid", cdst);
  }
  if (tuningContoursInLine) {
    if (contourGaussianKernel % 2 == 0) {
      contourGaussianKernel++;
    }
    if (contourSobelKernel % 2 == 0) {
      contourSobelKernel++;
    }
    if (contourSobelKernel < 3) {
      contourSobelKernel = 3;
    }
    if (contourSobelKernel > 7) {
      contourSobelKernel = 7;
    }
    if (edBlockSize % 2 == 0) {
      edBlockSize++;
    }
    if (edBlockSize <= 1) {
      edBlockSize = 3;
    }
  }
}

void setupTrackbars(const string& windowName) {
if (tuningCornerDetection) {
  // Slider for the Gaussian Blur
  // The kernel size has to be an odd number. 5 works ok.
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
  createTrackbar("Hough Resolution Pixels", windowName, &houghResolution, 10,
                 onTrackbar);
  createTrackbar("Hough Resolution Radians", windowName, &houghResolutionRad,
                 180, onTrackbar);
  createTrackbar("Hough Threshold", windowName, &houghThreshold,
                 255, onTrackbar);
  createTrackbar("Hough Min Line Length (P)", windowName, &houghMinLinLength,
                 100, onTrackbar);
  createTrackbar("Hough Max Line Gap (P)", windowName, &houghMaxLineGap,
                 100, onTrackbar);

}

if (tuningHarrisCorners) {
  // sliders for harris
  createTrackbar("harris block size", windowName, &blockSize, 10, onTrackbar);
  createTrackbar("harris aperture size", windowName, &apertureSize, 10, onTrackbar);
  createTrackbar("harris k", windowName, &k, 100, onTrackbar);
  createTrackbar("harris threshold", windowName, &harrisThreshold, 400, onTrackbar);
}
if (tuningGridLineDetection) {
  createTrackbar("grid line gaussian kernel", windowName,
     &gridLineGaussianKernel, 15, onTrackbar);
  createTrackbar("grid line threshold", windowName,
     &gridLineThresholdValue, 100, onTrackbar);
  createTrackbar("grid line threshold type", windowName,
     &gridLineThresholdType, 4, onTrackbar);
  createTrackbar("grid line otsu", windowName,
     &gridLineUseOtsu, 1, onTrackbar);
  createTrackbar("grid line canny min", windowName,
     &gridLineCannyMin, 255, onTrackbar);
  createTrackbar("grid line canny max", windowName,
     &gridLineCannyMax, 255, onTrackbar);
  createTrackbar("grid line sobel", windowName,
     &gridLineSobel, 7, onTrackbar);
  createTrackbar("grid line hough threshold", windowName,
     &gridLineHoughThreshold, 200, onTrackbar);
  createTrackbar("grid line hough min line", windowName,
     &gridLineHoughMinLinLength, 100, onTrackbar);
  createTrackbar("grid line hough max gap", windowName,
     &gridLineHoughMaxLineGap, 70, onTrackbar);
}
if (tuningContoursInLine) {
  createTrackbar("Gaussian Kernel Size", windowName, &contourGaussianKernel,
                 15, onTrackbar);
  createTrackbar("Threshold value", windowName, &contourThresholdValue, 255,
                 onTrackbar);
  createTrackbar("Threshold type", windowName, &contourThresholdType,
                 THRESH_TOZERO_INV, onTrackbar);
  createTrackbar("Use Otsu", windowName, &contourUseOtsu, 1, onTrackbar);

  createTrackbar("Canny Min", windowName, &contourCannyMin, 255, onTrackbar);
  createTrackbar("Canny Max", windowName, &contourCannyMax, 255, onTrackbar);
  createTrackbar("Sobel", windowName, &contourSobelKernel, 7, onTrackbar);
  createTrackbar("l2gradient", windowName, &contourL2Gradient, 1, onTrackbar);
  createTrackbar("houghThreshold", windowName, &contourHoughThreshold, 200,
                 onTrackbar);
  createTrackbar("hough min line", windowName,
     &contourHoughMLL, 100, onTrackbar);
  createTrackbar("hough max gap", windowName,
     &contourHoughMLG, 70, onTrackbar);
  createTrackbar("hough resolution", windowName,
     &contourHoughResolution, 100, onTrackbar);
  createTrackbar("hough rad", windowName,
     &contourHoughResolutionRad, 180, onTrackbar);
  
  createTrackbar("e-d threshold", windowName,
                 &edthreshold, 1, onTrackbar);
  createTrackbar("e-d block size", windowName,
                 &edBlockSize, 15, onTrackbar);
  createTrackbar("e-d subtract const", windowName,
                 &edSubtractMe, 10, onTrackbar);
  createTrackbar("e-d add const", windowName,
                 &edAddMe, 10, onTrackbar);
  createTrackbar("e-d horizontal size fudge", windowName,
                 &edHorizontalSizeFudge, 30, onTrackbar);
  createTrackbar("e-d horizontal height", windowName,
                 &edHorizontalHeight, 10, onTrackbar);
  createTrackbar("e-d vertical size fudge",
                 windowName, &edVerticalSizeFudge, 50, onTrackbar);
  createTrackbar("e-d vertical width", windowName,
                 &edVerticalWidth, 10, onTrackbar);
}
}

int main(int argc, char** argv) {
  //tuningGridLineDetection = true;
  //tuningCornerDetection = true;
  tuningContoursInLine = true;
  if (argc != 2) {
   cerr << "display_image.out <Path to Image>";
   return -1; 
  }
  Mat image;
  image = imread(argv[1], 1);
  if (!image.data) {
    cerr << "No image data.";
    return -1;
  }
  resize(image, image, Size(), 0.2, 0.2, INTER_AREA);
  namedWindow("Original Image", WINDOW_AUTOSIZE);
  namedWindow("Warped", WINDOW_AUTOSIZE);

  // everything starts with 'gray'.
  cvtColor(image, gray, COLOR_BGR2GRAY);
  imshow("Original Image", gray);

  namedWindow("Controls", WINDOW_AUTOSIZE);
  setupTrackbars("Controls");

  int input;

  {
  musicocr::CornerFinder cornerFinder;
  vector<Vec4i> lines = cornerFinder.find_lines(gray);
  vector<Point> corners = cornerFinder.find_corners(lines, gray.cols, gray.rows);

  // Crop to corners.
  Rect c(corners[0], corners[2]);
  rectangle(gray, corners[0], corners[2], Scalar(255, 255, 255), 6);

  warped = Mat(gray, c);

  // Somehow warping doesn't help that much, just cropping for now.
  imshow("Warped", warped);
  input = waitKeyEx(0);

  if (tuningGridLineDetection) {
    namedWindow("Canny", WINDOW_AUTOSIZE);
    namedWindow("hough", WINDOW_AUTOSIZE);
  }
  }

  // Now 'warped' contains the adjusted image, in grayscale.
  // Identify the printed horizontal lines, as well as any vertical lines.
  musicocr::Sheet sheet;

  {
  vector<Vec4i> lines = sheet.find_lines(warped);
  {
    bool horizontalp = musicocr::CornerFinder::mostLinesAreHorizontal(lines);

    if (!horizontalp) {
      cout << "should rotate this image" << endl;
      cv::rotate(warped, warped, cv::ROTATE_90_COUNTERCLOCKWISE);
      imshow("Warped", warped);
      // Could swap coordinates on lines if that's more efficient.
      // Alternatively, move the decision to rotate into the sheet class
      // and have sheet rotate warped.
      lines = sheet.find_lines(warped);
    }
  }

  Mat cdst;
  cvtColor(warped, cdst, COLOR_GRAY2BGR);

  sheet.analyseLines(lines, cdst);
  sheet.printSheetInfo();

  //namedWindow("Grid", WINDOW_AUTOSIZE);
  //imshow("Grid", cdst);
  }
  input = waitKeyEx(0);

  // Select the gridline to focus on (cursor keys).
  size_t gridLineCount = 0;
  const size_t lgCount = sheet.size();
  for (size_t i = 0; i < lgCount; i++) {
    gridLineCount += sheet.getNthLineGroup(i).size();
  }

  int lineIndex = 0;
  Mat focused;
  namedWindow("Focused", WINDOW_AUTOSIZE);

  // Get first sheetline.
  musicocr::LineGroup lg = sheet.getNthLineGroup(lineIndex);
  musicocr::SheetLine sl = lg.getNthVoice(lineIndex);
  focused = warped(sl.getBoundingBox());
  imshow("Focused", focused);

  namedWindow("processed", WINDOW_AUTOSIZE);
  namedWindow("canny", WINDOW_AUTOSIZE);
  namedWindow("hough", WINDOW_AUTOSIZE);
  namedWindow("what is this?", WINDOW_AUTOSIZE);

  // focused: the line from the original picture to process
  // processed: current state

  Mat processed;
  focused.copyTo(processed);

  // This holds contours, and should be a colour map.
  Mat cont;

  Mat tmp;

  bool quit = false;
  while(!quit) {
    input = waitKeyEx(0);
    bool navigate = false;
    bool process = false;
    cout << "input: " << input << endl;
    switch(input) {
      case 65364:  // cursor down
        navigate = true;
        lineIndex++;
        if (lineIndex >= gridLineCount) {
          lineIndex = 0; // wrap around
        }
        break;
      case 65362: // cursor up
        navigate = true;
        lineIndex--;
        if (lineIndex < 0) {
          lineIndex = gridLineCount - 1;
        }
        break;
      case 113: // q
        quit = true;
        break;
      default:
        // canny: c, adaptive threshold: t, erode, e: dilate: d,
        // equalizehist: h, distanceTransform: l, contour finding: f,
        // reset: r (processed = focus); number: quality rating
        process = true;
        break;
    }
    if (quit) { break; }
    if (navigate) {
     int maxLine = 0;
     for (size_t i = 0; i < lgCount; i++) {
       int nextMaxLine = maxLine + sheet.getNthLineGroup(i).size();
       if (nextMaxLine > lineIndex) {
         sl = sheet.getNthLineGroup(i).getNthVoice(lineIndex - maxLine);
         break;
       } else {
         maxLine = nextMaxLine;
       }
     }
     focused = warped(sl.getBoundingBox());
     focused.copyTo(processed);
     imshow("Focused", focused);
     imshow("processed", processed);
     continue;
    }
    if (process) {
      switch(input) {
        case 'a' : // adaptive threshold
           {
           const int additionConstant = (edSubtractMe > 0)
                     ? (0 - edSubtractMe) : edAddMe;
           tmp = Mat::zeros(processed.rows, processed.cols, processed.type());
           cout << "adaptiveThreshold 255, " << edthreshold
                << ", THRESH_BINARY, " << edBlockSize << ", "
                << additionConstant << endl; 
           adaptiveThreshold(processed, tmp, 255, edthreshold,
                             THRESH_BINARY, edBlockSize, additionConstant);
           tmp.copyTo(processed);
           imshow("processed", processed);
           }
          break;
        case 'b' : // gaussian blur
          {
          tmp = Mat::zeros(processed.rows, processed.cols, processed.type());
          cout << "gaussian blur " << contourGaussianKernel
               << ", 0, 0" << endl;
          GaussianBlur(processed, tmp, Size(contourGaussianKernel,
            contourGaussianKernel), 0, 0);
          tmp.copyTo(processed); 
          imshow("processed", processed);
          }
          break;
        case 'c' : // canny
          {
          cout << "canny " << contourCannyMin << ", "
               << contourCannyMax << ", " << contourSobelKernel
               << ", " << contourL2Gradient << endl;
          Canny(processed, processed, contourCannyMin, contourCannyMax,
                contourSobelKernel, (contourL2Gradient > 0));
          imshow("processed", processed);
          }
          break;
        case 'd': // dilate vertically
         {
         tmp = processed.clone();
         // With a right-sized structuring element (height about rows/7)
         // this finds the longer vertical lines including the bars.
         const int edVerticalHeight = tmp.rows / edVerticalSizeFudge;
         cout << "dilate vertically with "
              << edVerticalWidth << ", " 
              << edVerticalHeight << endl;
         Mat verticalStructure = getStructuringElement(MORPH_RECT,
           Size(edVerticalWidth, edVerticalHeight));
         dilate(tmp, tmp, verticalStructure, Point(-1, -1));
         tmp.copyTo(processed);
         imshow("processed", processed);
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
         imshow("processed", processed);
          }
          break;
        case 'x': // dilate horizontally
          // (x,y,s,c,f)
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
          imshow("processed", processed);
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
          imshow("processed", processed);
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
            drawContours(cont, contours, i, colour, 1, 8,
                         hierarchy, 0, Point(0, 0));
#if 0
            drawContours(cont, hull, i, Scalar(0, 0, 127), 2, 8,
                         vector<Vec4i>(), 0, Point(0, 0));
#endif
            convexHull(Mat(contours[i]), hull[i], false);
            rectangles[i] = boundingRect(Mat(hull[i]));
          }
          std::sort(rectangles.begin(), rectangles.end(), musicocr::rectLeft);
          for (int i = 0; i < rectangles.size(); i++) {
            rectangle(cont, rectangles[i], Scalar(0, 0, 127), 2);
            imshow("processed", cont);

            Mat partial = Mat(focused, rectangles[i]);
            cout << "showing contour at coordinates " << rectangles[i].tl()
                 << " to " << rectangles[i].br() << endl;
            Mat scaleup;
            resize(partial, scaleup, Size(), 2.0, 2.0, INTER_CUBIC);

            imshow("what is this?", scaleup);

            // 'l': vertical line, 'h': note head, 's': sharp,
            // 'm': multiple items, 'k': skippable, 'b': long break,
            // 'd': double vertical line', 'w': character or number,
            // 'o': other, 'c': connecting line
            int cat = waitKeyEx(0);

            char fname[200];
            sprintf(fname, "training/data/f%d.png", i);
            cout << fname << " contains " << cat << endl;
                 
            bool result = imwrite(fname, partial);
           
          }
          imshow("processed", cont);
          }
          break;
        case 'h': // h, equalizehist
          equalizeHist(processed, processed);
          imshow("processed", processed);
          break;
        case 'i': // i, invert
          cout << "inverting " << endl;
          tmp = ~processed;
          tmp.copyTo(processed);
          imshow("processed", processed);
          break;
        case 'l': // hough line finding
          {
          vector<Vec4i> lines;
          HoughLinesP(processed, lines, contourHoughResolution, 
                     (float)contourHoughResolutionRad * CV_PI/180.0,
                      contourHoughThreshold,
                      contourHoughMLL, contourHoughMLG);
          cvtColor(processed, cont, COLOR_GRAY2BGR);
          for (const auto& l: lines) {
            line(cont, Point(l[0], l[1]), Point(l[2], l[3]),
                 Scalar(255, 0, 0), 1);
          }
          imshow("processed", cont);
          }
          break;
        case 'r': // r, reset
          cout << "resetting" << endl;
          focused.copyTo(processed);
          imshow("processed", processed);
          break;
        case 's': // processed := focused - proceessed
          cout << "subtract" << endl;
          processed = focused + ~processed;
          imshow("processed", processed);
          break;
        case 't' : // non-adaptive threshold
          {
            cout << "threshold " << (float)contourThresholdValue/255.0
                 << ", " << contourThresholdType << ", "
                 << contourUseOtsu << endl;
            threshold(processed, processed, (float)contourThresholdValue/255.0,
            255,
            (contourThresholdType + (contourUseOtsu ? THRESH_OTSU : 0)));
            imshow("processed", processed);
          }
          break;
        case 'z' : // obtain coordinate system for current line
          // One horizontal dilate/erode to get just the horizontal lines.
          {
          tmp = processed.clone();
          const int edHorizontalWidth = tmp.cols / edHorizontalSizeFudge;
          Mat horizontalStructure = getStructuringElement(MORPH_RECT,
            Size(edHorizontalWidth, edHorizontalHeight));
          dilate(tmp, tmp, horizontalStructure, Point(-1, -1));
          erode(tmp, tmp, horizontalStructure, Point(-1, -1));

          GaussianBlur(tmp, tmp,
                       Size(contourGaussianKernel, contourGaussianKernel),
                       0, 0);

          threshold(tmp, tmp, (float)contourThresholdValue/255.0, 255,
            (contourThresholdType + (contourUseOtsu ? THRESH_OTSU : 0)));

          Canny(tmp, tmp, contourCannyMin, contourCannyMax,
                contourSobelKernel, (contourL2Gradient > 0));
          imshow("canny", tmp);
          vector<Vec4i> lines;
          // 2.0 might help a bit with mild rotations.
          //HoughLinesP(tmp, lines, 1, 2.0 * CV_PI/180.0, contourHoughThreshold,
          //            contourHoughMLL, contourHoughMLG);
          HoughLinesP(tmp, lines, 1, 2.0 * CV_PI/180.0, 70,
                      23, 32);
          std::sort(lines.begin(), lines.end(), musicocr::moreRight);
          std::sort(lines.begin(), lines.end(), musicocr::moreTop);
          vector<Vec4i> horizontals;
          const Scalar colour1(0, 0, 255);
          const Scalar colour2(255, 0, 0);
          // The horizontal lines are across about 25 pts vertically, with
          // 4-8 distance between lines. start around 25, go up to about 480.
          Scalar currentColour = colour1;
          for (size_t i = 0; i < lines.size(); i++) {
            const auto& l = lines[i];
            int currentHeight = l[1];
            Point lp(l[0], l[1]); Point rp(l[2], l[3]);
            int top = std::min(l[1], l[3]), bottom = std::max(l[1], l[3]);
            for (size_t j = i+1; j < lines.size(); j++) {
              const auto& k = lines[j];
              if (k[1] - bottom <= 2) {
                if (k[2] <= lp.x || k[0] >= rp.x) {
                  if (k[2] <= lp.x) {
                    lp = Point(k[0], k[1]);
                  }
                  if (k[0] >= rp.x) {
                    rp = Point(k[2], k[3]);
                  }
                  top = std::min(top, std::min(k[1], k[3]));
                  bottom = std::max(bottom, std::max(k[1], k[3]));
                } else {
                  i = j + 1;
                  break;
                }
              } else {
                  i = j + 1;
                  break;
              }
            }
            horizontals.emplace_back(Vec4i(lp.x, lp.y, rp.x, rp.y));
          //  cout << "extended line at height " << currentHeight;  
          //  cout << " from " << lp << " to " << rp << endl;
            line(tmp, lp, rp, currentColour, 1);
            if (currentColour == colour1) currentColour = colour2;
            else currentColour = colour1;
          }
          vector<Vec4i> gridLinesH;
          int lastHeight = 0;
          int bestLeft = 500, bestRight = 0;
          for (size_t i = 0; i < horizontals.size(); i++) {
            const auto& candidate = horizontals[i];
            if (!gridLinesH.empty() && candidate[1] - lastHeight >= 12
                && lastHeight < 50) {
              cout << "resetting at height " << lastHeight << endl;
              bestLeft = 500, bestRight = 0;
              gridLinesH.clear();
            }
            gridLinesH.push_back(candidate);
            lastHeight = candidate[1];
            if (candidate[0] < bestLeft) bestLeft = candidate[0];
            if (candidate[2] > bestRight) bestRight = candidate[2];
          } 
          rectangle(tmp, Point(bestLeft, gridLinesH[0][1]),
                    Point(bestRight, gridLinesH.back()[3]),
                    Scalar(255, 255, 255), 2);
          imshow("processed", tmp);
            // try to find bar line candidates
            tmp = processed.clone();
            const int edVerticalHeight = tmp.rows / edVerticalSizeFudge;
            Mat verticalStructure = getStructuringElement(MORPH_RECT,
              Size(edVerticalWidth, edVerticalHeight));
            dilate(tmp, tmp, verticalStructure, Point(-1, -1));
            erode(tmp, tmp, verticalStructure, Point(-1, -1));
            threshold(tmp, tmp, (float)contourThresholdValue/255.0, 255,
              (contourThresholdType + (contourUseOtsu ? THRESH_OTSU : 0)));

            Canny(tmp, tmp, contourCannyMin, contourCannyMax,
                  contourSobelKernel, (contourL2Gradient > 0));
            imshow("canny", tmp);
            HoughLinesP(tmp, lines, 9, 180.0 * CV_PI/180.0,
                        24, 23, 29);
            cout << "found " << lines.size() << " vertical lines." << endl;
            Mat bla;
            cvtColor(tmp, bla, COLOR_GRAY2BGR);
            // there should be a bar line on the left and one on the right
            // edge. they should have the height of the grid. look for them,
            // and for other potential bar lines that also have the height
            // of the grid.
            for (const auto& l: lines) {
              if (l[0] < bestLeft - 2) continue;
              if (l[2] > bestRight + 2) continue;
  
              const int top = std::min(l[1], l[3]);
              const int bottom = std::max(l[1], l[3]);

              int gridLinesTop, gridLinesBottom;
              if (l[0] - bestLeft >= (bestRight - bestLeft)/2) {
                gridLinesTop = gridLinesH[0][3];
                gridLinesBottom = gridLinesH.back()[3];
              } else {
                gridLinesTop = gridLinesH[0][1];
                gridLinesBottom = gridLinesH.back()[1];
              }
              if (std::abs(top - gridLinesTop) < 3 &&
                  std::abs(bottom - gridLinesBottom) < 3) {
                // this has good precision for finding bar lines but
                // the recall could be better. Could collect these
                // in gridLinesV.
                line(bla, Point(l[0], l[1]), Point(l[2], l[3]),
                     Scalar(0, 255, 0), 1);
                continue;
              } 
              line(bla, Point(l[0], l[1]), Point(l[2], l[3]),
                   Scalar(255, 0, 0), 1);
            }
            imshow("hough", bla);
          }
          break;
        default: // is it a number?
          if ('0' < input && input <= '9') {
            // input - 48: quality rating
            cout << "quality rating" << input - '0' << endl;
          }
          break;
      }

      // Not sure about this.
#if 0
      Mat edges;
      adaptiveThreshold(vertical, edges, 255,
        ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);

      Mat kernel = Mat::ones(2, 2, CV_8UC1);
      dilate(edges, edges, kernel);

      Mat smooth;
      vertical.copyTo(smooth);
      blur(smooth, smooth, Size(2, 2));
      smooth.copyTo(vertical, edges);

      imshow("smooth", vertical);
#endif

  }
  }

  return 0;
}
