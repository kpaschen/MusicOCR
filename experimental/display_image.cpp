#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "corners.hpp"

using namespace cv;
using namespace std;

Mat gray;
int gaussianKernel = 15;
int thresholdValue = 0.0;
int thresholdType = THRESH_BINARY;
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
  if (houghResolution < 1) { houghResolution = 1; }
  config->houghResolution = houghResolution;
  config->houghResolutionRad = houghResolutionRad * CV_PI / 180;
  if (config->houghResolutionRad < 1) { config->houghResolutionRad = 1; }
  config->houghThreshold = houghThreshold;
  config->houghMinLinLength = houghMinLinLength;
  config->houghMaxLineGap = houghMaxLineGap;
}

musicocr::CornerConfig lastConfig;

void onTrackbar(int, void *) {
  makeCornerConfig(&lastConfig);
  musicocr::CornerFinder cornerFinder(lastConfig);

  vector<Vec4i> lines = cornerFinder.find_lines(gray);

  Mat cdst;
  cvtColor(gray, cdst, COLOR_GRAY2BGR);
  for (size_t i = 0; i < lines.size(); i++) {
     Vec4i l = lines[i];
     line(cdst, Point(l[0], l[1]), Point(l[2], l[3]),
          Scalar(255, 0, 0), 2);
  }

  vector<Point> corners = cornerFinder.find_corners(lines, gray.cols, gray.rows);
  line(cdst, corners[0], corners[1], Scalar(0, 126, 0), 3);   // top
  line(cdst, corners[1], corners[2], Scalar(0, 255, 0), 3);   // right
  line(cdst, corners[2], corners[3], Scalar(0, 0, 126), 3);   // bottom
  line(cdst, corners[3], corners[0], Scalar(0, 0, 255), 3);   // left
  imshow("Display Image", cdst);

  // Get a perspective transform using the corners and apply it.
  Mat warped = Mat::zeros(gray.rows, gray.cols, gray.type());
  cornerFinder.adjustToCorners(gray, warped, corners);

  // Then rotate counterclockwise by 90 degress if !horizontalp,
  bool horizontalp = cornerFinder.mostLinesAreHorizontal(lines);
  if (!horizontalp) {
    cout << "mostly vertical, should flip " << endl;
    cv::rotate(warped, warped, cv::ROTATE_90_COUNTERCLOCKWISE);
  }
  imshow("Original Image", warped);
}

void setupTrackbars(const string& windowName) {
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

int main(int argc, char** argv) {
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
  imshow("Original Image", image);

  cvtColor(image, gray, COLOR_BGR2GRAY);
  // everything starts with 'gray'.
  namedWindow("Display Image", WINDOW_AUTOSIZE);
  imshow("Display Image", gray);
  namedWindow("Controls", WINDOW_AUTOSIZE);
  setupTrackbars("Controls");

  int input = waitKeyEx(0);
  return 0;
}
