#include <gtest/gtest.h>
#include <iostream>
#include <unistd.h>

#include "corners.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

TEST(CornersTestSuite, TestOne) {
  const char *buffer = getcwd(NULL, 0);
  const std::string test_file = std::string(buffer) + "/test/data/sample1.jpg";
  cv::Mat image = cv::imread(test_file);
  ASSERT_TRUE(image.data != NULL);
  std::vector<cv::Point> points;
  musicocr::CornerConfig config;
  musicocr::CornerFinder finder(config);
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Vec4i> lines = finder.find_lines(gray);
  points = finder.find_corners(lines, gray.cols, gray.rows);
  ASSERT_EQ(4, points.size());
}

TEST(CornersTestSuite, TestHorizontality) {
  std::vector<cv::Vec4i> lines;
  musicocr::CornerConfig config;
  musicocr::CornerFinder finder(config);

  // The two x coordinates are the same, detect this as vertical.
  cv::Vec4i linev = {10, 15, 10, 16};

  lines.push_back(linev);
  ASSERT_TRUE(!finder.mostLinesAreHorizontal(lines));

  lines.clear();
  // Not exactly vertical, but should be good enough.
  cv::Vec4i linev2 = {10, 15, 11, 20};
  lines.push_back(linev2);
  ASSERT_TRUE(!finder.mostLinesAreHorizontal(lines));

  lines.clear();
  // vertical line.
  cv::Vec4i lineh = {10, 15, 20, 15};
  lines.push_back(lineh);
  ASSERT_TRUE(finder.mostLinesAreHorizontal(lines));

  lines.clear();
  // vertical line.
  cv::Vec4i lineh2 = {10, 15, 20, 16};
  lines.push_back(lineh2);
  ASSERT_TRUE(finder.mostLinesAreHorizontal(lines));

  // Add two vertical lines and one horizontal.
  lines.clear();
  lines.push_back(linev);
  lines.push_back(linev2);
  lines.push_back(lineh);
  ASSERT_TRUE(!finder.mostLinesAreHorizontal(lines));

  // Add another horizontal line.
  lines.push_back(lineh2);
  ASSERT_TRUE(finder.mostLinesAreHorizontal(lines));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
