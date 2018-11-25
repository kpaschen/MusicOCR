#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <unistd.h>

#include "corners.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

void initRegressionMap(std::map<std::string, std::vector<cv::Point>>& m) {
   m.emplace("sample1.jpg", std::vector<cv::Point>({
     cv::Point(0, 3), cv::Point(924,64), cv::Point(924,673),
     cv::Point(0,689)
   }));
   m.emplace("DSC_0177.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(924,0), cv::Point(924,679),
     cv::Point(0,679)
   }));
   m.emplace("DSC_0178.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(693,0), cv::Point(693,924),
     cv::Point(0,924)
   }));
   m.emplace("DSC_0179.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(924,0), cv::Point(924,693),
     cv::Point(0,693)
   }));
   m.emplace("DSC_0180.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(924,0), cv::Point(924,693),
     cv::Point(0,693)
   }));
   m.emplace("DSC_0181.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(924,0), cv::Point(924,693),
     cv::Point(0,693)
   }));
   m.emplace("DSC_0182.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(924,0), cv::Point(924,621),
     cv::Point(0,638)
   }));
   m.emplace("DSC_0184.jpg", std::vector<cv::Point>({
     cv::Point(0, 11), cv::Point(924,66), cv::Point(924,611),
     cv::Point(0,619)
   }));
   m.emplace("DSC_0186.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(924,0), cv::Point(924,693),
     cv::Point(0,693)
   }));
   m.emplace("DSC_0130.jpg", std::vector<cv::Point>({
     cv::Point(0, 116), cv::Point(692,113), cv::Point(692,909),
     cv::Point(0,908)
   }));
   m.emplace("DSC_0131.jpg", std::vector<cv::Point>({
     cv::Point(32, 1), cv::Point(693,1), cv::Point(693,747),
     cv::Point(32,746)
   }));
   m.emplace("DSC_0206.jpg", std::vector<cv::Point>({
     cv::Point(0, 1), cv::Point(693,1), cv::Point(693,909),
     cv::Point(0,909)
   }));
   m.emplace("DSC_0207.jpg", std::vector<cv::Point>({
     cv::Point(1, 5), cv::Point(659,2), cv::Point(659,904),
     cv::Point(1,904)
   }));
   m.emplace("DSC_0208.jpg", std::vector<cv::Point>({
     cv::Point(1, 0), cv::Point(744,0), cv::Point(744,685),
     cv::Point(18,688)
   }));
   m.emplace("DSC_0209.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(673,0), cv::Point(673,693),
     cv::Point(0,693)
   }));
   m.emplace("DSC_0212.jpg", std::vector<cv::Point>({
     cv::Point(1, 1), cv::Point(670,1), cv::Point(670,876),
     cv::Point(6,876)
   }));
   m.emplace("DSC_0213.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(691,0), cv::Point(691,924),
     cv::Point(0,924)
   }));
   // This is a printed page, does not get recognised correctly.
   //m.emplace("DSC_0214.jpg", std::vector<cv::Point>({
   //  cv::Point(102, 180), cv::Point(674,80), cv::Point(674,916),
   //  cv::Point(29,920)
   //}));
}

TEST(CornersTestSuite, TestCornerRegressions) {
  const char *buffer = getcwd(NULL, 0);
  std::map<std::string, std::vector<cv::Point>> expected;
  initRegressionMap(expected);
  cv::Mat image;
  cv::Mat gray;
  musicocr::CornerFinder finder;
  std::vector<cv::Point> points;
  for (const auto& it : expected) {
    const std::string test_file =
        std::string(buffer) + "/test/data/" + it.first;
    image = cv::imread(test_file);
    ASSERT_TRUE(image.data != NULL);
    resize(image, image, cv::Size(), 0.2, 0.2, cv::INTER_AREA);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Vec4i> lines = finder.find_lines(gray);
    points = finder.find_corners(lines, gray.cols, gray.rows);
    ASSERT_EQ(4, points.size());
    for (size_t i = 0; i < 4; i++) {
      EXPECT_EQ(it.second[i].x, points[i].x) << "dataset " << it.first
            << ", point " << i << ": expected " << it.second[i]
            << " but got " << points[i] << std::endl;
      EXPECT_EQ(it.second[i].y, points[i].y) << "dataset " << it.first
            << ", point " << i << ": expected " << it.second[i]
            << " but got " << points[i] << std::endl;
    }
  }
}

void initRotationMap(std::map<std::string, bool>& m) {
  m.emplace("sample1.jpg", true);
  m.emplace("DSC_0177.jpg", true);
  m.emplace("DSC_0178.jpg", false);
  m.emplace("DSC_0179.jpg", false);
  m.emplace("DSC_0180.jpg", true);
  m.emplace("DSC_0182.jpg", true);
  m.emplace("DSC_0184.jpg", true);
  m.emplace("DSC_0186.jpg", true);
  m.emplace("DSC_0130.jpg", false);
  m.emplace("DSC_0131.jpg", false);
  m.emplace("DSC_0206.jpg", false);
  m.emplace("DSC_0207.jpg", false);
  m.emplace("DSC_0208.jpg", false);
  m.emplace("DSC_0209.jpg", false);
  m.emplace("DSC_0212.jpg", false);
  m.emplace("DSC_0213.jpg", false);
  m.emplace("DSC_0214.jpg", false);
}

TEST(CornersTestSuite, TestShouldRotate) {
  const char *buffer = getcwd(NULL, 0);
  std::map<std::string, bool> expected;
  initRotationMap(expected);
  cv::Mat image;
  cv::Mat gray;
  musicocr::CornerFinder finder;
  for (const auto& it : expected) {
    const std::string test_file =
        std::string(buffer) + "/test/data/" + it.first;
    image = cv::imread(test_file);
    ASSERT_TRUE(image.data != NULL);
    resize(image, image, cv::Size(), 0.2, 0.2, cv::INTER_AREA);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    bool r = finder.shouldRotate(gray);
    EXPECT_EQ(it.second, r)
      << "wrong rotation for " << it.first << std::endl;
  }
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
