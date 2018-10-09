#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <unistd.h>

#include "corners.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

void initRegressionMap(std::map<std::string, std::vector<cv::Point>>& m) {
   m.emplace("sample1.jpg", std::vector<cv::Point>({
     cv::Point(25, 44), cv::Point(840,44), cv::Point(840,667),
     cv::Point(28,676)
   }));
   m.emplace("DSC_0177.jpg", std::vector<cv::Point>({
     cv::Point(5, 28), cv::Point(919,28), cv::Point(920,675),
     cv::Point(9,675)
   }));
   m.emplace("DSC_0178.jpg", std::vector<cv::Point>({
     cv::Point(0, 2), cv::Point(693,1), cv::Point(693,913),
     cv::Point(0,912)
   }));
   m.emplace("DSC_0179.jpg", std::vector<cv::Point>({
     cv::Point(68, 80), cv::Point(924,79), cv::Point(924,492),
     cv::Point(73,494)
   }));
   m.emplace("DSC_0180.jpg", std::vector<cv::Point>({
     cv::Point(40, 11), cv::Point(917,12), cv::Point(917,671),
     cv::Point(46,671)
   }));
   m.emplace("DSC_0181.jpg", std::vector<cv::Point>({
     cv::Point(0, 65), cv::Point(806,68), cv::Point(817,664),
     cv::Point(0,683)
   }));
   m.emplace("DSC_0182.jpg", std::vector<cv::Point>({
     cv::Point(16, 55), cv::Point(848,58), cv::Point(845,598),
     cv::Point(17,624)
   }));
   m.emplace("DSC_0184.jpg", std::vector<cv::Point>({
     cv::Point(50, 44), cv::Point(892,44), cv::Point(900,618),
     cv::Point(52,620)
   }));
   m.emplace("DSC_0186.jpg", std::vector<cv::Point>({
     cv::Point(28, 26), cv::Point(898,27), cv::Point(904,656),
     cv::Point(30,656)
   }));
   m.emplace("DSC_0130.jpg", std::vector<cv::Point>({
     cv::Point(0, 138), cv::Point(692,133), cv::Point(692,904),
     cv::Point(0,899)
   }));
   m.emplace("DSC_0131.jpg", std::vector<cv::Point>({
     cv::Point(0, 4), cv::Point(693,5), cv::Point(693,905),
     cv::Point(0,901)
   }));
   m.emplace("DSC_0206.jpg", std::vector<cv::Point>({
     cv::Point(0, 3), cv::Point(686,3), cv::Point(689,794),
     cv::Point(0,786)
   }));
   m.emplace("DSC_0207.jpg", std::vector<cv::Point>({
     cv::Point(0, 8), cv::Point(659,2), cv::Point(659,859),
     cv::Point(0,867)
   }));
   m.emplace("DSC_0208.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(744,0), cv::Point(744,679),
     cv::Point(16,683)
   }));
   m.emplace("DSC_0209.jpg", std::vector<cv::Point>({
     cv::Point(0, 0), cv::Point(661,0), cv::Point(667,693),
     cv::Point(0,693)
   }));
   m.emplace("DSC_0212.jpg", std::vector<cv::Point>({
     cv::Point(0, 1), cv::Point(655,6), cv::Point(659,876),
     cv::Point(4,876)
   }));
   // A regression on this would probably be an improvement.
   m.emplace("DSC_0213.jpg", std::vector<cv::Point>({
     cv::Point(0, 88), cv::Point(685,88), cv::Point(685,869),
     cv::Point(0,861)
   }));
   m.emplace("DSC_0214.jpg", std::vector<cv::Point>({
     cv::Point(14, 76), cv::Point(674,80), cv::Point(674,917),
     cv::Point(29,920)
   }));
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
