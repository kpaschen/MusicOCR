#include <gtest/gtest.h>
#include <iostream>
#include <unistd.h>

#include "corners.hpp"
#include "structured_page.hpp"
#include "opencv2/opencv.hpp"

void initVoiceMap(std::map<std::string, std::vector<int>>& m) {
  m.emplace("sample1.jpg",
    std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  // 363 horizontal, 68 vertical lines
  m.emplace("DSC_0177.jpg",
    std::vector<int>({4, 4, 4, 1}));
  // 399 horizontal, 21 vertical lines
  m.emplace("DSC_0178.jpg",
  // 200 horizontal, 4 vertical lines
    std::vector<int>({2, 3, 3, 1, 1, 1, 1}));
  m.emplace("DSC_0179.jpg",
  // 397 horizontal, 39 vertical lines
    std::vector<int>({1, 1, 1, 1, 1}));
  m.emplace("DSC_0182.jpg",
    std::vector<int>({1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  m.emplace("DSC_0184.jpg",
    // should be: std::vector<int>({1, 1, 2, 2, 2, 2, 2, 2, 1, 1}));
    std::vector<int>({1, 1, 4, 2, 4, 1, 1}));
  // 363 horizontal, 10 vertical lines
  m.emplace("DSC_0186.jpg",
    std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  // 223 horizontal, 49 vertical lines
  m.emplace("DSC_0130.jpg",
    std::vector<int>({1, 1, 1, 1, 1, 1, 1}));
  // 206 horizontal, 41 vertical lines
  m.emplace("DSC_0131.jpg",
    std::vector<int>({1, 1, 1, 1, 1}));
  m.emplace("DSC_0206.jpg",
    std::vector<int>({2, 2, 2, 1, 2}));
  m.emplace("DSC_0207.jpg",
   // should be: std::vector<int>({2, 2, 2, 2, 1, 1, 1, 1}));
    std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0208.jpg",
    // should be: std::vector<int>({2, 2, 2, 2}));
    std::vector<int>({2, 1, 1, 1, 1, 2}));
  m.emplace("DSC_0209.jpg",
    // should be: std::vector<int>({2, 2, 2, 2, 1, 1}));
    std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0212.jpg",
    // should be: std::vector<int>({2, 2, 1, 1}));
    std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0213.jpg",
    // std::vector<int>({2, 2, 2, 2}));
    std::vector<int>({1, 2, 2, 2, 1, 1, 1}));
  m.emplace("DSC_0214.jpg",
    // std::vector<int>({2, 2, 2, 2}));
    std::vector<int>({1, 1, 2, 4, 2, 1}));
}

TEST(StructuredPageTestSuite, TestVoiceGrouping) {
  const char *buffer = getcwd(NULL, 0);
  std::map<std::string, std::vector<int>> expected;
  initVoiceMap(expected);
  musicocr::CornerFinder cornerFinder;
  for (const auto& it : expected) {
    cv::Mat image, gray, tmp;
    const std::string test_file =
    std::string(buffer) + "/test/data/" + it.first;
    image = cv::imread(test_file);
    ASSERT_TRUE(image.data != NULL);
    resize(image, image, cv::Size(), 0.2, 0.2, cv::INTER_AREA);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cornerFinder.adjust(gray, tmp);
    musicocr::Sheet sheet;

    cv::Mat clines = tmp.clone();
    std::vector<cv::Vec4i> hlines = sheet.find_lines(clines);
    std::vector<cv::Vec4i> vlines = sheet.findVerticalLines(clines);
    std::cout << it.first << ": " << hlines.size() << " horizontal and "
         << vlines.size() << " vertical lines." << std::endl;
    
    sheet.analyseLines(hlines, vlines, tmp);
    std::vector<int> voices = sheet.getSheetInfo();
    sheet.printSheetInfo();
    EXPECT_EQ(voices.size(), it.second.size()) << it.first
              << ": wrong number of voices." << std::endl;
  }
}

//int main(int argc, char **argv) {
//  testing::InitGoogleTest(&argc, argv);
//  return RUN_ALL_TESTS();
//}
