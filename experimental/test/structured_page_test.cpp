#include <gtest/gtest.h>
#include <iostream>
#include <unistd.h>

#include "corners.hpp"
#include "shapes.hpp"
#include "structured_page.hpp"
#include "opencv2/opencv.hpp"

void initLineCountMap(std::map<std::string, int>& m) {
  m.emplace("sample1.jpg", 12);
  m.emplace("DSC_0177.jpg", 11);  // fixme
  m.emplace("DSC_0178.jpg", 12);
  m.emplace("DSC_0179.jpg", 5);
  m.emplace("DSC_0182.jpg", 12);
  m.emplace("DSC_0184.jpg", 7);
  m.emplace("DSC_0186.jpg", 12);
  m.emplace("DSC_0130.jpg", 6);
  m.emplace("DSC_0131.jpg", 6);
  m.emplace("DSC_0206.jpg", 11);
  m.emplace("DSC_0207.jpg", 12);
  m.emplace("DSC_0208.jpg", 9);
  m.emplace("DSC_0209.jpg", 10);
  m.emplace("DSC_0212.jpg", 13);  // fixme
  m.emplace("DSC_0213.jpg", 11);
  m.emplace("DSC_0214.jpg", 5);
}

TEST(StructuredPageTestSuite, TestLineFinding) {
  const char *buffer = getcwd(NULL, 0);
  musicocr::CornerFinder cornerFinder;
  std::map<std::string, int> expected;
  initLineCountMap(expected);
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
    std::vector<cv::Rect> hlines = sheet.find_lines_outlines(tmp);
    sheet.createSheetLines(hlines, tmp);
    EXPECT_EQ(sheet.getLineCount(), it.second) << it.first
              << ": wrong number of lines." << std::endl;
  }
}

#if 0
void initVoiceMap(std::map<std::string, std::vector<int>>& m) {
  m.emplace("sample1.jpg",
    // Should be: all single lines.
    // std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    std::vector<int>({1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0177.jpg",
    std::vector<int>({4, 4, 4}));
  m.emplace("DSC_0178.jpg",
    std::vector<int>({2, 3, 3, 1, 1, 1, 1}));
  m.emplace("DSC_0179.jpg",
    std::vector<int>({1, 1, 1, 1}));
  m.emplace("DSC_0182.jpg",
    std::vector<int>({4, 1, 1, 1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0184.jpg",
    //should be: std::vector<int>({1, 2, 2, 2, 2, 2, 1}));
    std::vector<int>({1, 4, 2, 1}));
  m.emplace("DSC_0186.jpg",
    //std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    std::vector<int>({3, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0130.jpg",
    std::vector<int>({1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0131.jpg",
    //std::vector<int>({1, 1, 1, 1, 1}));
    std::vector<int>({1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0206.jpg",
    std::vector<int>({2, 2, 2, 1, 2}));
  m.emplace("DSC_0207.jpg",
   // should be: std::vector<int>({2, 2, 2, 2, 1, 1, 1, 1}));
    std::vector<int>({2, 1, 1, 2, 2, 1, 1, 1}));
  m.emplace("DSC_0208.jpg",
    std::vector<int>({2, 2, 2, 2}));
  m.emplace("DSC_0209.jpg",
    std::vector<int>({2, 2, 2, 2, 1, 1}));
  m.emplace("DSC_0212.jpg",
    // should be: std::vector<int>({2, 2, 1, 1}));
    std::vector<int>({2, 2, 1, 1, 1, 1, 1, 1, 1, 1}));
  m.emplace("DSC_0213.jpg",
    // std::vector<int>({2, 2, 2, 2}));
    std::vector<int>({1, 2, 2, 2, 1, 1, 1}));
  //m.emplace("DSC_0214.jpg",
    // std::vector<int>({2, 2, 2, 2}));
  //  std::vector<int>({1, 1, 2, 4, 2, 1}));
}
// TODO: this needs to be tested as part of global
// shape finding.
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

    std::vector<cv::Rect> hlines = sheet.find_lines_outlines(tmp);
    std::vector<cv::Vec4i> vlines = sheet.findVerticalLines(tmp);
    std::cout << it.first << ": " << hlines.size() << " horizontal and "
         << vlines.size() << " vertical lines." << std::endl;
    
    sheet.analyseLines(hlines, vlines, tmp);
    // Each vector element is one line group.
    // Each line group is a vector of sheet line indices.
    std::vector<std::vector<int>> voices = sheet.getSheetInfo();
    sheet.printSheetInfo();
    // TODO: this only verifies the number of line groups found,
    // it should also look at the sheet line indices within each group.
    EXPECT_EQ(voices.size(), it.second.size()) << it.first
              << ": wrong number of voices." << std::endl;
  }
}
#endif

