#ifndef utils_hpp
#define utils_hpp

#include "opencv2/imgproc.hpp"

namespace musicocr {
// Very basic heuristic for orientation.
short lineIsHorizontal(const cv::Vec4i& vec);

// Returns true if one and two could be combined into one
// (vertical or horizontal) line. In theory, this could
// be achieved using hough parameters, but in practice,
// I couldn't get that to work.
bool maybeSameLine(const cv::Vec4i& one, const cv::Vec4i& two,
                   int maxDist);

// comparators for sorting.
bool moreTop(const cv::Vec4i& line1, const cv::Vec4i& line2);
bool moreBottom(const cv::Vec4i& line1, const cv::Vec4i& line2);
bool moreLeft(const cv::Vec4i& line1, const cv::Vec4i& line2);
bool moreRight(const cv::Vec4i& line1, const cv::Vec4i& line2);

bool rectLeft(const cv::Rect&, const cv::Rect&);
}  // end namespace musicocr
#endif
