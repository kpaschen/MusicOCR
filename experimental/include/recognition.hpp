#ifndef recognition_hpp
#define recognition_hpp

#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/text.hpp>

namespace musicocr {

class Scanner {
  public:
    void process(const cv::Mat&) const;  
};

}  // namespace musicocr

#endif
