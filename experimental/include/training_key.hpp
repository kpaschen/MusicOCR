#ifndef training_key_hpp
#define training_key_hpp

#include <iostream>
#include <map>

namespace musicocr {

class TrainingKey {
public:
  TrainingKey();

  const std::string& getCategoryName(int basicCategory) const;
  int getCategoryForStatModel(int basicCategory) const;

private:
  // This maps the known categories to human-readable names.
  // Also serves as directory of which categories exist.
  std::map<int, std::string> trainingKey;

  // This says which training keys are considered the same
  // for stat model training.
  std::map<int, int> trainingKeyForStatModels;

  static std::string unknown_category;
};

}  // end namespace musicocr

#endif
