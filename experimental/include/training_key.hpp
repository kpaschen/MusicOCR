#ifndef training_key_hpp
#define training_key_hpp

#include <iostream>
#include <map>

namespace musicocr {

class TrainingKey {
public:
  enum KeyMode {
    basic, statmodel
  };

  enum TopLevelCategory {
    unknown = 0, vline = 108, round = 100, hline = 99, composite = 109
  };
  
  enum Category {
    undefined = 0, character = 97, barbreak = 98, connector = 99,
    dot = 100, eighthbreak = 101,
    flat = 102, violinclef = 103, notehead = 104,
    bassclef = 105, bar = 106, speck = 107, vertical = 108, multiple = 109,
    note = 110, piece = 112, sharp = 115, undoaccidental = 117,
    quarterbreak = 120
  };

  TrainingKey();

  const std::string& getCategoryName(int basicCategory) const;
  int getCategory(int basicCategory, KeyMode mode) const;
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
