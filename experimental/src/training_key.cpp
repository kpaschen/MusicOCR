#include "training_key.hpp"

namespace musicocr {

  std::string TrainingKey::unknown_category = "Unknown Category";

  TrainingKey::TrainingKey() {
    trainingKey.emplace(101, "eighth break");  // 'e'
    trainingKey.emplace(120, "quarter break");  // 'x'
    trainingKey.emplace(98, "2-4 beats break"); // 'b'
  
    // horizontal or curved lines
    trainingKey.emplace(99, "connector piece"); // 'c'
  
    trainingKey.emplace(108, "vertical line"); // 'l'
    trainingKey.emplace(100, "dot"); // 'd'
  
    // accidentals
    trainingKey.emplace(102, "flat");  // 'f'
    trainingKey.emplace(115, "sharp");  // 's'
    trainingKey.emplace(117, "undo accidental"); // 'u'
  
    // filled or empty note head
    trainingKey.emplace(104, "note head");  // 'h'
    // note head and vertical line
    trainingKey.emplace(110, "note");

    // A thing to skip
    trainingKey.emplace(107, "speck"); // 'k'
  
    // This is mostly for badly segmented pieces.
    trainingKey.emplace(109, "complex"); // 'm'
    trainingKey.emplace(97, "character"); // 'a'
    trainingKey.emplace(103, "violin clef"); // 'g'
    trainingKey.emplace(105, "bass clef"); // 'i'
  
    // A small piece of a larger item that has no meaning by itself.
    trainingKey.emplace(112, "piece");  // 'p'

    // Now the projection map for the stat models.
    trainingKeyForStatModels.emplace(97, 109); // alphanum -> complex
    trainingKeyForStatModels.emplace(98, 109); // bar break
    trainingKeyForStatModels.emplace(101, 109); // eighth break
    trainingKeyForStatModels.emplace(102, 109); // flat
    trainingKeyForStatModels.emplace(103, 109); // violin clef
    trainingKeyForStatModels.emplace(105, 109); // bass clef
    trainingKeyForStatModels.emplace(109, 109); // complex
    trainingKeyForStatModels.emplace(110, 109); // note
    trainingKeyForStatModels.emplace(115, 109); // sharp
    trainingKeyForStatModels.emplace(117, 109); // undo accidental
    trainingKeyForStatModels.emplace(120, 109); // quarter break
    trainingKeyForStatModels.emplace(99, 99); // connector piece
    trainingKeyForStatModels.emplace(100, 100); // dot
    trainingKeyForStatModels.emplace(104, 100); // note head
    trainingKeyForStatModels.emplace(107, 100); // speck
    trainingKeyForStatModels.emplace(112, 100); // piece
    trainingKeyForStatModels.emplace(108, 108); // vertical line
  }

  const std::string& TrainingKey::getCategoryName(int basicCategory) const {
    const auto& cat = trainingKey.find(basicCategory);
    if (cat == trainingKey.end()) {
      return unknown_category;
    }
    return cat->second;
  }

  int TrainingKey::getCategoryForStatModel(int basicCategory) const {
    const auto& cat = trainingKeyForStatModels.find(basicCategory);
    if (cat == trainingKeyForStatModels.end()) {
      return -1;
    }
    return cat->second;
  }

}  // namespace musicocr
