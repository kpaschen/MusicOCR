#include "training_key.hpp"

namespace musicocr {

  std::string TrainingKey::unknown_category = "Unknown Category";

  TrainingKey::TrainingKey() {
    trainingKey.emplace(Category::eighthbreak, "eighth break");  // 'e'
    trainingKey.emplace(Category::quarterbreak, "quarter break");  // 'x'
    trainingKey.emplace(Category::barbreak, "2-4 beats break"); // 'b'
  
    // horizontal or curved lines
    trainingKey.emplace(Category::connector, "connector piece"); // 'c'
  
    trainingKey.emplace(Category::vertical, "vertical line"); // 'l'
    trainingKey.emplace(Category::bar, "bar line");
    trainingKey.emplace(Category::dot, "dot"); // 'd'
  
    // accidentals
    trainingKey.emplace(Category::flat, "flat");  // 'f'
    trainingKey.emplace(Category::sharp, "sharp");  // 's'
    trainingKey.emplace(Category::undoaccidental, "undo accidental"); // 'u'
  
    // filled or empty note head
    trainingKey.emplace(Category::notehead, "note head");  // 'h'
    // note head and vertical line
    trainingKey.emplace(Category::note, "note");

    // A thing to skip
    trainingKey.emplace(Category::speck, "speck"); // 'k'
  
    // This is mostly for badly segmented pieces.
    trainingKey.emplace(Category::multiple, "complex"); // 'm'
    trainingKey.emplace(Category::character, "character"); // 'a'
    trainingKey.emplace(Category::violinclef, "violin clef"); // 'g'
    trainingKey.emplace(Category::bassclef, "bass clef"); // 'i'
  
    // A small piece of a larger item that has no meaning by itself.
    trainingKey.emplace(Category::piece, "piece");  // 'p'

    // Now the projection map for the stat models.
    trainingKeyForStatModels.emplace(Category::character,
        TopLevelCategory::composite); // alphanum -> complex
    trainingKeyForStatModels.emplace(Category::barbreak,
        TopLevelCategory::composite); // bar break
    trainingKeyForStatModels.emplace(Category::eighthbreak,
        TopLevelCategory::composite); // eighth break
    trainingKeyForStatModels.emplace(Category::flat,
        TopLevelCategory::composite); // flat
    trainingKeyForStatModels.emplace(Category::violinclef,
        TopLevelCategory::composite); // violin clef
    trainingKeyForStatModels.emplace(Category::bassclef,
        TopLevelCategory::composite); // bass clef
    trainingKeyForStatModels.emplace(Category::multiple,
        TopLevelCategory::composite); // complex
    trainingKeyForStatModels.emplace(Category::note,
        TopLevelCategory::composite); // note
    trainingKeyForStatModels.emplace(Category::sharp,
        TopLevelCategory::composite); // sharp
    trainingKeyForStatModels.emplace(Category::undoaccidental,
        TopLevelCategory::composite); // undo accidental
    trainingKeyForStatModels.emplace(Category::quarterbreak,
        TopLevelCategory::composite); // quarter break
    trainingKeyForStatModels.emplace(Category::connector,
        TopLevelCategory::hline); // connector piece
    trainingKeyForStatModels.emplace(Category::dot,
        TopLevelCategory::round); // dot
    trainingKeyForStatModels.emplace(Category::notehead,
        TopLevelCategory::round); // note head
    trainingKeyForStatModels.emplace(Category::speck,
        TopLevelCategory::round); // speck
    trainingKeyForStatModels.emplace(Category::piece,
        TopLevelCategory::round); // piece
    trainingKeyForStatModels.emplace(Category::vertical,
        TopLevelCategory::vline); // vertical line
    trainingKeyForStatModels.emplace(Category::bar,
        TopLevelCategory::vline); // vertical line
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

int TrainingKey::getCategory(int basicCategory, KeyMode mode) const {
  switch(mode) {
    case basic: return basicCategory;
        break;
    case statmodel: return getCategoryForStatModel(basicCategory);
        break;
    default:
        return basicCategory;
        break;
  }
}

}  // namespace musicocr
