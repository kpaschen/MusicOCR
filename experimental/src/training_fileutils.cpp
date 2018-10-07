#include "training_fileutils.hpp"

namespace musicocr {

  using std::string;
  using std::vector;
  using std::map;
  using std::cout;
  using std::cerr;
  using std::endl;

bool SampleDataFiles::parseFilename(char* filename, char* imagesetname,
                   int *linenumber, int *index, int *xcoord, int *ycoord) {
  if (sscanf(filename, "%[^.].%d.%d.%d.%d.png", imagesetname,
             linenumber, index, xcoord, ycoord) == 5) {
    return true;
  }
  if (sscanf(filename, "responses.%[^.].%d", imagesetname,
             linenumber) == 2) {
      *index = -1;
      return true;
  }
  return false;
}

void SampleDataFiles::readFiles(const string& dirname, SampleData& collector) {
  readFiles(dirname, "", collector);
}

void SampleDataFiles::readFiles(const string& dirname, const string& dataset,
                                SampleData& collector) {
  DIR* dirp = opendir(dirname.c_str());
  struct dirent *dp;
  char trainingset[50];
  int lno, idx, xcoord, ycoord;

  while((dp = readdir(dirp)) != NULL) {
    char* filename = dp->d_name;
    if (!parseFilename(filename, trainingset, &lno, &idx, &xcoord, &ycoord)) {
      cerr << "Could not parse file name " << filename << ", skipping." << endl;
      continue;
    }
    if (dataset != "" && trainingset != dataset) {
      continue;
    }
    if (idx == -1) {  // responses file
      vector<int> resp;
      char rfilename[dirname.size() + strlen(filename) + 2];
      sprintf(rfilename, "%s/%s", dirname.c_str(), filename);
      std::ifstream r;
      r.open(rfilename);
      // could check r.good() here but then if the file does not exist
      // there's a bigger logical error and we'll fail downstream anyway.
      char l[9];
      int idx, cat;
      while (!r.eof()) {
        r.getline(l, 9);
        if (sscanf(l, "%d: %d", &idx, &cat) != 2) {
          cerr << "Bad response line: " << l << ", skipping." << endl;
          continue;
        }
        resp.push_back(cat);
      }
      if (responses.find(trainingset) == responses.end()) {
        responses.emplace(trainingset, map<int, vector<int>>());
      }
      map<int, vector<int>>& rmap = responses.find(trainingset)->second;
      // Not guarding against duplicate line numbers.
      cout << "adding " << resp.size() << " labels for line " << lno
           << " of " << trainingset << endl;
      rmap.insert(std::pair<int, vector<int>>(lno, resp));
    } else {  // sample image
      if (datasets.find(trainingset) == datasets.end()) {
        datasets.emplace(trainingset,
                         map<int, vector<std::pair<int, int>>>());
      }
      map<int, std::vector<std::pair<int, int>>>& tset =
        datasets.find(trainingset)->second;
      if (tset.find(lno) == tset.end()) {
        tset[lno] = std::vector<std::pair<int, int>>();
      }
      std::vector<std::pair<int, int>>& coords = tset.find(lno)->second;
      coords.emplace_back(xcoord, ycoord);
    }
  }
  for (const auto& sample : datasets) {
    const string& name = sample.first;
    const map<int, vector<std::pair<int, int>>>& tset = sample.second;
    const map<int, vector<int>>& lineresponses = responses[name];
    char rfilename[dirname.size() + name.size() + 20];
    for (const auto& tset_iter : tset) {
      // i is the line index, l is the number of samples for that line.
      const int i = tset_iter.first;
      const int l = tset_iter.second.size();
      const vector<int>& linelabels = lineresponses.find(i)->second;
      if (l != linelabels.size()) {
        cerr << "Wrong number of labels for line " << i << " of "
             << name << ": expected " << l << " but got " << linelabels.size()
             << endl;
        continue;
      }
      for (size_t j = 0; j < l; j++) {
        // TODO: to be really safe on sprintf here, should make sure
        // i and j aren't too big: usually i is at most 12 and j might reach 100.
        // rfilename isn't sized for more.
        const std::pair<int, int>& coords = tset_iter.second[j];
        sprintf(rfilename, "%s/%s.%d.%d.%d.%d.png", dirname.c_str(),
                name.c_str(), i, j, coords.first, coords.second);
        // This will dump core if the file does not exist, but then that
        // would be a major logical error.
        cv::Mat smat = cv::imread(rfilename, 0);

        // Add smat and linelabels[j] to data collector
        collector.addTrainingData(smat, linelabels[j], xcoord, ycoord);
      }
    }
  }
}
}  // namespace
