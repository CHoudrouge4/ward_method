#include "ward.h"

/*
* replace the matrix by a map(char, char) to double
*
*/

ward::ward(const std::string file_name) : hierarchical_clustering(file_name) {}

void ward::run() {
  falconn::LSHConstructionParameters params;

  uint n = pts.size(); // dimension;
  M = std::vector<point>(n, std::vector<double>(n));
  for(uint i = 0; i < n; ++i) {
    for(uint j = i + 1; j < n; ++j) {
  //    std::pair<, > p = std::make_pair(i, j);
    //  mp[p] = mp[p] = distance(i, j);
    }
  }
}
