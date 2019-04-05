#include "ward.h"

/*
* replace the matrix by a map(char, char) to double
*
*/

ward::ward(const std::string file_name) : HC(file_name) {}

void ward::run() {
  uint n = pts.size() / dimension;
  M = std::vector<point>(n, std::vector<double>(n));
  for(uint i = 0; i < n; ++i) {
    M[i][i] = 0;
    for(uint j = i + 1; j < n; ++j) {
      M[i][j] = M[j][i] = distance(i, j);
    }
  }

}
