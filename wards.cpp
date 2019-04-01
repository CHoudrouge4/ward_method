#include "wards.h"
#include <fstream>
#include <cmath>
#define x(i) 2 * i
#define y(i) 2 * i + 1

double wards::distance(uint i, uint j) {
  double dx = pts[x(i)] - pts[x(j)];
  double dy = pts[y(i)] - pts[y(j)];
  double d2 = dx * dx + dy * dy;
  return sqrt(d2);
}

wards::wards(const std::string file_name) {
  read_file(file_name);
}

void wards::read_file(const std::string file_name) {
  std::ifstream in(file_name);
  int number_points;
  in >> number_points >> dimension;
  pts = point(number_points * dimension);
  for(int i = 0; i < dimension * number_points; ++i) in >> pts[i];
  in.close();
}

void wards::clusster() {
  uint n = pts.size() / dimension;
  M = std::vector<point>(n, std::vector<double>(n));
  for(uint i = 0; i < n; ++i) {
    M[i][i] = 0;
    for(uint j = i + 1; j < n; ++j) {
      M[i][j] = M[j][i] = distance(i, j);
    }
  }
}
