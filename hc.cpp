#include "hc.h"
#include <fstream>
#include <cmath>
#define x(i) 2 * i
#define y(i) 2 * i + 1

double HC::distance(uint i, uint j) {
  double dx = pts[x(i)] - pts[x(j)];
  double dy = pts[y(i)] - pts[y(j)];
  double d2 = dx * dx + dy * dy;
  return sqrt(d2);
}

HC::HC(const std::string file_name) {
  read_file(file_name);
}

void HC::read_file(const std::string file_name) {
  std::ifstream in(file_name);
  int number_points;
  in >> number_points >> dimension;
  pts = point(number_points * dimension);
  for(int i = 0; i < dimension * number_points; ++i) in >> pts[i];
  in.close();
}
