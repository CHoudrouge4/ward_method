#include <iostream>
#include "flann/flann.hpp"
#include "flann/io/hdf5.h"
#include <fstream>
#include <stdlib.h>
#include <vector>

float * read_file(const std::string file_name, int &n, int &m) {
  std::ifstream in(file_name);
  in >> n >> m;
  float * array = (float *) malloc(n * m * sizeof(float));
  for (int i = 0; i < n; ++i) {
    for(int j = 0; j < m; ++j) {
      in >> *(array + i*m + j);
    }
  }
  in.close();
  return array;
}

int main () {

  int n;
  int d;
  float * result = read_file("data.in", n, d);
  flann::Matrix<float> dataset(result, n, d);
  for (int i = 0; i < n; ++i) {
     for (int j = 0; j < d; ++j) {
       std::cout << dataset[i][j] << ' ';
     }
     std::cout << '\n';
  }

  flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();

  std::vector<float> pt_(d);
  for(int i = 0; i < d; ++i)  {
     pt_[i] = 1.0;
     std::cout << "1d : " << pt_[i] << '\n';
  }

  flann::Matrix<float> pt(pt_.data(), 1, d);
  index.addPoints(pt);

  for (int j = 0; j < d; ++j) {
    std::cout << pt[0][j] << ' ';
  }
  std::cout << '\n';



  delete [] dataset.ptr();
  delete [] pt.ptr();
  return 0;
}
