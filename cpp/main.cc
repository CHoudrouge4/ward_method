#include <iostream>
#include "flann/flann.hpp"
#include "flann/io/hdf5.h"
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <random>

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

float * generate_random_matrix(int n, int m) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(1.0, 5.0);
  float * array = (float *) malloc(n * m * sizeof(float));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      * (array + i * m + j) = dis(gen);
    }
  }
  return array;
}

void printing_result (const std::vector<std::vector<int>> &indices, const std::vector<std::vector<float>> &dists) {
  std::cout << "Indices" << '\n';
  for (auto e: indices) {
      for (auto ee: e) std::cout << ee << '\n';
  }
  std::cout << "Distance" << '\n';
  for (auto d_: dists) {
      for (auto d: d_) std::cout << d << '\n';
  }
}

void print_matrix (flann::Matrix<float> &dataset, int n, int d) {
  for (int i = 0; i < n; ++i) {
     for (int j = 0; j < d; ++j) {
       std::cout << dataset[i][j] << ' ';
     }
     std::cout << '\n';
  }
}

int main () {

  int n = 5;
  int d = 2;
  float * result = generate_random_matrix(n, d);

  flann::Matrix<float> dataset(result, n, d);
  print_matrix(dataset, n, d);

  flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();

  std::vector<float> pt_(d);
  for(int i = 0; i < d; ++i)  {
     pt_[i] = dataset[0][i];
     std::cout << "1d : " << pt_[i] << '\n';
  }

  flann::Matrix<float> query(pt_.data(), 1, d);
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<float>> dists;

  index.knnSearch(query, indices, dists, 1,  flann::SearchParams(128));

  printing_result(indices, dists);

  index.removePoint(indices[0][0]);

  print_matrix(dataset, n, d);

  indices.clear();
  dists.clear();

  index.knnSearch(query, indices, dists, 1,  flann::SearchParams(128));
  printing_result(indices, dists);

  index.addPoints(query);

  index.knnSearch(query, indices, dists, 1,  flann::SearchParams(128));
  printing_result(indices, dists);

  index.removePoint(indices[0][0]);

  index.knnSearch(query, indices, dists, 1,  flann::SearchParams(128));
  printing_result(indices, dists);
  // index.addPoints(pt);
  //
  // for (int j = 0; j < d; ++j) {
  //   std::cout << pt[0][j] << ' ';
  // }
  // std::cout << '\n';
  //
  //

  delete [] dataset.ptr();
  delete [] query.ptr();
  return 0;
}
