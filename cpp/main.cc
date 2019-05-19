#include <iostream>
#include "flann/flann.hpp"
#include "flann/io/hdf5.h"
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <random>
#include <dlfcn.h>
#include "nn_cluster.h"
#include "hierarchical_clustering.h"
#include <time.h>
#include <limits>
#include <cmath>


clock_t start;
clock_t current;
clock_t elapsed = 0;

inline void begin_record_time() { start = clock(); current = start; }
inline void get_current_time() { current = clock(); }
inline clock_t elapsed_time() { return current - start;}

float power(float x, int y) {
    if (y == 0)
        return 1;
    else if (y % 2 == 0)
        return power(x, y / 2) * power(x, y / 2);
    else
        return x * power(x, y / 2) * power(x, y / 2);
}

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

float compute_min_distance(float * array, int n, int m) {
  float result = std::numeric_limits<float>::max();
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      float dist = 0.0;
      for (int k = 0; k < m; ++k) {
        float tmp = (*(array + i * m + k)) - (* (array + j * m + k));
        dist += tmp * tmp;
      }
      if(dist < result) result = dist;
    }
  }
  return result;
}

void print_array (float * array, int n, int m) {
  std::cout << "[" << ' ';
  for (int i = 0; i < n; ++i) {
    std::cout << '[';
    for(int j = 0; j < m; ++j) {
      std::cout << *(array + i*m + j) << ' ';
    }
    std::cout << ']' << '\n';
  }
  std::cout << "]" << '\n';
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

void print_matrix (const flann::Matrix<float> &dataset, const int n, const int d) {
  for (int i = 0; i < n; ++i) {
     for (int j = 0; j < d; ++j) {
       std::cout << dataset[i][j] << ' ';
     }
     std::cout << '\n';
  }
}

extern "C" typedef double (*func_t)(int n, int d, void * array);

void test_add_delete_cluster(nnCluster &index, const int n, const int d) {

  std::random_device rd;
  std::mt19937 gen(rd());
  int n_d_s = index.get_number_of_data_structures();
  std::uniform_int_distribution<> dis(1, n_d_s);
  begin_record_time();
  while(elapsed_time() < 100000000) {
    float * new_cluster_ = generate_random_matrix(1, d);
    int random_size = dis(gen);
    flann::Matrix<float> new_cluster(new_cluster_, 1, d);

    std::cout << ">>>> the random size is " << random_size << std::endl;;
    index.add_new_cluster(new_cluster, random_size);


    auto t = index.query(new_cluster, random_size, true);
    std::cout << "query result " << std::get<0>(t) << ' ' << std::get<1>(t) << ' ' << std::get<2>(t) << '\n';
    float * res = index.get_point(std::get<0>(t), std::get<2>(t));
    float dist = 0;
    for (int i = 0; i < d; ++i) {
      std::cout << new_cluster[0][i] << ' ' << *(new_cluster_ + i) << ' ' << *(res + i) << std::endl;
      float tmp = (*(new_cluster_ + i) -  (*(res + i)));
      dist += tmp * tmp;
    }
    std::cout << "distance "<< dist << '\n';
    for(int i = 0; i < d; ++i) {
      assert(*(new_cluster_ + i) == *(res + i));
    }
    get_current_time();
  }
}

void test_data_structure() {
  std::cout << "testing the data structure" << '\n';
  int n = 5;
  int d = 2;
  float * points = generate_random_matrix(n, d);
  double epsilon = 0.5;
  double gamma = 0.9;
  nnCluster index(points, n, d, epsilon, gamma);
  //index.compute_min_dist();
  //test_add_delete_cluster(index, n, d);
}

void test_HC() {
  int n;
  int d;
  //float * points = generate_random_matrix( n, d);
  float * points = read_file("KDD.in", n , d);

  //print_array(points, n, d);

  hierarchical_clustering hc(points, n, d, 0.5, 0.9);

  //free(points);
  clock_t start = clock();
  hc.build_hierarchy();
  clock_t end = clock();
  std::cout << (float)(end - start)/CLOCKS_PER_SEC << std::endl;
  // auto merges = hc.get_merges();
  // for(auto p : merges) {
  //   std::cout << (p.first).first << ' ' << (p.first).second << " # " << (p.second).first << ' ' << (p.second).second << '\n';
  // }

  hc.print_merges();
}

float logb(float num, float base) {
  return std::log10(num)/ std::log10(base);
}

int main () {
  //test_data_structure();
  test_HC();

  // for(int i = 1; i <= 150; ++i) {
  //   std::cout << "size " << i <<  " bucket " << floor(logb(i, 1.5)) << " bucket size " << power(1.5, floor(logb(i, 1.5))) << std::endl;
  // }



  // int n = 5;
  // int d = 2;
  // float * result = generate_random_matrix(n, d);
  //
  // flann::Matrix<float> dataset(result, n, d);
  // print_matrix(dataset, n, d);
  //
  // flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(4));
  // index.buildIndex();
  //
  // std::vector<float> pt_(d);
  // for(int i = 0; i < d; ++i)  {
  //    pt_[i] = dataset[0][i];
  //    std::cout << "1d : " << pt_[i] << '\n';
  // }
  //
  //  void * lib = dlopen("/home/hussein/projects/m2_thesis/ward_method/lib/librms.so", RTLD_LAZY);
  //  func_t func = (func_t)dlsym( lib, "radius_min_circle");
  //  double radius = func(n, d, (void *) result);
  //  std::cout << "radius " << radius << '\n';
  //
  // flann::Matrix<float> query(pt_.data(), 1, d);
  //
  // std::vector<std::vector<int>> indices;
  // std::vector<std::vector<float>> dists;
  //
  // index.knnSearch(query, indices, dists, 1,  flann::SearchParams(128));
  //
  // printing_result(indices, dists);
  //
  // index.removePoint(indices[0][0]);
  //
  // print_matrix(dataset, n, d);
  //
  // indices.clear();
  // dists.clear();
  //
  // index.knnSearch(query, indices, dists, 1,  flann::SearchParams(128));
  // printing_result(indices, dists);
  //
  // index.addPoints(query);
  //
  // index.knnSearch(query, indices, dists, 1,  flann::SearchParams(128));
  // printing_result(indices, dists);
  //
  // index.removePoint(indices[0][0]);
  //
  // index.knnSearch(query, indices, dists, 1,  flann::SearchParams(128));
  // printing_result(indices, dists);
  // // index.addPoints(pt);
  // //
  // // for (int j = 0; j < d; ++j) {
  // //   std::cout << pt[0][j] << ' ';
  // // }
  // // std::cout << '\n';
  // //
  // //
  //
  // delete [] dataset.ptr();
  // delete [] query.ptr();
  return 0;
}
