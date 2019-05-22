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
  int k;
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
  std::uniform_real_distribution<> dis(-10.0, 10.0);
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
  nnCluster index(points, n, d, epsilon, gamma, 16, 4);
  //index.compute_min_dist();
  //test_add_delete_cluster(index, n, d);
}

void test_HC() {
  int n;
  int d;
  //float * points = generate_random_matrix( n, d);
  //std::vector<std::string> data = {"./data/iris", "./data/cancer", "./data/digits", "./data/boston"};

  const int trees = 8;
  const int leaves = 200;
  std::vector<std::string> data = {"./data/data10000_50_50"};
  float epsilon;
  std::cin >> epsilon;
  std::cout << "epsilon read" << std::endl;
  for(auto&& data_name : data) {
  //  std::string data_name = "./data/data10000_0_100";
    float * points = read_file(data_name + ".in", n , d);
    std::cout << "done reading" << std::endl;

    hierarchical_clustering hc(points, n, d, epsilon, 0.9, trees, leaves);
    std::cout << "done initializing" << std::endl;
    clock_t start = clock();
    hc.build_hierarchy();
    clock_t end = clock();
    std::cout << (float)(end - start)/CLOCKS_PER_SEC << std::endl;
    epsilon = epsilon * 100;
    std::string output_file = data_name + std::to_string((int)floor((epsilon))) + "_" + std::to_string(trees) + "_"  + std::to_string(leaves) + ".out";
    std::cout << output_file << std::endl;
    hc.print_file(output_file);
    epsilon /= 100;
  }
}

void the_big_exp() {
  int n;
  int d = 2;
  int k = 10;
  std::ofstream out("perf.txt", std::ios_base::app);
  std::vector<int> trees = {4 , 16};
  std::vector<int> leaves = {5, 128};
  std::vector<float> epsilons = {5};

  for (auto&& e: epsilons) {
    for (auto&& tr: trees) {
      for(auto&& l: leaves) {
        for (int i = 1000; i < 10000; i += 500) {
          out << e << ' ' << tr << ' ' << l << ' ' << i << ' ' << d;
          for (int j = 0; j < 10; ++j) {
            std::string data_name = "data" + std::to_string(i) + '_' + std::to_string(j) + '_' + std::to_string(d) + '_' + std::to_string(k);
            std::string file_name = "./data/" + data_name + ".in";
            float * points = read_file(file_name, n, d);
            //std::cout << "done reading" << std::endl;
            hierarchical_clustering hc(points, n, d, e, 0.9, tr, l);
            //std::cout << "done initializing" << std::endl;
            clock_t start = clock();
            hc.build_hierarchy();
            clock_t end = clock();
            out << ' ' << (float)(end - start)/CLOCKS_PER_SEC;
            float epsilon = e * 100;
            std::string output_file = data_name + '_' + std::to_string((int)floor((epsilon))) + "_" + std::to_string(tr) + "_"  + std::to_string(l) + ".out";
            std::cout << output_file << std::endl;
            hc.print_file(output_file);
          }
          out << std::endl;
        }
      }
    }
  }

  out.close();
}

float logb(float num, float base) {
  return std::log10(num)/ std::log10(base);
}

int main () {
  // test_data_structure();
  // test_HC();
  the_big_exp();
  return 0;
}

/**

flann::Matrix<float> POINTS(points, n, d);
const flann::AutotunedIndexParams params(0.8);
std::vector<flann::AutotunedIndex<flann::L2<float>> *> nn_data_structures;
flann::AutotunedIndex<flann::L2<float>> * ds = new flann::AutotunedIndex<flann::L2<float>> (POINTS, params);
nn_data_structures.push_back(ds);
nn_data_structures[0]->buildIndex();

float * new_cluster_ = generate_random_matrix(1, d);
flann::Matrix<float> q (new_cluster_, 1, d);
std::vector<std::vector<float>> dist;
std::vector<std::vector<size_t>> idx;

nn_data_structures[0]->knnSearch(q, idx, dist, 1, flann::SearchParams(32));
auto p = nn_data_structures[0]->getPoint(0);
std::cout << p[0] << std::endl;

**/
