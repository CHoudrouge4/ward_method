#pragma once
#include "flann/flann.hpp"
#include "flann/io/hdf5.h"
#include <tuple>
#include <unordered_map>

typedef std::pair<int, int> pair_int;

namespace std {
  template <>
  struct hash<pair_int> {
    const size_t num = 65537;
    inline size_t operator()(const std::pair<int, int> &x) const {
        return (x.first * num) ^ x.second;
    }
  };
}

class nnCluster {

public:

  // Constructor
  nnCluster (float * points_, int n, int d, double epsilon_, double gamma_);

  std::tuple<int, float, int> query (const flann::Matrix<float> &query, int query_size, bool itself=false);

  /**
  * This function adds a cluster to the data clusters of the clusters of size
  * cluster size
  *
  */
  int add_cluster(flann::Matrix<float> &cluster, const int cluster_size);

  int add_cluster(flann::Matrix<float> &cluster, int cluster_size, int old_index, int new_index );

  void delete_cluster(int idx, int size);

  float * get_point(int idx, int size);

  int get_number_of_data_structures() const;

  float compute_min_dist();

  pair_int get_index(int index, int weight);

  ~nnCluster();
private:
  flann::Matrix<float> points;
  int size, dimension, number_of_data_structure;
  double epsilon, gamma;

  std::vector<flann::Index<flann::L2<float>>> nn_data_structures;
  std::vector<bool> build;
  std::vector<int> sizes;
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<float>> dists;

  float distance(int size_a, int size_b, float dist);

  std::unordered_map<std::pair<int, int>, int> cluster_weight;
  std::unordered_map<pair_int, pair_int> dict;
};
