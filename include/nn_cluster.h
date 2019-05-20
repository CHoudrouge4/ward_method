#pragma once
#include "flann/flann.hpp"
#include "flann/io/hdf5.h"
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <memory>

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

struct pairhash {
private:
  const size_t num = 65537;
public:
  inline size_t operator()(const std::pair<int, int> &x) const {
    return (x.first * num) ^ x.second;
  }
};

class nnCluster {

public:

  // Constructor
  nnCluster (float * points_, int n, int d, float epsilon_, float gamma_, const size_t &tree_number, int visited_leaf_);

  std::tuple<int, float, int> query (const flann::Matrix<float> &query, int query_size, bool itself=false);

  /**
  * This function adds a cluster to the data clusters of the clusters of size
  * cluster size
  *
  */
  int add_cluster(const flann::Matrix<float> &cluster, const int cluster_size);

  int add_cluster(const flann::Matrix<float> &cluster, int cluster_size, int old_index, int new_index );

  std::tuple<int, float, int> add_new_cluster(const flann::Matrix<float> &cluster, const int cluster_size);

  void delete_cluster(int idx, int size);

  float * get_point(int idx, int size);

  int get_number_of_data_structures() const;

  float compute_min_dist(std::unordered_set<pair_int> &unmerged_clusters, std::unordered_map<pair_int, bool, pairhash> &existed);

  pair_int get_index(int index, int weight);

  void update_dict(int new_idx, int new_weight, int old_idx, int old_weight);

  void update_size(int ds_index, int new_index, int size);

  ~nnCluster();
private:
  flann::Matrix<float> points;
  int size, dimension, number_of_data_structure, visited_leaf;
  double epsilon, gamma;


  std::vector<flann::Index<flann::L2<float>>> nn_data_structures;
  std::vector<bool> build;
  std::vector<int> sizes;
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<float>> dists;

  float distance(int size_a, int size_b, float dist);

  std::unordered_map<std::pair<int, int>, int> cluster_weight;
  std::unordered_map<pair_int, pair_int> dict;
  std::unordered_map<pair_int, int> idx_index;
};
