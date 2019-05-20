#pragma once
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "nn_cluster.h"
#include <memory>

typedef std::vector<double> point;
typedef unsigned int uint;
typedef std::pair<int, int> pair_int;

/**
* Hash function for the pair of integers.
*/
class hierarchical_clustering {

private:
  int dimension;
  int size;
  float epsilon;
  float gamma;
  double max_dist;
  double min_dist;
  double beta;

  nnCluster nnc;

  bool stop = false;

  std::vector<std::pair<pair_int, pair_int>> merges;

  std::unordered_map<int, int> index_weight;
  std::unordered_map<pair_int, bool, pairhash> existed;
  std::unordered_set<pair_int> magic;
  std::unordered_set<std::pair<int, int>> unmerged_clusters;

  float * merge(float * mu_a, float * mu_b, int size_a, int size_b);

  std::unordered_set<pair_int> helper(std::unordered_set<pair_int> &mp, float merge_value);

  std::unordered_map<pair_int, pair_int> dict;

  std::unordered_map<pair_int, std::pair<pair_int, pair_int>> rep;

  std::vector<std::tuple<pair_int, pair_int, pair_int>> output;

  std::vector<pair_int> to_erase;
public:
  hierarchical_clustering(float * data, int n, int d, float epsilon_, float gamma_, int tree_number, int visited_leaf);
  std::vector<std::pair<pair_int, pair_int>> get_merges() const;
  void print_merges();
  void build_hierarchy();
  void print_file(const std::string filename);
};
