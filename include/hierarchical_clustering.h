#pragma once
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "nn_cluster.h"
#include <memory>
#define x(i) 2 * i
#define y(i) 2 * i + 1

typedef std::vector<double> point;
typedef unsigned int uint;
typedef std::pair<int, int> pair_int;


/**
* Hash function for the pair of integers.
*/
struct pairhash {
private:
  const size_t num = 65537;
public:
  inline size_t operator()(const std::pair<int, int> &x) const {
    return (x.first * num) ^ x.second;
  }
};

namespace std {
  template <>
  struct hash<pair_int> {
    const size_t num = 65537;
    inline size_t operator()(const std::pair<int, int> &x) const {
        return (x.first * num) ^ x.second;
    }
  };
}

class hierarchical_clustering {

private:
  int dimension;
  int size;
  std::vector<std::pair<pair_int, pair_int>> merges;
  double epsilon;
  double gamma;
  nnCluster nnc;
  double max_dist;
  double min_dist;
  double beta;
  std::unordered_set<std::pair<int, int>> unmerged_clusters;

  std::vector<int> pts_index;
  std::unordered_map<int, int> index_weight;
  std::unordered_map<pair_int, bool, pairhash> existed;
  std::unordered_set <pair_int> magic;
  double compute_min_dist();
  float * merge(float * mu_a, float * mu_b, int size_a, int size_b);
//  void read_file(const std::string);
  std::unordered_set<pair_int> helper(std::unordered_set<pair_int> &mp, float merge_value);
  std::unordered_map<pair_int, pair_int> dict;
  std::unordered_map<pair_int, std::pair<pair_int, pair_int>> rep;
public:
  hierarchical_clustering(float * data, int n, int d, double epsilon_, double gamma_);
  std::vector<std::pair<pair_int, pair_int>> get_merges() const;
  void print_merges();
  void build_hierarchy();
};
