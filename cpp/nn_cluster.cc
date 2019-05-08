#include "nn_cluster.h"
#include <cmath>
#include <limits>

float log_base(float num, float base) {
  return log(num)/ log(base);
}

inline float distance(int size_a, int size_b, float dist) {
  float coef = size_a * size_b;
  coef = coef/(size_a + size_b);
  return coef * dist;
}

nnCluster::nnCluster (float * points_, int n, int d, double epsilon_, double gamma_):
      points(points_, n, d), size(n), dimension(d), epsilon(epsilon_) , gamma(gamma_) {
  number_of_data_structure = ceil(log_base(size, 1 + epsilon));
  flann::Index<flann::L2<float>> index(points, flann::KDTreeIndexParams(4));
  build = std::vector<bool>(number_of_data_structure, false);

  nn_data_structures.push_back(index);
  nn_data_structures[0].buildIndex();
  build[0] = true;
  //const IndexParams& params;
  for (int i = 1; i < number_of_data_structure; ++i) {
    flann::Index<flann::L2<float>> tmp(flann::KDTreeIndexParams(4));
    nn_data_structures.push_back(tmp);
  }
}

std::tuple<int, float, int> nnCluster::query (const flann::Matrix<float> &query, int query_size) {
  float min_distance = std::numeric_limits<float>::max();
  int res = -1;
  int res_size = 1;
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<float>> dists;
  for (int i = 0; i < number_of_data_structure; ++i) {
    if (!build[i]) continue;
    nn_data_structures[i].knnSearch(query, indices, dists, 1,  flann::SearchParams(128));

    int tmp_index = indices[0][0];
    int tmp_size = (int) floor(pow(1 + epsilon, i));
    float tmp_dist = distance(query_size, tmp_size, dists[0][0]);

    if (tmp_dist < min_distance) {
      min_distance = tmp_dist;
      res = tmp_index;
      res_size = tmp_size;
    }

    indices.clear();
    dists.clear();
  }

  return std::make_tuple(res, min_distance, res_size);
}

void nnCluster::add_cluster(flann::Matrix<float> &cluster, int cluster_size) {
      int idx = (int) floor(log_base(cluster_size, 1 + epsilon));
      if (!build[idx])
        nn_data_structures[idx].buildIndex(cluster);
      else
        nn_data_structures[idx].addPoints(cluster);
}

void nnCluster::delete_cluster(int idx, int size) {
      int i = (int) floor(log_base(size, 1 + epsilon));
      nn_data_structures[i].removePoint(idx);
}

float * nnCluster::get_point(int idx, int size) {
  int i = (int) floor(log_base(size, 1 + epsilon));
  if(build[i])
    return nn_data_structures[i].getPoint(idx);
  else
    return nullptr;
}
