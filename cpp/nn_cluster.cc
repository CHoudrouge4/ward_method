#include "nn_cluster.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <sstream>

template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
	if(v.size() == 0) {
		out << "()";
		return out;
	} else  {
		out << '(';
		out << ' ' << v[0] ;
		for(size_t i = 1; i < v.size(); ++i)
			out << " , " << v[i];
		out << ')';
	return out;
	}
}

float log_base(float num, float base) {
  return log(num)/ log(base);
}

float nnCluster::distance(int size_a, int size_b, float dist) {
  float coef = size_a * size_b;
  coef = coef/(size_a + size_b);
  return coef * dist;
}

nnCluster::nnCluster (float * points_, int n, int d, double epsilon_, double gamma_):
      points(points_, n, d), size(n), dimension(d), epsilon(epsilon_) , gamma(gamma_) {
  number_of_data_structure = (int)std::max(1.0f, ceil(log_base(size, 1 + epsilon)));

  flann::Index<flann::L2<float>> index(points, flann::KDTreeIndexParams(4));
  build = std::vector<bool>(number_of_data_structure, false);
	sizes = std::vector<int> (number_of_data_structure, 0);
	sizes[0] = n;
  assert(size == n);
  assert(d == dimension);
  assert(number_of_data_structure >= 1);
  assert(epsilon == epsilon_);
  assert(gamma == gamma_);

  nn_data_structures.push_back(index);
  nn_data_structures[0].buildIndex();
  build[0] = true;
  //const IndexParams& params;
  for (int i = 1; i < number_of_data_structure + 1; ++i) {
    flann::Index<flann::L2<float>> tmp(flann::KDTreeIndexParams(4));
    nn_data_structures.push_back(tmp);
  }
}

std::tuple<int, float, int> nnCluster::query (const flann::Matrix<float> &query, const int query_size, bool itself) {
  float min_distance = std::numeric_limits<float>::max();
  int res = -1;
  int res_size = 1;
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<float>> dists;
  for (int i = 0; i < number_of_data_structure; ++i) {
    if (!build[i] || sizes[i] <= 0) continue;
		int tmp_size = (int) floor(pow(1 + epsilon, i));
    nn_data_structures[i].knnSearch(query, indices, dists, 1,  flann::SearchParams(128));
		if(indices.size() == 0) continue;
		if(indices[0].size() == 0) continue;
    int tmp_index = indices[0][0];
    float tmp_dist;

    if(itself)
      tmp_dist = dists[0][0];
    else
      tmp_dist = distance(query_size, tmp_size, dists[0][0]);

    if (tmp_dist <= min_distance) {
      min_distance = tmp_dist;
      res = tmp_index;
      res_size = tmp_size;
    }

    indices[0].clear();
    dists[0].clear();
  }

  return std::make_tuple(res, min_distance, res_size);
}

void nnCluster::add_cluster(flann::Matrix<float> &cluster, int cluster_size) {
      int idx = ceil(log_base(cluster_size, 1 + epsilon));
      if (!build[idx]) {
        nn_data_structures[idx].buildIndex(cluster);
        build[idx] = true;
      } else {
        nn_data_structures[idx].addPoints(cluster);
      }
			sizes[idx] = sizes[idx] + 1;
}

void nnCluster::delete_cluster(int idx, int size) {
      int i = ceil(log_base(size, 1 + epsilon));// I think it is wrong
			assert(i >= 0);
			assert((size_t)i < nn_data_structures.size());
      nn_data_structures[i].removePoint(idx);
			sizes[i] = sizes[i] - 1;
}

float * nnCluster::get_point(int idx, int size) {
  int i = (int) ceil(log_base(size, 1 + epsilon));
	std::cout << "the size of the buket " << sizes[i] << std::endl;
  if(build[i]) {
    return nn_data_structures[i].getPoint(idx);
  } else {
	  return nullptr;
	}
}

int nnCluster::get_number_of_data_structures() const {
  return number_of_data_structure;
}

nnCluster::~nnCluster() {
  delete [] points.ptr();
}
