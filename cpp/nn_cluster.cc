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
  return std::log10(num) / std::log10(base);
}

float nnCluster::distance(int size_a, int size_b, float dist) {
  float coef = size_a * size_b;
  coef = coef/(size_a + size_b);
  return coef * dist;
}

nnCluster::nnCluster (float * points_, int n, int d, double epsilon_, double gamma_):
      points(points_, n, d), size(n), dimension(d), epsilon(epsilon_) , gamma(gamma_) {

  number_of_data_structure = (int) floor(log_base(n, 1 + epsilon));
  flann::Index<flann::L2<float>> index(points, flann::KDTreeIndexParams(10));
  build = std::vector<bool>(number_of_data_structure, false);
	sizes = std::vector<int> (number_of_data_structure, 0);
	sizes[0] = n;
  nn_data_structures.push_back(index);
  nn_data_structures[0].buildIndex();
  build[0] = true;
  for (int i = 1; i < number_of_data_structure; ++i) {
    flann::Index<flann::L2<float>> tmp(flann::KDTreeIndexParams(10));
    nn_data_structures.push_back(tmp);
  }
}

std::tuple<int, float, int> nnCluster::query (const flann::Matrix<float> &query, const int query_size, bool itself) {
  float min_distance = std::numeric_limits<float>::max();
  int res = -1;
  int res_index = 0;
  for (int i = 0; i < number_of_data_structure; ++i) {
    if (!build[i] || sizes[i] <= 0) continue;
		nn_data_structures[i].knnSearch(query, indices, dists, 1, flann::SearchParams(1));
		if(indices.size() == 0) continue;
		if(indices[0].size() == 0) continue;
    int tmp_index = indices[0][0];
		int tmp_size = cluster_weight[{i, tmp_index}];
	  float tmp_dist;
    if (itself)
      tmp_dist = dists[0][0];
    else
      tmp_dist = distance(query_size, tmp_size, dists[0][0]);

    if (tmp_dist < min_distance) {
      min_distance = tmp_dist;
      res = tmp_index;
      res_index = i;
			std::cout << "i " <<  i  <<  " res size " << res_index << std::endl;
	  }

    indices[0].clear();
    dists[0].clear();
  }
	//
  return std::make_tuple(res, min_distance, cluster_weight[{res_index, index_ds[{res, res_index}]}]);
}

int nnCluster::add_cluster(flann::Matrix<float> &cluster, int cluster_size) {
      int idx = floor(log_base(cluster_size, 1 + epsilon));
			std::cout << " cluster size " << cluster_size << ' ' << idx << "/" << number_of_data_structure << std::endl;
			assert(idx < number_of_data_structure);
			assert(idx >= 0);
		  if (!build[idx]) {
        nn_data_structures[idx].buildIndex(cluster);
        build[idx] = true;
      } else {
        nn_data_structures[idx].addPoints(cluster);
      }
			sizes[idx] = sizes[idx] + 1;
			return idx;
}

int nnCluster::add_cluster(flann::Matrix<float> &cluster, int cluster_size, int old_index, int new_index) {
	dict[{new_index, cluster_size}] = dict[{old_index, cluster_size}];
	int idx = add_cluster(cluster, cluster_size);
	index_ds[{new_index, idx}] = old_index;
	return idx;
}

void nnCluster::add_new_cluster(flann::Matrix<float> &cluster, const int cluster_size) {
	int idx = add_cluster(cluster, cluster_size);
	auto t = query(cluster, cluster_size, true);
	dict[{std::get<0>(t), cluster_size}] = {std::get<0>(t), cluster_size};
	cluster_weight[{idx, std::get<0>(t)}] = cluster_size;
}

void nnCluster::delete_cluster(int idx, int size) {
  int i = floor(log_base(size, 1 + epsilon));// I think it is wrong
	assert(i >= 0);
	assert((size_t)i < nn_data_structures.size());
  nn_data_structures[i].removePoint(idx);
	sizes[i] = sizes[i] - 1;
}

float * nnCluster::get_point(int idx, int size) {
  int i = (int) floor(log_base(size, 1 + epsilon));
	std::cout << "index ? " << i << std::endl;
	assert(i < number_of_data_structure);
	assert(i >= 0);
  if(build[i]) {
    return nn_data_structures[i].getPoint(idx);
  } else {
	  return nullptr;
	}
}

float nnCluster::compute_min_dist() {
  float min_dis = std::numeric_limits<float>::max();
  for (int i = 0; i < size; ++i) {
      float * res_ = get_point(i, 1);
      flann::Matrix<float> res(res_, 1, dimension);
      delete_cluster(i, 1);
      auto t = query(res, 1, true);
      add_cluster(res, 1);
      min_dis = std::min(std::get<1>(t), min_dis);
      t = query(res, 1);
      dict[{i, 1}] = {i, 1};
      dict[{std::get<0>(t), std::get<2>(t)}] = {i, 1};
			index_ds[{i, 0}] = i;
			index_ds[{std::get<0>(t), 0}] = {i};

			cluster_weight[{0, i}] = 1;
			std::cout << "( " << std::get<0>(t) << ' ' <<  std::get<2>(t) << " ) " << std::endl;
  }
  return min_dis;
}

int nnCluster::get_number_of_data_structures() const {
  return nn_data_structures.size();
}

pair_int nnCluster::get_index(int index, int weight) {
	return dict[std::make_pair(index, weight)];
}

// we can later use one weight
void nnCluster::update_dict(int new_idx, int new_weight, int old_idx, int old_weight) {
	dict[{new_idx, new_weight}] = dict[{old_idx, old_weight}];
}

nnCluster::~nnCluster() {
  delete [] points.ptr();
}
