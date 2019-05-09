#include "hierarchical_clustering.h"
#include <fstream>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <tuple>

//  TODO:
// check the float and double compatibility

float log_base_(float num, float base) {
  return log(num)/ log(base);
}

extern "C" typedef double (*func_t)(int n, int d, void * array);

 //std::cout << "radius " << radius << '\n';

hierarchical_clustering::hierarchical_clustering(float * data, int n, int d, double epsilon_, double gamma_):
                                                                  nnc(data, n, d, epsilon_, gamma_), dimension(d), size(n), epsilon(epsilon_), gamma(gamma_) {

  void * lib = dlopen("/home/hussein/projects/m2_thesis/ward_method/lib/librms.so", RTLD_LAZY);
  func_t func = (func_t)dlsym( lib, "radius_min_circle");
  double radius = func(n, d, (void *) data);
  max_dist = 2 * radius;
  min_dist = compute_min_dist();
  beta = ceil(log_base_((max_dist/min_dist) * n, 1 + epsilon)); // be carefull four the double / float
}

double hierarchical_clustering::compute_min_dist() {
  float min_dis = std::numeric_limits<float>::max();
  for (int i = 0; i < size; ++i) {
      float * res_ = nnc.get_point(i, 1);
      flann::Matrix<float> res(res_, 1, dimension);
      nnc.delete_cluster(i, 1);
      auto t = nnc.query(res, 1);
      nnc.add_cluster(res, 1);
      min_dis = std::min(std::get<1>(t), min_dis);
      t = nnc.query(res, 1);
      unmerged_clusters.insert (std::get<0>(t));
      pts_index[i] = std::get<0>(t);
      index_weight[std::get<0>(t)] = 1;
  }
  return min_dis;
}

float * hierarchical_clustering::merge(float * mu_a, float * mu_b, int size_a, int size_b) {
  int den = size_a + size_b;
  float coeff_a = size_a/den;
  float coeff_b = size_b/den;
  float * res = (float *) malloc(dimension * sizeof(float));
  for (int i = 0; i < dimension; ++i) {
    *(res + i) = coeff_a * ( * (mu_a + i)) + coeff_b * ( * (mu_b + i));
  }
  return res;
}

void hierarchical_clustering::build_hierarchy() {
  // we want to have a list of the unmerged value

  float merge_value;
  for (int i = 0; i < beta; ++i) {
    merge_value = pow(1 + epsilon, i);
    for (auto&& u : unmerged_clusters) {
      float * res_ = nnc.get_point(u, index_weight[u]);
      flann::Matrix<float> res(res_, 1, dimension);
      auto t = nnc.query(res, 1);
      float dist = std::get<1>(t);
      while(dist < merge_value) {
        float * nn_pt = nnc.get_point(std::get<0>(t), std::get<2>(t));
        float * merged_cluster_ = merge(res_, nn_pt, index_weight[u], std::get<2>(t)); //
        flann::Matrix<float> merged_cluster(merged_cluster_, 1, dimension);
        nnc.add_cluster(merged_cluster, index_weight[u] + std::get<2>(t));
        nnc.delete_cluster(u, index_weight[u]);
        nnc.delete_cluster(std::get<0>(t), std::get<2>(t));
        unmerged_clusters.erase(u);
        unmerged_clusters.erase(std::get<0>(t));
        merges.push_back({u, std::get<0>(t)});
        t = nnc.query(merged_cluster, index_weight[u] + std::get<2>(t));
        unmerged_clusters.insert(std::get<0>(t));
         // erase u and t(1)
      }

      if(unmerged_clusters.size() == 1) break;
    }
  }
}

// double hierarchical_clustering::distance(uint i, uint j) {
//   double dx = pts[x(i)] - pts[x(j)];
//   double dy = pts[y(i)] - pts[y(j)];
//   double d2 = dx * dx + dy * dy;
//   return sqrt(d2);
// }

// hierarchical_clustering::hierarchical_clustering(const std::string file_name) {
//   read_file(file_name);
// }

// void hierarchical_clustering::read_file(const std::string file_name) {
//   std::ifstream in(file_name);
//   int number_points;
//   in >> number_points >> dimension;
//   pts = point(number_points * dimension);
//   for(int i = 0; i < dimension * number_points; ++i)
//     in >> pts[i];
//   in.close();
// }

std::vector<std::pair<int, int>> hierarchical_clustering::get_merges() const {
  return merges;
}
