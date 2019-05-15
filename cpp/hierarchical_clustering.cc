#include "hierarchical_clustering.h"
#include <fstream>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <stdlib.h>
#include <tuple>

#define id first
#define w second

typedef std::pair<int, int> pair_int;
//  TODO:
// check the float and double compatibility
// Remove the sqrt from the library if it is uses.


void print_array (float * array, int n, int m, std::string msg) {
  std::cout << msg << ' ';
  std::cout << "[" << ' ';
  for (int i = 0; i < n; ++i) {
    std::cout << '[';
    for(int j = 0; j < m; ++j) {
      std::cout << *(array + i*m + j) << ' ';
    }
    std::cout << ']' << ' ';
  }
  std::cout << "]" << '\n';
}

float log_base_(float num, float base) {
  return log(num)/ log(base);
}

extern "C" typedef double (*func_t)(int n, int d, void * array);

hierarchical_clustering::hierarchical_clustering(float * data, int n, int d, double epsilon_, double gamma_):
                                                                  nnc(data, n, d, epsilon_, gamma_), dimension(d), size(n), epsilon(epsilon_), gamma(gamma_) {

  assert(size == n);
  assert(d == dimension);
  assert(epsilon == epsilon_);
  assert(gamma == gamma_);

  pts_index = std::vector<int>(size);
  void * lib = dlopen("/home/hussein/projects/m2_thesis/ward_method/lib/librms.so", RTLD_LAZY);
  func_t func = (func_t)dlsym( lib, "radius_min_circle");
  double radius = func(n, d, (void *) data);
  max_dist = 2 * radius;
  max_dist = max_dist * max_dist;
  min_dist = compute_min_dist();
  beta = ceil(log_base_((max_dist/min_dist) * n, 1 + epsilon)); // be carefull four the double / float
  unmerged_clusters.max_load_factor(std::numeric_limits<float>::infinity());
}

double hierarchical_clustering::compute_min_dist() {
  float min_dis = std::numeric_limits<float>::max();
  for (int i = 0; i < size; ++i) {
      float * res_ = nnc.get_point(i, 1);
      flann::Matrix<float> res(res_, 1, dimension);
      nnc.delete_cluster(i, 1);
      auto t = nnc.query(res, 1, true);
      nnc.add_cluster(res, 1);
    //  std::cout << "approx " << std::get<1>(t) << '\n';
      min_dis = std::min(std::get<1>(t), min_dis);
      t = nnc.query(res, 1);
      assert(std::get<0>(t) >= 0);
      unmerged_clusters.insert (std::make_pair(std::get<0>(t), 1));
      assert(i >= 0);
      assert((unsigned int)i < pts_index.size());
      pts_index[i] = std::get<0>(t);
      existed[{std::get<0>(t), std::get<2>(t)}] = true;
  }
  return min_dis;
}

float * hierarchical_clustering::merge(float * mu_a, float * mu_b, int size_a, int size_b) {
  float den = size_a + size_b;
  float coeff_a = size_a/den;
  float coeff_b = size_b/den;
  float * res = (float *) malloc(dimension * sizeof(float));
  for (int i = 0; i < dimension; ++i) {
    *(res + i) = coeff_a * ( * (mu_a + i)) + coeff_b * ( * (mu_b + i));
  }
  return res;
}

std::unordered_set<pair_int>  hierarchical_clustering::helper(std::unordered_set<pair_int> &mp, float merge_value) {
  std::unordered_set <pair_int> unchecked;
  std::vector<pair_int> to_erase;
  bool intered = false;
  for (auto&& p : mp) {
    if(existed[p]) {
      intered = true;
      bool ok = false;
      int u = p.id;
      int u_weight = p.w;

      // do a copy
      float * res_ = (float *) malloc(dimension * sizeof(float));
      float * tmp = nnc.get_point(u, u_weight);
      if(tmp == nullptr) std::cout << "tmp is nullptr" << std::endl;
      memcpy(res_, tmp, dimension * sizeof(float));

      flann::Matrix<float> res(res_, 1, dimension);
      std::cout << "delete the cluster " << u << ' ' << u_weight << '\n';
      nnc.delete_cluster(u , u_weight);

      std::cout << "u " << u << " u_weight " << u_weight <<  std::endl;
      if(res_ == nullptr) std::cout << "nullptr" << std::endl;
      print_array(res_, 1, dimension, "u coordinates before query ");

      auto t = nnc.query(res, u_weight);
      std::cout << "t: " << std::get<0>(t) << ' ' << std::get<1>(t) << ' ' << std::get<2>(t) << std::endl;
      float dist = std::get<1>(t); // getting the distance

      flann::Matrix<float> merged_cluster;
      int merged_weight;
      to_erase.push_back({u, u_weight});
      while (dist < merge_value) {
        ok = true;

        std::cout << "u weight " << u_weight << std::endl;
        assert(res_ != nullptr);
        print_array(res_, 1, dimension, "u coordinates");
        float * nn_pt = nnc.get_point(std::get<0>(t), std::get<2>(t));
        print_array(nn_pt, 1, dimension, "NN coords");


        // merging phase
        float * merged_cluster_ = merge(res_, nn_pt, u_weight, std::get<2>(t));
        merged_cluster = flann::Matrix<float>(merged_cluster_, 1, dimension);
        print_array(merged_cluster_, 1, dimension, "merged");
        merged_weight = u_weight + std::get<2>(t);
        std::cout << "weight after merge " << merged_weight  << std::endl;

        // Deleting phase
        // it is wrong because we dont have valid p
        if (u != -1) {
          existed[p] = false;
          to_erase.push_back({u, u_weight});
        }

        existed[{std::get<0>(t), std::get<2>(t)}] = false;
        to_erase.push_back({std::get<0>(t), std::get<2>(t)});
        nnc.delete_cluster(std::get<0>(t), std::get<2>(t));

        // swaping + continure on merging phase
        // getting the neighbor of the merged guy
        // the stuff of u has to be for merged
        t = nnc.query(merged_cluster, merged_weight);
        dist = std::get<1>(t);
        u = -1;
        u_weight = merged_weight;
        if(merged_weight == size) break;
        std::cout << "new distance " << dist << std::endl;
        float * nnn_pt = nnc.get_point(std::get<0>(t), std::get<2>(t));
        print_array(nnn_pt, 1, dimension, "NN coords");
        if(dist < merge_value) {
          res_ = merged_cluster_;
          res = merged_cluster;
        }
      }

      if(!ok) {
        assert(res_ != nullptr);
        print_array(res_, 1, dimension, "failed : ");
        nnc.add_cluster(res, u_weight);
        t = nnc.query(res, u_weight, true);
        existed[{std::get<0>(t), std::get<2>(t)}] = true;
        std::cout << "add to magic " << std::get<0>(t) << ' ' << std::get<2>(t) <<  std::endl;
        magic.insert({std::get<0>(t), std::get<2>(t)});
        //unchecked.insert({std::get<0>(t), std::get<2>(t)});
      } else {
        nnc.add_cluster(merged_cluster, merged_weight);
        t = nnc.query(merged_cluster, merged_weight, true);
        existed[{std::get<0>(t), std::get<2>(t)}] = true;
        unchecked.insert({std::get<0>(t), std::get<2>(t)});
      }
    }
  }

  for(size_t i = 0; i < to_erase.size(); ++i) {
    mp.erase(to_erase[i]);
  }
  return unchecked;
}

void hierarchical_clustering::build_hierarchy() {
  float merge_value;
  for (int i = 0; i < beta; ++i) {
    merge_value = pow(1 + epsilon, i); // find an efficient one
    std::cout << "merge value " << merge_value << std::endl;

    auto ss = helper(this->unmerged_clusters, merge_value); // these are the merges
    while (ss.size() > 1) {
      std::cout << "calling with ss" << std::endl;
      auto tmp = helper(ss, merge_value);
      ss.clear();
      for(auto p: tmp) {
        existed[p] = true;
        ss.insert(p);
      }
    }

    if(ss.size() == 1)
      unmerged_clusters.insert(*ss.begin());


    std::cout << " magic size " << magic.size() << std::endl;
    for(auto m: magic) {
      unmerged_clusters.insert(m);
    }
    magic.clear();
    std::cout << "magic is empty" << std::endl;

    std::cout << "remaining " << unmerged_clusters.size() << '\n';
    if(unmerged_clusters.size() <= 1) break;
    }
}

std::vector<std::pair<int, int>> hierarchical_clustering::get_merges() const {
  return merges;
}
