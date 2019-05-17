#include "hierarchical_clustering.h"
#include <fstream>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <stdlib.h>
#include <tuple>
#include <utility>
#include <sstream>

#define id first
#define w second

float power_(float x, int y) {
    if (y == 0)
        return 1;
    else if (y % 2 == 0)
        return power_(x, y / 2) * power_(x, y / 2);
    else
        return x * power_(x, y / 2) * power_(x, y / 2);
}

float logbase(float num, float base) {
  return std::log10(num)/ std::log10(base);
}

typedef std::pair<int, int> pair_int;

//  TODO:
// check the float and double compatibility
// Remove the sqrt from the library if it is uses.
// fix the last bound
std::string toString( const std::pair< size_t, size_t >& data) {
    std::ostringstream str;
    str << data.first << ", " << data.second;
    return str.str();
}

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

float log_base_(double num, double base) {
  return log(num)/ log(base);
}

extern "C" typedef double (*func_t)(int n, int d, void * array);

hierarchical_clustering::hierarchical_clustering(float * data, int n, int d, double epsilon_, double gamma_):
                                                                  nnc(data, n, d, epsilon_, gamma_), dimension(d), size(n), epsilon(epsilon_), gamma(gamma_) {

  void * lib = dlopen("/home/hussein/projects/m2_thesis/ward_method/lib/librms.so", RTLD_LAZY);
  func_t func = (func_t)dlsym( lib, "radius_min_circle");
  double radius = func(n, d, (void *) data);
  max_dist = 4 * radius;
  min_dist = nnc.compute_min_dist();
  beta = ceil(log_base_((max_dist/min_dist) * n, 1 + epsilon)); // be carefull four the double / float
  unmerged_clusters.max_load_factor(std::numeric_limits<float>::infinity());
  output.reserve(n - 1);
  max_merge_size = (int)power_(1 + epsilon, nnc.get_number_of_data_structures());
  std::cout << " max merge size " << max_merge_size << std::endl;
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

int hierarchical_clustering::compute_merge_weight(int x, int y) {
    int tx = (int) power_(1 + epsilon, x);
    int ty = (int) power_(1 + epsilon, y);
    std::cout << "tx: " << tx <<  " ty " << ty << std::endl;
    return (int) floor(logbase(tx + ty, 1 + epsilon));
}

std::unordered_set<pair_int>  hierarchical_clustering::helper(std::unordered_set<pair_int> &mp, float merge_value) {
  std::unordered_set <pair_int> unchecked;
  std::vector<pair_int> to_erase;

  for (auto&& p : mp) {
    if(existed[p]) {
      bool ok = false;
      bool flag = false;
      int u = p.id;
      int u_weight = p.w;
      if(u_weight >= max_merge_size) continue;
      float * res_ = (float *) malloc(dimension * sizeof(float));
      float * tmp = nnc.get_point(u, u_weight);
      if(tmp == nullptr) std::cout << " tmp is nullptr" << std::endl;
      memcpy(res_, tmp, dimension * sizeof(float));

      flann::Matrix<float> res(res_, 1, dimension);
      nnc.delete_cluster(u , u_weight);

      print_array(res_, 1, dimension, "u coordinates before query ");

      auto t = nnc.query(res, u_weight);
      float dist = std::get<1>(t); // getting the distance

      flann::Matrix<float> merged_cluster; // move to the outside later on
      int merged_weight;
      to_erase.push_back({u, u_weight});

      assert(u_weight <= max_merge_size);
      while (dist < merge_value) {
        ok = true;
        print_array(res_, 1, dimension, "u coordinates");
        std::cout << "asking for " << std::get<0>(t) << ' ' <<  std::get<2>(t) << std::endl;
        float * nn_pt = nnc.get_point(std::get<0>(t), std::get<2>(t));
        print_array(nn_pt, 1, dimension, "NN coords");

        // merging phase
        float * merged_cluster_ = merge(res_, nn_pt, u_weight, std::get<2>(t));
        merged_cluster = flann::Matrix<float>(merged_cluster_, 1, dimension);
        print_array(merged_cluster_, 1, dimension, "merged");

        //merged_weight = std::min(u_weight + std::get<2>(t), max_merge_size);

        if(flag) {
          merged_weight = compute_merge_weight(u_weight, std::get<2>(t));
        } else {
          merged_weight = u_weight + std::get<2>(t);
        }

        std::cout  << "u_weight " << u_weight << " get " << std::get<2>(t) << std::endl;
        merges.push_back({{u, u_weight}, {std::get<0>(t), std::get<2>(t)}});

        assert(std::get<2>(t) <= max_merge_size);
        assert(merged_weight <= max_merge_size);

        {
          // register the merge
          int u_tmp = u;
          int weight_tmp = u_weight;
          nnc.add_cluster(merged_cluster, merged_weight);
          auto p = nnc.query(merged_cluster, merged_weight, true);
          u = std::get<0>(p);
          std::cout << merged_weight << " == " << std::get<2>(p) << std::endl;
          assert(merged_weight == std::get<2>(p));
          u_weight = merged_weight;
          nnc.delete_cluster(u, u_weight);
          dict[{u, u_weight}] = {u, u_weight};
          std::cout << u << ' ' << u_weight << std::endl;
          std::cout << toString({u, merged_weight}) << " = " << toString(dict[{u_tmp, weight_tmp}]) << " --> " << toString(dict[{std::get<0>(t), std::get<2>(t)}]) << std::endl;
      //    rep[{u, merged_weight}] = std::make_pair(dict[{u_tmp, weight_tmp}], dict[{std::get<0>(t), std::get<2>(t)}]);
        }

        // Deleting phase
        // it is wrong because we dont have valid p
        //if (u != -1) {
          existed[p] = false;
          to_erase.push_back({u, u_weight});


        existed[{std::get<0>(t), std::get<2>(t)}] = false;
        to_erase.push_back({std::get<0>(t), std::get<2>(t)});
        nnc.delete_cluster(std::get<0>(t), std::get<2>(t));

        // swaping + continure on merging phase
        // getting the neighbor of the merged guy
        // the stuff of u has to be for merged
        if(merged_weight >= max_merge_size) break;
        t = nnc.query(merged_cluster, merged_weight);
        dist = std::get<1>(t);
        //u = -1;
        //u_weight = merged_weight;

        float * nnn_pt = nnc.get_point(std::get<0>(t), std::get<2>(t));
        if(nnn_pt == nullptr) break;
        print_array(nnn_pt, 1, dimension, "NN coords");
        if(dist < merge_value) {
          //free(res_);
          res_ = merged_cluster_;
          res = merged_cluster;
          flag = true;
        }
      }

      if(!ok) {
        assert(res_ != nullptr);
        print_array(res_, 1, dimension, "failed : ");
        nnc.add_cluster(res, u_weight);
        t = nnc.query(res, u_weight, true);
        dict[{std::get<0>(t), std::get<2>(t)}] = dict[{u, u_weight}];
        std::cout << toString({std::get<0>(t), std::get<2>(t)}) << " --> " << toString(dict[{u, u_weight}]) << std::endl;
        existed[{std::get<0>(t), std::get<2>(t)}] = true;
        magic.insert({std::get<0>(t), std::get<2>(t)});
      } else {
        nnc.add_cluster(merged_cluster, merged_weight);
        auto tt = nnc.query(merged_cluster, merged_weight, true);
        existed[{std::get<0>(tt), std::get<2>(tt)}] = true;
        unchecked.insert({std::get<0>(tt), std::get<2>(tt)});
        dict[{std::get<0>(tt), std::get<2>(tt)}] = dict[{u, u_weight}];
      }
    }
  }

  for(size_t i = 0; i < to_erase.size(); ++i) mp.erase(to_erase[i]);
  return unchecked;
}

void hierarchical_clustering::build_hierarchy() {

  for(int i = 0; i < size; ++i) {
    cluster_weight[{0, i}] = 1;
  }

  float merge_value;
  for (int i = 0; i < beta; ++i) {
    merge_value = pow(1 + epsilon, i); // find an efficient one
    //std::cout << "merge value " << merge_value << std::endl;

    auto ss = helper(this->unmerged_clusters, merge_value); // these are the merges
    while (ss.size() > 1) {
      auto tmp = helper(ss, merge_value);
      ss.clear();
      for(auto p: tmp) {
        existed[p] = true;
        ss.insert(p);
      }
    }

    if(ss.size() == 1) unmerged_clusters.insert(*ss.begin());
    for(auto m: magic) unmerged_clusters.insert(m);

    magic.clear();
    if(unmerged_clusters.size() <= 1) break;
  }
}

std::vector<std::pair<pair_int, pair_int>> hierarchical_clustering::get_merges() const {
  return merges;
}

void hierarchical_clustering::print_merges() {
  for (auto&& p : merges) {
    std::cout << dict[p.first].first << ' ' << dict[p.first].second << " || " <<
    dict[p.second].first << ' ' << dict[p.second].second << '\n';
  }

  // p = ((), ((), ()))
  for (auto&& p: rep) {
    std::cout << toString(p.first) << " = "  << toString(p.second.first) << " + " << toString(p.second.second) << '\n';
  }
}
