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
  unmerged_clusters.max_load_factor(std::numeric_limits<float>::infinity());
  min_dist = nnc.compute_min_dist(unmerged_clusters, existed);
  beta = ceil(log_base_((max_dist/min_dist) * n, 1 + epsilon)); // be carefull four the double / float
  output.reserve(n - 1);
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

  for (auto&& p : mp) {
    if(existed[p]) {
      bool ok = false;
      bool flag = false;
      int u = p.id;
      int u_weight = p.w;

      std::cout << "u: " << u << " u_weight " <<  u_weight << std::endl;

      float * res_ = (float *) malloc(dimension * sizeof(float));
      float * tmp = nnc.get_point(u, u_weight);
      //if(tmp == nullptr) std::cout << " tmp is nullptr" << std::endl;
      memcpy(res_, tmp, dimension * sizeof(float));

      flann::Matrix<float> res(res_, 1, dimension);
      nnc.delete_cluster(u, u_weight);

      print_array(res_, 1, dimension, "u coordinates before query ");

      auto t = nnc.query(res, u_weight);
      std::cout << "t: " << std::get<0>(t) << ' ' << std::get<1>(t) << ' ' << std::get<2>(t) << std::endl;

      float dist = std::get<1>(t); // getting the distance
      int t_weight = std::get<2>(t);

      flann::Matrix<float> merged_cluster; // move to the outside later on
      int merged_weight;
      to_erase.push_back({u, u_weight});

      while (dist < merge_value) {
        ok = true;

        print_array(res_, 1, dimension, "u coordinates");
        std::cout << "before NN " << std::get<0>(t) << " " << t_weight << std::endl;
        float * nn_pt = nnc.get_point(std::get<0>(t), t_weight);
        print_array(nn_pt, 1, dimension, "NN coords");

        // merging phase
        float * merged_cluster_ = merge(res_, nn_pt, u_weight, t_weight);
        merged_cluster = flann::Matrix<float>(merged_cluster_, 1, dimension);
        print_array(merged_cluster_, 1, dimension, "merged");

        merged_weight = u_weight + t_weight;

        //merges.push_back({{u, u_weight}, {std::get<0>(t), t_weight}});

        {
          int u_tmp = u;
          int weight_tmp = u_weight;
          std::cout << "before add new" << std::endl;
          nnc.add_new_cluster(merged_cluster, merged_weight);
          auto p = nnc.query(merged_cluster, merged_weight, true);

          u = std::get<0>(p);
          std::cout << " std::get 2 " << std::get<2>(p) << std::endl;
          u_weight = merged_weight;
          nnc.delete_cluster(u, u_weight);

          nnc.update_dict(u, u_weight, u, u_weight);
          //std::cout << toString({u, merged_weight}) << " = " << toString(dict[{u_tmp, weight_tmp}]) << " --> " << toString(dict[{std::get<0>(t), std::get<2>(t)}]) << std::endl;
          //rep[{u, merged_weight}] = std::make_pair(dict[{u_tmp, weight_tmp}], dict[{std::get<0>(t), std::get<2>(t)}]);
        }


        existed[p] = false;
        to_erase.push_back({u, u_weight});

        existed[{std::get<0>(t), t_weight}] = false;
        to_erase.push_back({std::get<0>(t), t_weight});
        nnc.delete_cluster(std::get<0>(t), t_weight);

        std::cout << ">>>> merged weight " <<  merged_weight << std::endl;
        assert(merged_weight <= size);
        if(merged_weight == size) break;

        t = nnc.query(merged_cluster, merged_weight);
        dist = std::get<1>(t);
        t_weight = std::get<2>(t);
        std::cout << "before NNN " << std::get<0>(t) << ' ' << t_weight  << std::endl;
        float * nnn_pt = nnc.get_point(std::get<0>(t), t_weight);
        if(nnn_pt == nullptr) break;
        print_array(nnn_pt, 1, dimension, "NNN coords");
        if(dist < merge_value) {
          res_ = merged_cluster_;
          res = merged_cluster;
          flag = true;
        }
      }
      if(u_weight == size) break;
      if(!ok) {
        assert(res_ != nullptr);
        print_array(res_, 1, dimension, "failed : ");
        int idx = nnc.add_cluster(res, u_weight);
        t = nnc.query(res, u_weight, true);
        nnc.update_size(idx, std::get<0>(t), u_weight);
        t_weight = std::get<2>(t);
        //dict[{std::get<0>(t), t_weight} ] = dict[{u, u_weight}];
        nnc.update_dict(std::get<0>(t), t_weight, u, u_weight);
        //std::cout << toString({std::get<0>(t), std::get<2>(t)}) << " --> " << toString(dict[{u, u_weight}]) << std::endl;
        existed[{std::get<0>(t), u_weight}] = true;
        magic.insert({std::get<0>(t), u_weight});
      } else {
        std::cout << "else" << std::endl;
        int idx = nnc.add_cluster(merged_cluster, merged_weight);
        auto tt = nnc.query(merged_cluster, merged_weight, true);
        nnc.update_size(idx, std::get<0>(tt), merged_weight);

        tt = nnc.query(merged_cluster, merged_weight, true);
        std::cout << "else query true " << std::get<2>(tt) << std::endl;

        existed[{std::get<0>(tt), merged_weight}] = true;
        unchecked.insert({std::get<0>(tt), merged_weight});
        //dict[, std::get<2>(tt)}] = dict[{u, u_weight}];
        nnc.update_dict(std::get<0>(tt), merged_weight, u, u_weight);
      }
    }
  }

  for(size_t i = 0; i < to_erase.size(); ++i) mp.erase(to_erase[i]);
  return unchecked;
}

void hierarchical_clustering::build_hierarchy() {

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
