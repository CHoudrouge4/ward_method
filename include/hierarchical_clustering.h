#include <vector>
#include <string>
#include "nn_cluster.h"
#define x(i) 2 * i
#define y(i) 2 * i + 1

typedef std::vector<double> point;
typedef unsigned int uint;

class hierarchical_clustering {
private:
  int dimension;
  int size;
  std::vector<std::pair<int, int>> merges;
  double epsilon;
  double gamma;
  nnCluster nnc;
  double max_dist;
  double min_dist;
  double beta;

  double compute_min_dist();
  float * merge(float * mu_a, float * mu_b, int size_a, int size_b);
//  void read_file(const std::string);
public:
  hierarchical_clustering(float * data, int n, int d, double epsilon_, double gamma_);
  virtual void run() = 0;
  std::vector<std::pair<int, int>> get_merges() const;
  void build_hierarchy();
};
