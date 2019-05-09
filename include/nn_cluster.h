#include "flann/flann.hpp"
#include "flann/io/hdf5.h"
#include <tuple>

class nnCluster {

public:

  // Constructor
  nnCluster (float * points_, int n, int d, double epsilon_, double gamma_);

  std::tuple<int, float, int> query (const flann::Matrix<float> &query, int query_size, bool itself=false);

  /**
  * This function adds a cluster to the data clusters of the clusters of size
  * cluster size
  *
  */
  void add_cluster(flann::Matrix<float> &cluster, const int cluster_size);

  void delete_cluster(int idx, int size);

  float * get_point(int idx, int size);

  int get_number_of_data_structures() const;

  ~nnCluster();
private:
  flann::Matrix<float> points;
  int size, dimension;
  double epsilon;
  double gamma;
  int number_of_data_structure;
  std::vector<flann::Index<flann::L2<float>>> nn_data_structures;
  std::vector<bool> build;
};
