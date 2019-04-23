#include <vector>
#include <string>
#define x(i) 2 * i
#define y(i) 2 * i + 1

typedef std::vector<double> point;
typedef unsigned int uint;

class hierarchical_clustering {
protected:
  std::vector<double> pts;
  std::vector<std::pair<int, int>> merges;

  int dimension;
  void read_file(const std::string);
  double distance(uint i, uint j);
public:
  hierarchical_clustering(const std::string);
  virtual void run() = 0;
  std::vector<std::pair<int, int>> get_merges() const;
};
