#include <vector>
#include <string>
#define x(i) 2 * i
#define y(i) 2 * i + 1

typedef std::vector<double> point;
typedef unsigned int uint;

class wards {
private:
  std::vector<double> pts;
  std::vector<point> M; // dissimilarity matrix,
  int dimension;

  void read_file(const std::string);
  double distance(uint i, uint j);
  void clusster();
public:
  wards(const std::string);
};
