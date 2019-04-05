#include <vector>
#include <string>
#define x(i) 2 * i
#define y(i) 2 * i + 1

typedef std::vector<double> point;
typedef unsigned int uint;

class HC {
protected:
  std::vector<double> pts;
  int dimension;

  void read_file(const std::string);
  double distance(uint i, uint j);
public:
  HC(const std::string);
  virtual void run() = 0;
};
