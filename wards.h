#include <vector>
#include <string>
#define x(i) 2 * i
#define y(i) 2 * i + 1

class wards {
private:
  std::vector<double> pts;
  int dimension;

  void read_file(const std::string);
  void clusster();
public:
  wards(const std::string, int);

};
