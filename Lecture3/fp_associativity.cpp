#include <iostream>
#include <iomanip>

int main(void) {
  double a = .1;
  double b = .2;
  double c = .3;

  double sum1 = (a + b) + c;
  double sum2 = a + (b + c);

  std::cout << std::setprecision(20) << sum1 << std::endl <<
               std::setprecision(20) << sum2 << std::endl;

  if (sum1 == .6)
    std::cout << "(.1 + .2) + .3 == .6" << std::endl;
  
  if (sum2 == .6)
    std::cout << ".1 + (.2 + .3) == .6" << std::endl;

  return 0;
}
