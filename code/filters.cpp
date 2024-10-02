#include "headers/filters.h"

#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

using namespace Eigen;
using namespace std;

MatrixXd getHav1() {
    MatrixXd Hav1;
  Hav1 << 0, 1, 0,
          1, 4, 1,
          0, 1, 0;
  Hav1 = Hav1 / 8.0; // Normalize the kernel
    return Hav1;
}

// Define the function to return Hav2
MatrixXd getHav2() {
    MatrixXd Hav2;
  Hav2 << 1, 1, 1,
          1, 1, 1,
          1, 1, 1;
  Hav2 = Hav2 / 9.0; // Normalize the kernel
    return Hav2;
}

MatrixXd getHsh2(){
    MatrixXd Hsh2;
    Hsh2 <<  0, -3,  0,
            -1,  9, -3,
             0, -1,  0;

    return Hsh2;
}