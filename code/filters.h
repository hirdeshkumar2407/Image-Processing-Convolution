#ifndef filters_H
#define filters_H

#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

using namespace Eigen;
using namespace std;

// Function to return the Hav2 matrix
Matrix3d getHav2();

Matrix3d getHsh2();

#endif