#include "headers/filters.h"

#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

using namespace Eigen;
using namespace std;

// -- smoothing kernel matrix
Matrix3d getHav1()
{
    Matrix3d Hav1;
    Hav1 << 0, 1, 0,
        1, 4, 1,
        0, 1, 0;
    Hav1 = Hav1 * (1. / 8.0); // Normalize the kernel
    return Hav1;
}

// -- smoothing kernel matrix
Matrix3d getHav2()
{
    Matrix3d Hav2;
    Hav2 << 1, 1, 1,
        1, 1, 1,
        1, 1, 1;
    Hav2 = Hav2 * (1. / 9.0); // Normalize the kernel
    return Hav2;
}

// -- sharpening kernel matrix
Matrix3d getHsh2()
{
    Matrix3d Hsh2;
    Hsh2 << 0, -3, 0,
        -1, 9, -3,
        0, -1, 0;

    return Hsh2;
}


// -- laplacian kernel matrix
Matrix3d getHlap()
{
    Matrix3d Hlap;
    Hlap << 0, -1, 0,
        -1, 4, -1,
        0, -1, 0;

    return Hlap;
}
