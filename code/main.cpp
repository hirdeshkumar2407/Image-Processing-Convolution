#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <tuple>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#include "headers/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "headers/stb_image_write.h"
#include "headers/filters.h"

using namespace Eigen;
using namespace std;

// Function to get image dimensions and load the image data
tuple<int, int, int, unsigned char *> task1_getImageDimensions(char *input_image_path)
{
    cout << "\n--------TASK 1----------\n";
    int width, height, channels;
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 0);
    if (!image_data)
    {
        cerr << "Error: Could not load image " << input_image_path << endl;
        exit(1);
    }
    cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;
    return make_tuple(width, height, channels, image_data);
}

// export image in .png format
void exportimagenotnormalise(unsigned char *image_data, const string &image_name, const string &ext, const MatrixXd &image, int width, int height)
{

    // Create an Eigen matrix to hold the transformed image in unsigned char format
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> transform_image(height, width);

    // Map the matrix values to the grayscale range [0, 255]
    transform_image = image.unaryExpr([](double val) -> unsigned char
                                      {
                                          return static_cast<unsigned char>(std::min(std::max(val, 0.0), 255.0)); // Ensure values stay in [0, 255]
                                      });

    string output_image_path = "results/" + image_name + "." + ext;

    cout << "image created: " << output_image_path << endl;
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, transform_image.data(), width) == 0)
    {
        cerr << "Error: Could not save grayscale image" << endl;
    }
}

// Function to add noise to the image
MatrixXd task2_addNoisetoImage(int width, int height, MatrixXd image_matrix)
{
    cout << "\n--------TASK 2----------\n";
    MatrixXd noise(height, width);
    noise.setRandom();                   // Fill the matrix with random values between -1 and 1
    noise = noise * 50.0;                // Scale the noise to be between -50 and 50
    image_matrix = image_matrix + noise; // Add the noise to the original image

    return image_matrix;
}

// Function to convert MatrixXd to a VectorXd
VectorXd convertMatrixToVector(const MatrixXd &image_matrix)
{
    VectorXd flattened_image = Map<const VectorXd>(image_matrix.data(), image_matrix.size());
    // Use Map<const VectorXd> instead of Map<VectorXd> because image_matrix.data() returns a const double* (read-only pointer).
    // Eigen allows you to use the Map class to reinterpret the matrix's underlying data as a vector without explicitly copying the data.
    // image_matrix.data() gives a pointer to the matrix's data stored in memory.
    // image_matrix.size() gives the total number of elements in the matrix (i.e., rows * cols).
    // Map<VectorXd> constructs a VectorXd that views the matrix's data as a 1D vector.
    return flattened_image;
}

SparseMatrix<double> populateSparseMatrix(int image_size, Matrix3d smoothing_matrix, int width, int height)
{
    SparseMatrix<double> A1(image_size, image_size);

    vector<Triplet<double>> nonzero_values;

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            int pixel_index = i * height + j; // index for the current pixel
            //  Apply the 3x3 kernel to each pixel and its neighbors
            for (int ki = -1; ki <= 1; ki++)
            {
                for (int kj = -1; kj <= 1; kj++)
                {
                    int next_row = i + ki;
                    int next_col = j + kj;
                    int next_index = next_row * height + next_col;
                    if (next_row < 0 || next_col < 0 || next_row >= width || next_col >= height)
                    {
                        continue;
                    }
                    // Add the kernel value to the corresponding entry in A1
                    double kernel_value = smoothing_matrix(ki + 1, kj + 1);
                    nonzero_values.push_back(Triplet<double>(pixel_index, next_index, kernel_value));
                    // break;
                }
            }
        }
    }

    A1.setFromTriplets(nonzero_values.begin(), nonzero_values.end());
    return A1;
}

/*
 smoothing kernel Hav2 as a matrix vector multiplication
 between a matrix A1 having size mnXmn and the image vector.
*/
pair<MatrixXd, SparseMatrix<double>> task4_imageSmoothing(MatrixXd image_matrix, Matrix3d smoothing_matrix, int width, int height)
{
    cout << "\n--------TASK 4----------\n";
    int image_size = image_matrix.size();

    //  sparse matrix multiplication requires the image to be treated as a 1D vector
    VectorXd flattened_image = convertMatrixToVector(image_matrix);

    SparseMatrix<double> A1 = populateSparseMatrix(image_size, smoothing_matrix, width, height);
    cout << "Number of non-zero entries in A1: " << A1.nonZeros() << endl;

    VectorXd smoothed_image = A1 * flattened_image;
    MatrixXd result = Map<MatrixXd>(smoothed_image.data(), height, width);

    cout << "smoothed image size: " << result.rows() << "x" << result.cols() << endl;
    return make_pair(result, A1);
}

MatrixXd task5_matrixvectorMultiplication(const SparseMatrix<double> &A1, const VectorXd &w, int width, int height)
{
    cout << "\n--------TASK 5----------\n";
    VectorXd result = A1 * w;
    MatrixXd resultmat = Map<MatrixXd>(result.data(), height, width);
    return resultmat;
}

// Main function
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }

    char *input_image_path = argv[1];

    // -- Task 1 -- Load image dimensions and data
    auto [width, height, channels, image_data] = task1_getImageDimensions(input_image_path);

    // Prepare Eigen matrix for the image
    MatrixXd image_matrix(height, width);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int index = (i * width + j) * channels;
            image_matrix(i, j) = static_cast<double>(image_data[index]); // Keep values in [0, 255]
        }
    }

    // -- Task 2 --
    MatrixXd t2_image_matrix = task2_addNoisetoImage(width, height, image_matrix);
    exportimagenotnormalise(image_data, "noisy_image", "png", t2_image_matrix, width, height);

    // -- Task 3 -- Convert the image matrix to a vector
    cout << "\n--------TASK 3----------\n";
    VectorXd v = convertMatrixToVector(image_matrix);    // Original image as a vector
    VectorXd w = convertMatrixToVector(t2_image_matrix); // Noisy image as a vector

    cout << "original width and height:" << (width * height) << endl; // Difference between original and noisy images as a vector
    cout << "v mn:" << v.size() << endl;
    cout << "w mn:" << w.size() << endl;
    cout << "Euclidean norm of v norm: " << v.norm() << endl;

    // -- Task 4 --
    Matrix3d Hav2 = getHav2();
    pair<MatrixXd, SparseMatrix<double>> result = task4_imageSmoothing(image_matrix, Hav2, width, height);
    //exportimagenotnormalise(image_data, "smoothed_image", "png", result.first, width, height);

    // -- Task 5 --
    SparseMatrix<double> A1 = result.second;
    MatrixXd resulmat = task5_matrixvectorMultiplication(A1, w, width, height);
    exportimagenotnormalise(image_data, "smoothed_noisy_image", "png", resulmat, width, height);

    // Free the image data after use
    stbi_image_free(image_data);

    return 0;
}