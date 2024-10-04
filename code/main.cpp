#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "headers/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "headers/stb_image_write.h"
#include "headers/filters.h"

using namespace Eigen;
using namespace std;

void printNonZeroEntries(const SparseMatrix<double>& sparseMatrix, int stopper) {
    cout << "Printing non-zero entries of the sparse matrix:" << endl;
    for (int k = 0; k < sparseMatrix.outerSize()-stopper; ++k) {
        cout << "Outer index: " << k << endl;
        for (SparseMatrix<double>::InnerIterator it(sparseMatrix, k); it; ++it) {
            cout << "Non-zero value at (" << it.row() << ", " << it.col() << ") = " << it.value() << endl;
        }
    }
}
// Function to perform normalisation to sparse matrix between 0 and 255


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
    SparseMatrix<double> sp_matrix(image_size, image_size);

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

                   if(kernel_value != 0){
                   nonzero_values.push_back(Triplet<double>(pixel_index, next_index, kernel_value));
                   }
                    
                    // break;
                }
            }
        }
    }

    sp_matrix.setFromTriplets(nonzero_values.begin(), nonzero_values.end());

    // printNonZeroEntries(sp_matrix, 359);
    return sp_matrix;
}



MatrixXd normaliseafterConvSimMatrix(MatrixXd& matrix) {
    double minValue = 0.0;
    double maxValue = 255.0;
int rows = matrix.rows();
    int cols = matrix.cols();
    // Iterate through each element in the dense matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double value = matrix(i, j);
            // Clamp the value to the range [minValue, maxValue]
            if (value < minValue) {
                matrix(i, j) = minValue;
            }
            else if (value > maxValue) {
                matrix(i, j) = maxValue;
            }
        }
    }
    // Optionally, you can print the normalized matrix or specific non-zero values
    //cout << "Normalized matrix: \n" << matrix << endl;
    return matrix;
    

}

SparseMatrix<double> normaliseafterConvSpaMatrix(SparseMatrix<double>& sparseMatrix) {
    for (int k = 0; k < sparseMatrix.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(sparseMatrix, k); it; ++it) {
            // Access the current non-zero element
            double value = it.value();

            if (value < 0) {
               sparseMatrix.coeffRef(it.row(), it.col()) = 0;
            }
            else if (value > 255) {
                sparseMatrix.coeffRef(it.row(), it.col()) = 255;
            }




        }
    }

    //printNonZeroEntries(sparseMatrix, 341);

    return sparseMatrix;
}




MatrixXd spMatrixVectorMultiplication(const SparseMatrix<double> &sp_matrix, const VectorXd &vec, int width, int height)
{
    VectorXd result = sp_matrix * vec;
    MatrixXd resultmat = Map<MatrixXd>(result.data(), height, width);
    return resultmat;
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

    MatrixXd result = spMatrixVectorMultiplication(A1, flattened_image, width, height);
    //cout << "smoothed image size: " << result.rows() << "x" << result.cols() << endl;
    return make_pair(result, A1);
}
bool isSymmetric(const SparseMatrix<double>& matrix) {
    // Check if the matrix has the same number of rows and columns
    if (matrix.rows() != matrix.cols()) {
        return false;  // A non-square matrix cannot be symmetric
    }

    // Compare the matrix with its transpose
    SparseMatrix<double> transposeMatrix = matrix.transpose();

    // Check if all the elements are the same
    return matrix.isApprox(transposeMatrix);
}

pair<MatrixXd, SparseMatrix<double>> task6_imageSharpening(MatrixXd image_matrix, Matrix3d sharpening_matrix, int width, int height)
{
    cout << "\n--------TASK 6----------\n";
    int image_size = image_matrix.size();

    //  sparse matrix multiplication requires the image to be treated as a 1D vector
    VectorXd flattened_image = convertMatrixToVector(image_matrix);

    SparseMatrix<double> A2 = populateSparseMatrix(image_size,  sharpening_matrix, width, height);
   
    cout << "Number of non-zero entries in A2: " << A2.nonZeros() << endl;
    
    if (!isSymmetric(A2)) {
        cout << "Matrix A2 is not symmetric" << endl;
    }else{
        cout << "Matrix A2 is symmetric" << endl;
    }
    MatrixXd result = spMatrixVectorMultiplication(A2, flattened_image, width, height);
    //cout << "smoothed image size: " << result.rows() << "x" << result.cols() << endl;
    return make_pair(result, A2);
}


pair<MatrixXd, SparseMatrix<double>> task10_edge_detection(MatrixXd image_matrix, Matrix3d edge_matrix, int width, int height)
{
   
    int image_size = image_matrix.size();

    //  sparse matrix multiplication requires the image to be treated as a 1D vector
    VectorXd flattened_image = convertMatrixToVector(image_matrix);

    SparseMatrix<double> A3 = populateSparseMatrix(image_size,  edge_matrix, width, height);
    // SparseMatrix<double> A3_norm=normaliseafterConvSpaMatrix(A3);
    cout << "Number of non-zero entries in A3: " << A3.nonZeros() << endl;
    
    if (!isSymmetric(A3)) {
        cout << "Matrix A3 is not symmetric" << endl;
    }else{
        cout << "Matrix A3 is symmetric" << endl;
    }


    MatrixXd result = spMatrixVectorMultiplication(A3, flattened_image, width, height);
    //MatrixXd result_norm=normaliseafterConvSimMatrix(result);
    result = result.cwiseMin(255).cwiseMax(0);
    //cout << "smoothed image size: " << result.rows() << "x" << result.cols() << endl;
    return make_pair(result, A3);
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
    cout << "\n--------TASK 5----------\n";
    SparseMatrix<double> A1 = result.second;
    MatrixXd resulmat = spMatrixVectorMultiplication(A1, w, width, height);
    exportimagenotnormalise(image_data, "smoothed_noisy", "png", resulmat, width, height);

    // -- Task 6 --
    Matrix3d Hsh2 = getHsh2();
    pair<MatrixXd, SparseMatrix<double>> result2 = task6_imageSharpening(image_matrix, Hsh2, width, height);

  
    cout << "\n--------TASK 7----------\n";
    // -- Task 7 --
    SparseMatrix<double> A2 = result2.second;
    MatrixXd resulmat2 = spMatrixVectorMultiplication(A2, v, width, height);
   exportimagenotnormalise(image_data, "sharpened_orignal_image", "png", resulmat2, width, height);

   // -- Task 8 --
   //cout << "\n--------TASK 8----------\n";
   // -- Task 9 --
   // cout << "\n--------TASK 9----------\n";
 
   // -- Task 10 --
   cout << "\n--------TASK 10----------\n";
      Matrix3d Hlap = getHlap();
    pair<MatrixXd, SparseMatrix<double>> result3 =task10_edge_detection(image_matrix, Hlap, width, height);
    
     cout << "\n--------TASK 11----------\n";
    // -- Task 11 --
    SparseMatrix<double> A3 = result3.second;
    MatrixXd resulmat3 = spMatrixVectorMultiplication(A3, v, width, height);
   exportimagenotnormalise(image_data, "edge_detection_image", "png", resulmat3, width, height);
    // Free the image data after use
    stbi_image_free(image_data);

    return 0;
}