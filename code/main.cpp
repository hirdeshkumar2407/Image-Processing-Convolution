#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <cstring>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <algorithm>
#include <fstream>
//#include "headers/iterative.h"

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
    return sp_matrix;
}

// MatrixXd normalizeMatrix(MatrixXd& matrix) {
//     double minValue = 0.0;
//     double maxValue = 255.0;
//     int rows = matrix.rows();
//     int cols = matrix.cols();
    
//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             double value = matrix(i, j);
//             // Clamp the value to the range [minValue, maxValue]
//             if (value < minValue)
//             {
//                 matrix(i, j) = minValue;
//             }
//             else if (value > maxValue)
//             {
//                 matrix(i, j) = maxValue;
//             }
//         }
//     }
//     return matrix;
// }

// SparseMatrix<double> normalizeSparseMatrix(SparseMatrix<double>& sparseMatrix) {
//     for (int k = 0; k < sparseMatrix.outerSize(); ++k) {
//         for (SparseMatrix<double>::InnerIterator it(sparseMatrix, k); it; ++it) {
//             // Access the current non-zero element
//             double value = it.value();

//             if (value < 0) {
//                sparseMatrix.coeffRef(it.row(), it.col()) = 0;
//             }
//             else if (value > 255) {
//                 sparseMatrix.coeffRef(it.row(), it.col()) = 255;
//             }

//         }
//     }
//     return sparseMatrix;
// }


MatrixXd spMatrixVectorMultiplication(const SparseMatrix<double> &sp_matrix, const VectorXd &vec, int width, int height)
{
    VectorXd result = sp_matrix * vec;
    MatrixXd resultmat = Map<MatrixXd>(result.data(), height, width);
    return resultmat;
}

// pair<MatrixXd, SparseMatrix<double>> task4_imageSmoothing(MatrixXd image_matrix, Matrix3d smoothing_matrix, int width, int height)
// {
//     cout << "\n--------TASK 4----------\n";
//     int image_size = image_matrix.size();

//     //  sparse matrix multiplication requires the image to be treated as a 1D vector
//     VectorXd flattened_image = convertMatrixToVector(image_matrix);

//     SparseMatrix<double> A1 = populateSparseMatrix(image_size, smoothing_matrix, width, height);
//     cout << "Number of non-zero entries in A1: " << A1.nonZeros() << endl;

//     MatrixXd result = spMatrixVectorMultiplication(A1, flattened_image, width, height);
//     //cout << "smoothed image size: " << result.rows() << "x" << result.cols() << endl;
//     return make_pair(result, A1);
// }

bool isSymmetric(const SparseMatrix<double>& matrix) {
    if (matrix.rows() != matrix.cols()) {
        return false;  // A non-square matrix cannot be symmetric
    }

    SparseMatrix<double> transposeMatrix = matrix.transpose();
    return matrix.isApprox(transposeMatrix);
}

pair<MatrixXd, SparseMatrix<double>> imageConvolution(VectorXd flattened_image, Matrix3d kernel_matrix, string spMatrix_name, int width, int height, bool normalized, bool symmetry)
{
    int image_size = height * width;

    SparseMatrix<double> sparseMatrix = populateSparseMatrix(image_size,  kernel_matrix, width, height);
   
    cout << "Number of non-zero entries in " << spMatrix_name << ": " << sparseMatrix.nonZeros() << endl;
    
    if(symmetry){
        if (!isSymmetric(sparseMatrix))
        {
            cout << spMatrix_name << " is not symmetric" << endl;
        }
        else
        {
            cout << spMatrix_name << " is symmetric" << endl;
        }
    }
    
    MatrixXd result = spMatrixVectorMultiplication(sparseMatrix, flattened_image, width, height);
    result = normalized?(result.cwiseMin(255).cwiseMax(0)):result;
    return make_pair(result, sparseMatrix);
}

void exportVectortomtx(const VectorXd &v, const char*  filename)
{
    int n = v.size();
    FILE* out = fopen(filename,"w");
    fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out,"%d\n", n);
    for (int i=0; i<n; i++) {
        fprintf(out,"%d %f\n", i ,v(i));
    }
    fclose(out);
        cout << filename << " vector exported successfully"  << endl;

}

void exportSparseMatrixToMTX(const SparseMatrix<double>& sparseMatrix, const std::string& filename) {
    std::ofstream outFile(filename+".mtx");
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header for the Matrix Market file
    outFile << "%%MatrixMarket matrix coordinate real general" << std::endl;
    outFile << sparseMatrix.rows() << " " << sparseMatrix.cols() << " " << sparseMatrix.nonZeros() << std::endl;

    // Write non-zero values (in 1-based indexing)
    for (int k = 0; k < sparseMatrix.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(sparseMatrix, k); it; ++it) {
            // Write row, column (in 1-based indexing), and value
            outFile << it.row() + 1 << " " << it.col() + 1 << " " << it.value() << std::endl;
        }
    }

    outFile.close();
    cout << filename << " matrix exported successfully to .mtx"  << endl;
}

void task8_exportmatrixes(SparseMatrix<double> &A2, VectorXd &w)
{
    exportSparseMatrixToMTX(A2, "A2");
    exportVectortomtx(w, "w.mtx");
   
}

VectorXd loadVectorFromMTX(const string &filename) {
    // Load the vector from file
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return VectorXd();
    }

    string line;
    getline(file, line); // Skip the first line (MatrixMarket header)

    // Read the number of entries
    int numEntries;
    file >> numEntries;

    VectorXd vec(numEntries);
    for (int i = 0; i < numEntries; ++i) {
        int index;
        double value;

        file >> index >> value;
        vec[index - 1] = value;  // Convert to 0-based indexing
    }

    file.close();
    return vec;
}
 
MatrixXd exportVectorMTXto2DMatrix(const char*  filename, int height, int width)
{
    VectorXd vec=loadVectorFromMTX(filename);
 
    MatrixXd resultmat = Map<MatrixXd>(vec.data(), height, width);
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
    cout << "\n--------TASK 4----------\n";
    Matrix3d Hav2 = getHav2();
    pair<MatrixXd, SparseMatrix<double>> result = imageConvolution(v, Hav2, "A1", width, height, false, false);
    //exportimagenotnormalise(image_data, "smoothed_image", "png", result.first, width, height);

    cout << "\n--------TASK 5----------\n";
    SparseMatrix<double> A1 = result.second;
    MatrixXd resulmat = spMatrixVectorMultiplication(A1, w, width, height);
    exportimagenotnormalise(image_data, "smoothed_noisy", "png", resulmat, width, height);

    // -- Task 6 --
    cout << "\n--------TASK 6----------\n";
    Matrix3d Hsh2 = getHsh2();
    pair<MatrixXd, SparseMatrix<double>> sharpened_image = imageConvolution(v, Hsh2, "A2", width, height, false, true);
    SparseMatrix<double> A2 = sharpened_image.second;

    cout << "\n--------TASK 7----------\n";
    exportimagenotnormalise(image_data, "sharpened_image", "png", sharpened_image.first, width, height);

    // -- Task 8 --
    cout << "\n--------TASK 8----------\n";
    task8_exportmatrixes(A2, w);
    const char *commandtask8 = "mpirun -n 1 ./challenge1 A2.mtx w.mtx x-sol.mtx hist.txt -i bicg -tol 1.0e-9 -p ilu";
    system(commandtask8);

    // -- Task 9 --
    cout << "\n--------TASK 9----------\n";
    MatrixXd resultmat9 = exportVectorMTXto2DMatrix("x-sol.mtx", height, width);
    exportimagenotnormalise(image_data, "x_image", "png", resultmat9, width, height);

    // -- Task 10 --
    cout << "\n--------TASK 10----------\n";
    Matrix3d Hlap = getHlap();
    pair<MatrixXd, SparseMatrix<double>> edge_detection_img = imageConvolution(v, Hlap, "A3", width, height, true, true);
    SparseMatrix<double> A3 = edge_detection_img.second;

    cout << "\n--------TASK 11----------\n";
    exportimagenotnormalise(image_data, "edge_detection_image", "png", edge_detection_img.first, width, height);


    // Free the image data after use
    stbi_image_free(image_data);

    return 0;
}