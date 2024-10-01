#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <tuple>
// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

// Function to get image dimensions and load the image data
tuple<int, int, int, unsigned char*> task1_getImageDimensions(char* input_image_path) {
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 0);
    if (!image_data) {
        cerr << "Error: Could not load image " << input_image_path << endl;
        exit(1);
    }
    cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;
    return make_tuple(width, height, channels, image_data);
}
void exportimagenotnormalise(unsigned char* image_data, const string& image_name, const string& ext, const MatrixXd& image, int width, int height) {
    // Do not free image_data here, only free it after exporting the image
    // stbi_image_free(image_data);

    // Create an Eigen matrix to hold the transformed image in unsigned char format
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> transform_image(height, width);

    // Map the matrix values to the grayscale range [0, 255]
    transform_image = image.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(std::min(std::max(val, 0.0), 255.0)); // Ensure values stay in [0, 255]
    });

    // Construct the output file name
    string output_image_path = image_name + "." + ext;

    // Save the image as a PNG file
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, transform_image.data(), width) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;
    }

    // Now free the image data after all processing is done
   // stbi_image_free(image_data);
}

// Function to add noise to the image
MatrixXd task2(int width, int height, MatrixXd image_matrix) {
    MatrixXd noise(height, width);
    noise.setRandom();  // Fill the matrix with random values between -1 and 1
    noise = noise * 50.0;  // Scale the noise to be between -50 and 50
    image_matrix = image_matrix + noise;  // Add the noise to the original image
    
    return image_matrix;
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }

    char* input_image_path = argv[1];

    // Load image dimensions and data
    auto [width, height, channels, image_data] = task1_getImageDimensions(input_image_path);
   /*input_image_path as if it's the image data itself, but it's actually just the file path. When you attempt to access input_image_path[index], you're reading from the string pointer, not the actual pixel data.*/


    // Prepare Eigen matrix for the image
    MatrixXd image_matrix(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = (i * width + j) * channels;
            image_matrix(i, j) = static_cast<double>(image_data[index]);  // Keep values in [0, 255]
        }
    }

    // Add noise to the image matrix
    MatrixXd t2_image_matrix = task2(width, height, image_matrix);

    // Task 2 Output the image matrices
   // cout << "Image before noise addition: " << endl << image_matrix << endl;
   // cout << "Image after noise addition: " << endl << t2_image_matrix << endl;


    exportimagenotnormalise(image_data, "noisy_image", "png", t2_image_matrix, width, height);
    // Free the image data after use
    stbi_image_free(image_data);

    return 0;
}
