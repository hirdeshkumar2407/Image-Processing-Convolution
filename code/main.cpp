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

tuple<int, int, int> task1_getImageDimensions(char* input_image_path) {
  int width, height, channels;
//   tuple<int width, int height, int> image_dimensions;
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 0);
  if (!image_data) {
    cerr << "Error: Could not load image " << input_image_path << endl;
    exit(1);
  }
  cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;
  stbi_image_free(image_data);

  return make_tuple(width, height, channels);
}

void exportimage(unsigned char* image_data, string image_name, string ext, MatrixXd image, int width, int height) {
     stbi_image_free(image_data);


  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> transform_image(height, width);
  // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
  transform_image = image.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });
/*
    const string output_image_path1 = image_name+"."+ext;
  if (stbi_write_png(output_image_path1.c_str(), width, height, 1,
                     transform_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;

    return 1;
  }*/

}

void task2(int width,int height){
    MatrixXd image(width, height);
    image.setRandom(); // fill the matrix with random values between -1 and 1
    cout << "Mean: " << image << endl;


    
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <image_path>" << endl;
    return 1;
  }

   char* input_image_path = argv[1];

  tuple<int, int, int> dimensions = task1_getImageDimensions(input_image_path);
  task2(get<0>(dimensions), get<1>(dimensions));

  return 0;
}