//
// Created by junbug331 on 22. 8. 15.
//

#ifndef CUDA_GAUSSIAN_BLUR_GAUSSIAN_BLUR_H
#define CUDA_GAUSSIAN_BLUR_GAUSSIAN_BLUR_H
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <iomanip>

void gaussianBlur(cv::Mat &input, cv::Mat &output, float sigma, int kSize);
template <typename T>
void gaussianBlur_(cv::Mat &input, cv::Mat &output, float sigma, int ksize);
float* generateKernel(float sigma, int kSize);


template <typename T>
void printMat(T *mat, int width, int height)
{
    T sum = 0.0;
    for (int i=0; i<height; ++i)
    {
        for (int j=0; j<width; ++j)
        {
            std::cout << std::setw(4) << std::setprecision(2) << mat[i*width + j] << '\t';
            sum += mat[i*width + j];
        }
        std::cout << std::endl;
    }
}

#endif //CUDA_GAUSSIAN_BLUR_GAUSSIAN_BLUR_H
