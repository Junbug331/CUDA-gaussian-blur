#include "convolution_kernel.h"
#include "gaussian_blur.h"
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>

void gaussianBlur(cv::Mat &input, cv::Mat &output, float sigma, int kSize)
{
    if (input.type() % 8 == CV_8U)
        gaussianBlur_<uchar>(input, output, sigma, kSize);
    else if (input.type() % 8 == CV_8S)
        gaussianBlur_<schar>(input, output, sigma, kSize);
    else if (input.type() % 8 == CV_16U)
        gaussianBlur_<ushort>(input, output, sigma, kSize);
    else if (input.type() % 8 == CV_16S)
        gaussianBlur_<short>(input, output, sigma, kSize);
    else if (input.type() % 8 == CV_32S)
        gaussianBlur_<int>(input, output, sigma, kSize);
    else if (input.type() % 8 == CV_32F)
        gaussianBlur_<float>(input, output, sigma, kSize);
    else if (input.type() % 8 == CV_64F)
        gaussianBlur_<double>(input, output, sigma, kSize);
}

template <typename T>
void gaussianBlur_(cv::Mat &input, cv::Mat &output, float sigma, int kSize)
{
    int width = input.cols;
    int height = input.rows;
    int chs = input.channels();
    size_t imgSize = width * height * chs;

//    std::vector<T> buf;
//    if (input.isContinuous())
//        buf.assign(input.data, input.data + input.total()*input.channels());
//    else
//    {
//        for (int i=0; i<input.rows; ++i)
//            buf.insert(buf.end(), input.ptr<T>(i), input.ptr<T>(i)+input.cols*input.channels());
//    }
    auto deletor = [](T *ptr){ cudaFreeHost(ptr);};
    std::shared_ptr<T> in(new T[imgSize], deletor);
    std::shared_ptr<T> out(new T[imgSize], deletor);
    std::unique_ptr<float[]> kernel(generateKernel(sigma, kSize));

    // Pin Host memory IMPORTANT
    cudaCheckError(cudaMallocHost((void **)&in, sizeof(T)*imgSize));
    cudaCheckError(cudaMallocHost((void **)&out, sizeof(T)*imgSize));
    memcpy(in.get(), input.data, sizeof(T)*imgSize);

    // Call CUDA function
    convolution2DCUDA(in.get(), out.get(), kernel.get(), kSize, width, height, chs);

    // Save result
    output = cv::Mat(height, width, input.type(), out.get());
}

float* generateKernel(float sigma, int kSize)
{
    float *kernel = new float[kSize*kSize];
    int radius = kSize >> 1;
    float s = 2.*sigma*sigma;
    float sum = 0.0;

    for (int y=-radius; y<=radius; y++)
    {
        for (int x=-radius; x<=radius; x++)
        {
            kernel[(y+radius)*kSize + (x+radius)] =
                    exp(-(x*x + y*y)/s) / M_PI*s;
            sum += kernel[(y+radius)*kSize + (x+radius)];
        }
    }
    std::transform(kernel, kernel+(kSize*kSize), kernel,
                   [&](float x){return x/sum;});
    return kernel;
}

template void gaussianBlur_<uchar>(cv::Mat&, cv::Mat&, float, int);
template void gaussianBlur_<schar>(cv::Mat&, cv::Mat&, float, int);
template void gaussianBlur_<ushort>(cv::Mat&, cv::Mat&, float, int);
template void gaussianBlur_<short>(cv::Mat&, cv::Mat&, float, int);
template void gaussianBlur_<int>(cv::Mat&, cv::Mat&, float, int);
template void gaussianBlur_<float>(cv::Mat&, cv::Mat&, float, int);
template void gaussianBlur_<double>(cv::Mat&, cv::Mat&, float, int);