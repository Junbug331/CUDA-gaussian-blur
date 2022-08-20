#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "gaussian_blur.h"
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

using namespace std;
using namespace cv;

int main()
{

    Mat img = imread(RESOURCES_PATH + string("/pic.jpg"));
    cv::resize(img, img, cv::Size(640, 480));
    spdlog::stopwatch sw;

    Mat out1;
    sw.reset();
    gaussianBlur(img, out1, 5.0, 5);
    spdlog::info("CUDA gaussian blur took: {} sec", sw);
    imshow("gaussian blur\n", out1);
    waitKey();
    destroyAllWindows();

    Mat out2;
    sw.reset();
    cv::GaussianBlur(img, out2, cv::Size(5, 5), 5.0);
    spdlog::info("OpenCV gaussian blur took: {} sec", sw);

    return 0;
}