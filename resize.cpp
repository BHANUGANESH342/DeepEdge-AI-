
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>

using namespace cv;
using namespace std;
using namespace chrono;

void customResize(const Mat& src, Mat& dst, Size newSize) {
    dst.create(newSize, src.type());
    float scaleX = (float)src.cols / newSize.width;
    float scaleY = (float)src.rows / newSize.height;

    for (int y = 0; y < newSize.height; ++y) {
        for (int x = 0; x < newSize.width; ++x) {
            int x_start = cvFloor(x * scaleX);
            int y_start = cvFloor(y * scaleY);
            int x_end = cvCeil((x + 1) * scaleX);
            int y_end = cvCeil((y + 1) * scaleY);

            x_start = std::max(0, x_start);
            y_start = std::max(0, y_start);
            x_end = std::min(src.cols, x_end);
            y_end = std::min(src.rows, y_end);

            Vec3f sum(0, 0, 0);
            int count = 0;

            for (int yy = y_start; yy < y_end; ++yy) {
                for (int xx = x_start; xx < x_end; ++xx) {
                    sum += src.at<Vec3b>(yy, xx);
                    ++count;
                }
            }

            dst.at<Vec3b>(y, x) = sum / count;
        }
    }
}

double calculatePSNR(const Mat& I1, const Mat& I2) {
    Mat s1;
    absdiff(I1, I2, s1);        
    s1.convertTo(s1, CV_32F);    
    s1 = s1.mul(s1);            

    Scalar s = sum(s1);         
    double sse = s.val[0] + s.val[1] + s.val[2]; 
    if (sse <= 1e-10) {
        return 0;  
    } else {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

int main() {
    
    Mat inputImage = imread("G178_2-1080.BMP", IMREAD_COLOR);
    if (inputImage.empty()) {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }

    Size newSize(inputImage.cols / 2, inputImage.rows / 2);

    
    Mat outputNearest, outputLinear, outputCubic, outputCustom;

   
    auto start = high_resolution_clock::now();
    resize(inputImage, outputNearest, newSize, 0, 0, INTER_NEAREST);
    auto end = high_resolution_clock::now();
    auto durationNearest = duration_cast<milliseconds>(end - start);

    start = high_resolution_clock::now();
    resize(inputImage, outputLinear, newSize, 0, 0, INTER_LINEAR);
    end = high_resolution_clock::now();
    auto durationLinear = duration_cast<milliseconds>(end - start);

    start = high_resolution_clock::now();
    resize(inputImage, outputCubic, newSize, 0, 0, INTER_CUBIC);
    end = high_resolution_clock::now();
    auto durationCubic = duration_cast<milliseconds>(end - start);

    start = high_resolution_clock::now();
    customResize(inputImage, outputCustom, newSize);
    end = high_resolution_clock::now();
    auto durationCustom = duration_cast<milliseconds>(end - start);


    cout << "Time taken for INTER_NEAREST: " << durationNearest.count() << " ms" << endl;
    cout << "Time taken for INTER_LINEAR: " << durationLinear.count() << " ms" << endl;
    cout << "Time taken for INTER_CUBIC: " << durationCubic.count() << " ms" << endl;
    cout << "Time taken for Custom Resize: " << durationCustom.count() << " ms" << endl;

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", inputImage);

    namedWindow("Nearest Neighbor", WINDOW_NORMAL);
    imshow("Nearest Neighbor", outputNearest);

    namedWindow("Linear", WINDOW_NORMAL);
    imshow("Linear", outputLinear);

    namedWindow("Cubic", WINDOW_NORMAL);
    imshow("Cubic", outputCubic);

    namedWindow("Custom Resize", WINDOW_NORMAL);
    imshow("Custom Resize", outputCustom);

    
    Mat diffNearest, diffLinear, diffCubic, diffCustom;
    absdiff(outputNearest, outputLinear, diffNearest);
    absdiff(outputNearest, outputCubic, diffLinear);
    absdiff(outputNearest, outputCustom, diffCubic);
    absdiff(outputLinear, outputCustom, diffCustom);

    vector<Mat> diffs = { diffNearest, diffLinear, diffCubic, diffCustom };

    for (size_t i = 0; i < diffs.size(); ++i) {
        double mean, stddev;
        meanStdDev(diffs[i], mean, stddev);
        cout << "Difference: Mean = " << mean << ", StdDev = " << stddev << endl;
    }

  
    cout << "PSNR (Nearest vs Linear): " << calculatePSNR(outputNearest, outputLinear) << " dB" << endl;
    cout << "PSNR (Nearest vs Cubic): " << calculatePSNR(outputNearest, outputCubic) << " dB" << endl;
    cout << "PSNR (Nearest vs Custom): " << calculatePSNR(outputNearest, outputCustom) << " dB" << endl;
    cout << "PSNR (Linear vs Custom): " << calculatePSNR(outputLinear, outputCustom) << " dB" << endl;

    waitKey(0);

    destroyAllWindows();

    return 0;
}

