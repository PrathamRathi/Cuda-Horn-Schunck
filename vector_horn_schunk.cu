#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>

#ifdef USE_HIP

#include <hip/hip_runtime.h>
#include <iostream>
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaSetDevice          hipSetDevice
#define cudaDeviceSynchronize  hipDeviceSynchronize


#define cudaMalloc              hipMalloc 
#define cudaFree                hipFree

#define cudaHostMalloc           hipHostMalloc
#define cudaMemcpy              hipMemcpy

#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost

#define cudaError_t             hipError_t

#else

#include <cuda.h>

#endif

using namespace cv;
using namespace std;


template <typename T>
vector<T> matToVector(const Mat& mat) {
    if (mat.empty()) {
        throw runtime_error("Input matrix is empty.");
    }

    vector<T> vec(mat.rows * mat.cols * mat.channels());
    for (int y = 0; y < mat.rows; ++y) {
        const T* rowPtr = mat.ptr<T>(y);
        copy(rowPtr, rowPtr + mat.cols, vec.begin() + y * mat.cols);
    }
    return vec;
}

// Function to convert a std::vector back to cv::Mat
template <typename T>
Mat vectorToMat(const vector<T>& vec, int rows, int cols, int type) {
    Mat mat(rows, cols, type);
    for (int y = 0; y < rows; ++y) {
        T* rowPtr = mat.ptr<T>(y);
        copy(vec.begin() + y * cols, vec.begin() + (y + 1) * cols, rowPtr);
    }
    return mat;
}

void computeDerivatives(const Mat& im1, const Mat& im2, Mat& ix, Mat& iy, Mat& it) {
    // Define kernels for calculating derivatives
    Mat kernelX = (Mat_<double>(2, 2) << 0.25, -0.25, 0.25, -0.25); // Kernel for dx
    Mat kernelY = (Mat_<double>(2, 2) << 0.25, 0.25, -.25, -.25); // Kernel for dy
    Mat kernelT = (Mat_<double>(2, 2) << 0.25, 0.25, 0.25, 0.25);   // Kernel for dt

    // Convert images to double precision
    Mat im1_d, im2_d;
    im1.convertTo(im1_d, CV_64FC1);
    im2.convertTo(im2_d, CV_64FC1);

    // Compute derivatives
    Mat fx1, fx2, fy1, fy2, ft1, ft2;
    filter2D(im1_d, fx1, -1, kernelX);
    filter2D(im2_d, fx2, -1, kernelX);
    ix = fx1 + fx2;

    filter2D(im1_d, fy1, -1, kernelY);
    filter2D(im2_d, fy2, -1, kernelY);
    iy = fy1 + fy2;

    filter2D(im2_d, ft1, -1, -kernelT);
    filter2D(im1_d, ft2, -1, kernelT);
    it = ft1 + ft2;
}

// Compute average flow using vector operations
void computeNeighborAverage(const vector<double>& u, const vector<double>& v,
                                  vector<double>& uAvg, vector<double>& vAvg,
                                  int rows, int cols) {
    uAvg.assign(u.size(), 0.0);
    vAvg.assign(v.size(), 0.0);

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int idx = y * cols + x;
            uAvg[idx] = (
                u[idx - cols - 1] / 12 + u[idx - cols] / 6 + u[idx - cols + 1] / 12 +
                u[idx - 1] / 6 + u[idx + 1] / 6 +
                u[idx + cols - 1] / 12 + u[idx + cols] / 6 + u[idx + cols + 1] / 12
            );

            vAvg[idx] = (
                v[idx - cols - 1] / 12 + v[idx - cols] / 6 + v[idx - cols + 1] / 12 +
                v[idx - 1] / 6 + v[idx + 1] / 6 +
                v[idx + cols - 1] / 12 + v[idx + cols] / 6 + v[idx + cols + 1] / 12
            );
        }
    }
}

// Compute optical flow using Horn-Schunck method
void computeOpticalFlow(const Mat& frame1, const Mat& frame2, Mat& flowX, Mat& flowY,
                        double alpha = 0.01, int iterations = 20) {
    // Compute image derivatives
    Mat IxMat, IyMat, ItMat;
    computeDerivatives(frame1, frame2, IxMat, IyMat, ItMat);

    // Convert derivatives to vectors
    vector<double> Ix = matToVector<double>(IxMat);
    vector<double> Iy = matToVector<double>(IyMat);
    vector<double> It = matToVector<double>(ItMat);

    int rows = frame1.rows;
    int cols = frame1.cols;

    // Initialize flow fields as vectors
    vector<double> u(rows * cols, 0.0);
    vector<double> v(rows * cols, 0.0);
    vector<double> uAvg(rows * cols, 0.0);
    vector<double> vAvg(rows * cols, 0.0);

    // Iterative optimization
    for (int iter = 0; iter < iterations; ++iter) {
        // Compute average flow using vector operations
        computeNeighborAverage(u, v, uAvg, vAvg, rows, cols);

        // Update flow for each pixel
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                int idx = y * cols + x;
                double ix = Ix[idx];
                double iy = Iy[idx];
                double it = It[idx];

                double denom = alpha * alpha + ix * ix + iy * iy;
                double p = (ix * uAvg[idx] + iy * vAvg[idx] + it);
                u[idx] = uAvg[idx] - ix * (p / denom);
                v[idx] = vAvg[idx] - iy * (p / denom);
            }
        }
    }

    // Convert flow vectors back to matrices
    flowX = vectorToMat<double>(u, rows, cols, CV_64F);
    flowY = vectorToMat<double>(v, rows, cols, CV_64F);
}

// Visualize optical flow
void drawOpticalFlow(const Mat& flowX, const Mat& flowY, Mat& image, int scale = 3, int step = 16) {
    for (int y = 0; y < image.rows; y += step) {
        for (int x = 0; x < image.cols; x += step) {
            Point2f flow(flowX.at<double>(y, x), flowY.at<double>(y, x));
            Point start(x, y);
            Point end(cvRound(x + flow.x * scale), cvRound(y + flow.y * scale));
            arrowedLine(image, start, end, Scalar(0, 255, 0), 1, LINE_AA, 0, 0.2);
        }
    }
}

// Main function demonstrating usage
int main() {
    cout << "Running Horn-Schunck optical flow..." << endl;
   
    // string filename1 = "images/frame1.png";
    // string filename2 = "images/frame2.png";

    string filename1 = "images/car1.jpg";
    string filename2 = "images/car2.jpg";

    // Load two consecutive frames
    Mat frame1 = imread(filename1, 0);
    Mat frame2 = imread(filename2, 0);
   
    if (frame1.empty() || frame2.empty()) {
        cerr << "Error loading images!" << endl;
        cerr << "Make sure " << filename1 << " and " << filename2 << " exist in: " << filesystem::current_path() << endl;
        return -1;
    }
   
    cout << "Loaded images - Frame1: " << frame1.size() << " Frame2: " << frame2.size() << endl;
   
    // Compute optical flow
    Mat flowX, flowY;
    computeOpticalFlow(frame1, frame2, flowX, flowY, 1, 100);

     // Visualize optical flow
    Mat img_color;
    cvtColor(frame1, img_color, COLOR_GRAY2BGR);
    
    drawOpticalFlow(flowX, flowY, img_color);
    
    imwrite("outputs/Regular_optical_flow.png", img_color);
   
   
    return 0;
}