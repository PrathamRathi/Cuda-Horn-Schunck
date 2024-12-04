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

// Add this new visualization function
void visualizeFlowHSV(const Mat& flowU, const Mat& flowV, Mat& output) {
    Mat magnitude, angle;
    Mat hsv(flowU.size(), CV_8UC3);

    // Calculate magnitude and angle
    cartToPolar(flowU, flowV, magnitude, angle, true);

    // Normalize magnitude to the range [0, 255]
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);

    // Create separate channels
    vector<Mat> channels(3);

    // H = angle (hue represents direction)
    angle.convertTo(channels[0], CV_8U, 180.0 / CV_PI / 2.0);  // Scale to [0, 180] for OpenCV

    // S = 255 (full saturation)
    channels[1] = Mat::ones(flowU.size(), CV_8U) * 255;

    // V = normalized magnitude
    magnitude.convertTo(channels[2], CV_8U);

    // Merge channels
    merge(channels, hsv);

    // Convert HSV to BGR
    cvtColor(hsv, output, COLOR_HSV2BGR);
}

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

__global__ void compute_neighbor_average(double* __restrict__ u, double* __restrict__ v, 
                            double* __restrict__ uAvg, double* __restrict__ vAvg,
                               const int nx, const int ny) {     
    __shared__ double shared_u[18][18];
    __shared__ double shared_v[18][18];

   // Thread indices
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread indices
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    
    // Shared memory dimensions including halos
    const int shared_idx = local_y + 1;  // +1 to avoid the halo index at 0
    const int shared_jdx = local_x + 1;  // +1 to avoid the halo index at 0

    // Global index for linear arrays
    const int global_idx = global_y * nx + global_x;

    // Load main region into shared memory (including boundary elements)
    if (global_x < nx && global_y < ny) {
        shared_u[shared_jdx][shared_idx] = u[global_idx];
        shared_v[shared_jdx][shared_idx] = v[global_idx];
    } else {
        shared_u[shared_jdx][shared_idx] = 0.0;
        shared_v[shared_jdx][shared_idx] = 0.0;
    }

    // Load halos for neighboring elements from global memory
    if (local_x == 0 && global_x > 0) {
        shared_u[shared_jdx - 1][shared_idx] = u[global_idx - 1];
        shared_v[shared_jdx - 1][shared_idx] = v[global_idx - 1];
    }
    if (local_x == blockDim.x - 1 && global_x < nx - 1) {
        shared_u[shared_jdx + 1][shared_idx] = u[global_idx + 1];
        shared_v[shared_jdx + 1][shared_idx] = v[global_idx + 1];
    }
    if (local_y == 0 && global_y > 0) {
        shared_u[shared_jdx][shared_idx - 1] = u[global_idx - nx];
        shared_v[shared_jdx][shared_idx - 1] = v[global_idx - nx];
    }
    if (local_y == blockDim.y - 1 && global_y < ny - 1) {
        shared_u[shared_jdx][shared_idx + 1] = u[global_idx + nx];
        shared_v[shared_jdx][shared_idx + 1] = v[global_idx + nx];
    }

    // Corners (Halo corners)
    if (local_x == 0 && local_y == 0 && global_x > 0 && global_y > 0) {
        shared_u[shared_jdx - 1][shared_idx - 1] = u[global_idx - nx - 1];
        shared_v[shared_jdx - 1][shared_idx - 1] = v[global_idx - nx - 1];
    }
    if (local_x == blockDim.x - 1 && local_y == 0 && global_x < nx - 1 && global_y > 0) {
        shared_u[shared_jdx + 1][shared_idx - 1] = u[global_idx - nx + 1];
        shared_v[shared_jdx + 1][shared_idx - 1] = v[global_idx - nx + 1];
    }
    if (local_x == 0 && local_y == blockDim.y - 1 && global_x > 0 && global_y < ny - 1) {
        shared_u[shared_jdx - 1][shared_idx + 1] = u[global_idx + nx - 1];
        shared_v[shared_jdx - 1][shared_idx + 1] = v[global_idx + nx - 1];
    }
    if (local_x == blockDim.x - 1 && local_y == blockDim.y - 1 && global_x < nx - 1 && global_y < ny - 1) {
        shared_u[shared_jdx + 1][shared_idx + 1] = u[global_idx + nx + 1];
        shared_v[shared_jdx + 1][shared_idx + 1] = v[global_idx + nx + 1];
    }

    // Synchronize to ensure all threads have loaded shared memory
    __syncthreads();

    // Compute neighbor average for non-border threads
    if (global_x > 0 && global_x < nx - 1 && global_y > 0 && global_y < ny - 1) {
        uAvg[global_idx] = (
            shared_u[shared_jdx - 1][shared_idx - 1] / 12 + shared_u[shared_jdx][shared_idx - 1] / 6 + shared_u[shared_jdx + 1][shared_idx - 1] / 12 +
            shared_u[shared_jdx - 1][shared_idx] / 6 + shared_u[shared_jdx + 1][shared_idx] / 6 +
            shared_u[shared_jdx - 1][shared_idx + 1] / 12 + shared_u[shared_jdx][shared_idx + 1] / 6 + shared_u[shared_jdx + 1][shared_idx + 1] / 12
        );

        vAvg[global_idx] = (
            shared_v[shared_jdx - 1][shared_idx - 1] / 12 + shared_v[shared_jdx][shared_idx - 1] / 6 + shared_v[shared_jdx + 1][shared_idx - 1] / 12 +
            shared_v[shared_jdx - 1][shared_idx] / 6 + shared_v[shared_jdx + 1][shared_idx] / 6 +
            shared_v[shared_jdx - 1][shared_idx + 1] / 12 + shared_v[shared_jdx][shared_idx + 1] / 6 + shared_v[shared_jdx + 1][shared_idx + 1] / 12
        );
    }
}

__global__ void horn_schunk(double* __restrict__ u, double* __restrict__ v, 
                            double* __restrict__ uAvg, double* __restrict__ vAvg,
                            double* __restrict__ Ix, double* __restrict__ Iy, double* __restrict__ It,
                               double alpha, const int nx, const int ny) { 
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = global_y * nx + global_x;  
    
    if (global_x < nx && global_y < ny) {
        double ix = Ix[idx];
        double iy = Iy[idx];
        double it = It[idx];
        double uAvgVal = uAvg[idx];
        double vAvgVal = vAvg[idx];

        double denom = alpha * alpha + ix * ix + iy * iy;
        double p = (ix * uAvgVal + iy * vAvgVal + it);
        u[idx] = uAvgVal - ix * (p / denom);
        v[idx] = vAvgVal - iy * (p / denom);
    }
}

// Main function demonstrating usage
int main(int argc, char* argv[]) {
    cout << "Running Horn-Schunck optical flow..." << endl;

    string filename1 = argv[1];
    string filename2 = argv[2];
    string outputname = argv[3];

    // Load two consecutive frames
    Mat frame1 = imread(filename1, 0);
    Mat frame2 = imread(filename2, 0);
   
    if (frame1.empty() || frame2.empty()) {
        cerr << "Error loading images!" << endl;
        cerr << "Make sure " << filename1 << " and " << filename2 << " exist in: " << filesystem::current_path() << endl;
        return -1;
    }
   
    cout << "Loaded images - Frame1: " << frame1.size() << " Frame2: " << frame2.size() << endl;
    
    // Image size and grid sizes
    cudaError_t GPU_ERROR;
    int ny = frame1.rows;
    int nx = frame1.cols;
    size_t size = nx * ny * sizeof(double);
    int BLOCK_DIM_X = 16;
    int BLOCK_DIM_Y = 16;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);
    cout << "grid x dim:" << (nx + block.x - 1) / block.x << ", grid y dim:" << (ny + block.y - 1) / block.y << endl;

   // Compute image derivatives
    Mat IxMat, IyMat, ItMat;
    computeDerivatives(frame1, frame2, IxMat, IyMat, ItMat);

    // Convert derivatives to vectors
    vector<double> IxHost = matToVector<double>(IxMat);
    vector<double> IyHost = matToVector<double>(IyMat);
    vector<double> ItHost = matToVector<double>(ItMat);

    // Copy derivatives to host
    double *IxDevice, *IyDevice, *ItDevice;
    GPU_ERROR = cudaMalloc(&IxDevice, size);
    GPU_ERROR = cudaMalloc(&IyDevice, size);
    GPU_ERROR = cudaMalloc(&ItDevice, size);
    GPU_ERROR = cudaMemcpy(IxDevice, IxHost.data(), size, cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(IyDevice, IyHost.data(), size, cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(ItDevice, ItHost.data(), size, cudaMemcpyHostToDevice);
    cout << "Finished derivatives transfer" << endl;

    // Create average and flow vectors for device and host
    vector<double> uHost(nx * ny, 0.0);
    vector<double> vHost(nx * ny, 0.0);
    double *uDevice, *vDevice;
    GPU_ERROR = cudaMalloc(&uDevice, size);
    GPU_ERROR = cudaMalloc(&vDevice, size);
    GPU_ERROR = cudaMemcpy(uDevice, uHost.data(), size, cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(vDevice, vHost.data(), size, cudaMemcpyHostToDevice);

    double *uAverage, *vAverage;
    GPU_ERROR = cudaMalloc(&uAverage, size);
    GPU_ERROR = cudaMalloc(&vAverage, size);
    GPU_ERROR = cudaMemcpy(uAverage, uHost.data(), size, cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(vAverage, vHost.data(), size, cudaMemcpyHostToDevice);
    cout << "Copied over average and flow vectors" << endl;

    // Compute optical flow
    int currIteration = 0;
    int iterations = 200;
    double alpha = 1;
    while (currIteration < iterations){
        compute_neighbor_average<<<grid, block>>>(uDevice, vDevice, uAverage, vAverage, nx, ny);      
        GPU_ERROR = cudaDeviceSynchronize();

        horn_schunk<<<grid, block>>>(uDevice, vDevice, uAverage, vAverage, IxDevice, IyDevice, ItDevice, alpha, nx, ny);
        GPU_ERROR = cudaDeviceSynchronize();

        currIteration++;
    }
    cout << "Kernels finished" << endl;

    // Copy over flow results to host
    GPU_ERROR = cudaMemcpy(uHost.data(), uDevice, size, cudaMemcpyDeviceToHost);
    GPU_ERROR = cudaMemcpy(vHost.data(), vDevice, size, cudaMemcpyDeviceToHost);
    cout << "Copied results" << endl;

    // Visualize optical flow
    Mat img_color, flowX, flowY;
    cvtColor(frame1, img_color, COLOR_GRAY2BGR);
    flowX = vectorToMat<double>(uHost, ny, nx, CV_64F);
    flowY = vectorToMat<double>(vHost, ny, nx, CV_64F); 
    drawOpticalFlow(flowX, flowY, img_color);

    Mat flow_vis;
    visualizeFlowHSV(flowX, flowY, flow_vis);

    cout << "Writing optical flow images." << endl;
    imwrite("outputs/CUDA_optical_flow_" + outputname + ".png", img_color);
    imwrite("outputs/CUDA_optical_flow_hsv_" + outputname + ".png", flow_vis);
    return 0;
}