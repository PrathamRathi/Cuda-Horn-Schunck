#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>

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
#define cudaStream_t            hipStream_t
#define cudaStreamCreate        hipStreamCreate
#define cudaStreamDestroy       hipStreamDestroy
#define cudaStreamSynchronize   hipStreamSynchronize
#define cudaFreeHost            hipHostFree
#define cudaEventCreate         hipEventCreate
#define cudaEventRecord         hipEventRecord
#define cudaEventSynchronize    hipEventSynchronize
#define cudaEventElapsedTime    hipEventElapsedTime
#define cudaEventDestroy        hipEventDestroy
#define cudaEvent_t             hipEvent_t

#else

#include <cuda.h>

#endif

using namespace cv;
using namespace std;

// Hardware specifications for roofline model
const double PEAK_MEMORY_BANDWIDTH = 400e9;  // 400 GB/s
const double PEAK_FLOP_RATE = 2.5e12;       // 2.5 TFLOP/s

// Calculate theoretical peak performance based on arithmetic intensity
double calculate_roofline(double arithmetic_intensity) {
    return min(PEAK_FLOP_RATE, PEAK_MEMORY_BANDWIDTH * arithmetic_intensity);
}

// Modify the analyze_performance function to return the metrics
tuple<double, double, double> analyze_performance(int nx, int ny, int iterations, double elapsed_time, int gridDimX, int gridDimY) {
    // Calculate total pixels processed
    size_t total_pixels = nx * ny;
    
    int haloComputations = (gridDimX * nx + gridDimY * ny); // number of points involved in halo computations

    // Calculate FLOPs
    // compute_neighbor_average: 30 FLOPs per pixel for computation + 8 for halo indexing math
    // horn_schunk: 15 FLOPs per pixel
    double flops_per_pixel = 45.0;
    double total_flops = total_pixels * flops_per_pixel * iterations + (8 * haloComputations * iterations);
    
    // Calculate memory operations
    size_t bytes_per_pixel = (22 + 7) * sizeof(double); 
    double total_bytes = total_pixels * bytes_per_pixel * iterations + (4 * haloComputations * iterations);
    
    // Calculate metrics
    double arithmetic_intensity = total_flops / total_bytes;
    double achieved_tflops = total_flops / (elapsed_time * 1e12);  // Convert to TFLOPS
    double peak_tflops = calculate_roofline(arithmetic_intensity) / 1e12;  // Convert to TFLOPS
    
    return make_tuple(achieved_tflops, arithmetic_intensity, peak_tflops);
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


__global__ void horn_schunck(double* __restrict__ u, double* __restrict__ v, 
                            double* __restrict__ Ix, double* __restrict__ Iy, double* __restrict__ It,
                               double alpha, const int nx, const int ny) { 
     // Define halo width
    constexpr int HALO = 1;

    // Shared memory dimensions including halos
    __shared__ double s_u[18][18];
    __shared__ double s_v[18][18];
    
    // Global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local indices within the shared memory block
    int tx = threadIdx.x + HALO;
    int ty = threadIdx.y + HALO;
    
    // Global linear index
    int global_idx = y * nx + x;
    
    // Load center data
    if (x < nx && y < ny) {
        s_u[ty][tx] = u[global_idx];
        s_v[ty][tx] = v[global_idx];
    }
    
    // Load halo data
    // Top halo
    if (threadIdx.y == 0 && y > 0) {
        s_u[ty-HALO][tx] = u[global_idx - nx];
        s_v[ty-HALO][tx] = v[global_idx - nx];
    }
    
    // Bottom halo
    if (threadIdx.y == blockDim.y - 1 && y < ny - 1) {
        s_u[ty+HALO][tx] = u[global_idx + nx];
        s_v[ty+HALO][tx] = v[global_idx + nx];
    }
    
    // Left halo
    if (threadIdx.x == 0 && x > 0) {
        s_u[ty][tx-HALO] = u[global_idx - 1];
        s_v[ty][tx-HALO] = v[global_idx - 1];
    }
    
    // Right halo
    if (threadIdx.x == blockDim.x - 1 && x < nx - 1) {
        s_u[ty][tx+HALO] = u[global_idx + 1];
        s_v[ty][tx+HALO] = v[global_idx + 1];
    }
    
    // Corner halos
    // Top-left
    if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0) {
        s_u[ty-HALO][tx-HALO] = u[global_idx - nx - 1];
        s_v[ty-HALO][tx-HALO] = v[global_idx - nx - 1];
    }
    
    // Top-right
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && x < nx - 1 && y > 0) {
        s_u[ty-HALO][tx+HALO] = u[global_idx - nx + 1];
        s_v[ty-HALO][tx+HALO] = v[global_idx - nx + 1];
    }
    
    // Bottom-left
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && x > 0 && y < ny - 1) {
        s_u[ty+HALO][tx-HALO] = u[global_idx + nx - 1];
        s_v[ty+HALO][tx-HALO] = v[global_idx + nx - 1];
    }
    
    // Bottom-right
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && x < nx - 1 && y < ny - 1) {
        s_u[ty+HALO][tx+HALO] = u[global_idx + nx + 1];
        s_v[ty+HALO][tx+HALO] = v[global_idx + nx + 1];
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    double uAvg = 0;
    double vAvg = 0;
    // Compute averages only for interior threads
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        // Compute uAvg using 3x3 neighborhood with weighted average
        uAvg = (
            s_u[ty-1][tx-1] / 12.0 + 
            s_u[ty-1][tx]   / 6.0  + 
            s_u[ty-1][tx+1] / 12.0 + 
            s_u[ty][tx-1]   / 6.0  + 
            s_u[ty][tx+1]   / 6.0  + 
            s_u[ty+1][tx-1] / 12.0 + 
            s_u[ty+1][tx]   / 6.0  + 
            s_u[ty+1][tx+1] / 12.0
        );
        
        // Compute vAvg using 3x3 neighborhood with weighted average
        vAvg = (
            s_v[ty-1][tx-1] / 12.0 + 
            s_v[ty-1][tx]   / 6.0  + 
            s_v[ty-1][tx+1] / 12.0 + 
            s_v[ty][tx-1]   / 6.0  + 
            s_v[ty][tx+1]   / 6.0  + 
            s_v[ty+1][tx-1] / 12.0 + 
            s_v[ty+1][tx]   / 6.0  + 
            s_v[ty+1][tx+1] / 12.0
        );
    }
    if (x < nx && y < ny) {
        double ix = Ix[global_idx];
        double iy = Iy[global_idx];
        double it = It[global_idx];

        double denom = alpha * alpha + ix * ix + iy * iy;
        double p = (ix * uAvg + iy * vAvg + it);
        u[global_idx] = uAvg - ix * (p / denom);
        v[global_idx] = vAvg - iy * (p / denom);
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
    int gridDimX = (nx + block.x - 1) / block.x;
    int gridDimY = (ny + block.y - 1) / block.y;
    dim3 grid(gridDimX, gridDimY);
    cout << "grid x dim: " << gridDimX << ", grid y dim: " << gridDimY << endl;

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

    cout << "Copied over average and flow vectors" << endl;

    // Compute optical flow
    int currIteration = 0;
    int iterations = 200;
    double alpha = 1;
    
    // Add timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    cudaEventRecord(start);
    while (currIteration < iterations){
        horn_schunck<<<grid, block>>>(uDevice, vDevice, IxDevice, IyDevice, ItDevice, alpha, nx, ny);
        GPU_ERROR = cudaDeviceSynchronize();

        currIteration++;
    }
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    double elapsed_time_s = elapsed_time_ms / 1000.0;
    cout << "Kernels finished in " << elapsed_time_s << " seconds." << endl;

    // Get performance metrics
    auto [measured_tflops, ai, peak_tflops] = analyze_performance(nx, ny, iterations, elapsed_time_s, gridDimX, gridDimY);
    
    // Print performance table
    printf("\nGrid Size | TFLOPS | AI | Peak TFLOPS | Time(s) | Iterations\n");
    printf("---------|--------|----|-----------  |---------|------------|\n");
    printf("%4dx%4d | %6.6f | %6.6f | %6.6f | %7.5f | %10d |\n",
            nx, ny, measured_tflops, ai, peak_tflops, elapsed_time_s, iterations);
    
    // Additional bottleneck information
    printf("Bottleneck: %s\n", 
            (PEAK_MEMORY_BANDWIDTH * ai < PEAK_FLOP_RATE ? "Memory" : "Compute"));

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