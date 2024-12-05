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
#define cudaStream_t            hipStream_t
#define cudaStreamCreate        hipStreamCreate
#define cudaStreamDestroy       hipStreamDestroy
#define cudaStreamSynchronize   hipStreamSynchronize
#define cudaMemcpyAsync         hipMemcpyAsync
#define cudaMemset              hipMemset
#define cudaFreeHost            hipHostFree
#define cudaSuccess             hipSuccess

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
    
    // Compute averages only for interior threads
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        // Compute uAvg using 3x3 neighborhood with weighted average
        uAvg[global_idx] = (
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
        vAvg[global_idx] = (
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

    // Setup for streams and chunked memory
    const int k = 10;  // number of streams
    size_t chunk_size = (nx * ny + k - 1) / k;  // elements per chunk
    size_t chunk_bytes = chunk_size * sizeof(double);

    // Validate chunk size
    cout << "Total size: " << nx * ny << ", Chunk size: " << chunk_size << ", Number of chunks: " << k << endl;
    if (chunk_size * k < nx * ny) {
        cerr << "Error: Chunks don't cover entire data" << endl;
        return -1;
    }

    // Calculate grid dimensions for chunks
    size_t chunk_rows = (chunk_size + nx - 1) / nx;  // Calculate rows for this chunk
    dim3 chunk_grid((nx + block.x - 1) / block.x,
                    (chunk_rows + block.y - 1) / block.y);
    
    cout << "chunk grid x dim:" << chunk_grid.x << ", chunk grid y dim:" << chunk_grid.y << endl;

    // Arrays of device pointers for each chunk
    double **u_d = new double*[k];
    double **v_d = new double*[k];
    double **uAvg_d = new double*[k];
    double **vAvg_d = new double*[k];
    
    // Arrays of pinned host memory for each chunk
    double **u_h = new double*[k];
    double **v_h = new double*[k];
    
    // Create streams and allocate memory for each chunk
    cudaStream_t *streams = new cudaStream_t[k];
    for(int i = 0; i < k; i++) {
        GPU_ERROR = cudaStreamCreate(&streams[i]);
        if (GPU_ERROR != cudaSuccess) {
            cerr << "Failed to create stream " << i << endl;
            return -1;
        }
        
        // Allocate device memory for each chunk
        GPU_ERROR = cudaMalloc((void**) &u_d[i], chunk_bytes);
        GPU_ERROR = cudaMalloc((void**) &v_d[i], chunk_bytes);
        GPU_ERROR = cudaMalloc((void**) &uAvg_d[i], chunk_bytes);
        GPU_ERROR = cudaMalloc((void**) &vAvg_d[i], chunk_bytes);
        
        // Initialize device memory to zero
        GPU_ERROR = cudaMemset(u_d[i], 0, chunk_bytes);
        GPU_ERROR = cudaMemset(v_d[i], 0, chunk_bytes);
        GPU_ERROR = cudaMemset(uAvg_d[i], 0, chunk_bytes);
        GPU_ERROR = cudaMemset(vAvg_d[i], 0, chunk_bytes);
        
        // Allocate pinned host memory for each chunk
        GPU_ERROR = cudaHostMalloc(&u_h[i], chunk_bytes);
        GPU_ERROR = cudaHostMalloc(&v_h[i], chunk_bytes);
        
        // Initialize host memory to zero
        memset(u_h[i], 0, chunk_bytes);
        memset(v_h[i], 0, chunk_bytes);
    }

    // Compute image derivatives
    Mat IxMat, IyMat, ItMat;
    computeDerivatives(frame1, frame2, IxMat, IyMat, ItMat);

    // Convert derivatives to vectors
    vector<double> IxHost = matToVector<double>(IxMat);
    vector<double> IyHost = matToVector<double>(IyMat);
    vector<double> ItHost = matToVector<double>(ItMat);

    double *IxDevice, *IyDevice, *ItDevice;
    // Allocate memory on device
    GPU_ERROR = cudaMalloc(&IxDevice, size);
    GPU_ERROR = cudaMalloc(&IyDevice, size);
    GPU_ERROR = cudaMalloc(&ItDevice, size);
    
    // Copy derivatives to device asynchronously
    GPU_ERROR = cudaMemcpyAsync(IxDevice, IxHost.data(), size, cudaMemcpyHostToDevice, streams[0]);
    GPU_ERROR = cudaMemcpyAsync(IyDevice, IyHost.data(), size, cudaMemcpyHostToDevice, streams[0]);
    GPU_ERROR = cudaMemcpyAsync(ItDevice, ItHost.data(), size, cudaMemcpyHostToDevice, streams[0]);

    // Ensure derivatives are copied before starting computation
    GPU_ERROR = cudaStreamSynchronize(streams[0]);

    cout << "Finished derivatives transfer" << endl;

    // Compute optical flow
    int currIteration = 0;
    int iterations = 200;
    double alpha = 1;
    while (currIteration < iterations) {
        for(int i = 0; i < k; i++) {
            size_t start_idx = i * chunk_size;
            size_t current_chunk_size = min(chunk_size, nx * ny - start_idx);
            
            compute_neighbor_average<<<chunk_grid, block, 0, streams[i]>>>(
                u_d[i], v_d[i], uAvg_d[i], vAvg_d[i], nx, ny);      

            horn_schunk<<<chunk_grid, block, 0, streams[i]>>>(
                u_d[i], v_d[i], uAvg_d[i], vAvg_d[i], 
                IxDevice + start_idx, IyDevice + start_idx, ItDevice + start_idx, 
                alpha, nx, ny);
        }
        currIteration++;
    }
    cout << "Kernels finished" << endl;

    // Copy over flow results to host
    for(int i = 0; i < k; i++) {
        size_t start_idx = i * chunk_size;
        size_t current_chunk_size = min(chunk_size, nx * ny - start_idx);
        
        GPU_ERROR = cudaMemcpyAsync(u_h[i], u_d[i], current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
        GPU_ERROR = cudaMemcpyAsync(v_h[i], v_d[i], current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams
    for (int i = 0; i < k; i++) {
        GPU_ERROR = cudaStreamSynchronize(streams[i]);
    }
    cout << "Copied results" << endl;

    // Combine results from all chunks
    vector<double> u_result(nx * ny);
    vector<double> v_result(nx * ny);
    for(int i = 0; i < k; i++) {
        size_t start_idx = i * chunk_size;
        size_t current_chunk_size = min(chunk_size, nx * ny - start_idx);
        memcpy(&u_result[start_idx], u_h[i], current_chunk_size * sizeof(double));
        memcpy(&v_result[start_idx], v_h[i], current_chunk_size * sizeof(double));
    }

    // Visualize optical flow
    Mat img_color, flowX, flowY;
    cvtColor(frame1, img_color, COLOR_GRAY2BGR);
    flowX = vectorToMat<double>(u_result, ny, nx, CV_64F);
    flowY = vectorToMat<double>(v_result, ny, nx, CV_64F); 
    drawOpticalFlow(flowX, flowY, img_color);

    Mat flow_vis;
    visualizeFlowHSV(flowX, flowY, flow_vis);

    cout << "Writing optical flow images." << endl;
    imwrite("outputs/CUDA_optical_flow_" + outputname + ".png", img_color);
    imwrite("outputs/CUDA_optical_flow_hsv_" + outputname + ".png", flow_vis);

    // Cleanup derivative arrays first
    GPU_ERROR = cudaFree(IxDevice);
    GPU_ERROR = cudaFree(IyDevice);
    GPU_ERROR = cudaFree(ItDevice);

    // Cleanup chunk arrays and streams
    for(int i = 0; i < k; i++) {
        GPU_ERROR = cudaFree(u_d[i]);
        GPU_ERROR = cudaFree(v_d[i]);
        GPU_ERROR = cudaFree(uAvg_d[i]);
        GPU_ERROR = cudaFree(vAvg_d[i]);
        
        GPU_ERROR = cudaFreeHost(u_h[i]);
        GPU_ERROR = cudaFreeHost(v_h[i]);
        
        GPU_ERROR = cudaStreamDestroy(streams[i]);
    }
    
    delete[] u_d;
    delete[] v_d;
    delete[] uAvg_d;
    delete[] vAvg_d;
    delete[] u_h;
    delete[] v_h;
    delete[] streams;

    return 0;
}