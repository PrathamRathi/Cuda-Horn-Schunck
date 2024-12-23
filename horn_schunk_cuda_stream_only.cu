#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <tuple>

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
#define cudaEventCreate         hipEventCreate
#define cudaEventRecord         hipEventRecord
#define cudaEventSynchronize    hipEventSynchronize
#define cudaEventElapsedTime    hipEventElapsedTime
#define cudaEventDestroy        hipEventDestroy
#define cudaEvent_t             hipEvent_t
#define cudaHostAlloc           hipHostMalloc
#define cudaHostAllocDefault    hipHostMallocDefault
#define cudaMemcpyHostToHost    hipMemcpyHostToHost

#else

#include <cuda.h>
#include <cuda_runtime.h>

#endif

using namespace cv;
using namespace std;

// Hardware specifications for roofline model
const double PEAK_MEMORY_BANDWIDTH = 1.6e12;  //  1.6 TB/s
const double PEAK_FLOP_RATE = 2.5e12;       // 2.5 TFLOP/s
const int NUM_STREAMS = 16;

// Calculate theoretical peak performance based on arithmetic intensity
double calculate_roofline(double arithmetic_intensity) {
    return min(PEAK_FLOP_RATE, PEAK_MEMORY_BANDWIDTH * arithmetic_intensity);
}

// Modify the analyze_performance function to return the metrics
tuple<double, double, double> analyze_performance(int nx, int ny, int iterations, double elapsed_time, int gridDimX, int gridDimY) {
    size_t total_pixels = nx * ny;

    int haloComputations = ny * (gridDimX - 1) + nx * (gridDimY - 1);

    // FLOPs calculation (keeping 45 FLOPs per pixel as verified)
    double flops_per_pixel = 45.0;
    double total_flops = total_pixels * flops_per_pixel * iterations + (8 * haloComputations * iterations);
    
    // Corrected memory operations (23 doubles per pixel)
    size_t bytes_per_pixel = 23 * sizeof(double); 
    double total_bytes = total_pixels * bytes_per_pixel * iterations + (4 * haloComputations * iterations);
    
    // Calculate metrics
    double arithmetic_intensity = total_flops / total_bytes;
    double achieved_tflops = total_flops / (elapsed_time * 1e12);
    double peak_tflops = calculate_roofline(arithmetic_intensity) / 1e12;
    
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

__global__ void fused_horn_schunk(double* __restrict__ u, double* __restrict__ v,
                                 double* __restrict__ Ix, double* __restrict__ Iy, 
                                 double* __restrict__ It,
                                 double alpha, const int nx, const int section_height) {     
    // Define halo width
    constexpr int HALO = 1;

    // Shared memory dimensions including halos
    __shared__ double s_u[18][18];
    __shared__ double s_v[18][18];
    
    // Global indices within this section
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local indices within the shared memory block
    int tx = threadIdx.x + HALO;
    int ty = threadIdx.y + HALO;
    
    // Global linear index
    int global_idx = y * nx + x;
    
    // Load center data
    if (x < nx && y < section_height) {
        s_u[ty][tx] = u[global_idx];
        s_v[ty][tx] = v[global_idx];
    }
    
    // Load halos (only if we're at the edge of the section)
    if (x < nx && y < section_height) {
        // Top halo
        if (threadIdx.y == 0 && y > 0) {
            s_u[ty-HALO][tx] = u[global_idx - nx];
            s_v[ty-HALO][tx] = v[global_idx - nx];
        }
        
        // Bottom halo
        if (threadIdx.y == blockDim.y - 1 && y < section_height - 1) {
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
    }
    
    __syncthreads();
    
    // Compute flow updates only for interior points
    if (x > 0 && x < nx - 1 && y > 0 && y < section_height - 1) {
        // First compute weighted averages
        double uAvg = (
            s_u[ty-1][tx-1] / 12.0 + 
            s_u[ty-1][tx]   / 6.0  + 
            s_u[ty-1][tx+1] / 12.0 + 
            s_u[ty][tx-1]   / 6.0  + 
            s_u[ty][tx+1]   / 6.0  + 
            s_u[ty+1][tx-1] / 12.0 + 
            s_u[ty+1][tx]   / 6.0  + 
            s_u[ty+1][tx+1] / 12.0
        );
        
        double vAvg = (
            s_v[ty-1][tx-1] / 12.0 + 
            s_v[ty-1][tx]   / 6.0  + 
            s_v[ty-1][tx+1] / 12.0 + 
            s_v[ty][tx-1]   / 6.0  + 
            s_v[ty][tx+1]   / 6.0  + 
            s_v[ty+1][tx-1] / 12.0 + 
            s_v[ty+1][tx]   / 6.0  + 
            s_v[ty+1][tx+1] / 12.0
        );

        // Then compute Horn-Schunck update
        double ix = Ix[global_idx];
        double iy = Iy[global_idx];
        double it = It[global_idx];

        double denom = alpha * alpha + ix * ix + iy * iy;
        double p = (ix * uAvg + iy * vAvg + it);
        
        // Write final results directly
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
    Mat frame1_original = imread(filename1, 0);
    Mat frame2_original = imread(filename2, 0);
   
    if (frame1_original.empty() || frame2_original.empty()) {
        cerr << "Error loading images!" << endl;
        cerr << "Make sure " << filename1 << " and " << filename2 << " exist in: " << filesystem::current_path() << endl;
        return -1;
    }

    // Define sizes to test
    vector<int> test_sizes = {128, 256, 512, 1024, 2048, 4096};  // Test smaller sizes first
    
    for (int size : test_sizes) {
        cout << "\nProcessing size: " << size << "x" << size << endl;
        
        // Resize images
        Mat frame1, frame2;
        resize(frame1_original, frame1, Size(size, size));
        resize(frame2_original, frame2, Size(size, size));
        
        cout << "Resized images - Frame1: " << frame1.size() << " Frame2: " << frame2.size() << endl;
        
        // Image size and grid sizes
        cudaError_t GPU_ERROR;
        int ny = frame1.rows;
        int nx = frame1.cols;
        size_t size_bytes = nx * ny * sizeof(double);
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
        double *IxHost, *IyHost, *ItHost;
        double *uHost, *vHost;

        // Allocate pinned memory with error checking
        GPU_ERROR = cudaHostAlloc(&IxHost, size_bytes, cudaHostAllocDefault);
        if (GPU_ERROR != cudaSuccess) {
            cerr << "Failed to allocate pinned memory for IxHost" << endl;
            return -1;
        }
        GPU_ERROR = cudaHostAlloc(&IyHost, size_bytes, cudaHostAllocDefault);
        if (GPU_ERROR != cudaSuccess) {
            cerr << "Failed to allocate pinned memory for IyHost" << endl;
            return -1;
        }
        GPU_ERROR = cudaHostAlloc(&ItHost, size_bytes, cudaHostAllocDefault);
        if (GPU_ERROR != cudaSuccess) {
            cerr << "Failed to allocate pinned memory for ItHost" << endl;
            return -1;
        }
        GPU_ERROR = cudaHostAlloc(&uHost, size_bytes, cudaHostAllocDefault);
        if (GPU_ERROR != cudaSuccess) {
            cerr << "Failed to allocate pinned memory for uHost" << endl;
            return -1;
        }
        GPU_ERROR = cudaHostAlloc(&vHost, size_bytes, cudaHostAllocDefault);
        if (GPU_ERROR != cudaSuccess) {
            cerr << "Failed to allocate pinned memory for vHost" << endl;
            return -1;
        }

        // Copy data with error checking
        GPU_ERROR = cudaMemcpy(IxHost, IxMat.ptr<double>(), size_bytes, cudaMemcpyHostToHost);
        GPU_ERROR = cudaMemcpy(IyHost, IyMat.ptr<double>(), size_bytes, cudaMemcpyHostToHost);
        GPU_ERROR = cudaMemcpy(ItHost, ItMat.ptr<double>(), size_bytes, cudaMemcpyHostToHost);

        double *IxDevice, *IyDevice, *ItDevice;
        // Allocate memory on device
        GPU_ERROR = cudaMalloc(&IxDevice, size_bytes);
        if (GPU_ERROR != cudaSuccess) {
            cerr << "Failed to allocate Ix device memory" << endl;
            continue;  // Skip to next size if allocation fails
        }
        GPU_ERROR = cudaMalloc(&IyDevice, size_bytes);
        if (GPU_ERROR != cudaSuccess) {
            cudaFree(IxDevice);
            cerr << "Failed to allocate Iy device memory" << endl;
            continue;
        }
        GPU_ERROR = cudaMalloc(&ItDevice, size_bytes);
        if (GPU_ERROR != cudaSuccess) {
            cudaFree(IxDevice);
            cudaFree(IyDevice);
            cerr << "Failed to allocate It device memory" << endl;
            continue;
        }

        // Copy derivatives to device
        GPU_ERROR = cudaMemcpy(IxDevice, IxHost, size_bytes, cudaMemcpyHostToDevice);
        GPU_ERROR = cudaMemcpy(IyDevice, IyHost, size_bytes, cudaMemcpyHostToDevice);
        GPU_ERROR = cudaMemcpy(ItDevice, ItHost, size_bytes, cudaMemcpyHostToDevice);

        // Allocate device memory for results
        double *u_d, *v_d;
        GPU_ERROR = cudaMalloc(&u_d, size_bytes);
        GPU_ERROR = cudaMalloc(&v_d, size_bytes);

        // Initialize device memory to zero
        GPU_ERROR = cudaMemset(u_d, 0, size_bytes);
        GPU_ERROR = cudaMemset(v_d, 0, size_bytes);

        // Add timing variables
        cudaEvent_t start, stop;
        GPU_ERROR = cudaEventCreate(&start);
        GPU_ERROR = cudaEventCreate(&stop);
        
        // Start timing
        GPU_ERROR = cudaEventRecord(start);

        // Compute optical flow with fused kernel
        int iterations = 200;
        double alpha = 1;
        
        // Create streams
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) {
            GPU_ERROR = cudaStreamCreate(&streams[i]);
            if (GPU_ERROR != cudaSuccess) {
                cerr << "Failed to create stream " << i << endl;
                return -1;
            }
        }

        // Divide the image into sections for parallel processing
        int rows_per_section = ny / NUM_STREAMS;
        // Ensure rows_per_section is a multiple of block size
        rows_per_section = ((rows_per_section + block.y - 1) / block.y) * block.y;
        size_t section_bytes = nx * rows_per_section * sizeof(double);

        // Process iterations
        for(int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < NUM_STREAMS; i++) {
                int start_row = i * rows_per_section;
                int global_offset = start_row * nx;
                
                // Launch kernel asynchronously
                dim3 section_grid((nx + block.x - 1) / block.x, 
                                 (rows_per_section + block.y - 1) / block.y);
                
                fused_horn_schunk<<<section_grid, block, 0, streams[i]>>>(
                    u_d + global_offset, 
                    v_d + global_offset, 
                    IxDevice + global_offset, 
                    IyDevice + global_offset, 
                    ItDevice + global_offset,
                    alpha, nx, rows_per_section
                );
            }
            
            cudaDeviceSynchronize();
        }

        // Stop timing
        GPU_ERROR = cudaEventRecord(stop);
        GPU_ERROR = cudaEventSynchronize(stop);
        
        float elapsed_time_ms;
        GPU_ERROR = cudaEventElapsedTime(&elapsed_time_ms, start, stop);
        double elapsed_time_s = elapsed_time_ms / 1000.0;
        
        // Get performance metrics
        auto [measured_tflops, ai, peak_tflops] = analyze_performance(nx, ny, iterations, elapsed_time_s, gridDimX, gridDimY);
        
        // Print performance table
        printf("\nGrid Size | TFLOPS | AI | Peak TFLOPS | Time(s) | Iterations\n");
        printf("---------|--------|----|-----------  |---------|------------|\n");
        printf("%4dx%4d | %6.6f | %6.6f | %6.6f | %7.6f | %10d |\n",
               nx, ny, measured_tflops, ai, peak_tflops, elapsed_time_s, iterations);

        // Copy results back to host
        vector<double> u_result(nx * ny);
        vector<double> v_result(nx * ny);
        GPU_ERROR = cudaMemcpy(u_result.data(), u_d, size_bytes, cudaMemcpyDeviceToHost);
        GPU_ERROR = cudaMemcpy(v_result.data(), v_d, size_bytes, cudaMemcpyDeviceToHost);

        // Visualize and save results
        Mat img_color, flowX, flowY;
        cvtColor(frame1, img_color, COLOR_GRAY2BGR);
        flowX = vectorToMat<double>(u_result, ny, nx, CV_64F);
        flowY = vectorToMat<double>(v_result, ny, nx, CV_64F); 
        drawOpticalFlow(flowX, flowY, img_color);

        Mat flow_vis;
        visualizeFlowHSV(flowX, flowY, flow_vis);

        string size_str = to_string(size);
        imwrite("outputs/CUDA_optical_flow_" + outputname + "_" + size_str + "x" + size_str + ".png", img_color);
        imwrite("outputs/CUDA_optical_flow_hsv_" + outputname + "_" + size_str + "x" + size_str + ".png", flow_vis);

        // Cleanup
        if (cudaEventDestroy(start) != cudaSuccess) {
            cerr << "Error destroying start event" << endl;
        }
        if (cudaEventDestroy(stop) != cudaSuccess) {
            cerr << "Error destroying stop event" << endl;
        }
        if (cudaFree(IxDevice) != cudaSuccess) {
            cerr << "Error freeing IxDevice" << endl;
        }
        if (cudaFree(IyDevice) != cudaSuccess) {
            cerr << "Error freeing IyDevice" << endl;
        }
        if (cudaFree(ItDevice) != cudaSuccess) {
            cerr << "Error freeing ItDevice" << endl;
        }
        if (cudaFree(u_d) != cudaSuccess) {
            cerr << "Error freeing u_d" << endl;
        }
        if (cudaFree(v_d) != cudaSuccess) {
            cerr << "Error freeing v_d" << endl;
        }

        // Free pinned memory
        if (cudaFreeHost(IxHost) != cudaSuccess) {
            cerr << "Error freeing IxHost" << endl;
        }
        if (cudaFreeHost(IyHost) != cudaSuccess) {
            cerr << "Error freeing IyHost" << endl;
        }
        if (cudaFreeHost(ItHost) != cudaSuccess) {
            cerr << "Error freeing ItHost" << endl;
        }
        if (cudaFreeHost(uHost) != cudaSuccess) {
            cerr << "Error freeing uHost" << endl;
        }
        if (cudaFreeHost(vHost) != cudaSuccess) {
            cerr << "Error freeing vHost" << endl;
        }
        
        // Force synchronization before next iteration
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cerr << "Error in device synchronization" << endl;
        }

        // Clean up streams
        for (int i = 0; i < NUM_STREAMS; i++) {
            if (cudaStreamDestroy(streams[i]) != cudaSuccess) {
                cerr << "Error destroying stream " << i << endl;
            }
        }
    }

    return 0;
}