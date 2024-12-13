#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <tuple>
#include <mpi.h>

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

// Error checking macros
#ifdef USE_HIP
#define CHECK_CUDA(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error in %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#else
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

#define CHECK_MPI(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char error_string[MPI_MAX_ERROR_STRING]; \
            int length; \
            MPI_Error_string(err, error_string, &length); \
            fprintf(stderr, "MPI error in %s:%d: %s\n", \
                    __FILE__, __LINE__, error_string); \
            MPI_Abort(MPI_COMM_WORLD, err); \
        } \
    } while(0)

// Hardware specifications for roofline model
const double PEAK_MEMORY_BANDWIDTH = 160e9;  // 160 GB/s
const double PEAK_FLOP_RATE = 2.0e12;       // 2.0 TFLOP/s

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
    if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0) {
        s_u[ty-HALO][tx-HALO] = u[global_idx - nx - 1];
        s_v[ty-HALO][tx-HALO] = v[global_idx - nx - 1];
    }
    
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && x < nx - 1 && y > 0) {
        s_u[ty-HALO][tx+HALO] = u[global_idx - nx + 1];
        s_v[ty-HALO][tx+HALO] = v[global_idx - nx + 1];
    }
    
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && x > 0 && y < ny - 1) {
        s_u[ty+HALO][tx-HALO] = u[global_idx + nx - 1];
        s_v[ty+HALO][tx-HALO] = v[global_idx + nx - 1];
    }
    
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && x < nx - 1 && y < ny - 1) {
        s_u[ty+HALO][tx+HALO] = u[global_idx + nx + 1];
        s_v[ty+HALO][tx+HALO] = v[global_idx + nx + 1];
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Compute flow updates only for interior points
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
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

// Add these new structures and functions
struct ImageQuadrant {
    int start_x, start_y;
    int width, height;
    int halo_width;
    vector<double> data;
    cudaStream_t stream;
};

struct HaloRegion {
    vector<double> top, bottom, left, right;
};

ImageQuadrant getQuadrantInfo(int rank, int nx, int ny, int halo_width) {
    ImageQuadrant quad;
    quad.halo_width = halo_width;
    
    // Calculate base dimensions for each quadrant
    int base_width = nx / 2;
    int base_height = ny / 2;
    
    // Determine quadrant position based on rank
    quad.start_x = (rank % 2) * base_width;
    quad.start_y = (rank / 2) * base_height;
    quad.width = base_width;
    quad.height = base_height;
    
    // Create stream for this quadrant
    CHECK_CUDA(cudaStreamCreate(&quad.stream));
    
    return quad;
}

void exchangeHalos(ImageQuadrant& quad, double* u_d, double* v_d, int rank, MPI_Comm comm) {
    int world_size;
    MPI_Comm_size(comm, &world_size);
    
    HaloRegion send_halos, recv_halos;
    
    // Prepare halo data
    int w = quad.width;
    int h = quad.height;
    int hw = quad.halo_width;
    
    // Resize halo buffers
    send_halos.top.resize(w * hw);
    send_halos.bottom.resize(w * hw);
    send_halos.left.resize(h * hw);
    send_halos.right.resize(h * hw);
    recv_halos = send_halos;  // Same sizes for receiving
    
    // Determine neighbor ranks
    int top_rank = (rank >= 2) ? rank - 2 : -1;
    int bottom_rank = (rank < 2) ? rank + 2 : -1;
    int left_rank = (rank % 2 == 1) ? rank - 1 : -1;
    int right_rank = (rank % 2 == 0) ? rank + 1 : -1;
    
    // Asynchronously copy halo data from device to host
    CHECK_CUDA(cudaMemcpyAsync(send_halos.top.data(), 
                              u_d + quad.start_y * quad.width,
                              quad.width * quad.halo_width * sizeof(double),
                              cudaMemcpyDeviceToHost, 
                              quad.stream));
    
    // Similar for other directions...
    
    // Ensure device-to-host transfers are complete
    CHECK_CUDA(cudaStreamSynchronize(quad.stream));
    
    // Non-blocking MPI communications
    vector<MPI_Request> requests;
    
    if (top_rank != -1) {
        CHECK_MPI(MPI_Isend(send_halos.top.data(), w * hw, MPI_DOUBLE, 
                           top_rank, 0, comm, &requests.emplace_back()));
        CHECK_MPI(MPI_Irecv(recv_halos.top.data(), w * hw, MPI_DOUBLE, 
                           top_rank, 0, comm, &requests.emplace_back()));
    }
    
    // Similar for other directions...
    
    // Wait for all MPI communications
    if (!requests.empty()) {
        CHECK_MPI(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
    }
    
    // Asynchronously copy received halos back to device
    CHECK_CUDA(cudaMemcpyAsync(u_d + (quad.start_y - quad.halo_width) * quad.width,
                              recv_halos.top.data(),
                              quad.width * quad.halo_width * sizeof(double),
                              cudaMemcpyHostToDevice, 
                              quad.stream));
    
    // Similar for other directions...
    
    // Ensure all copies are complete
    CHECK_CUDA(cudaStreamSynchronize(quad.stream));
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    CHECK_MPI(MPI_Init(&argc, &argv));
    
    int world_rank, world_size;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    
    if (world_size != 4) {
        if (world_rank == 0) {
            cerr << "This program requires exactly 4 MPI processes" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Select GPU based on rank
    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    CHECK_CUDA(cudaSetDevice(world_rank % num_gpus));

    // Load images only on rank 0
    Mat frame1_original, frame2_original;
    int nx, ny;
    
    if (world_rank == 0) {
        frame1_original = imread(argv[1], 0);
        frame2_original = imread(argv[2], 0);
        
        if (frame1_original.empty() || frame2_original.empty()) {
            cerr << "Error loading images!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }
        
        nx = frame1_original.cols;
        ny = frame1_original.rows;
    }
    
    // Broadcast dimensions to all processes
    CHECK_MPI(MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD));

    // Get quadrant information for this process
    ImageQuadrant quad = getQuadrantInfo(world_rank, nx, ny, 1);
    
    // Allocate memory for this quadrant
    size_t quad_size = (quad.width + 2 * quad.halo_width) * 
                      (quad.height + 2 * quad.halo_width) * sizeof(double);
    
    double *u_d, *v_d, *IxDevice, *IyDevice, *ItDevice;
    CHECK_CUDA(cudaMalloc(&u_d, quad_size));
    CHECK_CUDA(cudaMalloc(&v_d, quad_size));
    CHECK_CUDA(cudaMalloc(&IxDevice, quad_size));
    CHECK_CUDA(cudaMalloc(&IyDevice, quad_size));
    CHECK_CUDA(cudaMalloc(&ItDevice, quad_size));

    // Main computation loop
    for(int iter = 0; iter < iterations; iter++) {
        // Exchange halos between processes
        exchangeHalos(quad, u_d, v_d, world_rank, MPI_COMM_WORLD);
        
        // Launch kernel in quadrant's stream
        fused_horn_schunk<<<grid, block, 0, quad.stream>>>(
            u_d, v_d, IxDevice, IyDevice, ItDevice, 
            alpha, quad.width, quad.height
        );
        
        CHECK_CUDA(cudaGetLastError());
    }

    // Gather results
    vector<double> u_result(quad.width * quad.height);
    vector<double> v_result(quad.width * quad.height);
    
    CHECK_CUDA(cudaMemcpyAsync(u_result.data(), u_d, 
                              quad.width * quad.height * sizeof(double),
                              cudaMemcpyDeviceToHost, quad.stream));
    CHECK_CUDA(cudaMemcpyAsync(v_result.data(), v_d, 
                              quad.width * quad.height * sizeof(double),
                              cudaMemcpyDeviceToHost, quad.stream));
    
    CHECK_CUDA(cudaStreamSynchronize(quad.stream));

    if (world_rank == 0) {
        vector<double> full_u(nx * ny), full_v(nx * ny);
        
        // Copy rank 0's results
        for (int y = 0; y < quad.height; y++) {
            memcpy(&full_u[y * nx + quad.start_x],
                  &u_result[y * quad.width],
                  quad.width * sizeof(double));
            memcpy(&full_v[y * nx + quad.start_x],
                  &v_result[y * quad.width],
                  quad.width * sizeof(double));
        }
        
        // Gather results from other ranks
        for (int i = 1; i < world_size; i++) {
            ImageQuadrant other_quad = getQuadrantInfo(i, nx, ny, 1);
            vector<double> temp_u(other_quad.width * other_quad.height);
            vector<double> temp_v(other_quad.width * other_quad.height);
            
            CHECK_MPI(MPI_Recv(temp_u.data(), temp_u.size(), MPI_DOUBLE, 
                              i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            CHECK_MPI(MPI_Recv(temp_v.data(), temp_v.size(), MPI_DOUBLE, 
                              i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            
            for (int y = 0; y < other_quad.height; y++) {
                memcpy(&full_u[(y + other_quad.start_y) * nx + other_quad.start_x],
                      &temp_u[y * other_quad.width],
                      other_quad.width * sizeof(double));
                memcpy(&full_v[(y + other_quad.start_y) * nx + other_quad.start_x],
                      &temp_v[y * other_quad.width],
                      other_quad.width * sizeof(double));
            }
        }
        
        // Visualize results
        Mat flowX = Mat(ny, nx, CV_64F, full_u.data());
        Mat flowY = Mat(ny, nx, CV_64F, full_v.data());
        
        Mat img_color;
        cvtColor(frame1_original, img_color, COLOR_GRAY2BGR);
        drawOpticalFlow(flowX, flowY, img_color);
        
        Mat flow_vis;
        visualizeFlowHSV(flowX, flowY, flow_vis);
        
        imwrite("outputs/MPI_CUDA_flow_" + string(argv[3]) + ".png", img_color);
        imwrite("outputs/MPI_CUDA_flow_hsv_" + string(argv[3]) + ".png", flow_vis);
    } else {
        // Send results to rank 0
        CHECK_MPI(MPI_Send(u_result.data(), u_result.size(), MPI_DOUBLE, 
                          0, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Send(v_result.data(), v_result.size(), MPI_DOUBLE, 
                          0, 1, MPI_COMM_WORLD));
    }

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(quad.stream));
    CHECK_CUDA(cudaFree(u_d));
    CHECK_CUDA(cudaFree(v_d));
    CHECK_CUDA(cudaFree(IxDevice));
    CHECK_CUDA(cudaFree(IyDevice));
    CHECK_CUDA(cudaFree(ItDevice));
    
    MPI_Finalize();
    return 0;
}