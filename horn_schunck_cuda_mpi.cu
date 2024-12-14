#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>
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
#define cudaFreeHost            hipHostFree
#define cudaEventCreate         hipEventCreate
#define cudaEventRecord         hipEventRecord
#define cudaEventSynchronize    hipEventSynchronize
#define cudaEventElapsedTime    hipEventElapsedTime
#define cudaEventDestroy        hipEventDestroy
#define cudaEvent_t             hipEvent_t
#define cudaHostAlloc           hipHostMalloc
#define cudaHostAllocDefault    hipHostMallocDefault
#define cudaMemcpyHostToHost    hipMemcpyHostToHost
#define cudaMemcpy2D            hipMemcpy2D

#else

#include <cuda.h>

#endif

using namespace cv;
using namespace std;

// Hardware specifications for roofline model
const double PEAK_MEMORY_BANDWIDTH = 400e9;  // 400 GB/s
const double PEAK_FLOP_RATE = 2.5e12;       // 2.5 TFLOP/s
const int GRID_SIZE = 2;
const int HALO_CELLS = 1;

// Calculate theoretical peak performance based on arithmetic intensity
double calculate_roofline(double arithmetic_intensity) {
    return min(PEAK_FLOP_RATE, PEAK_MEMORY_BANDWIDTH * arithmetic_intensity);
}

// Modify the analyze_performance function to return the metrics
tuple<double, double, double> analyze_performance(int nx, int ny, int iterations, double elapsed_time, int gridDimX, int gridDimY) {
    // Calculate total pixels processed
    size_t total_pixels = 4 * nx * ny;
    
    int haloComputations = 4 * (gridDimX * nx + gridDimY * ny); // number of points involved in halo computations

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
    Mat kernelX = (Mat_<double>(2, 2) << 0.25, -0.25, 0.25, -0.25); // Kernel for dx
    Mat kernelY = (Mat_<double>(2, 2) << 0.25, 0.25, -.25, -.25); // Kernel for dy
    Mat kernelT = (Mat_<double>(2, 2) << 0.25, 0.25, 0.25, 0.25);   // Kernel for dt

    Mat im1_d, im2_d;
    im1.convertTo(im1_d, CV_64FC1);
    im2.convertTo(im2_d, CV_64FC1);

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
                            double alpha, const int nx, const int ny, const int rank) {
    // Define halo width
    constexpr int HALO = 1;
    
    // Determine MPI grid position (2x2 grid)
    const int row = rank / 2;    // 0 for ranks 0,1 and 1 for ranks 2,3
    const int col = rank % 2;    // 0 for ranks 0,2 and 1 for ranks 1,3

    // Shared memory dimensions including halos
    __shared__ double s_u[18][18];
    __shared__ double s_v[18][18];
    
    // Global indices within the local block (including halo cells)
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Local indices within the shared memory block
    int tx = threadIdx.x + HALO;
    int ty = threadIdx.y + HALO;
    
    // Global linear index
    int global_idx = y * nx + x;
    
    // Calculate local boundaries considering MPI rank position
    bool isLeftBoundary = (col == 0);
    bool isRightBoundary = (col == 1);
    bool isTopBoundary = (row == 0);
    bool isBottomBoundary = (row == 1);

    // Load center data (including halo cells)
    if (x < nx && y < ny) {
        s_u[ty][tx] = u[global_idx];
        s_v[ty][tx] = v[global_idx];
    }
    
    // Load halo data - only load if not at global boundary
    // Top halo
    if (threadIdx.y == 0 && y > 0 && !(isTopBoundary && y == 1)) {
        s_u[ty-HALO][tx] = u[global_idx - nx];
        s_v[ty-HALO][tx] = v[global_idx - nx];
    }
    
    // Bottom halo
    if (threadIdx.y == blockDim.y - 1 && y < ny - 1 && !(isBottomBoundary && y == ny - 2)) {
        s_u[ty+HALO][tx] = u[global_idx + nx];
        s_v[ty+HALO][tx] = v[global_idx + nx];
    }
    
    // Left halo
    if (threadIdx.x == 0 && x > 0 && !(isLeftBoundary && x == 1)) {
        s_u[ty][tx-HALO] = u[global_idx - 1];
        s_v[ty][tx-HALO] = v[global_idx - 1];
    }
    
    // Right halo
    if (threadIdx.x == blockDim.x - 1 && x < nx - 1 && !(isRightBoundary && x == nx - 2)) {
        s_u[ty][tx+HALO] = u[global_idx + 1];
        s_v[ty][tx+HALO] = v[global_idx + 1];
    }
    
    // Corner halos - only load if not at global boundaries
    // Top-left
    if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0 && 
        !(isLeftBoundary && x == 1) && !(isTopBoundary && y == 1)) {
        s_u[ty-HALO][tx-HALO] = u[global_idx - nx - 1];
        s_v[ty-HALO][tx-HALO] = v[global_idx - nx - 1];
    }
    
    // Top-right
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && x < nx - 1 && y > 0 && 
        !(isRightBoundary && x == nx - 2) && !(isTopBoundary && y == 1)) {
        s_u[ty-HALO][tx+HALO] = u[global_idx - nx + 1];
        s_v[ty-HALO][tx+HALO] = v[global_idx - nx + 1];
    }
    
    // Bottom-left
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && x > 0 && y < ny - 1 && 
        !(isLeftBoundary && x == 1) && !(isBottomBoundary && y == ny - 2)) {
        s_u[ty+HALO][tx-HALO] = u[global_idx + nx - 1];
        s_v[ty+HALO][tx-HALO] = v[global_idx + nx - 1];
    }
    
    // Bottom-right
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && x < nx - 1 && y < ny - 1 && 
        !(isRightBoundary && x == nx - 2) && !(isBottomBoundary && y == ny - 2)) {
        s_u[ty+HALO][tx+HALO] = u[global_idx + nx + 1];
        s_v[ty+HALO][tx+HALO] = v[global_idx + nx + 1];
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Skip halo cells and global boundaries
    bool isHaloCell = (x < HALO || x >= nx - HALO || y < HALO || y >= ny - HALO);
    bool isGlobalBoundary = (isLeftBoundary && x == HALO) || 
                           (isRightBoundary && x == nx - HALO - 1) ||
                           (isTopBoundary && y == HALO) || 
                           (isBottomBoundary && y == ny - HALO - 1);
    
    double uAvg = 0;
    double vAvg = 0;

    // Compute averages only for interior points
    if (!isHaloCell && !isGlobalBoundary) {
        // Compute weighted averages using 3x3 neighborhood
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

        // Calculate index for Ix, Iy, It (which don't have halo cells)
        int interior_idx = (y - HALO) * (nx - 2*HALO) + (x - HALO);
        
        double ix = Ix[interior_idx];
        double iy = Iy[interior_idx];
        double it = It[interior_idx];

        double denom = alpha * alpha + ix * ix + iy * iy;
        double p = (ix * uAvg + iy * vAvg + it);
        
        // Update u and v in global memory
        u[global_idx] = uAvg - ix * (p / denom);
        v[global_idx] = vAvg - iy * (p / denom);
    }
}

struct GridDimensions {
    int total_rows;
    int total_cols;
    int local_rows;
    int local_cols;
};

void broadcast_dimensions(GridDimensions& dims, int rank) {
    int dims_array[4] = {dims.total_rows, dims.total_cols, dims.local_rows, dims.local_cols};
    MPI_Bcast(dims_array, 4, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        dims.total_rows = dims_array[0];
        dims.total_cols = dims_array[1];
        dims.local_rows = dims_array[2];
        dims.local_cols = dims_array[3];
    }
}

void get_grid_position(int rank, int& row, int& col) {
    row = rank / GRID_SIZE;
    col = rank % GRID_SIZE;
}

// Function to distribute data from rank 0 to all ranks
vector<double> distribute_data(const vector<double>& flatData, int rank, const GridDimensions& dims) {
    vector<double> localData(dims.local_rows * dims.local_cols);
    int row, col;
    get_grid_position(rank, row, col);
    
    if (rank == 0) {
        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                int current_rank = r * GRID_SIZE + c;
                vector<double> gridData(dims.local_rows * dims.local_cols);
                for (int i = 0; i < dims.local_rows; i++) {
                    for (int j = 0; j < dims.local_cols; j++) {
                        int global_i = r * dims.local_rows + i;
                        int global_j = c * dims.local_cols + j;
                        gridData[i * dims.local_cols + j] = 
                            flatData[global_i * dims.total_cols + global_j];
                    }
                }               
                if (current_rank != 0) {
                    MPI_Send(gridData.data(), dims.local_rows * dims.local_cols,
                            MPI_DOUBLE, current_rank, 0, MPI_COMM_WORLD);
                } else {
                    localData = gridData;
                }
            }
        }
    } else {
        // Receive data from rank 0
        MPI_Recv(localData.data(), dims.local_rows * dims.local_cols,
                 MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    return localData;
}


vector<double> gather_data(const vector<double>& localData, const GridDimensions& dims, int rank) {
    vector<double> globalData;
    
    if (rank == 0) {
        globalData.resize(dims.total_rows * dims.total_cols);
        
        int row_start = 0;
        int col_start = 0;
        // Copy rank 0's data (skipping halo cells)
        for (int i = 0; i < dims.local_rows; i++) {
            for (int j = 0; j < dims.local_cols; j++) {
                // Calculate source index (with halo cells)
                int local_idx = ((i + HALO_CELLS) * (dims.local_cols + 2*HALO_CELLS)) + 
                              (j + HALO_CELLS);
                              
                // Calculate destination index in global data
                int global_idx = (row_start + i) * dims.total_cols + 
                               (col_start + j);
                               
                globalData[global_idx] = localData[local_idx];
            }
        }

        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                int src_rank = r * GRID_SIZE + c;
                int row_start = r * dims.local_rows;
                int col_start = c * dims.local_cols;
                if (src_rank != 0) {
                    vector<double> recvData((dims.local_rows + 2*HALO_CELLS) * 
                                          (dims.local_cols + 2*HALO_CELLS));
                                          
                    MPI_Recv(recvData.data(), recvData.size(),
                            MPI_DOUBLE, src_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Copy data excluding halo cells
                    for (int i = 0; i < dims.local_rows; i++) {
                        for (int j = 0; j < dims.local_cols; j++) {
                            // Calculate source index (with halo cells)
                            int local_idx = ((i + HALO_CELLS) * (dims.local_cols + 2*HALO_CELLS)) + 
                                          (j + HALO_CELLS);
                                          
                            // Calculate destination index in global data
                            int global_idx = (row_start + i) * dims.total_cols + 
                                           (col_start + j);
                                           
                            globalData[global_idx] = recvData[local_idx];
                        }
                    }
                }
            }
        }
    } else {
        // Send local data (including halo cells) to rank 0
        cout << "rank: " << rank << " is sending data." << endl;
        MPI_Send(localData.data(), localData.size(),
                 MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    return globalData;
}
void printInnerPoints(const std::vector<double>& flatArray, int rows, int cols, int rank) {
    int count = 0;
    
    // Start from (1,1) and avoid edges
    for (int i = 50; i < 75; i++) {
        for (int j = 50; j < 75; j++) {
            int index = i * cols + j;
            cout << "Rank " << rank << ": " 
                      << flatArray[index]
                      << " at position (" << i << "," << j << ")" 
                      << endl;
            count++;
        }
    }
}

void exchangeHalos(double* d_matrix,           
                   const int local_nx,          
                   const int local_ny,         
                   const int halo_width,        
                   const int grid_dims[2],     
                   const int rank,             
                   MPI_Comm comm) {           
    cudaError_t GPU_ERROR;
    const int padded_nx = local_nx + 2 * halo_width;
    const int padded_ny = local_ny + 2 * halo_width;
    
    // Create MPI type for a single column (non-contiguous)
    MPI_Datatype column_type;
    MPI_Type_vector(local_ny,           // count - number of blocks
                    1,                   // blocklength - elements per block
                    padded_nx,          // stride - elements between start of each block
                    MPI_DOUBLE,          // base type
                    &column_type);
    MPI_Type_commit(&column_type);
    
    // Create MPI type for a single row (contiguous)
    MPI_Datatype row_type;
    MPI_Type_contiguous(local_nx,       // count - number of elements
                        MPI_DOUBLE,      // base type
                        &row_type);
    MPI_Type_commit(&row_type);

    double *h_send_left, *h_send_right, *h_recv_left, *h_recv_right;
    cudaHostAlloc(&h_send_left, local_ny * sizeof(double),cudaHostAllocDefault);   // Pinned allocation
    cudaHostAlloc(&h_send_right, local_ny * sizeof(double),cudaHostAllocDefault);
    cudaHostAlloc(&h_recv_left, local_ny * sizeof(double),cudaHostAllocDefault);
    cudaHostAlloc(&h_recv_right, local_ny * sizeof(double),cudaHostAllocDefault);
    
    // Get grid position
    int row = rank / grid_dims[1];
    int col = rank % grid_dims[1];
    
    // Left-Right exchanges
    if (grid_dims[1] > 1) {  // Only if multiple columns
        // Get neighbor ranks
        int left_rank = (col == 0) ? rank + grid_dims[1] - 1 : rank - 1;
        int right_rank = (col == grid_dims[1] - 1) ? rank - grid_dims[1] + 1 : rank + 1;
        
        // Copy left edge column to host
        GPU_ERROR = cudaMemcpy2D(h_send_left, sizeof(double),
                     d_matrix + halo_width, padded_nx * sizeof(double),
                     sizeof(double), local_ny, cudaMemcpyDeviceToHost);
        
        // Copy right edge column to host
         GPU_ERROR = cudaMemcpy2D(h_send_right, sizeof(double),
                     d_matrix + halo_width + local_nx - 1, padded_nx * sizeof(double),
                     sizeof(double), local_ny, cudaMemcpyDeviceToHost);
        
        // Exchange with left neighbor
        MPI_Sendrecv(h_send_left, 1, column_type, left_rank, 0,
                     h_recv_right, 1, column_type, right_rank, 0,
                     comm, MPI_STATUS_IGNORE);
        
        // Exchange with right neighbor
        MPI_Sendrecv(h_send_right, 1, column_type, right_rank, 1,
                     h_recv_left, 1, column_type, left_rank, 1,
                     comm, MPI_STATUS_IGNORE);
        
        // Copy received data to device halos
         GPU_ERROR = cudaMemcpy2D(d_matrix, padded_nx * sizeof(double),
                     h_recv_left, sizeof(double),
                     sizeof(double), local_ny, cudaMemcpyHostToDevice);
                     
         GPU_ERROR = cudaMemcpy2D(d_matrix + padded_nx - 1, padded_nx * sizeof(double),
                     h_recv_right, sizeof(double),
                     sizeof(double), local_ny, cudaMemcpyHostToDevice);
    }
    
    // Top-Bottom exchanges using MPI types directly
    if (grid_dims[0] > 1) {  // Only if multiple rows
        // Get neighbor ranks
        int top_rank = (row == 0) ? rank + local_ny * (grid_dims[0] - 1) : rank - grid_dims[1];
        int bottom_rank = (row == grid_dims[0] - 1) ? rank - local_ny * (grid_dims[0] - 1) : rank + grid_dims[1];
        
        // Exchange with top neighbor - send from first row of data, receive into top halo
        double* top_send = d_matrix + (halo_width * padded_nx) + halo_width;
        double* top_recv = d_matrix + halo_width;
        MPI_Sendrecv(top_send, 1, row_type, top_rank, 2,
                     top_recv, 1, row_type, top_rank, 2,
                     comm, MPI_STATUS_IGNORE);
        
        // Exchange with bottom neighbor - send from last row of data, receive into bottom halo
        double* bottom_send = d_matrix + ((halo_width + local_ny - 1) * padded_nx) + halo_width;
        double* bottom_recv = d_matrix + ((padded_ny - 1) * padded_nx) + halo_width;
        MPI_Sendrecv(bottom_send, 1, row_type, bottom_rank, 3,
                     bottom_recv, 1, row_type, bottom_rank, 3,
                     comm, MPI_STATUS_IGNORE);
    }
    
    // Cleanup
    delete[] h_send_left;
    delete[] h_send_right;
    delete[] h_recv_left;
    delete[] h_recv_right;
    MPI_Type_free(&column_type);
    MPI_Type_free(&row_type);
}

// Main function demonstrating usage
int main(int argc, char* argv[]) {
    cout << "Running Horn-Schunck optical flow..." << endl;
    string filename1 = argv[1];
    string filename2 = argv[2];
    string outputname = argv[3];
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    GridDimensions dims;
    vector<double> inputData;
    vector<double>IxMain;
    vector<double>IyMain;
    vector<double>ItMain;

    Mat frame1;
    Mat frame2;
    if (rank == 0) {
        // Load two consecutive frames
        frame1 = imread(filename1, 0);
        frame2 = imread(filename2, 0);
    
        if (frame1.empty() || frame2.empty()) {
            cerr << "Error loading images!" << endl;
            cerr << "Make sure " << filename1 << " and " << filename2 << " exist in: " << filesystem::current_path() << endl;
            return -1;
        }
   
        cout << "Loaded images - Frame1: " << frame1.size() << " Frame2: " << frame2.size() << endl;

        dims.total_rows = frame1.rows;
        dims.total_cols = frame1.cols;
        dims.local_rows = dims.total_rows / GRID_SIZE;
        dims.local_cols = dims.total_cols / GRID_SIZE;

        // Compute image derivatives
        Mat IxMat, IyMat, ItMat;
        computeDerivatives(frame1, frame2, IxMat, IyMat, ItMat);

        // Convert derivatives to vectors
        IxMain = matToVector<double>(IxMat);
        IyMain = matToVector<double>(IyMat);
        ItMain = matToVector<double>(ItMat);
    }

    // // Distribute dimensions to all ranks
    broadcast_dimensions(dims, rank);
    
    // // Distribute the data
    vector<double> localIx = distribute_data(IxMain, rank, dims);
    vector<double> localIy = distribute_data(IyMain, rank, dims);
    vector<double> localIt = distribute_data(ItMain, rank, dims);
    MPI_Barrier(MPI_COMM_WORLD);
    cout<< "data distributed!" << endl;

    int row = 0;
    int col = 0;
    get_grid_position(rank, row, col);
   
    // Image size and grid sizes
    cudaError_t GPU_ERROR;
    int nxDer = dims.local_cols;
    int nyDer = dims.local_rows;
    int nx = dims.local_cols + 2; // add 2 for halo cells
    int ny = dims.local_rows + 2; // add 2 for halo cells
    size_t sizeDer = nxDer * nyDer * sizeof(double);
    size_t size = nx * ny * sizeof(double);

    int BLOCK_DIM_X = 16;
    int BLOCK_DIM_Y = 16;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    int gridDimX = (nx + block.x - 1) / block.x;
    int gridDimY = (ny + block.y - 1) / block.y;
    dim3 grid(gridDimX, gridDimY);
    if(rank == 0){
        cout << "grid x dim: " << gridDimX << ", grid y dim: " << gridDimY << endl;
    }

    // Copy derivatives to host
    double *IxDevice, *IyDevice, *ItDevice;
    GPU_ERROR = cudaMalloc(&IxDevice, sizeDer);
    GPU_ERROR = cudaMalloc(&IyDevice, sizeDer);
    GPU_ERROR = cudaMalloc(&ItDevice, sizeDer);
    GPU_ERROR = cudaMemcpy(IxDevice, localIx.data(), sizeDer, cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(IyDevice, localIy.data(), sizeDer, cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(ItDevice, localIt.data(), sizeDer, cudaMemcpyHostToDevice);
    if(rank == 0){
        cout << "Finished derivatives transfer" << endl;
    }
    // Create average and flow vectors for device and host
    vector<double> uHost(nx * ny, 0.0);
    vector<double> vHost(nx * ny, 0.0);
    double *uDevice, *vDevice;
    GPU_ERROR = cudaMalloc(&uDevice, size);
    GPU_ERROR = cudaMalloc(&vDevice, size);
    GPU_ERROR = cudaMemcpy(uDevice, uHost.data(), size, cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(vDevice, vHost.data(), size, cudaMemcpyHostToDevice);

    if(rank == 0){
        cout << "Copied over average and flow vectors" << endl;
    }
    // Compute optical flow
    int currIteration = 0;
    int iterations = 200;
    double alpha = 1;
    
    // Add timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    int gridDims[2] = {2,2};
    GPU_ERROR = cudaEventRecord(start);
    while (currIteration < iterations){
        horn_schunck<<<grid, block>>>(uDevice, vDevice, IxDevice, IyDevice, ItDevice, alpha, nx, ny, rank);
        GPU_ERROR = cudaDeviceSynchronize();
        currIteration++;
    }
    // Stop timing
    GPU_ERROR = cudaEventRecord(stop);
    GPU_ERROR = cudaEventSynchronize(stop);

    // Performance metrics
    float elapsed_time_ms;
    GPU_ERROR = cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    float elapsed_time_s = elapsed_time_ms / 1000.0;
    if(rank == 0){
        cout << "Kernels finished in " << elapsed_time_s << " seconds." << endl;
        auto [measured_tflops, ai, peak_tflops] = analyze_performance(nx, ny, iterations, elapsed_time_s, gridDimX, gridDimY);
        printf("\nGrid Size | TFLOPS | AI | Peak TFLOPS | Time(s) | Iterations\n");
        printf("---------|--------|----|-----------  |---------|------------|\n");
        printf("%4dx%4d | %6.6f | %6.6f | %6.6f | %7.5f | %10d |\n",
                nx * 2, ny * 2, measured_tflops, ai, peak_tflops, elapsed_time_s, iterations);
        printf("Bottleneck: %s\n", 
                (PEAK_MEMORY_BANDWIDTH * ai < PEAK_FLOP_RATE ? "Memory" : "Compute"));
    }
   
    // // Copy over flow results to host
    GPU_ERROR = cudaMemcpy(uHost.data(), uDevice, size, cudaMemcpyDeviceToHost);
    GPU_ERROR = cudaMemcpy(vHost.data(), vDevice, size, cudaMemcpyDeviceToHost);
    
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Copied results!" << endl;
    

    // Gather the processed data
    vector<double> uMain = gather_data(uHost, dims, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    vector<double> vMain = gather_data(vHost, dims, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        cout << "Data gathered!" << endl;
    }
    // Visualize optical flow
    if(rank == 0){
        Mat img_color, flowX, flowY;
        cvtColor(frame1, img_color, COLOR_GRAY2BGR);
        flowX = vectorToMat<double>(uMain, dims.total_rows, dims.total_cols, CV_64F);
        flowY = vectorToMat<double>(vMain, dims.total_rows, dims.total_cols, CV_64F); 
        drawOpticalFlow(flowX, flowY, img_color);

        Mat flow_vis;
        visualizeFlowHSV(flowX, flowY, flow_vis);

        cout << "Writing optical flow images with correct metrics" << endl;
        imwrite("outputs/CUDA_MPI_optical_flow_" + outputname + ".png", img_color);
        imwrite("outputs/CUDA_MPI_optical_flow_hsv_" + outputname + ".png", flow_vis);
    }
    MPI_Finalize();
    return 0;
}