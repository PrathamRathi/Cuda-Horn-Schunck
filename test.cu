#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
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

#else

#include <cuda.h>

#endif
using namespace cv;
using namespace std;

const int GHOST_CELLS = 1;  // Number of ghost cells on each side
const double INTENSITY_MULTIPLIER = 1.5;  // Constant to multiply pixel values
const int GRID_SIZE = 2;  // 2x2 grid

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            cout << "This program requires exactly 4 processes." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate grid position for each rank
    int row = rank / GRID_SIZE;    // 0 for ranks 0,1 and 1 for ranks 2,3
    int col = rank % GRID_SIZE;    // 0 for ranks 0,2 and 1 for ranks 1,3

    vector<double> localData;
    int total_rows, total_cols;
    int local_rows, local_cols;
    
    if (rank == 0) {
        // Read image on rank 0
        Mat image = imread("car_grayscale.jpg", IMREAD_GRAYSCALE);
        if (image.empty()) {
            cout << "Error: Could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        total_rows = image.rows;
        total_cols = image.cols;

        // Calculate local dimensions (without ghost cells)
        local_rows = total_rows / GRID_SIZE;
        local_cols = total_cols / GRID_SIZE;

        // Send dimensions to all processes
        int dims[4] = {total_rows, total_cols, local_rows, local_cols};
        MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);

        // Prepare temporary data for distribution
        vector<vector<double>> gridData(4);
        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                int current_rank = r * GRID_SIZE + c;
                gridData[current_rank].resize(local_rows * local_cols);
                
                // Copy appropriate section of the image
                for (int i = 0; i < local_rows; i++) {
                    for (int j = 0; j < local_cols; j++) {
                        int global_i = r * local_rows + i;
                        int global_j = c * local_cols + j;
                        gridData[current_rank][i * local_cols + j] = 
                            static_cast<double>(image.at<uchar>(global_i, global_j));
                    }
                }
                
                // Send to other ranks
                if (current_rank != 0) {
                    MPI_Send(gridData[current_rank].data(), local_rows * local_cols,
                            MPI_DOUBLE, current_rank, 0, MPI_COMM_WORLD);
                }
            }
        }

        // Copy rank 0's data to local buffer with ghost cells
        localData.resize((local_rows + 2 * GHOST_CELLS) * (local_cols + 2 * GHOST_CELLS), 0.0);
        // Copy the actual data to the interior (non-ghost) region
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                localData[(i + GHOST_CELLS) * (local_cols + 2 * GHOST_CELLS) + (j + GHOST_CELLS)] = 
                    gridData[0][i * local_cols + j];
            }
        }

    } else {
        // Receive dimensions
        int dims[4];
        MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);
        total_rows = dims[0];
        total_cols = dims[1];
        local_rows = dims[2];
        local_cols = dims[3];

        // Receive data from rank 0
        vector<double> tempData(local_rows * local_cols);
        MPI_Recv(tempData.data(), local_rows * local_cols,
                 MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy received data to local buffer with ghost cells
        localData.resize((local_rows + 2 * GHOST_CELLS) * (local_cols + 2 * GHOST_CELLS), 0.0);
        // Copy the actual data to the interior (non-ghost) region
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                localData[(i + GHOST_CELLS) * (local_cols + 2 * GHOST_CELLS) + (j + GHOST_CELLS)] = 
                    tempData[i * local_cols + j];
            }
        }
    }

    // Calculate neighbor ranks (for future use)
    int up = (row > 0) ? rank - GRID_SIZE : MPI_PROC_NULL;
    int down = (row < GRID_SIZE - 1) ? rank + GRID_SIZE : MPI_PROC_NULL;
    int left = (col > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (col < GRID_SIZE - 1) ? rank + 1 : MPI_PROC_NULL;

    // Process the data (multiply intensity) - only process non-ghost cells
    for (int i = GHOST_CELLS; i < local_rows + GHOST_CELLS; i++) {
        for (int j = GHOST_CELLS; j < local_cols + GHOST_CELLS; j++) {
            localData[i * (local_cols + 2 * GHOST_CELLS) + j] *= INTENSITY_MULTIPLIER;
            // Clamp values to valid range [0, 255]
            localData[i * (local_cols + 2 * GHOST_CELLS) + j] = 
                min(255.0, max(0.0, localData[i * (local_cols + 2 * GHOST_CELLS) + j]));
        }
    }

    if (rank == 0) {
        // Create output image
        Mat outputImage(total_rows, total_cols, CV_8UC1);
        
        // Copy rank 0's data (excluding ghost cells)
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                outputImage.at<uchar>(i, j) = static_cast<uchar>(
                    localData[(i + GHOST_CELLS) * (local_cols + 2 * GHOST_CELLS) + (j + GHOST_CELLS)]);
            }
        }

        // Receive and copy data from other ranks
        vector<double> recvData(local_rows * local_cols);
        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                int current_rank = r * GRID_SIZE + c;
                if (current_rank != 0) {
                    MPI_Recv(recvData.data(), local_rows * local_cols,
                            MPI_DOUBLE, current_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    // Copy to appropriate position in output image
                    for (int i = 0; i < local_rows; i++) {
                        for (int j = 0; j < local_cols; j++) {
                            int global_i = r * local_rows + i;
                            int global_j = c * local_cols + j;
                            outputImage.at<uchar>(global_i, global_j) = 
                                static_cast<uchar>(recvData[i * local_cols + j]);
                        }
                    }
                }
            }
        }

        // Write output image
        imwrite("output_gray.jpg", outputImage);
        cout << "Processing complete. Output saved to 'output.jpg'" << endl;
    } else {
        // Prepare data to send back (excluding ghost cells)
        vector<double> sendData(local_rows * local_cols);
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                sendData[i * local_cols + j] = 
                    localData[(i + GHOST_CELLS) * (local_cols + 2 * GHOST_CELLS) + (j + GHOST_CELLS)];
            }
        }
        // Send processed data back to rank 0
        MPI_Send(sendData.data(), local_rows * local_cols,
                 MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}