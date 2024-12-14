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