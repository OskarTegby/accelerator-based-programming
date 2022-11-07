#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cmath>
#include "cublas_v2.h"

/*
** Features to implement
* 1. Implement A^T * x for M = N.
* 2. Implement A * B natively and using cuBLAS.
* 3. Write the report. The results were recorded earlier.
*/

/*
 * Execution example: ./cublas -min 100 -max 10000 -repeat 20
 */

const int block_size = 128;
#define BLOCK_SIZE 16
#define VERBOSE
#define CUBLAS

//#define DEBUG
//#define DEBUG_SIZE 4

/*
 * @brief	a struct for storing the relevant properties of the (sub)matrix.
 *
 * @param	width		the width of the matrix,
 * @param	height		the height of the matrix,
 * @param	stride		the stride belonging to this matrix,
 * @param	elements 	the elements in this matrix.
 */
typedef struct {
    unsigned int width;
    unsigned int heigth;
    unsigned int stride;
    float* elements; 
} Matrix;

/**************************************************************************
 ************************* PRINT AND SET FUNCTIONS ************************
 **************************************************************************/

/*
 * @brief      setting elements in a vector
 * @param       N       the number of elements,
 * @param       val     the value to set everywhere,
 * @param       x       the vector to set values in.
 */
__global__ void setVector(const int N,
			  const float val,
			  float *x)
{
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    if (val == 0) {
      x[idx] = val;
    } else { 
#ifdef DEBUG // inc vals
      x[idx] = idx + 1;
#else
      x[idx] = val;
#endif
    }
  }
}

/*
 * @brief      setting elements row-major-wise in a matrix
 * @param       N       the number of rows,
 * @param       M       the number of columns,
 * @param       val     the value to set everywhere,
 * @param       x       the matrix to set values in.
 */
__global__ void setMatrixRowmaj(const int M,
				const int N,
				const float val,
				float *x)
{
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < M)
    for (unsigned int j = 0; j < N; j++) {
#ifdef DEBUG // inc vals
      x[i * N + j] = i * N + j + 1;
#else
      x[i * N + j] = val;
#endif
   }
}
  
/*
 * @brief      setting elements column-major-wise in a matrix
 * @param       M       the number of columns,
 * @param       N       the number of rows,
 * @param       val     the value to set everywhere,
 * @param       x       the matrix to set values in.
 */
__global__ void setMatrixColmaj(const int M,
				const int N,
				const float val,
				float *x)
{
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < M)
    for (unsigned int j = 0; j < N; j++) {
#ifdef DEBUG // inc vals
      x[i * N + j] = i * N + j + 1;
#else
      x[i * N + j] = val;
#endif
  }
}

/*
 * @brief       printing a matrix (in column-major format)
 * @param       M       the number of columns,
 * @param       N       the number of rows,
 * @param       x       the matrix to print.
 */
void printMatrix(const int M,
		 const int N,
		 std::vector<float> x)
{
  printf("\n");
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++)
      printf("%f ", x.at(i + j * M));
    printf("\n");
  }
  printf("\n");
}

/*
 * @brief	printing a vector
 * @param	N	the vector length,
 * @param	x	the vector to print.
 */
void printVector(const int N,
		 std::vector<float> x)
{
  printf("[");
  for (int i = 0; i < N - 1; i++) {
    printf("%f, ", x.at(i));
  }
  printf("%f]\n\n", x.at(N - 1));
}

/**************************************************************************
 **************************** MATRIX FUNCTIONS ****************************
 **************************************************************************/

/*
 * @brief	Extracts a sub-matrix from the given row and column.
		Hence, we are row and col number of submatrices
		down and to the right from the upper-left corner
		of the given matrix, respectively.
 * @param	A	the matrix which we extract the submatrix from,
 * @param	row	the row we start extracting from,
 * @param	col	the column we start subtracting from. 
 */
__device__ Matrix getSubMatrix(Matrix A,
			       const int row,
			       const int col)
{
   Matrix Asub;
   Asub.height	 = BLOCK_SIZE;
   Asub.width	 = BLOCK_SIZE;
   Asub.stride	 = A.stride;
   // we multiply by the stride to end up on the right row
   Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
					+ BLOCK_SIZE * col];

   return Asub;
}

/*
 * @brief	Extracts the element on the given row and column
		from the given matrix.
 * @param	A	the matrix which we extract the element from,
 * @param	row	the row we are extracting from,
 * @param	col	the column we are subtracting from. 
 */
__device__ float getElement(const Matrix A,
		 	    const int row,
		 	    const int col)
{
    return A.elements[row * A.stride + col];
}

/*
 * @brief	Updates a value in the matrix on a given row and column.
 * @param	A	the matrix which we set the element in,
 * @param	row	the row of the updated element,
 * @param	col	the column of the updated element, 
 * @param	value	the value which we are setting.
 */
__device__ void setElement(Matrix A,
			   const int row,
			   const int col,
			   const float value)
{
    A.elements[row * A.stride + col] = value;
}

/*
 * @brief	computing a sub-matrix of the matrix-matrix product.
 * @param       A       the first multiplicand,
 * @param       B       the second multiplicand,
 * @param       C       the resulting matrix.
 */
__global__ void MatMulKernel(const Matrix A,
                             const Matrix B,
                             const Matrix C)
{
  // The row and column where the block starts
  const unsigned int blockCol = blockIdx.x;
  const unsigned int blockRow = blockIdx.y;

  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

  // The row and column within the block 
  const unsigned int col = threadIdx.x;
  const unsigned int row = threadIdx.y;

  // The variable to which we accumulate the results
  float Cvalue = 0;

  // Looping over all sub-matrices of A and B required
  // in order to compute Csub, and then multiplying
  // each pair of sub-matrices and accumulating the results.
  for (unsigned int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    // Loading the sub-matrix Asub of A
    Matrix Asub = getSubMatrix(A, blockRow, m);
    // Loading the sub-matrix Bsub of B
    Matrix Bsub = getSubMatrix(B, m, blockCol);

    // Allocating shared memory for Asub and Bsub for speed
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loading Asub and Bsub from device memory to the shared memory
    As[row][col] = getElement(Asub, row, col);
    Bs[row][col] = getElement(Bsub, row, col);
  
    // Synchronizing to ensure that the sub-matrices are loaded before compute
    __syncthreads();

    // Multiplying Asub and Bsub
    for (unsigned int e = 0; e < BLOCK_SIZE; ++e)
	Cvalue += As[row][e] * Bs[e][col];

    // Synchronizing to ensure that the computation is done before movig on
    __syncthreads();
  } 

  // Writing Csub to device memory
  setElement(Csub, row, col, Cvalue);
}

/**************************************************************************
 ************************** BENCHMARK FUNCTIONS ***************************
 **************************************************************************/

/*
 * @brief 	performs matrix and vector multiplications on the GPU
		according to the specified settings.		
 * @param	M	the number of rows in the A and C matrices,
 * @param	N	the number of rows in B and columns in A,
 * @param	K	the number of columns in the B and C matrices,
 * @param	repeat	repetitions used to minimize noise.	
 */
float benchmarkTriad(const std::size_t M,
                     const std::size_t N,
                     const std::size_t K,
                     const long repeat)
{
#ifdef CUBLAS
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  if (stat !=CUBLAS_STATUS_SUCCESS) {
  	std::cout << "CUBLAS initialization failed\n";
	std::abort();
  }

  float alpha = 1.f;
  float beta = 0.f;
#endif

  // d_A, d_B, and d_C have the same block size since M = N = K.
  unsigned int n_blocks = (M * N + block_size - 1) / block_size;

  // Loading d_A into device memory
  Matrix d_A;
  d_A.height  = M;
  d_A.width   = N;
  size_t size_A = d_A.width * d_A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size_A);
  setMatrixRowmaj<<<n_blocks, block_size>>>(M, N, 1.f, d_A);
 
  // Loading d_B into device memory 
  Matrix d_B;
  d_B.height  = N;
  d_B.width   = K;
  size_t size_B = d_B.width * d_B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size_B);
  setMatrixRowmaj<<<n_blocks, block_size>>>(N, K, 1.f, d_B);
 
  // Allocating memory for d_C
  Matrix d_C;
  d_C.height  = M;
  d_C.width   = K;
  size_t size_C = d_C.width * d_C.height * sizeof(float);
  cudaMalloc(&d_C.elements, size_C);
  setMatrixRowmaj<<<n_blocks, block_size>>>(M, K, 0.f, d_C);
  
  std::vector<float> h_A(M * N);
  std::vector<float> h_B(N * K);
  std::vector<float> h_C(M * K);

#ifndef CUBLAS
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
#endif

#ifdef DEBUG
  const unsigned int           n_tests = 1;
  const unsigned long long int n_repeat = 1;
#else
  const unsigned int           n_tests = 20;
  const unsigned long long int n_repeat = 20; // replace 20 with repeat
#endif

  float best = 1e10, worst = 0, avg = 0;
  
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();
      
      setMatrixRowmaj<<<n_blocks, block_size>>>(M, K, 0.f, d_C);
	
      for (unsigned int rep = 0; rep < n_repeat; ++rep) {
#ifdef CUBLAS	
	// Calling the kernel
	stat = cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, // fix this
				A, M, x, 1, &beta, b, 1);
	
	// Checking for success
	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cout << "CUBLAS operation failed\n";
		std::abort();
	}
#else
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
#endif
      }

      cudaDeviceSynchronize();
      // measure the time by taking the difference between the time point
      // before starting and now
      const float time =
        std::chrono::duration_cast<std::chrono::duration<float>>(
          std::chrono::steady_clock::now() - t1)
          .count();
      
      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }
    
  // copy the result back to the host
  cudaMemcpy(h_A.data(), d_A.elements, size_A, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B.data(), d_B.elements, size_B, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C.data(), d_C.elements, size_C, cudaMemcpyDeviceToHost);

#ifdef DEBUG
  printf("\nA =");
  printMatrix(M, N, h_A);
  printf("B = ");
  printMatrix(N, K, h_B);
  printf("C = ");
  printMatrix(M, K, h_C);

  float res = 180; // fix this
  if ((abs(result_host[0] + result_host[M - 1] - res)) > 0.00000001)
    std::cout << "Error in computation, got "
              << (result_host[0] + result_host[M - 1]) << " instead of "
              << res
              << std::endl;
  else
    std::cout << "*** Congratulations! You know basic linear algebra! ***\n" << std::endl;
#else
  if (abs((result_host[0] + result_host[M - 1] - 2 * N)) > 0.00000001)
    std::cout << "Error in computation, got "
              << (result_host[0] + result_host[M - 1]) << " instead of "
              << 2 * N
              << std::endl; // fix this
#endif

  // free the memory on the device
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);

#ifdef VERBOSE
  std::cout << "(M, N) = (" << M << ", " << N << ")"
            << std::setw(8) << " - min/avg/max: "
            << std::setw(11) << best << " / "
            << std::setw(11) << avg / n_tests << " / "
            << std::setw(11) << worst // fix this
            << " seconds, or " << std::setw(8) << M * 1e-6 / best
            << " MUPD/s, or " << std::setw(8) << (M * N + M + N) * sizeof(float) * 1e-9 / best
            << " GB/s, or " << std::setw(8) << M * N * 2 / best
            << " FLOP/s" << std::endl;
#endif

/*
 ** LEGEND FOR THE METRICS USED

 ** MUPDS/s:
 * M * 1e-6 for storing the results.

 ** GB/s:
 * M * N reads from the matrix,
 * N reads from the vector, and
 * M writes for the result.

 ** FLOP/s:
 * N multiplications and additions to get one element,
 * M such operations to get one column, and
 * K such operations to get row,
 * (for the matrix-vector multiplication K = 1)
 * divided by the shortest computation time.
*/

  // returning the GB/s to write to the csv file
  return (M * N + M + N) * sizeof(float) * 1e-9 / best; // fix this 
}

/*
 * @brief	Calling the kernel with the settings in argv.
 * @param	argc	The number of arguments used when calling main.
 * @param	argv	The argument values used. These are follows
 *		
 *	min	lower limit of N,
 *	max	upper limit of N,
 * 	repeat	repetitions used to minimize noise.
 */
int main(int argc, char **argv)
{
  if (argc % 2 == 0)
    {
      std::cout << "Error, expected odd number of common line arguments"
                << std::endl
                << "Expected line of the form" << std::endl
                << "-min 100 -max 10000 -repeat 20" << std::endl;
      std::abort();
    }

  long N_min  = 8;
  long N_max  = -1;
  long repeat = -1;
  long m, n, k;

  // parse from the command line
  for (unsigned int l = 1; l < argc; l += 2)
    {
      std::string option = argv[l];
      if (option == "-min")
        N_min = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-max")
        N_max = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-repeat")
        repeat = std::atoll(argv[l + 1]);
      else if (option == "-part")
        part = std::atoll(argv[l + 1]);
      else
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }
  
  if (N_min < 1)
    {
      std::cout << "Expected positive size for min argument, got " << N_min
                << std::endl;
      return 0;
    }

  if (N_max < N_min)
    N_max = N_min;

#ifdef CUBLAS
    /* saving the csvs in a seprate folder,
       please create it if you don't have it */
    myfile.open("/home/oskart/abp/a2/csv/cublas_mm_mult.csv");
#else
    myfile.open("/home/oskart/abp/a2/csv/native_mm_mult.csv");
#endif

#ifdef DEBUG
  n = DEBUG_SIZE;
  m = n;
  k = n;
  benchmarkTriad(m, n, k, repeat);
#else
  std::ofstream myfile;
  for (n = N_min; n <= N_max; n = (1 + n * 1.1))
  {
      // round up to nearest multiple of 8
      n = (n + 7) / 8 * 8; // this needs to be a multiple of the block size
      m = n;
      k = n;
      myfile << n;
      myfile << " ";
      myfile << benchmarkTriad(m, n, k, repeat);
      myfile << std::endl;
  }
#endif

  return 0;
}

