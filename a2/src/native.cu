#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cmath>

/*
** Features to implement
* 1. Implement cuBLAS.
* 2. Implement A^T * x for M = N.
* 3. Implement A * B natively and using cuBLAS.
* 4. Write the report. The results were recorded earlier.
*/

const int block_size = 128;
//#define DEBUG
//#define DEBUG_SIZE 4
#define VERBOSE

/*
 * Execution example: ./task1 -min 100 -max 10000 -repeat 20
 */

/*
 * @brief	computing matrix-vector product.
 * @param       M       the number of columns,
 * @param       N       the number of rows,
 * @param       A       the matrix multiplying with,
 * @param	x	the vector multiplying with,
 * @param	b	the resulting vector. 
 */
__global__ void compute_triad(const int    M,
                              const int    N,
                              const float *A,
                              const float *x,
                              float *      b)
{
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  float sum = 0;
  if (i < M) {
    for (unsigned int j = 0; j < N; ++j) {
      sum += A[i + j * M] * x[j];
#if defined(DEBUG) && defined(VERBOSE)
      printf("A[%d] * x[%d] = b[%d]: %f * %f = %f\n", \
              i + j * M, j, i, A[i + j * M], x[j], sum);
#endif
    }
    b[i] = sum;
    sum = 0;
  }
}





/*
 * @brief      setting elements in a vector
 * @param       N       the number of elements,
 * @param       val     the value to set everywhere,
 * @param       x       the vector to set values in.
 */
__global__ void set_vector(const int N, const float val, float *x)
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
 * @param       M       the number of columns,
 * @param       N       the number of rows,
 * @param       val     the value to set everywhere,
 * @param       x       the matrix to set values in.
 */
__global__ void set_matrix_rowmaj(const int M, const int N, const float val, float *x)
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
__global__ void set_matrix_colmaj(const int M, const int N, const float val, float *x)
{
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
    for (unsigned int j = 0; j < M; j++) {
#ifdef DEBUG // inc vals
      x[i * M + j] = i + j * N + 1;
#else
      x[i * M + j] = val;
#endif
  }
}





/*
 * @brief       printing a matrix (in row-major format)
 * @param       M       the number of columns,
 * @param       N       the number of rows,
 * @param       x       the matrix to print.
 */
void print_matrix(const int M, const int N, std::vector<float> x)
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
void print_vector(const int N, std::vector<float> x)
{
  printf("[");
  for (int i = 0; i < N - 1; i++) {
    printf("%f, ", x.at(i));
  }
  printf("%f]\n\n", x.at(N - 1));
}





/*
 * @brief 	performs matrix and vector multiplications on the GPU
		according to the specified settings.		
 * @param	M	the number of rows in the matrix,
 * @param	N	the number of columns in the matrix,
 * @param	repeat	repetitions used to minimize noise.	
 */
float benchmark_triad(const std::size_t M,
                     const std::size_t N,
                     const long long   repeat)
{
  float *A, *x, *b;

  // allocate memory on the device
  cudaMalloc(&A, M * N * sizeof(float));
  cudaMalloc(&x, N * sizeof(float));
  cudaMalloc(&b, M * sizeof(float));
  
  unsigned int n_blocks = (M * N + block_size - 1) / block_size;
  
  set_matrix_colmaj<<<n_blocks, block_size>>>(M, N, 1.f, A);
  n_blocks = (N + block_size - 1) / block_size;
  set_vector<<<n_blocks, block_size>>>(N, 1.f, x);
  n_blocks = (M + block_size - 1) / block_size;
  set_vector<<<n_blocks, block_size>>>(M, 0.f, b);
  
  std::vector<float> result_host(M);
  std::vector<float> A_host(M * N);
  std::vector<float> x_host(N);

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
      
      set_vector<<<n_blocks, block_size>>>(M, 0.f, b);

      for (unsigned int rep = 0; rep < n_repeat; ++rep)
        compute_triad<<<n_blocks, block_size>>>(M, N, A, x, b);

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
  cudaMemcpy(result_host.data(), b, M * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(A_host.data(), A, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(x_host.data(), x, N * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef DEBUG
  printf("\nA =");
  print_matrix(M, N, A_host);
  printf("x = ");
  print_vector(N, x_host);
  printf("b = ");
  print_vector(M, result_host);

  float res = 180;
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
              << std::endl;
#endif

  // free the memory on the device
  cudaFree(A);
  cudaFree(x);
  cudaFree(b);

#ifdef VERBOSE
  std::cout << "(M, N) = (" << M << ", " << N << ")"
            << std::setw(8) << " - min/avg/max: "
            << std::setw(11) << best << " / "
            << std::setw(11) << avg / n_tests << " / "
            << std::setw(11) << worst
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
  return (M * N + M + N) * sizeof(float) * 1e-9 / best; 
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
  long m, n;

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

#ifdef DEBUG
  m = DEBUG_SIZE;
  n = m;
  benchmark_triad(m, n, repeat);
#else
  int part = 2;
  std::ofstream myfile;

  if (part == 0)
  {
  /* saving the csvs in a seprate folder,
     please create it if you don't have it */
  myfile.open("/home/oskart/abp/a2/csv/native_square.csv");
    for (n = N_min; n <= N_max; n = (1 + n * 1.1))
    {   
        // round up to nearest multiple of 8
        n = (n + 7) / 8 * 8;
        m = n;
        myfile << n;
        myfile << " ";
        myfile << benchmark_triad(m, n, repeat);
        myfile << std::endl;
    }   
  }
  else if (part == 1)
  {
  myfile.open("/home/oskart/abp/a2/csv/native_rect1.csv");
    n = 10000;
    for (m = N_min; m <= N_max; m = (1 + m * 1.1))
    {   
        // round up to nearest multiple of 8
        m = (m + 7) / 8 * 8;
        myfile << m;
        myfile << " ";
        myfile << benchmark_triad(m, n, repeat);
        myfile << std::endl;
    }   
  }
  else if (part == 2)  
  {
    myfile.open("/home/oskart/abp/a2/csv/native_rect2.csv");
    m = 16384;
    for (n = N_min; n <= N_max; n = (1 + n * 1.1))
    {   
        // round up to nearest multiple of 8
        n = (n + 7) / 8 * 8;
        myfile << n;
        myfile << " ";
        myfile << benchmark_triad(m, n, repeat);
        myfile << std::endl;
    }   
  }
  myfile.close();
#endif

  return 0;
}
