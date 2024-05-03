
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <dpct/blas_utils.hpp>

// cuda libraries and helpers
#include <helper_cuda.h>
#include <cmath>

// configurable parameters
// dimension of matrix
#define N 4
#define BATCH_SIZE 10000

// use double precision data type
//#defind cuda device
  printf("> initializing..\n");
  int dev = findCudaDevice(argc, (const char**)argv);
  if (dev == -1) {
    return (EXIT_FAILURE);
  }

  // initialize cuBLAS
  status = DPCT_CHECK_ERROR(handle = new dpct::blas::descriptor());
  if (status != 0) {
    printf("> ERROR: cuBLAS initialization failed..\n");
    return (EXIT_FAILURE);
  }

#ifdef DOUBLE_PRECISION
  printf("> using DOUBLE precision..\n");
#else
  printf("> using SINGLE precision..\n");
#endif

#ifdef PIVOT
  printf("> pivot ENABLED..\n");
#else
  printf("> pivot DISABLED..\n");
#endif

  // allocate memory for host variables
  h_AarrayInput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);
  h_AarrayOutput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);

  h_pivotArray = (int*)xmalloc(N * BATCH_SIZE * sizeof(int));
  h_infoArray = (int*)xmalloc(BATCH_SIZE * sizeof(int));

  // allocate memory for device variables
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_Aarray = (float *)sycl::malloc_device(
                           BATCH_SIZE * matSize, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_pivotArray = sycl::malloc_device<int>(
                           N * BATCH_SIZE, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_infoArray =
          sycl::malloc_device<int>(BATCH_SIZE, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_ptr_array = (float **)sycl::malloc_device(
          BATCH_SIZE * sizeof(DATA_TYPE *), dpct::get_in_order_queue())));

  // fill matrix with random data
  printf("> generating random matrices..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    initRandomMatrix(h_AarrayInput + (i * N * N));
  }

  // copy data to device from host
  printf("> copying data from host memory to GPU memory..\n");
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(d_Aarray, h_AarrayInput, BATCH_SIZE * matSize)
          .wait()));

  // create pointer array for matrices
  for (int i = 0; i < BATCH_SIZE; i++) h_ptr_array[i] = d_Aarray + (i * N * N);

  // copy pointer array to device memory
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(d_ptr_array, h_ptr_array, BATCH_SIZE * sizeof(DATA_TYPE *))
          .wait()));

  // perform LU decomposition
  printf("> performing LU decomposition..\n");
#ifdef PIVOT
  status = cublasXgetrfBatched(handle, N, d_ptr_array, N, d_pivotArray,
                               d_infoArray, BATCH_SIZE);
#else
  status = cublasXgetrfBatched(handle, N, d_ptr_array, N, NULL, d_infoArray,
                               BATCH_SIZE);
#endif /* PIVOT */
  if (status != 0) {
    printf("> ERROR: cublasDgetrfBatched() failed with error %s..\n",
           _cudaGetErrorEnum(status));
    return (EXIT_FAILURE);
  }

  // copy data to host from device
  printf("> copying data from GPU memory to host memory..\n");
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(h_AarrayOutput, d_Aarray, BATCH_SIZE * matSize)
          .wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(h_infoArray, d_infoArray, BATCH_SIZE * sizeof(int))
          .wait()));
#ifdef PIVOT
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(h_pivotArray, d_pivotArray, N * BATCH_SIZE * sizeof(int))
          .wait()));
#endif /* PIVOT */

  // verify the result
  printf("> verifying the result..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    if (h_infoArray[i] == 0) {
      DATA_TYPE* A = h_AarrayInput + (i * N * N);
      DATA_TYPE* LU = h_AarrayOutput + (i * N * N);
      DATA_TYPE L[N * N];
      DATA_TYPE U[N * N];
      getLUdecoded(LU, L, U);

      // test P * A = L * U
      int* P = h_pivotArray + (i * N);
      DATA_TYPE Pmat[N * N];
#ifdef PIVOT
      getPmatFromPivot(Pmat, P);
#else
      initIdentityMatrix(Pmat);
#endif /* PIVOT */

      // perform matrix multiplication
      DATA_TYPE PxA[N * N];
      DATA_TYPE LxU[N * N];
      matrixMultiply(PxA, Pmat, A);
      matrixMultiply(LxU, L, U);

      // check for equality of matrices
      if (!checkRelativeError(PxA, LxU, (DATA_TYPE)MAX_ERROR)) {
        printf("> ERROR: accuracy check failed for matrix number %05d..\n",
               i + 1);
        err_count++;
      }

    } else if (h_infoArray[i] > 0) {
      printf(
          "> execution for matrix %05d is successful, but U is singular and "
          "U(%d,%d) = 0..\n",
          i + 1, h_infoArray[i] - 1, h_infoArray[i] - 1);
    } else  // (h_infoArray[i] < 0)
    {
      printf("> ERROR: matrix %05d have an illegal value at index %d = %lf..\n",
             i + 1, -h_infoArray[i],
             *(h_AarrayInput + (i * N * N) + (-h_infoArray[i])));
    }
  }

  // free device variables
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(d_ptr_array, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(d_infoArray, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(d_pivotArray, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_Aarray, dpct::get_in_order_queue())));

  // free host variables
  if (h_infoArray) free(h_infoArray);
  if (h_pivotArray) free(h_pivotArray);
  if (h_AarrayOutput) free(h_AarrayOutput);
  if (h_AarrayInput) free(h_AarrayInput);

  // destroy cuBLAS handle
  status = DPCT_CHECK_ERROR(delete (handle));
  if (status != 0) {
    printf("> ERROR: cuBLAS uninitialization failed..\n");
    return (EXIT_FAILURE);
  }

  if (err_count > 0) {
    printf("> TEST FAILED for %d matrices, with precision: %g\n", err_count,
           MAX_ERROR);
    return (EXIT_FAILURE);
  }

  printf("> TEST SUCCESSFUL, with precision: %g\n", MAX_ERROR);
  return (EXIT_SUCCESS);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
vb