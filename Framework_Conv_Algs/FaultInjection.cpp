#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include <string>
#include <ctime>


//#define DEBUG

template<typename T> bool almost_equal(const T a, const T b, const T epsilon)
{
    return std::abs(a - b) < epsilon;
}

bool compare_cudnn(
        const float* actual,
        const float* expected,
        const std::vector<int> &shape)
{

    const float epsilon = static_cast<float>(1E-3);
    const int N = shape[3] * shape[2]; //1*H*W

    std::cout << N << std::endl;

    int idx_numpy = 0;
    int idx_cudnn = 0;

    for (int i = 0; i < N; i++) {

        const bool comparison_result_1 = almost_equal(actual[idx_cudnn], expected[idx_numpy], epsilon);
        const bool comparison_result_2 = almost_equal(actual[idx_cudnn + N], expected[idx_numpy + 1], epsilon);

        if(!comparison_result_1) return false;
        if(!comparison_result_2) return false;

        idx_numpy += 2;
        idx_cudnn += 1;
    }

    return true;
}

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << " " << __LINE__ << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << cudnnGetErrorString(err) << " " << __LINE__ << std::endl; \
    std::exit(1); \
  } \
}


void print(const float *data, int n, int c, int h, int w) {
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << data[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

cudnnConvolutionFwdAlgo_t convolutionAlgorithms[] = {
    // IMPLICIT_GEMM ID 0
    // Expresses the convolution as a matrix product without actually forming
    // the matrix that holds the input tensor data.
    // Suitable for small to medium size tensors.
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,

    // IMPLICIT_PRECOMP_GEMM ID 1
    // Similar to IMPLICIT_GEMM but uses precomputed indices to speed up the
    // matrix multiplication.
    // Requires additional memory workspace for indices.
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,

    // GEMM ID 2
    // Expresses the convolution as a direct matrix product.
    // Requires significant memory to store the matrix of input tensor data.
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,

    // DIRECT ID 3
    // Computes the convolution directly without transforming it into matrix multiplication.
    // Optimal for small kernel sizes. Not implemented in CUDNN.
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,

    // FFT ID 4
    // Uses the Fast Fourier Transform to compute the convolution.
    // Effective for large input sizes but requires substantial memory for intermediate results.
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,

    // FFT_TILING ID 5
    // Uses FFT and splits the input into tiles to manage memory better than the standard FFT.
    // Suitable for very large input sizes with more manageable memory requirements.
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,

    // WINOGRAD ID 6
    // Utilizes the Winograd algorithm for minimal filtering operations.
    // Works best for small convolutional windows.
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,

    // WINOGRAD_NONFUSED ID 7
    // A variant of Winograd that does not use fused operations.
    // Can handle larger tiles and requires more memory than WINOGRAD.
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
};


/**
 * @brief Performs all the cudnn listed convolutions for a fixed input and weight tensor.
 ## Inputs
 * @param cudnnHandle: A pointer to an already initialized cudnnHandle
 * @param convolutionsAlgos: A pointer to an array containing the algorithms identifiers (of type cudnnConvolutionFwdAlgo_t) that should be attempted to be executed.
 * @param nAlgos: The number of algorithms present in the array referenced by convolutionsAlgos.
 * @param convInput: A pointer to a linear float array containing the input tensor data.
 * @param convInputShape: A pointer to a four element int array containing the shape of the input tensor. Each of the four element represents the size of the tensor dimensions in the following order: N C H W.
                          N is the batch size, C is the number of channels, H and W are respectively the height and the width of each channel.
                          This function will read all the elements of the convInput tensor, namely the first N * C * H * W elements starting from the address contained in convInput.
 * @param convWeights: A pointer to a linear float array containing the weight tensor data.
 * @param convWeightsShape: A pointer to a four element int array containing the shape of the weight tensor. Each of the four element represents the size of the tensor dimensions in the following order: K C FH FW.
                          K is the number of channels returned in output by the convolution, C is the number of input channels of the convolution (it MUST be equal to convInputShape's C), FH and FW are
                          respectively the filter Height and Width.
                          This function will read all the elements of the convInput tensor, namely the first K * C * FH * FW elements starting from the address contained in convWeights.
 ## Outputs
 * @param validConvolutionIds: A pointer to an allocated array that will contain after the execution of the function the ids of the convolution that were actually executed.
                               This is due to the fact that some convolutions may not be supported by cudnn.
                               The memory area allocated for this array must contain a number of integers equal to nAlgos, but it can contain less.
 * @param validConvolutionsCount: The number of valid elements present in the validConvolutionIds array.
 * @param convOutputs: A pointer to an array of pointers that will be allocated within the function. It will contain the results
 *                     for each supported convolution operation. Each pointer in the array points to a contiguous block of memory
 *                     holding the output data from the convolution in floating-point format. The number of elements in each block
 *                     is determined by the output tensor dimensions. The number of valid pointers in the arrays is equal to validConvolutionsCount and the pointer
                       at the i-th position of convOutputs points to the result of the convolution algorithm stored to validConvolutionsCount[i].
 *                     The function handles the allocation of memory for each output block. The user is responsible for
 *                     eventually freeing the memory area addressed by each pointer of the array of pointer (all nAlgos pointers must be freed regardless they contain a valid result or not).

 * @param convOutputShape: A pointer to a 4 element int array that will contain, after the execution of the function, the NCHW shape of all output tensors. The shape is the same for every algorithm.
                      Each tensor contained in convOutputs will have this shape.

 *
 * @return 0
 */
int performConvolutions(
  const cudnnHandle_t *cudnnHandle,
  const cudnnConvolutionFwdAlgo_t *convolutionsAlgos,
  const int nAlgos,
  const float * const convInput,
  const int * const convInputShape,
  const float * const convWeights,
  const int * const convWeightsShape,
  int * const validConvolutionIds,
  int * const validConvolutionsCount,
  float **convOutputs,
  int * const convOutputShape
) {
  cudnnHandle_t cudnn = *cudnnHandle;

  const int in_n = convInputShape[0];
  const int in_c = convInputShape[1];
  const int in_h = convInputShape[2];
  const int in_w = convInputShape[3];

  #ifdef DEBUG
    std::cout << "in_n: " << in_n << std::endl;
    std::cout << "in_c: " << in_c << std::endl;
    std::cout << "in_h: " << in_h << std::endl;
    std::cout << "in_w: " << in_w << std::endl;
    std::cout << std::endl;
    fflush(stdout);
  #endif

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));


  float *d_convInput;

  CUDA_CALL(cudaMalloc(
        &d_convInput, in_n * in_c * in_h * in_w * sizeof(float)));
  CUDA_CALL(cudaMemcpy(
        d_convInput, convInput, in_n * in_c * in_h * in_w * sizeof(float), cudaMemcpyHostToDevice));

  const int filt_k = convWeightsShape[0];
  const int filt_c = convWeightsShape[1];
  const int filt_h = convWeightsShape[2];
  const int filt_w = convWeightsShape[3];

  #ifdef DEBUG
    std::cout << "filt_k: " << filt_k << std::endl;
    std::cout << "filt_c: " << filt_c << std::endl;
    std::cout << "filt_h: " << filt_h << std::endl;
    std::cout << "filt_w: " << filt_w << std::endl;
    std::cout << std::endl;
  #endif

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *d_convFilt;

  CUDA_CALL(cudaMalloc(
      &d_convFilt, filt_k * filt_c * filt_h * filt_w * sizeof(float)));
  CUDA_CALL(cudaMemcpy(
        d_convFilt, convWeights, filt_k * filt_c * filt_h * filt_w * sizeof(float),
        cudaMemcpyHostToDevice));

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  #ifdef DEBUG
    std::cout << "pad_h: " << pad_h << std::endl;
    std::cout << "pad_w: " << pad_w << std::endl;
    std::cout << "str_h: " << str_h << std::endl;
    std::cout << "str_w: " << str_w << std::endl;
    std::cout << "dil_h: " << dil_h << std::endl;
    std::cout << "dil_w: " << dil_w << std::endl;
    std::cout << std::endl;
  #endif

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          conv_desc,
          pad_h, pad_w, str_h, str_w, dil_h, dil_w,
          CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // output
    int out_n;
    int out_c;
    int out_h;
    int out_w;

    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
          conv_desc, in_desc, filt_desc,
          &out_n, &out_c, &out_h, &out_w));

  #ifdef DEBUG
    std::cout << "out_n: " << out_n << std::endl;
    std::cout << "out_c: " << out_c << std::endl;
    std::cout << "out_h: " << out_h << std::endl;
    std::cout << "out_w: " << out_w << std::endl;
    std::cout << std::endl;
  #endif


  convOutputShape[0] = out_n;
  convOutputShape[1] = out_h;
  convOutputShape[2] = out_c;
  convOutputShape[3] = out_w;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));



  float *d_convOut;

  unsigned int convOutputTensorSize = out_n * out_c * out_h * out_w;
  *convOutputs = (float *) calloc(nAlgos, convOutputTensorSize * sizeof(float));


  if(! *convOutputs){
    std::cout << "Error on allocating host convOutputs variable (malloc)" << std::endl;
    return 1;
  }

  CUDA_CALL(cudaMalloc(
        &d_convOut, convOutputTensorSize * sizeof(float)));

  std::cout << "Number of algorithms: " << nAlgos << std::endl;
  *validConvolutionsCount = 0;

  for(int i = 0; i < nAlgos; i++) {

    cudnnConvolutionFwdAlgo_t algo = convolutionsAlgos[i];
#ifdef DEBUG
    std::cout << "Convolution algorithm id: " << algo << std::endl;
    fflush(stdout);
#endif

    size_t ws_size;
    float *ws_data;
    cudnnStatus_t wsErr = cudnnGetConvolutionForwardWorkspaceSize(
          cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);
    if(wsErr == CUDNN_STATUS_NOT_SUPPORTED) {
      // Algorithm is not supported for tensor size configuration and so it is skipped.
      // Not a critical error
      std::cout << "Convolution Algorithm " << algo << " not supported for current configuration!" << std::endl;
    } else if(wsErr != CUDNN_STATUS_SUCCESS) {
      // If the error is different than not supported, halt the program.
      // There is an unhandled error
      CUDNN_CALL(err)
    } else {
      // Convolution may be supported
#ifdef DEBUG
      CUDA_CALL(cudaDeviceSynchronize());
      std::cout << "Workspace size: " << ws_size << std::endl;
      fflush(stdout);
#endif

      if(ws_size > 0) {
        CUDA_CALL(cudaMalloc(&ws_data, ws_size));

      } else {
        ws_data = nullptr;
      }


      float alpha = 1.f;
      float beta = 0.f;

      cudnnStatus_t err;
      err = cudnnConvolutionForward(
          cudnn,
          &alpha,
          in_desc,
          d_convInput,
          filt_desc,
          d_convFilt,
          conv_desc,
          algo,
          ws_data,
          ws_size,
          &beta,
          out_desc,
          d_convOut);
      if(err == CUDNN_STATUS_NOT_SUPPORTED) {
        // Algorithm is not supported for tensor size configuration and so it is skipped.
        // Not a critical error
        std::cout << "Convolution Algorithm " << algo << " not supported for current configuration!" << std::endl;
      } else if(err != CUDNN_STATUS_SUCCESS) {
        // If the error is different than not supported, halt the program.
        // There is an unhandled error
        CUDNN_CALL(err)
      } else {
        // Convolution is supported
        std::cout << "Convolution Algorithm " << algo << " executed" << std::endl;
        // Add algorithm id to the current
        validConvolutionIds[*validConvolutionsCount] = algo;
        (*validConvolutionsCount)++;

#ifdef DEBUG
        CUDA_CALL(cudaDeviceSynchronize())
        std::cout << "Convolution Executed size: " << std::endl;
        fflush(stdout);
#endif
        CUDA_CALL(cudaMemcpy(
              *convOutputs + i * convOutputTensorSize, d_convOut, convOutputTensorSize * sizeof(float),
              cudaMemcpyDeviceToHost));

#ifdef DEBUG
        CUDA_CALL(cudaDeviceSynchronize());
        std::cout << "Result Copied to CPU!" << std::endl;
        fflush(stdout);
#endif
      }

      if(ws_size > 0) {
          CUDA_CALL(cudaFree(ws_data));
      }
    }
    std::cout << std::endl;
  }

  // finalizing
  CUDA_CALL(cudaFree(d_convOut));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(d_convFilt));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(d_convInput));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
#ifdef DEBUG
    CUDA_CALL(cudaDeviceSynchronize());
    std::cout << "Freed performConvolutions internal resources " << std::endl;
    fflush(stdout);
#endif

  std::cout << std::endl;

  return 0;
}


/**
 * Maps the algorithmId to its name
 *
 * @param algorithmId id of the algorithm
 * @return The algorithm name associated to algorithmId
 */
std::string algorithmIdToName(int algorithmId) {
    switch(algorithmId) {
        case 0:
            return "IMPLICIT_GEMM";
        case 1:
            return "IMPLICIT_PRECOMP_GEMM";
        case 2:
            return "GEMM";
        case 3:
            return "DIRECT";
        case 4:
            return "FFT";
        case 5:
            return "FFT_TILING";
        case 6:
            return "WINOGRAD";
        case 7:
            return "WINOGRAD_NONFUSED";
        default:
            return "ALGORITHM NOT AVAILABLE";
    }
}


/**
 * @brief Injects a fault into a convolutional weight tensor.
 *
 * This function flips a specific bit in the convolutional weights tensor at a given position.
 *
 * @param convWeights Pointer to the convolutional weights tensor.
 * @param convWeightsShape Pointer to the shape of the convolutional weights tensor.
 * @param n Batch index.
 * @param c Channel index.
 * @param h Height index.
 * @param w Width index.
 * @param bitPos Bit position to flip.
 */
void injectFault(float *convWeights, int *convWeightsShape, int n, int c, int h, int w, int bitPos){
	int linearIndex;
	int bit;

    linearIndex = w + convWeightsShape[3] * (h + convWeightsShape[2] * (c + convWeightsShape[1] * n));

	bit = 1 << bitPos;

    int* intValuePtr = reinterpret_cast<int*>(&convWeights[linearIndex]);
    *intValuePtr ^= bit;
    convWeights[linearIndex] = *reinterpret_cast<float*>(intValuePtr);
}


/**
 * @brief Finds the maximum value in a subset of a convolutional output tensor.
 *
 * This function finds the maximum value within a specific region of a convolutional output tensor.
 *
 * @param convOutput Pointer to the convolutional output tensor.
 * @param index Index of the subset to analyze.
 * @param size Size of the subset to analyze.
 * @return The maximum value within the specified subset.
 */
float maxVal(float *convOutput, int index, int size) {
    int i, baseIndex;
    float maxVal;

    baseIndex = size * index;
    maxVal = convOutput[baseIndex];

    for (i = 1 ; i < size ; i++) {
        if (convOutput[baseIndex + i] > maxVal) {
            maxVal = convOutput[baseIndex + i];
        }
    }

    return maxVal;
}


/**
 * @brief Finds the minimum value in a subset of a convolutional output tensor.
 *
 * This function finds the minimum value within a specific region of a convolutional output tensor.
 *
 * @param convOutput Pointer to the convolutional output tensor.
 * @param index Index of the subset to analyze.
 * @param size Size of the subset to analyze.
 * @return The minimum value within the specified subset.
 */
float minVal(float *convOutput, int index, int size) {
    int i, baseIndex;
    float minVal;

    baseIndex = size * index;
    minVal = convOutput[baseIndex];

    for (i = 1 ; i < size ; i++) {
        if (convOutput[baseIndex + i] < minVal) {
            minVal = convOutput[baseIndex + i];
        }
    }

    return minVal;
}


/**
 * @brief Calculates the Root Mean Square Error (RMSE) between two subsets of a convolutional output tensor.
 *
 * This function calculates the RMSE between two subsets of a convolutional output tensor.
 *
 * @param convOutput Pointer to the convolutional output tensor.
 * @param indexFirstAlg Index of the first subset to compare.
 * @param indexSecondAlg Index of the second subset to compare.
 * @param size Size of the subsets to compare.
 * @return The RMSE between the two subsets.
 */
double calcRootMediumSqErr(float *convOutput, int indexFirstAlg, int indexSecondAlg, int size){

    int i, baseIndexFirstAlg, baseIndexSecondAlg;
    double sumSquaredDiff, meanSquaredDiff, diff;

    baseIndexFirstAlg = size * indexFirstAlg;
    baseIndexSecondAlg = size * indexSecondAlg;

    sumSquaredDiff = 0.0;

    for (i = 0 ; i < size ; i++) {
        diff = convOutput[baseIndexFirstAlg + i] - convOutput[baseIndexSecondAlg + i];
        sumSquaredDiff += diff * diff;
    }

    meanSquaredDiff = sumSquaredDiff / size;

    return sqrt(meanSquaredDiff);
}


/**
 * @brief Calculates the maximum relative error between two subsets of a convolutional output tensor.
 *
 * This function calculates the maximum relative error between two subsets of a convolutional output tensor.
 *
 * @param convOutput Pointer to the convolutional output tensor.
 * @param indexFirstAlg Index of the first subset to compare.
 * @param indexSecondAlg Index of the second subset to compare.
 * @param size Size of the subsets to compare.
 * @return The maximum relative error between the two subsets.
 */
double calcMaxRelErr(float *convOutput, int indexFirstAlg, int indexSecondAlg, int size){
	int i, baseIndexFirstAlg, baseIndexSecondAlg;
	double maxRelErr, relativeError;

    baseIndexFirstAlg = size * indexFirstAlg;
    baseIndexSecondAlg = size * indexSecondAlg;

    maxRelErr = 0.0;

    for (i = 0 ; i < size ; i++) {
        relativeError = fabs((convOutput[baseIndexFirstAlg + i] - convOutput[baseIndexSecondAlg + i]) / convOutput[baseIndexFirstAlg + i]);

        if (relativeError > maxRelErr) {
            maxRelErr = relativeError;
        }
    }

    return maxRelErr;
}


/**
 * @brief Calculates various metrics for the resilience evaluation of convolutional algorithms.
 *
 * This function calculates several metrics (max, min, RMSE, max relative error) for pairs of valid convolutional outputs,
 * and writes the results to a file.
 *
 * @param outputFile Pointer to the output file.
 * @param injectionId Identifier for the injection.
 * @param validConvolutionIds Array of valid convolution IDs.
 * @param validConvolutionsCount Number of valid convolutions.
 * @param convOutputs Pointer to the convolutional outputs tensor.
 * @param convOutputShape Pointer to the shape of the convolutional outputs tensor.
 */
void calculateMetrics(FILE* outputFile, int injectionId, int * validConvolutionIds, int validConvolutionsCount, float *convOutputs, int * convOutputShape){
	int i, j, tensorSize;
	float maxFirstAlg, maxSecondAlg, minFirstAlg, minSecondAlg;
    double rootMediumSqErr, maxRelErr;

	for(i = 0 ; i < validConvolutionsCount ; i++){
		for(j = i + 1 ; j < validConvolutionsCount ; j++){

			tensorSize = convOutputShape[0] * convOutputShape[1] * convOutputShape[2] * convOutputShape[3];

            maxFirstAlg = maxVal(convOutputs, validConvolutionIds[i], tensorSize);
            maxSecondAlg = maxVal(convOutputs, validConvolutionIds[j], tensorSize);
            minFirstAlg = minVal(convOutputs, validConvolutionIds[i], tensorSize);
            minSecondAlg = minVal(convOutputs, validConvolutionIds[j], tensorSize);

            rootMediumSqErr = calcRootMediumSqErr(convOutputs, validConvolutionIds[i], validConvolutionIds[j], tensorSize);
            maxRelErr = calcMaxRelErr(convOutputs, validConvolutionIds[i], validConvolutionIds[j], tensorSize);

            std::string firstAlgorithmName = algorithmIdToName(validConvolutionIds[i]);
            std::string secondAlgorithmName = algorithmIdToName(validConvolutionIds[j]);
            std::cout << injectionId << " " << firstAlgorithmName << " " << secondAlgorithmName << " " << rootMediumSqErr
                    << " " << maxRelErr << " " << maxFirstAlg << " " << maxSecondAlg << " " << minFirstAlg << " "<< minSecondAlg << std::endl;
            fprintf(outputFile, "%d,%s,%s,%lf,%lf,%f,%f,%f,%f\n",
                      injectionId,
                      firstAlgorithmName.c_str(),
                      secondAlgorithmName.c_str(),
                      rootMediumSqErr,
                      maxRelErr,
                      maxFirstAlg,
                      maxSecondAlg,
                      minFirstAlg,
                      minSecondAlg);
		}
	}

}


int main(int argc, char **argv) {
    // Check if the correct number of arguments are provided
    if (argc != 4) {
        std::cerr << "Incorrect parameters passed" << std::endl;
        return 1;
    }

    // Assign the arguments to variables
    std::string tensorBinFilePath = argv[1];
    std::string faultListPath = argv[2];
    std::string outputPath = argv[3];


  // Create cudnn handle (pass this to perform convolutions)
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // Read binary data for tensors
  FILE *file = fopen(tensorBinFilePath, "rb");
  if (file == NULL) {
      perror("Failed to open file");
      return EXIT_FAILURE;
  }

  // Read first int for number of tensors
  int num_tensors;
  fread(&num_tensors, sizeof(int), 1, file);
  printf("Number of tensors: %d\n", num_tensors);

  // Allocate array of float double pointer, each pointer points to a float array containing linearized tensors
  float **allTensors = (float **) malloc(num_tensors * sizeof(float *));

  // Allocate array of arrays of four integers, Each array contains the shape of the corresponding 4d tensor
  int (*shapes)[4] = (int (*)[4]) malloc(num_tensors * sizeof(*shapes));

  for (int i = 0; i < num_tensors; i++) {
      // For each tensor, the first for ints are the NCHW shape for the tensor
      fread(shapes[i], sizeof(int), 4, file);
      printf("Tensor %d dimensions: %dx%dx%dx%d\n", i+1, shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3]);

      // Read tensor data according to the shape
      allTensors[i] = (float *) malloc(shapes[i][0] * shapes[i][1] * shapes[i][2] * shapes[i][3] * sizeof(float));
      fread(allTensors[i], sizeof(float), shapes[i][0] * shapes[i][1] * shapes[i][2] * shapes[i][3], file);
  }
  fclose(file);
  // Data read

 // HERE I GOT IN allTensors the tensors, and in shapes the shapes
  int nAlgos = sizeof(convolutionAlgorithms) / sizeof(convolutionAlgorithms[0]);

  // Allocate array of pointers. Each pointer points to an array containing the outputs of a convolution with a particular algorithm.
  // The array can be filled partially if there are convolutions that are not supported.
  float *convOutputs;
  // The shape of the output tensors (common to all algorithms).
  int convOutputShape[4];

  // NUmber of valid convolutions. This determines the size of the convOutputs array and validConvolutionIds array.
  int validConvolutionCount;
  // Array for storing the ids of convolution that where valid
  int *validConvolutionIds = (int *) malloc(nAlgos * sizeof(int));

  int injectionId;
  int tensorId;
  int n, c, h, w;
  int bitPos;
  int tmp;

  FILE *fileInput = fopen(faultListPath, "r");
  if (fileInput == NULL) {
      perror("Failed to open the cvs fault_list file");
      return EXIT_FAILURE;
  }

  FILE *fileOutput = fopen(outputPath, "w");
  if (fileOutput == NULL) {
      perror("Error");
      return EXIT_FAILURE;
  }

  // Header of the output csv
  fprintf(fileOutput, "%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                "injectionId",
                "firstAlgorithmName",
                "secondAlgorithmName",
                "rootMediumSqErr",
                "maxRelErr",
                "maxFirstAlg",
                "maxSecondAlg",
                "minFirstAlg",
                "minSecondAlg");

  while ((tmp = fgetc(fileInput)) != EOF && tmp != '\n') {
      // Ignoring the header of the input csv
  }

  // Iterating for every row in the csv input file
  while (fscanf(fileInput, "%d,%d,%d,%d,%d,%d,%d", &injectionId, &tensorId, &n, &c, &h, &w, &bitPos) == 7){

  	// Injecting the fault
  	injectFault(allTensors[tensorId * 2 + 1], shapes[tensorId * 2 + 1], n, c, h, w, bitPos);

    // Execute convolutions
    performConvolutions(
      &cudnn,
      convolutionAlgorithms,
      nAlgos,
      allTensors[tensorId * 2],
      shapes[tensorId * 2],
      allTensors[tensorId * 2 + 1],
      shapes[tensorId * 2 + 1],
      validConvolutionIds,
      &validConvolutionCount,
      &convOutputs,
      convOutputShape
    );

    // Calculating the metrics
    calculateMetrics(fileOutput, injectionId, validConvolutionIds, validConvolutionCount, convOutputs, convOutputShape);

    // Injecting the fault
  	injectFault(allTensors[tensorId * 2 + 1], shapes[tensorId * 2 + 1], n, c, h, w, bitPos);

    // Free resources allocated by performConvolutions (all the output tensors)
    free(convOutputs);
  }

  fclose(fileInput);
  fclose(fileOutput);

  // From here free all the resources
  for(int i = 0; i < num_tensors; i++) {
    free(allTensors[i]);
  }
  free(allTensors);

  free(convOutputs);
  free(validConvolutionIds);
  // At the end always destroy cuddn handle

  CUDNN_CALL(cudnnDestroy(cudnn));

}

