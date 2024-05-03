%%writefile template.cpp
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

std::string algorithmIdToName(int id) {
    switch(id) {
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
            return "NOT AVAILABLE";
    }
}


//TODO CHECK IF FLOAT * IS CORRECT
void injectFault(float * convWeights, int * convWeightsShape, int n, int c, int h, int w, int bitPos){
	int linearIndex;
	int bit;


	linearIndex = w + convWeightsShape[3] * (h + convWeightsShape[2] * (c + convWeightsShape[1] * n));

	bit = 1 << bitPos;
  std::cout << "Float Before Injection: " << convWeights[linearIndex] << std::endl;
  int* intValuePtr = reinterpret_cast<int*>(&convWeights[linearIndex]);
  std::cout << "Int Before Injection: " << intValuePtr << std::endl;
  *intValuePtr ^= bit;
  std::cout << "Int After Injection: " << intValuePtr << std::endl;
  convWeights[linearIndex] = *reinterpret_cast<float*>(intValuePtr);
  std::cout << "Float After Injection: " << convWeights[linearIndex] << std::endl;
}

float maxVal(float *convOutput, int size) {
    float maxVal = convOutput[0];
    int i;

    for (i = 1 ; i < size ; i++) {
        if (convOutput[i] > maxVal) {
            maxVal = convOutput[i];
        }
    }

    return maxVal;
}

float minVal(float *convOutput, int size) {
    float minVal = convOutput[0];
    int i;

    for (i = 1 ; i < size ; i++) {
        if (convOutput[i] < minVal) {
            minVal = convOutput[i];
        }
    }

    return minVal;
}

float calcRootMediumSqErr(float *convOutputFirstAlg, float *convOutputSecondAlg, int size){
	int i;
  float sumSquaredDiff, meanSquaredDiff, diff;

  sumSquaredDiff = 0.0;

  for (i = 0 ; i < size ; i++) {
      if(i == 0){
        std::cout << "OutputFirstAlg: " << convOutputFirstAlg[i] << std::endl;
        std::cout << "OutputSecondAlg: " << convOutputSecondAlg[i] << std::endl;
        std::cout << "Diff: " << convOutputFirstAlg[i] - convOutputSecondAlg[i] << std::endl;
      }
      diff = convOutputFirstAlg[i] - convOutputSecondAlg[i];
      sumSquaredDiff += diff * diff;
  }

  meanSquaredDiff = sumSquaredDiff / size;

  return sqrt(meanSquaredDiff);
}

float calcMaxRelErr(float *convOutputFirstAlg, float *convOutputSecondAlg, int size){
	int i;
	float maxRelErr, relativeError;

    maxRelErr = 0.0;

    for (i = 0 ; i < size ; i++) {
        relativeError = fabs((convOutputFirstAlg[i] - convOutputSecondAlg[i]) / convOutputFirstAlg[i]);

        if (relativeError > maxRelErr) {
            maxRelErr = relativeError;
        }
    }

    return maxRelErr;
}

void calculateMetrics(FILE* outputFile, int injectionId, int * validConvolutionIds, int validConvolutionsCount, float **convOutputs, int * convOutputShape){
	int i, j, tensorSize;
	float maxFirstAlg, maxSecondAlg, minFirstAlg, minSecondAlg, rootMediumSqErr, maxRelErr;

	for(i = 0 ; i < validConvolutionsCount ; i++){
		for(j = i + 1 ; j < validConvolutionsCount ; j++){

			//TODO CHECK IF THIS IS CORRECT
			tensorSize = convOutputShape[0] * convOutputShape[1] * convOutputShape[2] * convOutputShape[3];
      std::cout << "OutputSize: " << tensorSize << std::endl;

			maxFirstAlg = maxVal(convOutputs[i], tensorSize);
            maxSecondAlg = maxVal(convOutputs[j], tensorSize);
            minFirstAlg = minVal(convOutputs[i], tensorSize);
            minSecondAlg = minVal(convOutputs[j], tensorSize);

            rootMediumSqErr = calcRootMediumSqErr(convOutputs[i], convOutputs[j], tensorSize);
            maxRelErr = calcMaxRelErr(convOutputs[i], convOutputs[j], tensorSize);

            std::string firstAlgorithmName = algorithmIdToName(validConvolutionIds[i]);
            std::string secondAlgorithmName = algorithmIdToName(validConvolutionIds[j]);
            std::cout << injectionId << " " << firstAlgorithmName << " " << secondAlgorithmName << " " << maxFirstAlg
                      << " " << rootMediumSqErr << " " << maxRelErr <<std::endl;
            fprintf(outputFile, "%d, %s, %s, %f, %f, %f, %f, %f, %f, \n",
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
  // Process args (if you need it)

  // Create cudnn handle (pass this to perform convolutions)
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // Read binary data for tensors
  FILE *file = fopen("tensors.bin", "rb");
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

  //inj_id,tensor_id,k,c,h,w,bit_pos
  //PER OGNI INIEZIONE
  //1) SALVO I DATI CHE LEGGO DALLA RIGA (FREAD)
  //2) INIETTO IL GUASTO
  //3) FACCIO LE CONVOLUZIONI
  //4) CALCOLO LE METRICHE
  //5) RIFACCIO LO XOR COL GUASTO PER EVITARE GUASTI MULTIPLI
  //6) RILASCIO LA MEMORIA NELLO HEAP DEGLI OUTPUT
  // DUBBIO: COME FACCIO A CAPIRE SE UNA CONVOLUZIONE E' VALIDA O NO?
  // DUBBIO: PER OGNI FAULT INJECTION, DEVO FARE NUM_TENSORS/2 VOLTE CONVOLUZIONI O UNA PER INIEZIONE

  int injectionId;
  int tensorId;
  int n, c, h, w;
  int bitPos;
  int num_injections = 5;
  int tmp;

  FILE *fileInput = fopen("faultList.csv", "r");
  if (fileInput == NULL) {
      perror("Failed to open the cvs fault_list file");
      return EXIT_FAILURE;
  }

  FILE *fileOutput = fopen("FaultInjection.csv", "w");
  if (fileOutput == NULL) {
      perror("Error");
      return EXIT_FAILURE;
  }

  while ((tmp = fgetc(fileInput)) != EOF && tmp != '\n') {
      // Ignoring the header of the csv
  }

  for (int j = 0 ; j < num_injections ; j++){

  	// Reading the row from the csv fault list
  	fscanf(fileInput, "%d,%d,%d,%d,%d,%d,%d", &injectionId, &tensorId, &n, &c, &h, &w, &bitPos);
    std::cout << j << " injectionId:  " << injectionId << std::endl;
    std::cout << j << " tensordId: " << tensorId << std::endl;
    std::cout << j << " n: " << n << std::endl;
    std::cout << j << " c: " << c << std::endl;
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
    calculateMetrics(fileOutput, injectionId, validConvolutionIds, validConvolutionCount, &convOutputs, convOutputShape);

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

