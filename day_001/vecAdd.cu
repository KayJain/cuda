#include<stdio.h>

// kernel for vector addition
__global__ void vecAdd(float *A, float *B, float *C, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        C[idx] = A[idx] + B[idx];
    }
}

// code for cuda error handling
void checkCudaError(cudaError_t err, const char *file, int line){
    if (err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

int main(){
    // declare arrays
    int n = 1000;
    size_t size = n * sizeof(float);

    //allocate host memory
    float *A_h = (float*)malloc(size);
    float *B_h = (float*)malloc(size);
    float *C_h = (float*)malloc(size);

    //initialize host arrays A and B
    for(int i=0; i<n; i++){
        A_h[i] = i * 1.0f;
        B_h[i] = i * 2.0f;
    }

    // allocate device global memory for vectors.
    float *A_d, *B_d, *C_d;
    checkCudaError(cudaMalloc((void **)&A_d, size), __FILE__, __LINE__);
    checkCudaError(cudaMalloc((void **)&B_d, size), __FILE__, __LINE__);
    checkCudaError(cudaMalloc((void **)&C_d, size), __FILE__, __LINE__);

    //copy the vectors  from host to device
    checkCudaError(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice),  __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //call the kernel
    vecAdd<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    //copy the result from device to host
    checkCudaError(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    //verify the results
    for(int i=0; i<n; i++){
        if(C_h[i] != A_h[i] + B_h[i]){
            printf("calculation mismatch");
            break;
        }
    }
    printf("Vector addition completed successfully! \n");

    //Free device memory
    checkCudaError(cudaFree(A_d), __FILE__, __LINE__);
    checkCudaError(cudaFree(B_d), __FILE__, __LINE__);
    checkCudaError(cudaFree(C_d), __FILE__, __LINE__);

    //free host memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}