#include<stdio.h>

// kernel for vector addition
__global__ void vecAdd(float *A, float *B, float *C, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        C[idx] = A[idx] + B[idx];
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
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    //copy the vectors  from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    //call the kernel
    vecAdd<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    //copy the result from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    //verify the results
    for(int i=0; i<n; i++){
        if(C_h[i] != A_h[i] + B_h[i]){
            printf("calculation mismatch");
            break;
        }
    }
    printf("Vector addition completed successfully! \n");

    //Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    //free host memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}