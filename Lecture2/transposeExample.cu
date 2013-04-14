#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>

__global__
void simpleTranspose(int *array_in, int *array_out, int rows_in, int cols_in)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    int m = tid % cols_in;
    int n = tid / cols_in;

    array_out[m * rows_in + n] = array_in[n * cols_in + m];
}

__global__
void simpleTranspose2D(int *array_in, int *array_out, int rows_in, int cols_in)
{
    const int m = threadIdx.x + blockDim.x * blockIdx.x;
    const int n = threadIdx.y + blockDim.y * blockIdx.y;

    array_out[m * rows_in + n] = array_in[n * cols_in + m];
}

template<int numWarps>
__global__
void fastTranspose(int *array_in, int *array_out, int rows_in, int cols_in)
{
    const int warpId   = threadIdx.y;
    const int lane     = threadIdx.x;
    const int warpSize = 32;
    const int smemRows = 32;

    __shared__ int block[smemRows][warpSize + 1];

    int bc = blockIdx.x;
    int br = blockIdx.y;

    //load 32x32 block into shared memory
    for (int i = 0; i < smemRows / numWarps; ++i) {
        int gr = br * smemRows + i * numWarps + warpId;
        int gc = bc * warpSize + lane;

        block[i * numWarps + warpId][lane] = array_in[gr * cols_in + gc];
    }

    __syncthreads();

    //now we switch to each warp outputting a row, which will read
    //from a column in the shared memory
    //this way everything remains coalesced
    for (int i = 0; i < smemRows / numWarps; ++i) {
        int gr = br * smemRows + lane;
        int gc = bc * warpSize + i * numWarps + warpId;

        array_out[gc * rows_in + gr] = block[lane][i * numWarps + warpId];
    }
}

void check_error() {
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("error: %s\n", cudaGetErrorString(err));
}

void print_array(std::vector<int> &array, int M, int N) {
    for (int n = N -1; n != -1; --n) {
        for (int m = 0; m < M; ++m) {
            printf("%d ", array[n * M + m]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    const int side = 2048;

    std::vector<int> hIn (side * side);
    std::vector<int> hOut(side * side);

    for(int i = 0; i < side * side; ++i)
        hIn[i] = random() % 100;

//    print_array(hIn, side, side);

    int *dIn, *dOut;
    cudaMalloc(&dIn,  sizeof(int) * side * side);
    cudaMalloc(&dOut, sizeof(int) * side * side);

    cudaMemcpy(dIn, &hIn[0], sizeof(int) * side * side, cudaMemcpyHostToDevice);

    const int numThreads = 256;
    const int numBlocks = (side * side + numThreads - 1) / numThreads;

    simpleTranspose<<<numBlocks, numThreads>>>(dIn, dOut, side, side);
    check_error();

    cudaMemcpy(&hOut[0], dOut, sizeof(int) * side * side, cudaMemcpyDeviceToHost);

    //    print_array(hOut, side, side);

    for (int n = 0; n < side; ++n) {
        for (int m = 0; m < side; ++m) {
            assert(hOut[n * side + m] == hIn[m * side + n]);
        }
    }

    dim3 bDim(16, 16);
    dim3 gDim(side / 16, side / 16);

    simpleTranspose2D<<<gDim, bDim>>>(dIn, dOut, side, side);
    check_error();

    cudaMemcpy(&hOut[0], dOut, sizeof(int) * side * side, cudaMemcpyDeviceToHost);

    //    print_array(hOut, side, side);

    for (int n = 0; n < side; ++n) {
        for (int m = 0; m < side; ++m) {
            assert(hOut[n * side + m] == hIn[m * side + n]);
        }
    }

    bDim.x = 32;
    bDim.y = 4;
    gDim.x = side / 32;
    gDim.y = side / 32;
    fastTranspose<4><<<gDim, bDim>>>(dIn, dOut, side, side);
    check_error();
    cudaMemcpy(&hOut[0], dOut, sizeof(int) * side * side, cudaMemcpyDeviceToHost);

    //    print_array(hOut, N, M);

    for (int n = 0; n < side; ++n) {
        for (int m = 0; m < side; ++m) {
            assert(hOut[n * side + m] == hIn[m * side + n]);
        }
    }

    return 0;
}
