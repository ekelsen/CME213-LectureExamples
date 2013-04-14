__global__
void myKernel(int *in) {
    in[threadIdx.x] += 1;
}

int main(void) {
    int *dIn;
    cudaMalloc(&dIn, sizeof(int));

    myKernel<<<1, 128>>>(dIn);
    return 0;
}
