/**
 * reduce_scan_1block.cu - using dissemination reduction for reducing and scanning a small array with CUDA
 * Kevin Lundeen, Seattle University, CPSC 5600 demo program
 * Notes:
 * - only works for one block (maximum block size for all of CUDA is 1024 threads per block)
 * - eliminated the remaining data races that were in reduce_scan_naive.cu
 * - algo requires power of 2 so we pad with zeros up to 1024 elements
 * - now much faster using block shared memory during loops (which also handily exposed the data races we had before)
 */

#include <iostream>
using namespace std;

const int MAX_BLOCK_SIZE = 1024;
const int MAX_ELEMENTS = 4096; // FIXME: change this to 2^20

__global__ void blockScan(float *data, float *blockSums) {
    __shared__ float local[MAX_BLOCK_SIZE];
    int gindex = threadIdx.x;
    int bindex = blockIdx.x;
    int blockSize = blockDim.x;
    int index = bindex * blockSize + gindex;
    local[gindex] = data[index];
    for (int stride = 1; stride < blockSize; stride *= 2) {

        __syncthreads();  // cannot be inside the if-block 'cuz everyone has to call it!
        int addend = 0;
        if (stride <= gindex)
            addend = local[gindex - stride];

        __syncthreads();
        local[gindex] += addend;
    }
    data[index] = local[gindex];
    // Store the sum at the last index of the block
    if (gindex == blockSize - 1) {
        blockSums[bindex] = local[blockSize-1];
    }
}

__global__ void addCumulativeSums(float *data, float *blockSums) {
    int gindex = threadIdx.x;
    int bindex = blockIdx.x;
    int blockSize = blockDim.x;
    int index = bindex * blockSize + gindex;

    for(int i = 0 ; i<bindex ; i++) {
        data[index] += blockSums[i];
    }
}

void fillArray(float *data, int n, int sz) {
    for (int i = 0; i < n; i++)
        data[i] = (float)(i+1); // + (i+1)/1000.0;
    for (int i = n; i < sz; i++)
        data[i] = 0.0; // pad with 0.0's for addition
}

void printArray(float *data, int n, string title, int m=5) {
    cout << title << ":";
    for (int i = 0; i < m; i++)
        cout << " " << data[i];
    cout << " ...";
    for (int i = n - m; i < n; i++)
        cout << " " << data[i];
    cout << endl;
}

int main(void) {
    int n;
    float *data;
    int threads = MAX_BLOCK_SIZE;
    cout << "How many data elements? ";
    cin >> n;
    // if (n > threads) {
    // 	cerr << "Cannot do more than " << threads << " numbers with this simple algorithm!" << endl;
    // 	return 1;
    // }
    int numBlocks = (MAX_ELEMENTS + threads - 1) / threads;
    cout<<"BLOCKS"<<numBlocks<<endl;
    cudaMallocManaged(&data, MAX_ELEMENTS * sizeof(*data));
    fillArray(data, n, MAX_ELEMENTS);
    printArray(data, n, "Before");
    float* blockSums;
    cudaMallocManaged(&blockSums, numBlocks * sizeof(float));
    // allreduce<<<1, threads>>>(data);
    // cudaDeviceSynchronize();
    // printArray(data, n, "Reduce");
    // fillArray(data, n, threads);

    blockScan<<<numBlocks, threads>>>(data, blockSums);
    cudaDeviceSynchronize();

    addCumulativeSums<<<numBlocks, threads>>>(data,blockSums);
    cudaDeviceSynchronize();

    printArray(data, n, "Scan");
    cudaFree(data);
    cudaFree(blockSums);
    return 0;
}
