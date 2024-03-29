/**
 * Kevin Lundeen, Seattle University, CPSC 5600
 * bitonic_naive.cu - a bitonic sort that only works when the j-loop fits in a single block
 *                  - n must be a power of 2
 */
#include <iostream>
#include <random>
using namespace std;

/**
  * swaps the given elements in the given array
  * (note the __device__ moniker that says it can only
  *  be called from other device code; we don't need it
  *  here, but __device__ functions can return a value
  *  even though __global__'s cannot)
  */
__host__ __device__ void swap(float *data, int a, int b) {
    float temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

/**
  * inside of the bitonic sort loop for a particular value of i for a given value of k
  * (this function assumes j <= MAX_BLOCK_SIZE--bigger than that and we'd need to
  *  synchronize across different blocks)
  */
__global__ void bitonic(float *data, int k, int j, int block) {
    int i = blockDim.x * block + threadIdx.x;
    int ixj = i ^ j;
    // avoid data race by only having the lesser of ixj and i actually do the comparison
    if (ixj > i) {
        if ((i & k) == 0 && data[i] > data[ixj])
            swap(data, i, ixj);
        if ((i & k) != 0 && data[i] < data[ixj])
            swap(data, i, ixj);
    }
    // wait for all the threads to finish before the next comparison/swap
    __syncthreads();
}


int main() {
    const int MAX_BLOCK_SIZE = 1024; // true for all CUDA architectures so far
    int threads = MAX_BLOCK_SIZE;
    int n;
    cout << "n = ? (must be power of 2): ";
    cin >> n;
    if (pow(2,floor(log2(n))) != n) {
        cerr << "n must be power of 2 and <= " << MAX_BLOCK_SIZE << endl;
        return 1;
    }

    // use managed memory for the data array
    float *data;
    cudaMallocManaged(&data, n * sizeof(*data));
    int numBlocks = (n + threads - 1) / threads;
    // fill it with random values
    random_device r;
    default_random_engine gen(r());
    uniform_real_distribution<float> rand(-3.14, +3.14);
    for (int i = 0; i < n; i++)
        data[i] = rand(gen);

    for (int k = 2; k <= n; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            for(int b = 0; b<numBlocks; b++) {
                bitonic<<<1, MAX_BLOCK_SIZE>>>(data, k, j, b);
            }
            cudaDeviceSynchronize();
        }
    }
    cudaDeviceSynchronize();

    // print out results
    for (int i = 1; i < n; i++){
        if (i < 3 || i >= n - 3 || i % 100 == 0)
            cout << data[i] << " ";
        else
            cout << ".";
    }
    cout << endl;
    cudaFree(data);
    return 0;
}
