#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

using namespace std;

const int MAX_BLOCK_SIZE = 1024;

struct X_Y
{
    int index;
    float x;
    float y;
    float scan;
};

int nextPowerOfTwo(int x) {
    return pow(2, ceil(log2(x)));
}

vector<string> split(const string &s, char delimiter)
{
    vector<string> result;
    stringstream ss(s);
    string token;
    while (getline(ss, token, delimiter))
    {
        result.push_back(token);
    }
    return result;
}

X_Y *load_data(string filename, int &n, int &powerOfTwoN)
{
    ifstream input(filename);
    string line;
    vector<X_Y> data;
    int index = 1;
    getline(input, line); // skip header line
    float max_x = 0.0;
    while (getline(input, line)) {
        vector<string> fields = split(line, ',');
        float x = stof(fields[0]);
        max_x = max(max_x, x);
        X_Y record = {index, x, stof(fields[1]), 0.0};

        data.push_back(record);
        index++;
    }
    n = data.size();

    powerOfTwoN = nextPowerOfTwo(n);
    cout<<"Read " << n << " lines into X_Y array..." << endl;
    /*
     * The input needs to be a power of two for bitonic sort
     * so if the file size is not a power of two, we should fill
     * the rest of the array with maximum numbers to stay at the end
     * of the array even after sorting
     */
    X_Y *result = new X_Y[powerOfTwoN];
    copy(data.begin(), data.end(), result);
    max_x = max_x + 1.0f;
    for(int i = n ; i<powerOfTwoN; i++) {
        result[i] = {0, max_x, 0, 0.0};
    }

    if(powerOfTwoN != n) cout<< "Padded data to a size of: " << powerOfTwoN << endl;

    return result;
}

__global__ void blockScan(X_Y *data, float *blockSums) {
    __shared__ float local[MAX_BLOCK_SIZE];
    int gindex = threadIdx.x;
    int bindex = blockIdx.x;
    int blockSize = blockDim.x;
    int index = bindex * blockSize + gindex;
    local[gindex] = data[index].y;
    for (int stride = 1; stride < blockSize; stride *= 2) {
        __syncthreads(); // cannot be inside the if-block 'cuz everyone has to call it!
        float addend = 0.0;
        if (stride <= gindex)
            addend = local[gindex - stride];

        __syncthreads();
        local[gindex] += addend;
    }
    data[index].scan = local[gindex];

    // Store the sum at the last index of the block
    if (gindex == blockSize - 1) {
        blockSums[bindex] = local[blockSize - 1];
    }
}

__global__ void addCumulativeSums(X_Y *data, float *blockSums)
{
    int gindex = threadIdx.x;
    int bindex = blockIdx.x;
    int blockSize = blockDim.x;
    int index = bindex * blockSize + gindex;

    for (int i = 0; i < bindex; i++) {
        data[index].scan += blockSums[i];
    }
}

void scan(X_Y *d_data, int n, int threads, int numBlocks)
{
    float *blockSums;
    cudaMallocManaged(&blockSums, numBlocks * sizeof(float));

    blockScan<<<numBlocks, threads>>>(d_data, blockSums);
    cudaDeviceSynchronize();

    addCumulativeSums<<<numBlocks, threads>>>(d_data, blockSums);
    cudaDeviceSynchronize();

    // Free blockSums
    cudaFree(blockSums);
}

/**
  * swaps the given elements in the given array
  * (note the __device__ moniker that says it can only
  *  be called from other device code; we don't need it
  *  here, but __device__ functions can return a value
  *  even though __global__'s cannot)
  */
__device__ void swap(X_Y *data, int a, int b) {
    X_Y temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

/**
  * inside of the bitonic sort loop for a particular value of i for a given value of k
  * (this function assumes j <= MAX_BLOCK_SIZE--bigger than that and we'd need to
  *  synchronize across different blocks)
  */
__global__ void bitonic(X_Y *data, int k, int j, int block) {
    int i = blockDim.x * block + threadIdx.x;
    int ixj = i ^ j;
    // avoid data race by only having the lesser of ixj and i actually do the comparison
    if (ixj > i) {
        if ((i & k) == 0 && data[i].x > data[ixj].x)
            swap(data, i, ixj);
        if ((i & k) != 0 && data[i].x < data[ixj].x)
            swap(data, i, ixj);
    }
    // wait for all the threads to finish before the next comparison/swap
    __syncthreads();
}

void sort(X_Y *d_data, int n, int threads, int numBlocks) {
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            for(int b = 0; b<numBlocks; b++) {
                bitonic<<<1, MAX_BLOCK_SIZE>>>(d_data, k, j, b);
            }
        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}

void dump_data(string outfile, X_Y *data, int n) {
    ofstream file(outfile);

    // Write column headers
    file << "n,x,y,scan\n";

    // Write data to file
    for (int i = 0; i < n; i++)
    {
        file << data[i].index << ","
             << data[i].x << ","
             << data[i].y << ","
             << data[i].scan << "\n";
    }

    file.close();
}

bool areCsvFilesEqual(const std::string& filename1, const std::string& filename2) {
    std::ifstream file1(filename1);
    std::ifstream file2(filename2);
    std::vector<std::string> lines1;
    std::vector<std::string> lines2;
    std::string line;

    // Read the lines from the first file
    while (std::getline(file1, line)) {
        lines1.push_back(line);
    }

    // Read the lines from the second file
    while (std::getline(file2, line)) {
        lines2.push_back(line);
    }

    // Compare the size of both vectors
    if (lines1.size() != lines2.size()) {
        return false;
    }

    // Compare each element of both vectors
    for (size_t i = 0; i < lines1.size(); ++i) {
        if (lines1[i] != lines2[i]) {
            cout<<"1 " << lines1[i] <<" 2 " <<lines2[i]<<endl;
            // return false;
        }
    }

    // The files are the same
    return true;
}

int main(void)
{
    string infile = "x_y/x_y_scan.csv";
    string outfile = "./output.csv";
    int n = 0;
    int nPowerOfTwo = 0;
    X_Y *data = load_data(infile, n, nPowerOfTwo);
    X_Y *d_data;
    cudaMalloc(&d_data, nPowerOfTwo * sizeof(X_Y));

    // Copy data from host to device
    cudaMemcpy(d_data, data, nPowerOfTwo * sizeof(X_Y), cudaMemcpyHostToDevice);

    int threads = nPowerOfTwo < MAX_BLOCK_SIZE ? nPowerOfTwo : MAX_BLOCK_SIZE;
    int numBlocks = (nPowerOfTwo + threads - 1) / threads;

    sort(d_data, nPowerOfTwo, threads, numBlocks);
    scan(d_data, n, threads, numBlocks);

    cudaMemcpy(data, d_data, nPowerOfTwo * sizeof(X_Y), cudaMemcpyDeviceToHost);

    int index  = -1;
    // use data array here
    for (int i = 1; i < n; i++) {
        if(data[i].x<data[i-1].x) {
            index = i;
            break;
        }
        // cout << "index: " << data[i].index << ", x: " << data[i].x << ", y: " << data[i].y << ", scan: " << data[i].scan << endl;
    }

    // sanity check
    if(index == -1) {
        cout<<"[Sort Success] Output data has all the data sorted on the x column!" << endl;
    } else {
        cout<<"[Sort Fail] Output data has unsorted elements at index: " << index << endl;
    }

    dump_data(outfile, data, n);
    cout << "Scanned data is written successfully in: " << outfile << endl;
    // free the memory allocated by the load_data function
    if (areCsvFilesEqual("x_y/x_y_scan_100.csv", "./output.csv")) {
        std::cout << "The files are the same.\n";
    } else {
        std::cout << "The files are different.\n";
    }

    cudaFree(d_data);
    delete[] data;

    return 0;
}
