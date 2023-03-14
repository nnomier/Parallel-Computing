/**
 * This is a CUDA program that loads data from a CSV file,
 * sorts the data based on the X column using bitonic sort algorithm,
 * and calculates the cumulative sum of the Y column using block scan.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

using namespace std;

const int MAX_BLOCK_SIZE = 1024;

/**
 * Data type to hold values from csv file
 */
struct X_Y
{
    int index;
    float x;
    float y;
    float scan;
};

// calculates the smallest power of 2 number bigger than x
int nextPowerOfTwo(int x) {
    return pow(2, ceil(log2(x)));
}

// Splits a string on a given @param delimiter
// @returns a vector of resulted string tokens
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

/**
 * A function to count the number of lines in a csv file
 */
int count_lines_in_csv(const string& filename) {
    int count = 0;
    string line;

    ifstream file(filename);
    while (getline(file, line)) {
        count++;
    }

    return count-1; // to diregard the header line
}

/**
 * A function that loads the data in file with the name @param filename
 * into an X_Y array of size @param powerOfTwoN which is the next smallest
 * power of 2 number bigger than @param n
 * @returns the array of X_Y values
 */
X_Y *load_data(string filename, int &n, int &powerOfTwoN)
{
    n = count_lines_in_csv(filename);
    powerOfTwoN = nextPowerOfTwo(n);
    X_Y *data;
    cudaMallocManaged(&data, powerOfTwoN * sizeof(X_Y));

    ifstream input(filename);
    string line;
    int index = 1;
    getline(input, line); // skip header line
    float max_x = 0.0;
    while (getline(input, line)) {
        vector<string> fields = split(line, ',');
        float x = stof(fields[0]);
        max_x = max(max_x, x);
        X_Y record = {index, x, stof(fields[1]), 0.0};

        data[index-1] = (record);
        index++;
    }

    cout<<"Read " << n << " lines into X_Y array..." << endl;
    /*
     * The input needs to be a power of two for bitonic sort
     * so if the file size is not a power of two, we should fill
     * the rest of the array with maximum numbers to stay at the end
     * of the array even after sorting
     */
    max_x = max_x + 1.0f;
    for(int i = n ; i<powerOfTwoN; i++) {
        data[i] = {0, max_x, 0, 0.0};
    }

    if(powerOfTwoN != n) cout<< "Padded data to a size of: " << powerOfTwoN << endl;

    return data;
}

/**
 * A cuda kernel to calculate dissemination scan for each block of data
 * @param data X_Y array to perform prefix sum on its y values
 * @param blockSums an array of floats of size number of blocks
 * to store the sum at each block
 */
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

/**
 * Cuda device to accumulate block sums in @param blockSums
 * to the scan values of @param data
 */
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

/**
 * Calculate dissemination scan on @param data of size @param n
 */
void scan(X_Y *data, int n, int threads, int numBlocks)
{
    float *blockSums;
    cudaMallocManaged(&blockSums, numBlocks * sizeof(float));

    blockScan<<<numBlocks, threads>>>(data, blockSums);
    cudaDeviceSynchronize();

    addCumulativeSums<<<numBlocks, threads>>>(data, blockSums);
    cudaDeviceSynchronize();

    // Free blockSums
    cudaFree(blockSums);
}

/**
  * swaps the given elements in the given array
  */
__device__ void swap(X_Y *data, int a, int b) {
    X_Y temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

/**
  * inside of the bitonic sort loop for a particular value of i for a given value of k
  * ,a given j and block
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

/**
 * A function to sort @param data of size @param n on their x values
 */
void sort(X_Y *data, int n, int threads, int numBlocks) {
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            for(int b = 0; b<numBlocks; b++) {
                bitonic<<<1, MAX_BLOCK_SIZE>>>(data, k, j, b);
            }
        }
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}

/**
 * A fucntion to write the resulting @param data of size @param n
 * to an output file with the name @param output
 */
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

/**
 * A utility function to test if two csv files are different or not
 */
bool areCsvFilesEqual(const string& filename1, const string& filename2) {
    ifstream file1(filename1);
    ifstream file2(filename2);
    vector<string> lines1;
    vector<string> lines2;
    string line;

    // Read the lines from the first file
    while (getline(file1, line)) {
        lines1.push_back(line);
    }

    // Read the lines from the second file
    while (getline(file2, line)) {
        lines2.push_back(line);
    }

    // Compare the size of both vectors
    if (lines1.size() != lines2.size()) {
        return false;
    }

    // Compare each element of both vectors
    for (size_t i = 500 ; i < 1500; ++i) {
        if (lines1[i] != lines2[i]) {
            return false;
        }
    }

    // The files are the same
    return true;
}

int main(void)
{
    string infile = "x_y/x_y_100.csv";
    string outfile = "./output.csv";
    int n = 0;
    int nPowerOfTwo = 0;
    X_Y *data = load_data(infile, n, nPowerOfTwo);

    int threads = nPowerOfTwo < MAX_BLOCK_SIZE ? nPowerOfTwo : MAX_BLOCK_SIZE;
    int numBlocks = (nPowerOfTwo + threads - 1) / threads;

    sort(data, nPowerOfTwo, threads, numBlocks);
    scan(data, n, threads, numBlocks);

    int index  = -1;
    for (int i = 1; i < n; i++) {
        if(data[i].x<data[i-1].x) {
            index = i;
            break;
        }
    }

    // sorting sanity check
    if(index == -1) {
        cout<<"[Sort Success] Output data has all the data sorted on the x column!" << endl;
    } else {
        cout<<"[Sort Fail] Output data has unsorted elements at index: " << index << endl;
    }

    dump_data(outfile, data, n);
    cout << "Scanned data is written successfully in: " << outfile << endl;

    // if (areCsvFilesEqual("x_y/x_y_scan.csv", "./output.csv")) {
    //    cout << "The files are the same.\n";
    // } else {
    //    cout << "The files are different.\n";
    // }

    // free the memory allocated by the load_data function
    cudaFree(data);

    return 0;
}
