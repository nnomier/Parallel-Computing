/**
 *
 * @author Noha Nomier
 * @see "Seattle University, CPSC5600, Winter 2023"
 * Based on idea from Matthew Flatt, Univ of Utah
 *
 * This is a program that demonstrates the use of threads to
 * parallelize the computation of heavy work on a large dataset
 * The program uses two thread groups, one for encoding the input data
 * and one for decoding the input data.
 */
#include <iostream>
#include <string>
#include "ThreadGroup.h"

using namespace std;

const int N_THREADS = 2;

// A struct, SharedData, which is used to store the data that
// is shared between the threads.
struct SharedData {
    int dataLength;
    int* data;
};

/**
 * Function to simulate encoding by performing a time-consuming
 * operation on an integer input.
 *
 * @param v The integer input to be encoded.
 * @return The encoded value of the input.
 */
 int encode(int v) {
    // do something time-consuming (and arbitrary)
    for (int i = 0; i < 500; i++)
        v = ((v * v) + v) % 10;
    return v;
}

/**
 * Function to simulate decoding by performing a time-consuming
 * operation on an integer input.
 *
 * @param v The integer input to be decoded.
 * @return The decoded value of the input.
 */
int decode(int v) {
    // do something time-consuming (and arbitrary)
    return encode(v);
}

class DataThread {
public:
    /**
     * Calculates the chunk size for this thread and start/end indices
     * instead of typing the same code in both Thread types
     * @param id The id of the thread.
     * @param sharedData A pointer to the shared data struct that the thread will operate on.
     */
    void calculateChunk(int id, SharedData *sharedData) {
        int chunk_size = sharedData->dataLength / N_THREADS;
        start_index = id * chunk_size;
        end_index = id != N_THREADS - 1 ? start_index + chunk_size : sharedData->dataLength;
    }

protected:
    int start_index, end_index;
};


/**
 * Class which inherits from DataThread representing a thread
 * that performs encoding on a shared data array.
 *
 * It overloads the () operator to define the behavior of the thread which
 * is invoked in ThreadGroup code
 *
 * @param id The id of the thread.
 * @param sharedData A pointer to the shared data struct that the thread will operate on.
 */
class EncodeThread : public DataThread {
public:
    void operator()(int id, void *sharedData) {
        SharedData *ourData = (SharedData*) sharedData;
        calculateChunk(id, ourData);
        cout << "Encoder thread with id: [" << id << "] encoding data from index: " << start_index << " to " << end_index << std::endl;

        //  encoding on the assigned data chunk
        for (int i = start_index; i < end_index; i++) {
            ourData->data[i] = encode(ourData->data[i]);
        }
    }
};

/**
 * Class which inherits from DataThread representing a thread that
 * performs decoding on a shared data array.
 *
 * It overloads the () operator to define the behavior of the thread which
 * is invoked in ThreadGroup code
 *
 * @param id The id of the thread.
 * @param sharedData A pointer to the shared data struct that the thread will operate on.
 */
class DecodeThread : public DataThread {
public:
    void operator()(int id, void *sharedData) {
        SharedData *ourData = (SharedData *) sharedData;
        calculateChunk(id, ourData);
        cout << "Decoder thread with id [" << id << "] decoding data from index: " << start_index << " to " << end_index
             << std::endl;

        //  decoding on the assigned data chunk
        for (int i = start_index; i < end_index; i++) {
            ourData->data[i] = decode(ourData->data[i]);
        }
    }
};

/**
 * Computes the prefix sum of the input array using mutiple threads.
 *
 * First it encodes the input array using two EncodeThread threads,
 * then performs a prefix sum on the encoded array,
 * and finally decodes the array using two DecodeThread threads.
 *
 * @param data Pointer to the input array.
 * @param length Length of the input array.
 */
void prefixSums(int *data, int length) {
    cout << "Starting " << N_THREADS << "  encoder threads." << std::endl;
    ThreadGroup<EncodeThread> encodeGroup;
    SharedData ourData = {length, data};

    for (int i = 0; i < N_THREADS; i++)
        encodeGroup.createThread(i, &ourData);

    encodeGroup.waitForAll();

    for (int i = 1; i < length; i++) {
        data[i] = data[i-1] + data[i];
    }

    cout << "Starting " << N_THREADS << "  decoder threads." << std::endl;
    ThreadGroup<DecodeThread> decodeGroup;

    for (int i = 0; i < N_THREADS; i++)
        decodeGroup.createThread(i, &ourData);

    decodeGroup.waitForAll();
}

int main() {
    int length = 1000 * 1000;

    // make array
    int *data = new int[length];
    for (int i = 1; i < length; i++)
        data[i] = 1;
    data[0] = 6;

    // transform array into converted/deconverted prefix sum of original
    prefixSums(data, length);

    // printed out result is 6, 6, and 2 when data[0] is 6 to start and the rest 1
    cout << "[0]: " << data[0] << endl
         << "[" << length/2 << "]: " << data[length/2] << endl
         << "[end]: " << data[length-1] << endl;

    delete[] data;
    return 0;
}

