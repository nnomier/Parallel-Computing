/**
 * @author Noha Nomier
 * @see "Seattle University, CPSC5600, Winter 2023"
 *
 * This is a program that demonstrates the ladner fischer prefix sum
 * parallel algorithm using std::async
 * The interior nodes of the tree in the algorithm is implemented
 * using an array representation of a heap
 *
 * Bonus: It supports any input size even if not a power of 2 by taking
 * the input size and converting it to the next smallest power of two
 * number and fills them with zeros
 * when computing the actual sum/prefix sum, checks are added to resolve any
 * out of bounds exceptions
 *
 */

#include <iostream>
#include <vector>
#include <future>
#include <math.h>

using namespace std;

typedef vector<int> Data;

int MAX_FORK_LEVELS = 4;

/**
 * The Heaper class is the foundation of the heap data structure,
 * providing the basic functionality for storing data and calculating
 * the indices of the parent, left, and right children.
 */
class Heaper {
public:
    /**
     * Constructor for Heaper class
     * n is the size of input data rounded to the nearest power of two
     * if it's not already
     * n and realSize are the same in the case of input size power of two
     * @param data  data that the heap will be built on
     */
    Heaper(const Data *data) : n(nextPowerOfTwo(data->size())), realSize(data->size()), data(data) {
        interior = new Data(n - 1, 0);
    }

    virtual ~Heaper() {
        delete interior;
    }
public:
    int n; // n is size of data, n-1 is size of interior
    int realSize;
    const Data *data;
    Data *interior;

    virtual int size() {
        return (n-1) + n;
    }

    virtual int value(int i) {
        //it's an interior node
        if (i < n - 1)
            return interior->at(i);
        //total size is (n-1) interior nodes + (realSize) data elements
        //so a condition is added in case of overflow when it's not power of two
        if(i>=(n-1+realSize)) {
            return 0;
        }
        return data->at(i - (n - 1));
    }

    virtual int parent(int i) {

        return (i - 1) / 2;
    }

    virtual int left(int i) {
        return i * 2 + 1;
    }

    virtual int right(int i) {
        return left(i) + 1;
    }

    bool isLeaf(int i) {
        return i >= n-1;
    }

private:
    long nextPowerOfTwo(long x) {
        return pow(2, ceil(log2(x)));
    }

};

/**
 * SumHeap is a child class of the Heaper class that is responsible
 * for the pair-wise sum pass and the prefix sum
 */
class SumHeap: public Heaper {
public:
    /**
     * Constructor for the SumHeap class
     * @param data  data that the heap will be built on
    */
    SumHeap(const Data *data) : Heaper(data) {
        calcSum(0, 0);

    }
    int sum(int node=0) {
        return value(node);
    }
public:
    /**
     * a public function that calculates the prefix sums of the data
     * in parallel by calling another calcPrefix function
     * @param prefixes data structure to store prefixes of the data
     */
    void prefixSums(Data *prefixes) {
        calcPrefix(0, 0, 0,prefixes);
    }
private:
    /**
     * function that calculates the pair wise sum of the heap
     * recursively, in parallel storing each pair-sum in internal
     * nodes
     * @param i current node in the heap
     * @param level  current level of the heap
    */
    void calcSum(int i, int level) {
        if (isLeaf(i))
            return;

        if(level<MAX_FORK_LEVELS) {
            auto future = async(launch::async, &SumHeap::calcSum, this, left(i), level+1);
            calcSum(right(i), level+1);
            future.wait();
        }
        else {
            calcSum(left(i), level+1);
            calcSum(right(i), level+1);
        }

        interior->at(i) = value(left(i)) + value(right(i));
    }

private:
    /**
     *  function that calculates the prefix sums of the data recursively,
     *  in parallel using async
     * @param i  current node in the heap
     * @param sumPrior  sum of the parent node
     * @param level  current level of the heap
     * @param prefixes  prefix sums of the data
    */
    void calcPrefix(int i, int sumPrior, int level, Data *prefixes) {

        if(isLeaf(i)) {
            // checks if current i is out of bound for the case of non power of two input
            if(i-(n-1)>=(int)prefixes->size()) return;

            prefixes->at(i - (n -1)) = sumPrior + value(i);

        } else if (level<MAX_FORK_LEVELS){
            // fork a thread for the left child
            auto leftFuture = async(launch::async,&SumHeap::calcPrefix, this, left(i), sumPrior, level+1, prefixes);
            // call right child from main thread recursively
            calcPrefix(right(i), sumPrior + value(left(i)),level+1, prefixes);
            // wait for  threads to complete
            leftFuture.wait();
        } else {
            calcPrefix(left(i), sumPrior,level+1, prefixes);
            calcPrefix(right(i), sumPrior + value(left(i)),level+1, prefixes);
        }
    }
};

const int N = 1<<26;

int main() {
    Data data(N, 1);  // put a 1 in each element of the data array
    data[0] = 10;

    Data prefix(N, 1);

    // start timer
    auto start = chrono::steady_clock::now();

    cout << "Initializing Heap Data and calculating Pair-wise Sum ... " << endl;
    SumHeap heap(&data);
    cout << "Calculating prefix sum for the input.." << endl;

    heap.prefixSums(&prefix);
    cout << "Done!" << endl;

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double,milli>(end-start).count();

    int check = 10;
    bool correct = true;
    for (int elem: prefix) {
        if (elem != check++) {
            correct = false;
            cout << "FAILED RESULT at " << check - 1;
            break;
        }
    }
    if(correct)
        cout << "Prefix sum is calculated correctly in total time of:  " << elpased << "ms" << endl;

    return 0;
}
