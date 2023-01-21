#include <iostream>
#include <vector>
#include <future>

using namespace std;

typedef vector<int> Data;

int MAX_FORK_LEVELS = 4;

class Heaper {
public:
    Heaper(const Data *data) : n(data->size()), data(data) {

        interior = new Data(n - 1, 0);
    }
    virtual ~Heaper() {
        delete interior;
    }
public:
    int n; // n is size of data, n-1 is size of interior
    const Data *data;
    Data *interior;

    virtual int size() {
        return (n-1) + n;
    }

    virtual int value(int i) {
        if (i < n - 1)
            return interior->at(i);
        else
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
};

class SumHeap: public Heaper {
public:
    SumHeap(const Data *data) : Heaper(data) {
        calcSum(0, 0);
    }
    int sum(int node=0) {
        return value(node);
    }
public:
    void prefixSums(Data *prefixes) {
        calcPrefix(0, 0, 0,prefixes);
    }
private:
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
    void calcPrefix(int i, int sumPrior, int level, Data *prefixes) {
        if(isLeaf(i)) {
            prefixes->at(i - (n -1)) = sumPrior + value(i); // why i+1-n
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

const int N = 1<<26;  // FIXME must be power of 2 for now

//const int N = 4;

int main() {
    Data data(N, 1);  // put a 1 in each element of the data array
    data[0] = 10;
    Data prefix(N, 1);

    // start timer
    auto start = chrono::steady_clock::now();

    SumHeap heap(&data);

    heap.prefixSums(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double,milli>(end-start).count();

    int check = 10;
    for (int elem: prefix)
        if (elem != check++) {
            cout << "FAILED RESULT at " << check-1;
            break;
        }
    cout << "in " << elpased << "ms" << endl;
    return 0;
}
