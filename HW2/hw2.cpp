#include <iostream>
#include <vector>

using namespace std;

typedef vector<int> Data;

int N_THREADS = 16

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
        calcSum(0);
    }
    int sum(int node=0) {
        return value(node);
    }
public:
    void prefixSums(Data *prefixes) {
        calcPrefix(0, 0, prefixes);
    }
private:
    void calcSum(int i) {
        if (isLeaf(i))
            return;
        calcSum(left(i));
        calcSum(right(i));

        interior->at(i) = value(left(i)) + value(right(i));
    }
private:
    void calcPrefix(int i, int sumPrior, Data *prefixes) {
        if(isLeaf(i)) {
            prefixes->at(i - (n -1)) = sumPrior + value(i); // why i+1-n
        } else {
            calcPrefix(left(i), sumPrior, prefixes);
            calcPrefix(right(i), sumPrior + value(left(i)), prefixes);
        }

    }
};

const int N = 1<<4;  // FIXME must be power of 2 for now

//const int N = 4;

int main() {
    Data data(N, 1);  // put a 1 in each element of the data array
    data[0] = 10;
    Data prefix(N, 1);

    // start timer
    auto start = chrono::steady_clock::now();

    SumHeap heap(&data);

    heap.prefixSums(&prefix);
    cout << "Data vector values:" << endl;
    for (auto elem : *heap.data) {
        cout << elem << " ";
    }
    cout << endl;

    cout << "Interior vector values:" << endl;
    for (auto elem : *heap.interior) {
        cout << elem << " ";
    }
    cout << endl;

    cout << "Prefix vector values:" << endl;
    for (auto elem : prefix) {
        cout << elem << " ";
    }
    cout << endl;

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
//            13
//    11             2
//10      1       1       1
