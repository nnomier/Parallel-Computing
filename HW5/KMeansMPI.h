#pragma once  // only process the first time it is included; ignore otherwise
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <set>
#include <array>
#include <mpi.h>

using namespace std;

template <int k, int d>
class KMeansMPI {
public:
    // some type definitions to make things easier
    typedef std::array<u_char,d> Element;
    class Cluster;
    typedef std::array<Cluster,k> Clusters;
    const int MAX_FIT_STEPS = 300;

    // debugging
    const bool VERBOSE = false;  // set to true for debugging output
#define V(stuff) if(VERBOSE) {using namespace std; stuff}

    /**
     * Expose the clusters to the client readonly.
     * @return clusters from latest call to fit()
     */
    virtual const Clusters& getClusters() {
        return clusters;
    }

    /**
     * fit() is the main k-means algorithm
    */
    virtual void fit(const Element *data, int data_n) {
        elements = data;
        n = data_n;
        fitWork(ROOT);
        cout<<"ALL ELEMENTS: " <<endl;
        for (int i = 0; i < n; i++) {
            std::cout << "Element " << i << ": (";
            for (int j = 0; j < d; j++) {
                std::cout << (int)elements[i][j];
                if (j != d-1) {
                    std::cout << ", ";
                }
            }
            std::cout << ")" << std::endl;
        }
    }

    /**
     * This is a per-process work for the fitting
     * @param rank within MPI_COMM_WORLD
     * @pre n and elements are set in the ROOT process; all p processes call fitWork simultaneously
     * @post clusters are now stable ( or we gave up after MAX_FIT_STEPS)
     */
    virtual void fitWork(int rank) {
        MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

        int *sendcounts = nullptr;
        int *displs =  nullptr;
        partition_size = n / n_procs;
        int last_partition_size = partition_size + n % n_procs;
        my_partition_size = rank == n_procs-1 ? last_partition_size:partition_size; //max_size
        auto *buffer = new u_char[n*d];

        cout<<"my size:  "<<my_partition_size<<endl;
        if(rank == ROOT) {
            sendcounts = new int[n_procs];
            displs = new int[n_procs];

            // Compute the send counts and displacements for each process
            for (int i = 0; i < n_procs; i++) {
                sendcounts[i] = (i == n_procs - 1) ? last_partition_size * d : partition_size * d;
            }
            displs[0] = 0;
            for (int i = 1; i < n_procs; i++) {
                displs[i] = displs[i-1] + sendcounts[i-1];
            }

            int buffer_i = 0;
            for(int i=0 ; i<n ;i++) {
                for(int j=0; j<d; j++) {
                    buffer[buffer_i++] = elements[i][j];
                }
            }
        }

        my_elements = new Element[last_partition_size];
        MPI_Scatterv(buffer, sendcounts, displs, MPI_UNSIGNED_CHAR,
                     my_elements, last_partition_size * d, MPI_UNSIGNED_CHAR,
                     ROOT, MPI_COMM_WORLD);

        dist.resize(my_partition_size);

        if(rank == ROOT) {
            reseedClusters();
            // for (int j = 0; j < k; j++) {
            //     std::cout << "Cluster " << j << ": (";
            //     for (int jd = 0; jd < d; jd++) {
            //         std::cout << (int)clusters[j].centroid[jd];
            //         if (jd != d - 1) {
            //             std::cout << ", ";
            //         }
            //     }
            //     std::cout << ")" << std::endl;
            // }
        }
        bcastCentroids(rank); // send the initial values of centroids to everyone
        // for (int j = 0; j < k; j++) {
        //     std::cout << "Cluster " << j << ": (";
        //     for (int jd = 0; jd < d; jd++) {
        //         std::cout << (int)clusters[j].centroid[jd];
        //         if (jd != d-1) {
        //             std::cout << ", ";
        //         }
        //     }
        //     std::cout << ")" << std::endl;
        // }
        Clusters prior = clusters; // save old clusters with the centroids and the elements please
        prior[0].centroid[0]++;  // just to make it different the first time
        int generation = 0;

        while(generation++ < MAX_FIT_STEPS && prior != clusters) {
            V(cout << rank << "working on generation" << generation << endl;)
            cout << rank << "working on generation" << generation << endl;
            updateDistances();  //calc distances for my partition [0..m]
            prior = clusters; // prepare check for convergence
            updateClusters(); // calc k clusters from my partition only
            mergeClusters(rank); // reduce all the processes' clusters
            bcastCentroids(rank);  // everyone needs to know the reduced centroids calculated by ROOT
        }
        delete[] my_elements;
        delete[] sendcounts;
        delete[] displs;
    }

    virtual void bcastCentroids(int rank) {
        //  V(cout<<" "<<rank<<" bcastCentroids"<<endl;)
        int count = k * d;
        auto *buffer = new u_char[count];
        if(rank == ROOT) {
            int i = 0;
            for(int j = 0; j<k; j++) {
                for(int jd = 0; jd<d; jd++) {
                    buffer[i++] = clusters[j].centroid[jd];
                }
            }
            //  V(cout<<" "<<rank<<" sending centroids: "<<clusters;)
        }

        MPI_Bcast(buffer, count, MPI_UNSIGNED_CHAR, ROOT, MPI_COMM_WORLD);

        if (rank!= ROOT) {
            int i = 0;
            for (int j = 0; j < k; j++) {
                for (int jd = 0; jd < d; jd++) {
                    clusters[j].centroid[jd] = buffer[i++];
                }
            }
        }
        delete[] buffer;
    }

    /**
     * The algorithm constructs k clusters and attempts to populate them with like neighbors.
     * This inner class, Cluster, holds each cluster's centroid (mean) and the index of the objects
     * belonging to this cluster.
     */
    struct Cluster {
        Element centroid;  // the current center (mean) of the elements in the cluster
        std::vector<int> elements;

        // equality is just the centroids, regarless of elements
        friend bool operator==(const Cluster& left, const Cluster& right) {
            return left.centroid == right.centroid;  // equality means the same centroid, regardless of elements
        }
    };

protected:
    const Element *elements = nullptr;       // set of elements to classify into k categories (supplied to latest call to fit())
    Element *my_elements = nullptr;    // set of elements each process is responsible for
    int n = 0;                               // number of elements in this->elements
    Clusters clusters;                       // k clusters resulting from latest call to fit()
    std::vector<std::array<double,k>> dist;  // dist[i][j] is the distance from elements[i] to clusters[j].centroid
    int my_partition_size = 0; // size of partition
    int n_procs = 0;
    int partition_size = 0;
    const int ROOT = 0;

    /**
     * Get the initial cluster centroids.
     * Default implementation here is to just pick k elements at random from the element
     * set
     * @return list of clusters made by using k random elements as the initial centroids
     */
    virtual void reseedClusters() {
        std::vector<int> seeds;
        std::vector<int> candidates(n);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto random = std::mt19937{std::random_device{}()};
        // Note that we need C++20 for std::sample
        std::sample(candidates.begin(), candidates.end(), back_inserter(seeds), k, random);
        for (int i = 0; i < k; i++) {
            clusters[i].centroid = elements[seeds[i]];
            clusters[i].elements.clear();
        }
    }

    /**
     * Calculate the distance from each element to each centroid.
     * Place into this->dist which is a k-vector of distances from each element to the kth centroid.
     */
    virtual void updateDistances() {
        for (int i = 0; i < my_partition_size; i++) {
            V(cout<<"distances for "<<i<<"(";for(int x=0;x<d;x++)printf("%02x",elements[i][x]);)
            for (int j = 0; j < k; j++) {
                dist[i][j] = distance(clusters[j].centroid, my_elements[i]);
                V(cout<<" " << dist[i][j];)
            }
            V(cout<<endl;)
        }
    }

    /**
     * Recalculate the current clusters based on the new distances shown in this->dist.
     */
    virtual void updateClusters() {
        // reinitialize all the clusters
        for (int j = 0; j < k; j++) {
            clusters[j].centroid = Element{};
            clusters[j].elements.clear();
        }
        // for each element, put it in its closest cluster (updating the cluster's centroid as we go)
        for (int i = 0; i < my_partition_size; i++) {
            int min = 0;
            for (int j = 1; j < k; j++)
                if (dist[i][j] < dist[i][min])  // here I am checking the minimum distance between each element and each centroid
                    min = j;
            accum(clusters[min].centroid, clusters[min].elements.size(), my_elements[i], 1);
            clusters[min].elements.push_back(i);
        }
    }

    void mergeClusters(int rank) {
        // construct my buffer
        int my_buffer_size = k+my_partition_size;
        auto *mybuffer = new int[my_buffer_size];
        int iterator = 0;
        cout<<"RANK "<<rank<<" buffer size"<<my_buffer_size <<endl;
        for (int i = 0; i < k ;i++) {
            int centroid_elements = clusters[i].elements.size();
            cout<<"rank "<<rank<<" elements size"<<centroid_elements<<endl;
            mybuffer[iterator++] = centroid_elements;

            for(int j=0; j<centroid_elements; j++) {
                cout<<rank<<" elindex " << clusters[i].elements[j] + (partition_size * rank)<<endl;
                mybuffer[iterator++] = clusters[i].elements[j] + (partition_size * rank); //multiply by rank to find the real index in elements
            }
        }

        for(int i=0;i<my_buffer_size;i++){
            cout<<mybuffer[i]<<" ";
        }
        cout<<endl;
        int *recvcounts = nullptr;
        int *displs = nullptr;
        int total_size = 0;
        total_size = k * n_procs + n;
        auto *buffer = new int[total_size];
        // Combine all of the clusters
        if (rank == ROOT) {
            // Gather all the elements into the root process
            recvcounts = new int[n_procs];
            displs = new int[n_procs];
            // int buffer_size = k * 2 * partition_size;
            for (int i = 0; i < n_procs; i++) {
                recvcounts[i] = partition_size + k;
                displs[i] = i * recvcounts[i];
            }
            recvcounts[n_procs - 1] += (n % n_procs);
        }

        MPI_Gatherv(mybuffer, my_buffer_size, MPI_INT,
                    buffer, recvcounts, displs, MPI_INT,
                    ROOT, MPI_COMM_WORLD);

        iterator = 0;
        if(rank == ROOT) {
            for (auto& cluster : clusters) {
                cluster.elements.clear();
            }
            for (int p = 0; p < n_procs; p++) {
                for (int c = 0; c < k; c++) {
                    int num_elements = buffer[iterator++];

                    for (int j = 0; j < num_elements; j++) {
                        int element_index = buffer[iterator++];
                        accum(clusters[c].centroid, clusters[c].elements.size(), elements[element_index], 1);
                        clusters[c].elements.push_back(element_index);
                    }
                }
            }
        }


        // Recalculate the centroid for the merged elements
//                for (int j = 0; j < d; j++) {
//                    double sum = 0;
//                    for (int elem : clusters[i].elements) {
//                        sum += (double)elements[elem][j];
//                    }
//                    clusters[i].centroid[j] = (u_char)(sum / clusters[i].elements.size());
//                }

    }


    /**
     * Method to update a centroid with an additional element(s)
     * @param centroid   accumulating mean of the elements in a cluster so far
     * @param centroid_n number of elements in the cluster so far
     * @param addend     another element(s) to be added; if multiple, addend is their mean
     * @param addend_n   number of addends represented in the addend argument
     */
    //     accum(clusters[min].centroid, clusters[min].elements.size(), elements[i], 1);
    //     clusters[min].elements.push_back(i);
    virtual void accum(Element& centroid, int centroid_n, const Element& addend, int addend_n) const {
        int new_n = centroid_n + addend_n;
        for (int i = 0; i < d; i++) {
            double new_total = (double) centroid[i] * centroid_n + (double) addend[i] * addend_n;
            centroid[i] = (u_char)(new_total / new_n);
        }
    }

    /**
     * Subclass-supplied method to calculate the distance between two elements
     * @param a one element
     * @param b another element
     * @return distance from a to b (or more abstract metric); distance(a,b) >= 0.0 always
     */
    virtual double distance(const Element& a, const Element& b) const = 0;

};
