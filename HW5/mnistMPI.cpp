/**
 * @file mnistMPI.h - implementation of k-means clustering algorithm using MPI on
 * MNIST Dataset
 * @author Noha Nomier
 * @see "Seattle University, CPSC5600, Winter 2023"
 *
 * To run this file, you can run make all or make run_mnist
 * This file first reads dataset from 2 file that exist under data/ in this directory
 * data/t10k-images-idx3-ubyte and data/t10k-labels-idx1-ubyte
 * It also depends on MnistKMeansMPI.h which contains the fit() and distance() implementation
 * this reads 1000 images but the size which runs on the algorithm
 * is passed by a constant NUM_IMAGES which you change to a value up to 1000,
 * I chose a smaller number for better visualization
 * The result is printed to the console as well as an HTML file "Mnist_mpi.html"
 * Analysis is also done by calculating the most seen number in each centroid and dividing
 * the count by the total number of elements in a certain centroid
 */

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <iterator>
#include "mpi.h"
#include "MnistKMeansMPI.h"

using namespace std;

const int K = 10;
const int NUM_IMAGES = 500; // Number of images to be tested, should be less than 1000

void to_html(const MnistKMeansMPI<K, 784>::Clusters& clusters, const vector<array<u_char, 784>>& images, const vector<pair<int, double>> analysis, const string& filename);

vector<pair<int, double>> count_labels(const MnistKMeansMPI<K, 784>::Clusters& clusters, const vector<int>& labels);

/**
 * A helper function that takes an unsigned 32-bit integer x and swaps its byte order
 * (from little-endian to big-endian or vice versa) and returns the swapped value.
 * Used to convert the byte order of integers read from binary files.
**/
uint32_t swap(uint32_t x) {
    return (x >> 24) |
           ((x << 8) & 0x00FF0000) |
           ((x >> 8) & 0x0000FF00) |
           (x << 24);
}

/**
 * A function that reads a binary file of MNIST dataset images and returns a vector of vectors of
  unsigned 8-bit integers that represents the images.
 * @param path string that contains the path of the binary file.
 * @throws a runtime_error if it fails to open the file or if the magic number in the file is invalid.
 * @returns the vector of vectors of unsigned 8-bit integers that represents the images.
**/
vector<vector<unsigned char>> read_mnist_images(const string& path) {
    ifstream file(path, ios::binary);

    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + path);
    }

    uint32_t magic_number, num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap(magic_number); // Convert to big-endian if necessary
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = swap(num_images);
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    num_rows = swap(num_rows);
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    num_cols = swap(num_cols);

    if (magic_number != 2051) {
        throw runtime_error("Invalid MNIST image file: " + path);
    }

    int image_size = num_rows*num_cols;
    vector<vector<unsigned char>> images(num_images, vector<unsigned char>(image_size));

    for (uint32_t i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), image_size);
    }

    return images;
}

/**
 * A function that reads a binary file of MNIST dataset labels and returns a vector of integers that represents the labels.
 * @param path string that contains the path of the binary file.
 * @throws a runtime_error if it fails to open the file or if the magic number in the file is invalid.
 * @returns the vector of integers that represents the labels.
**/
vector<int> read_mnist_labels(const string& path) {
    ifstream file(path, ios::binary);

    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + path);
    }

    uint32_t magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap(magic_number); // Convert to big-endian if necessary
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = swap(num_labels);

    if (magic_number != 2049) {
        throw runtime_error("Invalid MNIST label file: " + path);
    }

    vector<int> labels(num_labels);

    for (uint32_t i = 0; i < num_labels; ++i) {
        labels[i] = file.get();
    }

    return labels;
}

int main() {
    MPI_Init(nullptr, nullptr);
    // Example usage:

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    vector<int> labels{};
    vector<array<u_char, 784>> images_converted;
    // Set up k-means
    MnistKMeansMPI<K, 784> kMeans;
    if (rank == 0) {
        auto images = read_mnist_images("data/t10k-images-idx3-ubyte");
        labels = read_mnist_labels("data/t10k-labels-idx1-ubyte");
        cout << "Loaded " << images.size() << " images and " << labels.size() << " labels." << endl;

        int n_images = images.size();
        int num_labels = labels.size();

        for (int i = 0; i < NUM_IMAGES; ++i) {
            images_converted.push_back({});
            copy(images[i].begin(), images[i].end(), images_converted[i].begin());
        }

        if (n_images != num_labels) {
            throw runtime_error("INVALID DATA: Number of images is not equal to the number of labels");
        }

        kMeans.fit(images_converted, NUM_IMAGES);

    } else {
        kMeans.fitWork(rank);
        MPI_Finalize();
        return 0;
    }

    // get the result
    MnistKMeansMPI<K, 784>::Clusters clusters = kMeans.getClusters();
    vector<pair<int, double>> analysis = count_labels(clusters, labels);
    to_html(clusters, images_converted, analysis, "Mnist_mpi.html");

    // Report the result to console
    int i = 0;
    for (int c = 0; c < K; c++) {
        const auto& cluster = clusters[c];
        cout << endl << endl << "cluster #" << ++i  << ": Most Common Label is: "
             << analysis[c].first << " Accuracy: " << analysis[c].second << "%" << endl;

        int col_width = 5;
        int row_count = 0;

        for (int j: cluster.elements) {
            if (row_count % 10 == 0) {
                cout << endl;
            }

            cout << setw(col_width) << static_cast<int>(labels[j]) << " ";

            row_count++;
        }
    }

    MPI_Finalize();
    return 0;
}

/**
  * A function that prints the KMeans clusters result to html
  * each image is treated as a 28x28 table with varying colors to visualize the image
  * The result is stored in @param filename
**/
void to_html(const MnistKMeansMPI<K, 784>::Clusters& clusters, const vector<array<u_char, 784>>& images, const vector<pair<int, double>> analysis, const string& filename) {
    cout << "HTML file generated: " << filename << endl;
    ofstream f(filename);
    f << "<html><body><table border=1>" << endl;

    int i = 0;
    for (int c = 0; c < K; c++) {
        const auto& cluster = clusters[c];
        // Print centroid in a separate row with a label
        array<u_char, 784> centroid = cluster.centroid;
        f << "<tr><td colspan=29><b>Centroid #" << i << "</b></td><td><b>Most Frequent Number: " <<analysis[c].first
          <<"</b></td><td><b> Accuracy: "<< analysis[c].second<<"</b></td></tr>" << endl;
        f << "<tr><td><table>";
        for (int j = 0; j < 28; ++j) {
            f << "<tr>";
            for (int k = 0; k < 28; ++k) {
                int pixel_value = static_cast<int>(centroid[j * 28 + k]);
                f << "<td style=\"background-color:rgb(" << pixel_value << "," << pixel_value << "," << pixel_value << ");\"></td>";
            }
            f << "</tr>";
        }
        f << "</table></td></tr>";

        // Print image values in separate cells
        f << "<tr>";
        int cell_count = 0;
        for (const auto& image_id : cluster.elements) {
            if (cell_count > 10) {
                f << "</tr>" << endl << "<tr>";
                cell_count = 0;
            }
            array<u_char, 784> image = images[image_id];
            f << "<td><table>";
            for (int j = 0; j < 28; ++j) {
                f << "<tr>";
                for (int k = 0; k < 28; ++k) {
                    int pixel_value = static_cast<int>(image[j * 28 + k]);
                    f << "<td style=\"background-color:rgb(" << pixel_value << "," << pixel_value << "," << pixel_value << ");\"></td>";
                }
                f << "</tr>";
            }
            f << "</table></td>";
            cell_count++;
        }
        f << "</tr>" << endl;
        f << "<tr><td colspan=29>&nbsp;</td></tr>" << endl;

        ++i;
    }

    f << "</table></body></html>" << endl;
}

/**
  * This function is used for analysis, It counts the most occured number (label) in each cluster
  * and calculate the accuracy by dividing the count of that number by the total number of elements
  * contained in that cluster
  * @returns vector<pair<int, double>> of size K where each pair consists of the most occured number
  * in that cluster and its accuracy
**/
vector<pair<int, double>> count_labels(const MnistKMeansMPI<K, 784>::Clusters& clusters, const vector<int>& labels) {
    vector<pair<int, double>> results;
    for (const auto& cluster : clusters) {
        vector<int> label_counts(10, 0);
        for (int j : cluster.elements) {
            int label = labels[j];
            label_counts[label]++;
        }
        int max_count = 0;
        int max_label = -1;
        for (int i = 0; i < 10; ++i) {
            if (label_counts[i] > max_count) {
                max_count = label_counts[i];
                max_label = i;
            }
        }
        double accuracy = static_cast<double>(max_count) / cluster.elements.size() * 100.0;
        results.push_back(make_pair(max_label, accuracy));
    }
    return results;
}
