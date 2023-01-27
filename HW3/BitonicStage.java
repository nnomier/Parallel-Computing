/**
 * A class that performs the bitonic sort algorithm on a given input.
 * The input is divided into two halves both sorted ascendingly
 * The sorted halves are adjusted to become a valid bitonic sequence
 * then merged into an output sorted array
 *
 * @author Noha Nomier
 */

import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

public class BitonicStage implements Runnable {
    private String threadName;
    private SynchronousQueue<double[]> firstHalfQueue, secondHalfQueue, output;
    private static final int timeout = 10;  // in seconds

    public BitonicStage(SynchronousQueue<double[]> firstHalfQueue, SynchronousQueue<double[]> secondHalfQueue, SynchronousQueue<double[]> outputQueue, String threadName) {
        this.threadName = threadName;
        this.firstHalfQueue = firstHalfQueue;
        this.secondHalfQueue = secondHalfQueue;
        this.output = outputQueue;
    }

    public BitonicStage() {

    }

    enum Direction {
        ASCENDING(), DESCENDING();
    }

    /**
     * retrieves elements from the firstHalfQueue and secondHalfQueue continuously,
     * calls process() to perform BitonicSort
     * and then adds the result to the output queue.
     */
    @Override
    public void run() {
        double[] halfOne = new double[1];
        double[] halfTwo = new double[1];
        double[] bitonic;
        while (halfOne != null && halfTwo != null) {
            try {
                halfOne = firstHalfQueue.poll(timeout * 1000, TimeUnit.MILLISECONDS);
                halfTwo = secondHalfQueue.poll(timeout * 1000, TimeUnit.MILLISECONDS);

                if (halfOne != null && halfTwo != null) {
                    bitonic = process(halfOne, halfTwo);
                    output.offer(bitonic, timeout * 1000, TimeUnit.MILLISECONDS);
                } else {
                    System.out.println(getClass().getName() + " " + threadName + " got null array");
                }
            } catch (InterruptedException e) {
                return;
            }
        }
    }

    /**
     * process the input by merging and sorting it using BitonicSort
     * @param firstHalf the first half of the input
     * @param secondHalf the second half of the input
     * @return the sorted and merged input
     */
    public double[] process(double[] firstHalf, double[] secondHalf) {
        int totalLength = firstHalf.length + secondHalf.length;
        double[] bitonic = new double[totalLength];

        // copy the first half of the first input array
        for (int i = 0; i < firstHalf.length; i++)
            bitonic[i] = firstHalf[i];

        // reverse the second input array and copy it to the second half of the bitonic array
        for (int i = 0; i < secondHalf.length; i++)
            bitonic[i + firstHalf.length] = secondHalf[secondHalf.length - i - 1];


        // sort the bitonic array in ascending order
        bitonicSort(bitonic, 0, totalLength, Direction.ASCENDING);

        return bitonic;
    }


    /**
     * Recursive method for sorting the input array in bitonic order.
     * @param arr the input array
     * @param low the starting index of the portion of the array to be sorted
     * @param count the number of elements in the portion of the array to be sorted
     * @param dir the direction of the sort (ascending or descending)
     */
    private void bitonicSort(double[] arr, int low, int count, Direction dir) {
        if (count > 1) {
            int k = count / 2;
            bitonicMerge(arr, low, count, dir);
            bitonicSort(arr, low, k, dir);
            bitonicSort(arr, low + k, k, dir);
        }
    }

    /**
     * Merges the two halves of the input array in bitonic order.
     * @param arr the input array
     * @param low the starting index of the portion of the array to be merged
     * @param count the number of elements in the portion of the array to be merged
     * @param dir the direction of the sort (ascending or descending)
     */
    private void bitonicMerge(double[] arr, int low, int count, Direction dir) {
        for (int i = 0; i < count / 2; i++) {
            compare(arr, low + i, low + (count / 2) + i, dir);
        }

    }

    /**
     * Compares two elements in the input array and swaps them if necessary.
     * @param arr the input array
     * @param i the index of the first element to be compared
     * @param j the index of the second element to be compared
     * @param dir the direction of the sort (ascending or descending)
     */
    private void compare(double[] arr, int i, int j, Direction dir) {
        if ((dir == Direction.ASCENDING && arr[i] > arr[j]) ||
                (dir == Direction.DESCENDING && arr[i] < arr[j])) {
            //swap
            double temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
}
