/**
 * BitonicLoops class represents a worker that sorts a portion of an array using bitonic sorting algorithm
 * and synchronized via barriers.
 *
 * @author Noha Nomier
 */

import java.util.Map;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class BitonicLoops implements Runnable {
    private int startWire;
    private int endWire;
    private int threadId;
    private int total_length;
    private double[] data;
    private CyclicBarrier[] barriers;
    private Map<Integer, Integer> threadBarrierMap;

    /**
     * Constructor for BitonicLoops class.
     * @param start the start index of the portion of the array to be sorted.
     * @param end the end index of the portion of the array to be sorted.
     * @param threadBarrierMap the map associating each thread to a barrier.
     * @param barriers the barriers to be used for synchronization.
     * @param data the data to be sorted.
     * @param size the total length of the data to be sorted.
     * @param threadId the ID of the thread.
     **/
    public BitonicLoops(int start, int end, Map<Integer, Integer> threadBarrierMap, CyclicBarrier[] barriers, double[] data, int size, int threadId) {
        this.startWire = start;
        this.endWire = end;
        this.barriers = barriers;
        this.data = data;
        this.total_length = size;
        this.threadBarrierMap = threadBarrierMap;
        this.threadId = threadId;
    }

    @Override
    public void run() {
        try {
            sort();
        } catch (BrokenBarrierException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Method that sorts the portion of the data using the Bitonic sorting algorithm.
     * @throws BrokenBarrierException if a barrier fails during the sort process.
     * @throws InterruptedException if the thread is interrupted during the sort process.
     */
    private void sort() throws BrokenBarrierException, InterruptedException {
        for (int k = 2; k <= total_length; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                checkBarriers(j);
                for (int i = startWire; i < endWire; i++) {
                    int ixj = i ^ j;
                    if (ixj > i) {
                        compare(i, k, ixj);
                    }
                }
            }
        }
    }

    /**
     * Method that checks the barriers for synchronization.
     * @param j the current comparison distance (yarn) to be used for barrier synchronization.
     * @throws BrokenBarrierException if a barrier fails during the sort process.
     * @throws InterruptedException if the thread is interrupted during the sort process.
     */
    private void checkBarriers(int j) throws BrokenBarrierException, InterruptedException {

        int granularity = (int) (Math.log(barriers.length + 1) / Math.log(2)); //barriers.length = (1<<granularity)-1

        if (j > 4 || barriers.length == 1) {
            barriers[0].await(); //biggest barrier in all cases
        } else if ((j <= 4 && granularity == 2) || (j < 4 && granularity == 3)) {
            barriers[threadBarrierMap.get(threadId)].await();
        } else if (j == 4 && granularity == 3) {
            if (threadId < 4) barriers[1].await();
            else if (threadId >= 4) {
                barriers[2].await();
            }

        }
    }

    /**
     * Compares elements of the array at indices i and ixj and swaps them if they are in the wrong order based on
     * the value of k.
     * @param i the index of the first element to be compared
     * @param k used to determine the order of the elements being compared
     * @param ixj the index of the second element to be compared
     */
    private void compare(int i, int k, int ixj) {
        if (((i & k) == 0 && data[i] > data[ixj]) || ((i & k) != 0 && data[i] < data[ixj])) {
            double temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}
