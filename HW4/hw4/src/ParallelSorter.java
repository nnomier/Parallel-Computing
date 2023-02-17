/**
 * The ParallelSorter class provides an implementation of sorting arrays in parallel using Bitonic sort algorithm.
 * The class uses a set of worker threads to sort the input arrays and keeps track of the number of arrays sorted.
 * It uses a CyclicBarrier for synchronization between threads during sorting.
 *
 * Granularity values can be 1, 2, 3:
 * 1 is maximum where it will have barrier between each column (j)
 * 2 means each column can be divided into two separate barrier groups (when possible)
 * 3 means each column can be divided into four separate barrier groups (when possible)
 *
 * Benchmarking Results with N = 1 << 22 and TIME_ALLOWED = 10 SECONDS:
 * P                    Granularity             Sorted_Arrays
 * 1                    1                       5
 *
 * 2                    1                       9
 *
 * 4                    1                       15
 * 2                                            15
 *
 * 8                    1                       18
 * 2                                            17
 * 3                                            18
 */

import java.util.*;
import java.util.concurrent.CyclicBarrier;

public class ParallelSorter {
    protected Thread[] workers;
    protected int piece;
    protected CyclicBarrier[] barriers;
    protected Map<Integer, Integer> threadBarrierMap; //Map of thread indices and the barrier they need to wait on
    private static final Set<Integer> ALLOWED_GRANUL = new HashSet<>(Arrays.asList(1, 2, 3));
    private static final int N_THREADS = 8;
    private static final int GRANUL = 3; // can be 1, 2, 3
    private static final int N = 1 << 22;
    private static final int TIME_ALLOWED = 10;  // seconds

    /**
     * Constructs a new instance of the ParallelSorter class.
     * Initializes the barriers and worker threads.
     */
    public ParallelSorter() {
        barriers = new CyclicBarrier[(1 << GRANUL) - 1];
        threadBarrierMap = new HashMap<>();
        initBarriers();
        piece = N / N_THREADS;
        workers = new Thread[N_THREADS];
    }

    /**
     * Initializes the barriers for synchronization between worker threads.
     * Example when N_THREADS = 8  and GRANUL = 3
     * barriers[0] = 8 , barriers[1,2] = 4, barriers[3,4,5,6] = 2
     * Throws an IllegalArgumentException if the GRANUL value is not in ALLOWED_GRANUL.
     */
    private void initBarriers() {
        if (!ALLOWED_GRANUL.contains(GRANUL)) {
            throw new IllegalArgumentException("Error, The only allowed GRANULARITY values are " +
                    Arrays.toString(ALLOWED_GRANUL.toArray()));
        }
        int nextPowerOfTwo = 1;
        int barrierIndex = 0;
        int numberOfThreadsPerBarrier = N_THREADS;

        while (barrierIndex < barriers.length) {
            barriers[barrierIndex] = new CyclicBarrier(numberOfThreadsPerBarrier);
            int checkValue = (1 << nextPowerOfTwo) - 2;
            if (barrierIndex == checkValue && numberOfThreadsPerBarrier > 1) {
                numberOfThreadsPerBarrier /= 2;
                nextPowerOfTwo++;
            }
            barrierIndex++;
        }
        fillThreadsMap();
    }

    /**
     * Fills the threadBarrierMap with the thread index as the key and the barrier index as the value.
     */
    private void fillThreadsMap() {
        int barrierPiece = N_THREADS / GRANUL;
        for (int i = 0; i < N_THREADS; i++) {
            int firstPart = ((i / barrierPiece) + GRANUL);
            if (GRANUL == 2) firstPart--;

            int barrierIdx = firstPart % barriers.length;

            threadBarrierMap.put(i, barrierIdx);
        }
//        threadBarrierMap.forEach((key, value) -> System.out.println(key + " : " + value));
    }

    public static void main(String[] args) throws InterruptedException {
        ParallelSorter parallelSorter = new ParallelSorter();

        long start = System.currentTimeMillis();
        int work = 0;
        while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {
            double[] arr = RandomArrayGenerator.getArray(N);
            for (int i = 0; i < N_THREADS; i++) {
                int currStart = i * parallelSorter.piece;
                int currEnd = i != N_THREADS - 1 ? currStart + parallelSorter.piece : N;
                parallelSorter.workers[i] = new Thread(new BitonicLoops(currStart, currEnd, parallelSorter.threadBarrierMap, parallelSorter.barriers, arr, N, i));
                parallelSorter.workers[i].start();
            }

            for (int i = 0; i < N_THREADS; i++) {
                parallelSorter.workers[i].join();
            }

            if (RandomArrayGenerator.isSorted(arr)) work++;
            else if (!RandomArrayGenerator.isSorted(arr) || N != arr.length)
                System.out.println("failed");

        }

        System.out.println("sorted " + work + " arrays (each: " + N + " doubles) in "
                + TIME_ALLOWED + " seconds");
    }
}
