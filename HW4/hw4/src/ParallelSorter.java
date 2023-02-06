/**
 * 
 */

import java.util.*;
import java.util.concurrent.CyclicBarrier;

public class ParallelSorter {
    protected Thread[] workers;
    protected int piece;
    protected CyclicBarrier[] barriers;
    protected Map<Integer, Integer> threadBarrierMap;
    private static final Set<Integer> ALLOWED_GRANUL = new HashSet<>(Arrays.asList(1, 2, 3));
    private static final int N_THREADS = 8;
    private static final int GRANUL = 1; // can be 1, 2, 3
    private static final int N = 1 << 22;
    private static final int TIME_ALLOWED = 10;  // seconds

    public ParallelSorter() {
        barriers = new CyclicBarrier[(1 << GRANUL) - 1];
        threadBarrierMap = new HashMap<>();
        initBarriers();
        piece = N / N_THREADS;
        workers = new Thread[N_THREADS];
    }

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
//        for (CyclicBarrier b : barriers) {
//            System.out.println(b.getParties());
//        }

        fillThreadsMap();
    }

    private void fillThreadsMap() {
        int barrierPiece = N_THREADS / GRANUL;
        for (int i = 0; i < N_THREADS; i++) {
            int firstPart = ((i / barrierPiece) + GRANUL);
            if(GRANUL==2) firstPart--;

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
                parallelSorter.workers[i] = new Thread(new BitonicLoops(currStart, currEnd,parallelSorter.threadBarrierMap, parallelSorter.barriers, arr, N, i));
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
