import java.util.concurrent.CyclicBarrier;

public class ParallelSorter {
    protected Thread[] workers;
    protected int piece;
    protected CyclicBarrier barrier;
    private static final int N_THREADS = 6;
    private static final int N = 1 << 22;
    private static final int TIME_ALLOWED = 10;  // seconds

    public ParallelSorter() {
        barrier = new CyclicBarrier(N_THREADS);
        piece = N/N_THREADS;
        workers = new Thread[N_THREADS];
    }

    public static void main(String[] args) throws InterruptedException {
        ParallelSorter parallelSorter = new ParallelSorter();
        long start = System.currentTimeMillis();
        int work = 0;
        while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {
        double[] arr = RandomArrayGenerator.getArray(N);
        for(int i = 0 ; i<N_THREADS ; i++) {
            int currStart = i *  parallelSorter.piece;
            int currEnd =  i != N_THREADS - 1 ? currStart + parallelSorter.piece : N;
            parallelSorter.workers[i] = new Thread(new BitonicLoops(currStart, currEnd, parallelSorter.barrier, arr, N));
            parallelSorter.workers[i].start();
        }

        for(int i = 0 ; i<N_THREADS ; i++) {
            parallelSorter.workers[i].join();
        }

        if(RandomArrayGenerator.isSorted(arr)) work++;
        else if (!RandomArrayGenerator.isSorted(arr) || N != arr.length)
            System.out.println("failed");

        }

        System.out.println("sorted " + work + " arrays (each: " + N + " doubles) in "
                + TIME_ALLOWED + " seconds");
    }
}
