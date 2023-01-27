/**
 * The BitonicPipeline class creates a pipeline of worker threads that sort an
 * array of double values using the bitonic sort algorithm.
 * The pipeline is divided into 3 stages:
 * RandomArrayGenerator threads generate random arrays of double values
 * StageOne threads sort the arrays generated in stage 1
 * BitonicStage threads merge and sort the arrays from stage 2 using BitonicSorting
 * The class also has a runPipeline() method that initializes the threads, runs the pipeline
 * and evaluates the sorting process.
 * @author Noha Nomier
 */

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

public class BitonicPipeline {
    private SynchronousQueue<double[]>[] dataQueues;
    private Thread[] workerThreads;
    private static final int N = 1 << 22;  // size of the final sorted array (power of two)
    private static final int TIME_ALLOWED = 10;  // seconds
    private static final int ARRAY_GENERATOR_THREADS = 4;
    private static final int STAGE_ONE_THREADS = ARRAY_GENERATOR_THREADS;
    private static final int BITONIC_STAGE_THREADS = 2;
    private static final int TOTAL_THREADS = ARRAY_GENERATOR_THREADS + STAGE_ONE_THREADS + BITONIC_STAGE_THREADS + 1; // add 1 for last stage

    public BitonicPipeline() {
        dataQueues = new SynchronousQueue[TOTAL_THREADS];
        workerThreads = new Thread[TOTAL_THREADS];
    }

    /**
     * Runs the pipeline and performs the sorting.
     *
     * @throws InterruptedException
     */
    public void runPipeline() throws InterruptedException {
        initializeThreads();
        evaluateSorting();
        threadCleanup();
    }

    /**
     * Initializes the worker threads for each pipeline stage and
     * creates SynchronousQueue objects to pass data between stages
     */
    private void initializeThreads() {
        // Create SynchronousQueue objects and worker threads for each pipeline stage

        for (int i = 0; i < TOTAL_THREADS; i++) {
            dataQueues[i] = new SynchronousQueue<>();
        }

        // Initialize RandomArrayGenerator Threads to generate random array sections
        for (int i = 0; i < ARRAY_GENERATOR_THREADS; i++) {
            workerThreads[i] = new Thread(new RandomArrayGenerator(N / 4, dataQueues[i]));
        }

        // Initialize StageOne Threads to sort the generated arrays
        for (int i = 0; i < STAGE_ONE_THREADS; i++) {
            int index = i + ARRAY_GENERATOR_THREADS;
            workerThreads[index] = new Thread(new StageOne(dataQueues[i], dataQueues[index],
                    "THREAD [" + index + "]"));
        }

        // Initialize BitonicStage Threads
        for (int i = 0; i < BITONIC_STAGE_THREADS; i++) {
            int stageOneIndex = (i * 2) + ARRAY_GENERATOR_THREADS;
            int index = i + ARRAY_GENERATOR_THREADS + STAGE_ONE_THREADS;
            workerThreads[index] = new Thread(new BitonicStage(dataQueues[stageOneIndex],
                    dataQueues[stageOneIndex + 1], dataQueues[index],
                    "THREAD[" + index + "]"));
        }

        // Initialize last BitonicStage Thread to sort the final sequence
        int finalStageIndex = TOTAL_THREADS - 1;
        workerThreads[finalStageIndex] = new Thread(new BitonicStage(dataQueues[finalStageIndex - 2],
                dataQueues[finalStageIndex - 1], dataQueues[finalStageIndex],
                "THREAD[" + finalStageIndex + "]"));
    }

    /**
     * Evaluates the sorting by checking the results obtained from the final queue
     * @throws InterruptedException
     */
    private void evaluateSorting() throws InterruptedException {
        long start = System.currentTimeMillis();

        // Start threads
        for (int i = 0; i < TOTAL_THREADS; i++) {
            workerThreads[i].start();
        }

        int numberOfSortedArrays = 0;
        boolean isSuccessful = true;
        while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {
            double[] res = dataQueues[TOTAL_THREADS-1].poll(10 * 1000, TimeUnit.MILLISECONDS);
            if (res != null) {
                if (!RandomArrayGenerator.isSorted(res) || N != res.length) {
                    System.out.println("Sorting was not successfull");
                    isSuccessful = false;
                    break;
                }
                numberOfSortedArrays++;
            }
        }
        if (isSuccessful)
            System.out.println("sorted " + numberOfSortedArrays + " arrays (each: " + N + " doubles) in "
                    + TIME_ALLOWED + " seconds");
    }

    /**
     * Interrupts all threads that are still running
     */
    private void threadCleanup() {
        for (Thread thread : this.workerThreads) {
            if (thread.isAlive()) {
                thread.interrupt();
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        BitonicPipeline pipeline = new BitonicPipeline();
        pipeline.runPipeline();
    }
}
