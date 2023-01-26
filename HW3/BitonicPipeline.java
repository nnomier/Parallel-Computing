import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

public class BitonicPipeline {
    public static final int N = 1 << 22;  // size of the final sorted array (power of two)
    public static final int TIME_ALLOWED = 10;  // seconds
    public static final int ARRAY_GENERATOR_THREADS = 4;
    public static final int STAGE_ONE_THREADS = ARRAY_GENERATOR_THREADS;
    public static final int BITONIC_STAGE_THREADS = 2;

    public static final int TOTAL_THREADS = ARRAY_GENERATOR_THREADS + STAGE_ONE_THREADS + BITONIC_STAGE_THREADS + 1; // add 1 for last stage


    public static void main(String[] args) throws InterruptedException {
        // Create SynchronousQueue objects for each pipeline stage
            long start = System.currentTimeMillis();

            List<SynchronousQueue<double[]>> dataQueues = new ArrayList<>();

            Thread[] workerThreads = new Thread[TOTAL_THREADS];
            for (int i = 0; i < TOTAL_THREADS; i++) {
                dataQueues.add(new SynchronousQueue<>());
            }
            for (int i = 0; i < ARRAY_GENERATOR_THREADS; i++) {
                workerThreads[i] = new Thread(new RandomArrayGenerator(N / 4, dataQueues.get(i)));
            }


            for (int i = 0; i < STAGE_ONE_THREADS; i++) {
                int index = i + ARRAY_GENERATOR_THREADS;
                workerThreads[index] = new Thread(new StageOne(dataQueues.get(i), dataQueues.get(index), "THREAD ["+index+"]"));
            }

            for (int i = 0; i < BITONIC_STAGE_THREADS; i++) {
                int stageOneIndex = (i*2) + ARRAY_GENERATOR_THREADS;
                int index = i + ARRAY_GENERATOR_THREADS + STAGE_ONE_THREADS;
//                System.out.println("index is" + index);
                workerThreads[index] = new Thread(new BitonicStage(dataQueues.get(stageOneIndex), dataQueues.get(stageOneIndex+1), dataQueues.get(index), "THREAD[" + index + "]"));
            }

            workerThreads[TOTAL_THREADS-1] = new Thread(new BitonicStage(dataQueues.get(TOTAL_THREADS-3), dataQueues.get(TOTAL_THREADS-2), dataQueues.get(TOTAL_THREADS-1), "THREAD[" + (TOTAL_THREADS - 1) + "]"));
            for (int i = 0; i < TOTAL_THREADS; i++) {
                workerThreads[i].start();
            }


//        double[] data = new double[2];
        int count = 0;
        while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {
            double [] res = dataQueues.get(10).poll(10 * 1000, TimeUnit.MILLISECONDS);
            if( res != null) {
                if (!RandomArrayGenerator.isSorted(res) || N != res.length)
                    System.out.println("failed");
                count++;
                System.out.println("elarray" + res[0]);
            }


        }

        System.out.println("total= " + count);
//        for (int i = 0; i < 3; i++) {
//
//
//        }
//        System.out.println(Arrays.toString(data));

//        for (int i = 0; i < ARRAY_GENERATOR_THREADS; i++) {
//                workerThreads[i].join();
//            }


//        for (int i = 0; i < ARRAY_GENERATOR_THREADS; i++) {
//            workerThreads[i].interrupt();
//        }
            // Start the main thread
//        long start = System.currentTimeMillis();
//        int work = 0;
//        while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {
//            // Wait for the output from the final pipeline thread
//            double[] sortedArray = outputQueue6.take();
//
//            // Check if the array is sorted and discard it if not
//            if (!RandomArrayGenerator.isSorted(sortedArray) || N != sortedArray.length) {
//                System.out.println("failed");
//            } else {
//                work++;
//            }
//        }
//        System.out.println("sorted " + work + " arrays (each: " + N + " doubles) in " + TIME_ALLOWED + " seconds");
//
//        // Shut down the threads
        }

    }

