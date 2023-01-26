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
        ASCENDING(0), DESCENDING(1);

        private final int value;

        private Direction(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    @Override
    public void run() {
        double[] halfOne = new double[1];
        double[] halfTwo = new double[1];
        double[] bitonic;
        while (halfOne != null && halfTwo != null) {
            try {
                halfOne = firstHalfQueue.poll(timeout * 1000, TimeUnit.MILLISECONDS);
                halfTwo = secondHalfQueue.poll(timeout * 1000, TimeUnit.MILLISECONDS);

                if (halfOne != null && halfTwo !=null) {
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
    // merge two sorted arrays (a and b) into a bitonic sequence
    public double[] process(double[] firstHalf, double[] secondHalf) {
        int totalLength = firstHalf.length + secondHalf.length;
        double[] bitonic = new double[totalLength];

        // copy the first half of the first input array
        for (int i = 0; i < firstHalf.length; i++)
            bitonic[i] = firstHalf[i];

        // reverse the second input array and copy it to the second half of the bitonic array
        for (int i = 0; i < secondHalf.length; i++)
            bitonic[i + firstHalf.length] = secondHalf[secondHalf.length - i -1];


        // sort the bitonic array in ascending order
        bitonicSort(bitonic, 0, totalLength, Direction.ASCENDING);

        return bitonic;
    }


    // recursive bitonic sort function
    private void bitonicSort(double[] arr, int low, int count, Direction dir) {
        if (count > 1) {
            int k = count / 2;
            bitonicMerge(arr, low, count, dir);
            bitonicSort(arr, low, k, dir);
            bitonicSort(arr, low + k, k, dir);
        }
    }

    private void bitonicMerge(double[] arr, int low, int count, Direction dir) {
        for (int i = 0; i < count / 2; i++) {
            compare(arr, low + i, low + (count / 2) + i, dir);
        }

    }

    // compare two elements and swap them if necessary
    private void compare(double[] arr, int i, int j, Direction dir) {
        if ((dir == Direction.ASCENDING && arr[i] > arr[j]) || (dir == Direction.DESCENDING && arr[i] < arr[j])) {
            //swap
            double temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;

        }
    }
}
