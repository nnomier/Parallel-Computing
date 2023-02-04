import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class BitonicLoops implements Runnable{
    private int startWire;
    private int endWire;
    private int total_length;
    private double[] data;


    private CyclicBarrier barrier;

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

    public BitonicLoops(int start, int end, CyclicBarrier barrier, double[] data, int size) {
        this.startWire = start;
        this.endWire = end;
        this.barrier = barrier;
        this.data = data;
        this.total_length = size;
    }

    private void sort() throws BrokenBarrierException, InterruptedException {
        for (int k = 2 ; k <= total_length ; k *=2) {
            for(int j = k/2 ; j>0 ; j /=2) {
                barrier.await();
                for(int i = startWire ; i<endWire ; i++) {
                    int ixj = i ^j;
                    if (ixj > i) {
                        compare(i, k, data, ixj);
                    }
                }
            }
        }
    }

    private void compare(int i, int k, double[] data, int ixj) {
        if (((i & k) == 0 && data[i] > data[ixj]) || ((i & k) != 0 && data[i] < data[ixj])) {
            double temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}
