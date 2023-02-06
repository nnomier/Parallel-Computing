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

    private void sort() throws BrokenBarrierException, InterruptedException {
        for (int k = 2; k <= total_length; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                checkBarriers(j);
                for (int i = startWire; i < endWire; i++) {
                    int ixj = i ^ j;
                    if (ixj > i) {
                        compare(i, k, data, ixj);
                    }
                }
            }
        }
    }

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
//        if(j<4) {
//            barriers[threadBarrierMap.get(threadId)].await();
//        } else if (j==4) {
//            if(threadId < 4) barriers[1].await();
//            else barriers[2].await();
//        } else {
//            barriers[0].await();
//        }
//        if(j>4) {
//            barriers[0].await();
//        } else {
//            barriers[threadBarrierMap.get(threadId)+1].await();
//        }
    }

    private void compare(int i, int k, double[] data, int ixj) {
        if (((i & k) == 0 && data[i] > data[ixj]) || ((i & k) != 0 && data[i] < data[ixj])) {
            double temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}
