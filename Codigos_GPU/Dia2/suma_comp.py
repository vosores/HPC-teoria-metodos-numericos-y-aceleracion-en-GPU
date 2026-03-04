import numpy as np
import cupy as cp
import time
from numba import cuda, float32

@cuda.jit
def reduce_sum(x, out, start, ncount):
    sm = cuda.shared.array(512, dtype=float32)
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    bid  = cuda.blockIdx.x

    k = bid * (2 * bdim) + tid

    s = float32(0.0)

    i = start + k
    if k < ncount:
        s += x[i]

    k2 = k + bdim
    j = start + k2
    if k2 < ncount:
        s += x[j]

    sm[tid] = s
    cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            sm[tid] += sm[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out[bid] = sm[0]


def sum_device(x_dev, start=0, ncount=None, tpb=256):
    if ncount is None:
        ncount = x_dev.size - start

    blocks = (ncount + (2 * tpb) - 1) // (2 * tpb)
    curr = cuda.device_array(shape=blocks, dtype=np.float32)

    reduce_sum[blocks, tpb](x_dev, curr, start, ncount)

    curr_n = blocks
    while curr_n > 1:
        blocks = (curr_n + (2 * tpb) - 1) // (2 * tpb)
        nxt = cuda.device_array(shape=blocks, dtype=np.float32)
        reduce_sum[blocks, tpb](curr, nxt, 0, curr_n)
        curr = nxt
        curr_n = blocks

    cuda.synchronize()
    return float(curr.copy_to_host()[0])


def time_gpu_sum_numba(x_dev, start=0, ncount=None, tpb=256, K=50):
    _ = sum_device(x_dev, start, ncount, tpb)  # warm-up

    start_ev = cuda.event()
    end_ev = cuda.event()

    start_ev.record()
    for _ in range(K):
        val = sum_device(x_dev, start, ncount, tpb)
    end_ev.record()
    end_ev.synchronize()

    ms_total = cuda.event_elapsed_time(start_ev, end_ev)
    return val, (ms_total / K) / 1000.0

def time_gpu_sum_cupy(x_dev, start=0, ncount=None, K=50):
    u_cp = cp.asarray(x_dev)
    if ncount is None:
        ncount = u_cp.size - start

    _ = cp.sum(u_cp[start:start+ncount])
    cp.cuda.Stream.null.synchronize()

    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()

    start_ev.record()
    for _ in range(K):
        s = cp.sum(u_cp[start:start+ncount])
    end_ev.record()
    end_ev.synchronize()

    ms_total = cp.cuda.get_elapsed_time(start_ev, end_ev)
    return float(s.get()), (ms_total / K) / 1000.0

def time_cpu_sum_numpy(x_host, start=0, ncount=None, K=10):
    if ncount is None:
        ncount = x_host.size - start

    sl = x_host[start:start+ncount]
    _ = np.sum(sl)

    t0 = time.perf_counter()
    for _ in range(K):
        val = float(np.sum(sl))
    t1 = time.perf_counter()

    return val, (t1 - t0) / K

for N in [1_000_000, 10_000_000, 100_000_000]:
    start = 0
    ncount = N - start
    tpb = 256

    x_host = np.random.randn(N).astype(np.float32)
    x_dev = cuda.to_device(x_host)

    K_gpu = 50
    K_cpu = 50

    val_numba, t_numba = time_gpu_sum_numba(x_dev, start, ncount, tpb, K_gpu)
    val_cupy,  t_cupy  = time_gpu_sum_cupy(x_dev, start, ncount, K_gpu)
    val_numpy, t_numpy = time_cpu_sum_numpy(x_host, start, ncount, K_cpu)

    print("\nsum(x[start:start+ncount])")
    print(f"Numba GPU : {val_numba:.8e} | {t_numba*1e3:8.3f} ms")
    print(f"CuPy  GPU : {val_cupy :.8e} | {t_cupy *1e3:8.3f} ms")
    print(f"NumPy CPU : {val_numpy:.8e} | {t_numpy*1e3:8.3f} ms")

    print("\nDiferencias:")
    print("Numba vs NumPy:", abs(val_numba - val_numpy))