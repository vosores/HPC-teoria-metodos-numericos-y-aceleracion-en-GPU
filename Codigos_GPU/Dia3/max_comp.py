import numpy as np
import cupy as cp
import time
from numba import cuda, float32

# Reducción GPU: max(abs(x))
@cuda.jit
def reduce_max_abs(x, out, start, ncount):
    sm = cuda.shared.array(512, dtype=float32) # aqui uso shared memory para la reducción dentro del bloque
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    bid  = cuda.blockIdx.x

    k = bid * (2 * bdim) + tid

    m = float32(0.0)

    i = start + k
    if k < ncount:
        m = abs(x[i])

    k2 = k + bdim
    j = start + k2
    if k2 < ncount:
        v = abs(x[j])
        if v > m:
            m = v

    sm[tid] = m
    cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            v = sm[tid + stride]
            if v > sm[tid]:
                sm[tid] = v
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out[bid] = sm[0]


def max_abs_device(x_dev, start=0, ncount=None, tpb=256):
    if ncount is None:
        ncount = x_dev.size - start

    blocks = (ncount + (2 * tpb) - 1) // (2 * tpb)
    curr = cuda.device_array(shape=blocks, dtype=np.float32)

    reduce_max_abs[blocks, tpb](x_dev, curr, start, ncount)

    curr_n = blocks
    while curr_n > 1:
        blocks = (curr_n + (2 * tpb) - 1) // (2 * tpb)
        nxt = cuda.device_array(shape=blocks, dtype=np.float32)
        reduce_max_abs[blocks, tpb](curr, nxt, 0, curr_n)
        curr = nxt
        curr_n = blocks

    cuda.synchronize()
    return float(curr.copy_to_host()[0])


def time_gpu_max_abs_numba(x_dev, start=0, ncount=None, tpb=256, K=50):
    _ = max_abs_device(x_dev, start, ncount, tpb)  # warm-up

    start_ev = cuda.event()
    end_ev = cuda.event()

    start_ev.record()
    for _ in range(K):
        val = max_abs_device(x_dev, start, ncount, tpb)
    end_ev.record()
    end_ev.synchronize()

    ms_total = cuda.event_elapsed_time(start_ev, end_ev)
    return val, (ms_total / K) / 1000.0



# CuPy GPU: max(abs(x))
def time_gpu_max_abs_cupy(x_dev, start=0, ncount=None, K=50):
    u_cp = cp.asarray(x_dev)
    if ncount is None:
        ncount = u_cp.size - start

    _ = cp.max(cp.abs(u_cp[start:start+ncount]))
    cp.cuda.Stream.null.synchronize()

    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()

    start_ev.record()
    for _ in range(K):
        m = cp.max(cp.abs(u_cp[start:start+ncount]))
    end_ev.record()
    end_ev.synchronize()

    ms_total = cp.cuda.get_elapsed_time(start_ev, end_ev)
    return float(m.get()), (ms_total / K) / 1000.0



# NumPy CPU
def time_cpu_max_abs_numpy(x_host, start=0, ncount=None, K=10):
    if ncount is None:
        ncount = x_host.size - start

    sl = x_host[start:start+ncount]
    _ = np.max(np.abs(sl))

    t0 = time.perf_counter()
    for _ in range(K):
        val = float(np.max(np.abs(sl)))
    t1 = time.perf_counter()

    return val, (t1 - t0) / K


# ============================================================
# Benchmark
# ============================================================

for N in [1000000, 10000000, 100000000]:
    start = 0
    ncount = N - start
    tpb = 256

    x_host = np.random.randn(N).astype(np.float32)
    print("x:", x_host)
    x_dev = cuda.to_device(x_host)

    K_gpu = 50
    K_cpu = 50

    val_numba, t_numba = time_gpu_max_abs_numba(x_dev, start, ncount, tpb, K_gpu)
    val_cupy,  t_cupy  = time_gpu_max_abs_cupy(x_dev, start, ncount, K_gpu)
    val_numpy, t_numpy = time_cpu_max_abs_numpy(x_host, start, ncount, K_cpu)

    print("\nmax(abs(x[start:start+ncount]))")
    print(f"Numba GPU : {val_numba:.8e} | {t_numba*1e3:8.3f} ms")
    print(f"CuPy  GPU : {val_cupy :.8e} | {t_cupy *1e3:8.3f} ms")
    print(f"NumPy CPU : {val_numpy:.8e} | {t_numpy*1e3:8.3f} ms")

    print("\nDiferencias:")
    print("Numba vs NumPy:", abs(val_numba - val_numpy))
