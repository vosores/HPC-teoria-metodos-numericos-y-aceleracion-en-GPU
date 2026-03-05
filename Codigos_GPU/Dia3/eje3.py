import numpy as np
import time
from numba import cuda

def matvec_py(A, x, y):
    m = A.shape[0]
    n = A.shape[1]
    for i in range(m):
        s = 0.0
        for j in range(n):
            s = s + A[i, j] * x[j]
        y[i] = s


@cuda.jit
def matvec_rowmajor(A, x, y):
    i = cuda.grid(1)
    m = A.shape[0]
    n = A.shape[1]
    if i < m:
        s = 0.0
        for j in range(n):
            s = s + A[i, j] * x[j]
        y[i] = s

n = 10000
A = np.random.rand(n, n).astype(np.float32)
x = np.random.rand(n).astype(np.float32)
print("A MB:", A.nbytes / 1024**2)
print("x MB:", x.nbytes / 1024**2)
y = np.empty(n, dtype=np.float32)

d_A = cuda.to_device(A)
d_x = cuda.to_device(x)
d_y = cuda.device_array_like(y)

threads = 256
blocks = (n + threads - 1) // threads
# GPU warm-up (compila kernel)
matvec_rowmajor[blocks, threads](d_A, d_x, d_y)
cuda.synchronize()
d_y.copy_to_host(y)

# GPU timing (solo kernel)
t0 = time.perf_counter()
matvec_rowmajor[blocks, threads](d_A, d_x, d_y)
cuda.synchronize()
t1 = time.perf_counter()
gpu_time = t1 - t0
print("Resultado (GPU):", y[:5])
print("Tiempo GPU:", gpu_time)

# CPU warm-up
matvec_py(A, x, y)

t0 = time.perf_counter()
matvec_py(A, x, y)
t1 = time.perf_counter()
print("Resultado (CPU):", y[:5])
cpu_time = t1 - t0
print("Tiempo CPU:", cpu_time)

print("Speedup GPU vs CPU:", cpu_time / gpu_time)